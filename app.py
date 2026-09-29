"""
R8tinova - Inference service (dijalankan di Render).

Perannya SATU saja: terima gambar retina dari Laravel, jalankan model
TFLite, kembalikan JSON. Tidak menyimpan apa pun (filesystem Render free
bersifat ephemeral), tidak punya user/login/database.

Endpoint:
  GET  /ping      -> "ok", ringan, TANPA auth (untuk uptime monitor)
  GET  /health    -> info lingkungan & model  (butuh X-API-Key)
  GET  /selftest  -> 1 inference dummy + waktu (butuh X-API-Key)
  POST /predict   -> form-data "image" (butuh X-API-Key)

Environment variable:
  R8_API_KEY        WAJIB. Rahasia bersama dengan Laravel.
  R8_MODEL_VERSION  opsional, default "v1". Naikkan tiap ganti model.
  R8_MODEL_PATH     opsional, default models/retina360.tflite
"""

import os

# PENTING: batasi thread SEBELUM numpy di-import. Tanpa ini OpenBLAS
# mencoba membuat puluhan thread dan bisa macet di server dengan batas
# proses/CPU kecil (kasus yang terjadi di shared hosting).
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import hmac  # noqa: E402
import io  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402
from PIL import Image  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get(
    "R8_MODEL_PATH", os.path.join(BASE_DIR, "models", "retina360.tflite")
)
MODEL_VERSION = os.environ.get("R8_MODEL_VERSION", "v1")
API_KEY = os.environ.get("R8_API_KEY", "")

# HARUS sama persis urutannya dengan TARGET_CLASSES di src/config.py
CLASS_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {
    "N": "Normal",
    "D": "Diabetes",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "Age-related Macular Degeneration",
    "H": "Hypertension",
    "M": "Pathological Myopia",
    "O": "Other diseases/abnormalities",
}
IMAGE_SIZE = 224
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SIDE_BEFORE_PROCESS = 1600

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_interpreter = None
_lock = threading.Lock()  # Interpreter TFLite tidak thread-safe


def _authorized() -> bool:
    if not API_KEY:  # server salah konfigurasi -> tolak semua, jangan buka
        return False
    sent = request.headers.get("X-API-Key", "")
    return hmac.compare_digest(sent, API_KEY)


def _deny():
    return jsonify({"error": "Unauthorized"}), 401


def _peak_memory_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def get_interpreter():
    global _interpreter
    if _interpreter is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        from ai_edge_litert.interpreter import Interpreter

        interp = Interpreter(
            model_path=MODEL_PATH,
            num_threads=int(os.environ.get("R8_TFLITE_THREADS", "1")),
        )
        interp.allocate_tensors()
        _interpreter = interp
    return _interpreter


# ── Preprocessing (meniru src/preprocessing/image_processor.py) ──────────
def crop_retina(img: Image.Image) -> Image.Image:
    gray = np.asarray(img.convert("L"))
    mask = gray > 10
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    h, w = gray.shape
    if (x1 - x0 + 1) < w * 0.3 or (y1 - y0 + 1) < h * 0.3:
        return img
    return img.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE_BEFORE_PROCESS:
        img.thumbnail((MAX_SIDE_BEFORE_PROCESS, MAX_SIDE_BEFORE_PROCESS))
    img = crop_retina(img)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0


def run_model(x: np.ndarray) -> np.ndarray:
    with _lock:
        interp = get_interpreter()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        interp.set_tensor(inp["index"], x[np.newaxis, ...].astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out["index"])[0]


# ── Endpoint ──────────────────────────────────────────────────────────────
@app.get("/ping")
def ping():
    return "ok", 200


@app.get("/health")
def health():
    if not _authorized():
        return _deny()
    info = {
        "python": platform.python_version(),
        "libc": " ".join(platform.libc_ver()),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "model_version": MODEL_VERSION,
        "model_exists": os.path.exists(MODEL_PATH),
        "peak_memory_mb": _peak_memory_mb(),
    }
    try:
        import ai_edge_litert  # noqa: F401

        info["litert_import"] = "OK"
    except Exception as e:  # noqa: BLE001
        info["litert_import"] = f"GAGAL: {e}"
    return jsonify(info)


@app.get("/selftest")
def selftest():
    if not _authorized():
        return _deny()
    try:
        t0 = time.time()
        get_interpreter()
        load_s = time.time() - t0
        dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        t1 = time.time()
        probs = run_model(dummy)
        infer_s = time.time() - t1
        return jsonify(
            {
                "status": "OK",
                "model_version": MODEL_VERSION,
                "model_load_seconds": round(load_s, 3),
                "inference_seconds": round(infer_s, 3),
                "output_shape": list(probs.shape),
                "num_classes_expected": len(CLASS_CODES),
                "peak_memory_mb": _peak_memory_mb(),
            }
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"status": "GAGAL", "error": str(e)}), 500


@app.post("/predict")
def predict():
    if not _authorized():
        return _deny()

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "Kirim foto lewat form-data field 'image'."}), 400
    try:
        img = Image.open(io.BytesIO(file.read()))
        img.load()
    except Exception:  # noqa: BLE001
        return jsonify({"error": "File bukan gambar yang valid."}), 400

    try:
        t0 = time.time()
        probs = run_model(preprocess(img))
        elapsed_ms = int((time.time() - t0) * 1000)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Inference gagal: {e}"}), 500

    if len(probs) != len(CLASS_CODES):
        return (
            jsonify(
                {
                    "error": f"Model mengeluarkan {len(probs)} nilai, "
                    f"tapi CLASS_CODES berisi {len(CLASS_CODES)} kelas."
                }
            ),
            500,
        )

    results = [
        {
            "classCode": code,
            "displayName": CLASS_NAMES[code],
            "probability": float(p),
        }
        for code, p in zip(CLASS_CODES, probs)
    ]
    top = max((r for r in results if r["classCode"] != "N"), key=lambda r: r["probability"])
    return jsonify(
        {
            "model_version": MODEL_VERSION,
            "inference_ms": elapsed_ms,
            "top_finding": top,
            "results": results,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
