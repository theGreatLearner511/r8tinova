# R8tinova Inference Service (Render)

Service kecil: terima gambar dari Laravel -> jalankan model TFLite -> balas JSON.
Tidak menyimpan apa pun. Filesystem Render free itu ephemeral.

## Deploy ke Render

1. Taruh model di `models/retina360.tflite` (pakai `retina360_float16.tflite`
   dari Step 6, di-rename). File ini HARUS ikut di repo/Git, karena
   filesystem Render hilang setiap restart/spin-down.
2. Push folder ini ke repo GitHub (private).
3. Render -> New -> Web Service -> pilih repo.
   - Runtime: Python (versi dibaca dari `.python-version` = 3.12)
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT`
   - Instance type: Free
4. Environment variables:
   - `R8_API_KEY` = string acak panjang (buat: `php -r "echo bin2hex(random_bytes(32));"`)
   - `R8_MODEL_VERSION` = `v1` (naikkan tiap ganti model)
5. Tes (ganti URL & key):
   ```bash
   curl https://NAMA.onrender.com/ping
   curl -H "X-API-Key: KEY" https://NAMA.onrender.com/selftest
   curl -H "X-API-Key: KEY" -F "image=@1_left.jpg" https://NAMA.onrender.com/predict
   ```

## Format respons /predict

```json
{
  "model_version": "v1",
  "inference_ms": 240,
  "top_finding": {"classCode": "D", "displayName": "Diabetes", "probability": 0.72},
  "results": [{"classCode": "N", "displayName": "Normal", "probability": 0.18}, "..."]
}
```

## Contoh sisi Laravel (Http client)

```php
use Illuminate\Support\Facades\Http;

$res = Http::withHeaders(['X-API-Key' => config('services.r8_inference.key')])
    ->timeout(90)            // cold start Render bisa ~1 menit
    ->retry(2, 2000)         // ulangi 2x kalau gagal sesaat
    ->attach('image', file_get_contents($path), 'retina.jpg')
    ->post(config('services.r8_inference.url') . '/predict');

if ($res->failed()) {
    // balas error rapi ke Flutter, jangan simpan record setengah jadi
}

$data = $res->json();
// simpan: $data['model_version'], $data['top_finding'], $data['results'], $data['inference_ms']
```

`config/services.php`:
```php
'r8_inference' => [
    'url' => env('R8_INFERENCE_URL'),   // https://NAMA.onrender.com
    'key' => env('R8_INFERENCE_KEY'),
],
```

## Catatan

- Jangan panggil Render langsung dari Flutter: key akan bocor di APK.
  Selalu lewat Laravel.
- Simpan foto di disk PRIVATE Laravel (bukan `public`), sajikan lewat
  route ber-auth. Foto retina = data kesehatan.
- Jangan log isi gambar/hasil di Render.
- Ganti model: ganti file `models/retina360.tflite`, naikkan
  `R8_MODEL_VERSION`, push. Riwayat lama tetap tercatat versi model-nya.
