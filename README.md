# Magic Frame Free v9.3 — non-blocking CartoonGAN Worker

Wersja v9.3 izoluje TensorFlow.js/CartoonGAN od głównego wątku przez Web Worker.

## Najważniejsze zmiany
- MediaPipe Hands i requestAnimationFrame nie czekają na GAN.
- CartoonGAN działa w `src/cartoon-worker.js`.
- WebGPU jest używany w Workerze, z fallbackiem do CPU.
- `executeAsync()` + `tensor.data()` zamiast `tf.browser.toPixels()` na main thread.
- latest-frame-wins: brak kolejki starych klatek.
- dynamiczne 160/192/224 px.
- telemetry: AI total, compute, readback, Hands FPS, Render FPS.

## Uruchomienie
```powershell
npm install
npm run dev
```

## Build Vercel
```powershell
npm run build
vercel --prod --yes
```
