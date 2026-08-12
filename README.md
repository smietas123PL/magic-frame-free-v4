# Magic Frame Free v9.4 — non-blocking CartoonGAN Worker

Wersja v9.4 izoluje TensorFlow.js/CartoonGAN od głównego wątku przez Web Worker.

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


## v9.4 Speed AI
- Windows/Edge: prefer WebGL over WebGPU for CartoonGAN TFJS.
- GraphModel uses execute() instead of executeAsync().
- Adaptive inference: 96/128/160 px, starting at 128 px.
- Web Worker + latest-frame-wins preserved.
