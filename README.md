# Magic Frame Free v9.2 — CartoonGAN Light / TensorFlow.js

Wersja v9.2 wymienia wolny AnimeGAN ONNX na lekki CartoonGAN przeznaczony do przeglądarki.

## Co zostało zachowane

- MediaPipe Hand Landmarker 30 FPS
- freeform quad z P0–P3
- self-crossing `|><|`
- low-latency smoothing i outlier rejection
- latest-frame-wins (bez kolejki starych klatek)
- nagrywanie canvas

## Nowy silnik anime

- TensorFlow.js GraphModel
- CartoonGAN Light Shinkai / Paprika
- lokalne modele kopiowane z `local-tfjs-models@0.0.3` podczas builda
- automatyczny benchmark WebGPU vs WebGL
- adaptacyjna rozdzielczość inference: 160 / 192 / 224 px
- telemetria AI FPS / ms / hand FPS / render FPS

## Build

```powershell
npm install
npm run build
```

`prepare-assets.mjs` kopiuje MediaPipe WASM oraz modele CartoonGAN do `public/`, więc po deploymencie przeglądarka nie pobiera modelu z zewnętrznego CDN.

## Vercel

Projekt jest zwykłym Vite SPA. `vercel --prod --yes` wystarczy, jeśli folder jest już połączony z istniejącym projektem Vercel.

## Pochodzenie modeli

CartoonGAN TFJS assets: `local-tfjs-models`, wywodzące się z `mnicnc404/CartoonGan-tensorflow`.
