# Magic Frame Free v8 — Anime focus

Darmowa aplikacja webowa realtime: MediaPipe Hands + Face Landmarker + freeform/self-crossing quad + lokalny WebGL2 anime renderer.

## Co zmieniono w v8
- anime-only pipeline inspirowany klasycznym lookiem 2D: większe oczy, węższa szczęka, krótszy dół twarzy, skin smoothing, posterization i kontury,
- face-aware deformation wykonywana lokalnie w shaderze WebGL2,
- seam-free composite dla self-crossing quad: efekt jest rysowany raz przez maskę z dwóch trójkątów,
- outlier rejection narożników bez zwiększania bezwładności,
- nagrywanie samego canvas (bez UI) przez MediaRecorder,
- fallback Canvas, jeśli WebGL2 jest niedostępny.

## Uruchomienie
```powershell
npm install
npm run dev
```

## Build / Vercel
```powershell
npm install
npm run build
vercel --prod --yes
```

Modele MediaPipe są pobierane podczas `npm run build` przez `scripts/prepare-assets.mjs` i serwowane lokalnie z deploymentu.

## Ważne
To nie jest diffusion/video-to-video. Nie generuje od zera nowej twarzy anime. To darmowa, lokalna stylizacja i deformacja oparta o Face Landmarker + WebGL2, zaprojektowana pod małe opóźnienie.
