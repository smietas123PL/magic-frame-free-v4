# Magic Frame Free v7.1

Darmowy prototyp realtime pod Vercel: kamera + MediaPipe Hand/Face Landmarker + stabilna Virtual Frame + lokalny WebGL toon renderer.

## Co zmienia v7
- ramka nie jest już surowym quadem z 4 fingertipów,
- dłonie sterują pozycją, szerokością i obrotem stabilnego wirtualnego okna,
- wysokość jest stabilizowana i ograniczana sensownymi proporcjami,
- 700 ms hold przy chwilowym zgubieniu dłoni,
- WebGL toon renderer: Anime, Comic, Clay, Cyberpunk, B&W,
- Face Landmarker wzmacnia oczy/kontur dla efektów twarzy,
- Debug pokazuje landmarki, uchwyty, oś virtual frame i maskę twarzy.

## Lokalnie
npm install
npm run dev

## Build / Vercel
npm run build

Build kopiuje WASM MediaPipe i pobiera modele Hand Landmarker + Face Landmarker do public/models.


## v7.1 — aggressive / low-latency tracking

- Hand tracking target: 30 detections/s.
- Camera requests up to 60 FPS when the device supports it.
- Virtual-frame position uses 72–96% of the newest measurement depending on motion speed.
- Angle follows at 88% per update.
- Size follows at 68% with only a tiny dead-zone for jitter.
- Dropout hold reduced from 700 ms to 120 ms.
- Short motion prediction compensates roughly half a detection frame.
- Debug mode: cyan dashed quad = raw MediaPipe-derived frame; solid frame = final rendered frame.
