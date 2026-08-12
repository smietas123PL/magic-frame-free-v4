# Magic Frame Free v7

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
