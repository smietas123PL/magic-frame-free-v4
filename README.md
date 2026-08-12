# Magic Frame Free v8.1 — Anime 2D

Darmowy prototyp realtime pod Vercel: kamera + MediaPipe Hand/Face Landmarker + freeform/self-crossing quad + lokalny WebGL2 anime renderer.

## Co zmienia v8.1

- Tracking/freeform z v8 pozostaje bez przebudowy.
- Mocniejszy face-aware warp: większe oczy, węższa żuchwa i krótszy podbródek.
- Silniejsze wygładzanie twarzy, żeby usuwać fotograficzną mikrostrukturę skóry i zarostu.
- Ograniczona, 5-stopniowa paleta barw i jaśniejsze anime midtones.
- Ciemne obszary włosów/zarostu są scalane w większe graficzne plamy zamiast zachowywania pojedynczych włosków.
- Dedykowany line-art z Face Landmarker: kontur twarzy, brwi, oczy, nos i usta.
- Dodatkowe anime iris/catch-light na oczach.
- Self-crossing quad nadal jest dozwolony.
- Outlier rejection narożników pozostaje aktywny.
- Nagrywanie preferuje MP4/H.264 tam, gdzie MediaRecorder przeglądarki to obsługuje, z fallbackiem WebM.

## Lokalnie

```powershell
npm install
npm run dev
```

## Build / Vercel

```powershell
npm run build
```

Build kopiuje WASM MediaPipe z npm i pobiera oficjalne modele Hand Landmarker oraz Face Landmarker do `public/models`.

Nie jest potrzebny żaden płatny API key.
