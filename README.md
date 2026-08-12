# Magic Frame Free v7.2

Darmowy prototyp realtime pod Vercel: kamera + MediaPipe Hand/Face Landmarker + freeform/self-crossing quad + lokalny WebGL toon renderer.

## Co zmienia v7.2

- P0 = lewy index tip, P1 = prawy index tip, P2 = prawy thumb tip, P3 = lewy thumb tip.
- Punkty nie są sortowane po obwodzie i nie są prostowane do prostokąta.
- Dozwolone są quady wypukłe, wklęsłe i samoprzecinające (bow-tie / `|><|`).
- Każdy narożnik ma osobny ultra-lekki filtr: 88–100% nowego pomiaru zależnie od szybkości ruchu.
- Krótka predykcja ruchu kompensuje część jednej klatki opóźnienia.
- Dropout hold skrócony do 90 ms.
- Efekt jest kompozytowany jako dwa stałe trójkąty, więc przecięcie krawędzi nie jest automatycznie naprawiane.
- Tożsamość lewej/prawej dłoni jest podtrzymywana także podczas krzyżowania rąk.
- Debug pokazuje raw P0–P3 (turkus, przerywana linia) i finalny quad (niebieski).
- Hand tracking celuje w 30 detekcji/s, kamera prosi o maks. 60 FPS.

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
