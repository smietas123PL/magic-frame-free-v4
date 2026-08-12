# Magic Frame Free v5

Wersja Vite + npm przygotowana pod Vercel. Nie używa dynamicznego importu MediaPipe z jsDelivr.

## Co jest lokalne po deploymencie

- JavaScript MediaPipe jest bundlowany przez Vite z `@mediapipe/tasks-vision`.
- pliki WASM są kopiowane z `node_modules` do `public/mediapipe/wasm` podczas builda,
- oficjalny model `hand_landmarker.task` jest pobierany podczas builda i trafia do `public/models`,
- w czasie działania przeglądarka pobiera wszystko z Twojej domeny Vercel.

## Lokalnie (PowerShell)

```powershell
npm install
npm run dev
```

Otwórz adres pokazany przez Vite, zwykle `http://localhost:5173`.

## Vercel

Po wrzuceniu repo do GitHub zaimportuj projekt w Vercel. Vercel wykryje Vite. Build command: `npm run build`, output: `dist`.

Możesz też użyć CLI:

```powershell
npm install
npm run build
vercel --prod
```

## Diagnostyka

Na dole obrazu jest pasek:

`JS: OK · kamera: ... · tracker: ...`

Dzięki temu od razu wiadomo, czy problem dotyczy kamery, WASM, modelu czy inicjalizacji trackera.


## Zmiany v5
- luźniejsza detekcja gestu: dwie dłonie + wyciągnięty kciuk i wskazujący
- brak wymogu idealnego kąta 90°
- 260 ms podtrzymania ramki przy krótkim zgubieniu landmarków
- szybszy tracking (do ok. 24 detekcji/s)
