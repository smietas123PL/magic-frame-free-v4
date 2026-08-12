# Magic Frame Free v6.1

Darmowa aplikacja realtime pod Vercel: Vite + MediaPipe Hand Landmarker + Face Landmarker.

## Zmiany v6.1

- ramka aktywuje się natychmiast po wykryciu 2 dłoni,
- brak wymogu gestu L i wyprostowanych palców,
- status pokazuje `0/2`, `1/2` lub `2/2 dłonie`,
- mocniejsze wygładzanie ruchu ramki,
- 550 ms podtrzymania ramki przy chwilowym zgubieniu dłoni,
- zachowane efekty i Face Landmarker z v6,
- brak płatnych API.

## Lokalnie

```powershell
npm install
npm run dev
```

## Build / Vercel

```powershell
npm install
npm run build
vercel --prod
```

Build kopiuje MediaPipe WASM z npm oraz pobiera oficjalne modele dłoni i twarzy do `public/models`.
