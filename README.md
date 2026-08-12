# Magic Frame Free v9.1 — Performance

Darmowa webowa aplikacja realtime: MediaPipe Hands + freeform/self-crossing quad + AnimeGANv2 uruchamiany lokalnie przez ONNX Runtime Web.

## Co zmieniono w v9.1

- `Anime FAST 256 · Shinkai` jest trybem domyślnym i wykonuje inference na 256×256.
- `Anime QUALITY 512 · Face Portrait v2` pozostaje jako opcja jakościowa.
- Latest-frame-wins: aplikacja nie kolejkuje klatek AI.
- Tracking dłoni i compositing działają niezależnie od inference.
- Wynik AI jest mapowany do aktualnego położenia wielokąta, a nie położenia z momentu rozpoczęcia inference.
- Telemetria pokazuje backend, ms/klatkę AI, FPS AI, FPS renderera, FPS dłoni i rozdzielczość modelu.

## Lokalnie

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

Podczas `npm run build` skrypt pobiera modele i kopiuje runtime WASM do `public/`.

## Licencje modeli

Sprawdź warunki licencyjne AnimeGANv2 przed użyciem komercyjnym. Ten projekt jest przeznaczony do prototypowania i testów.
