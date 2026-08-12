# Magic Frame Free v9 — AnimeGANv2 Face Portrait

Darmowy prototyp realtime: kamera + MediaPipe Hands + freeform/self-crossing quad + prawdziwy model AnimeGANv2 uruchamiany lokalnie w przeglądarce przez ONNX Runtime Web.

## Co zmieniło się względem v8.1

- Usunięty ręcznie robiony „anime shader”.
- Dodany gotowy model **AnimeGANv2 Face Portrait v2** (`face_paint_512_v2_0.onnx`, ok. 8.6 MB).
- ONNX Runtime Web używa **WebGPU** na Edge/Chrome, jeśli model i GPU to obsługują.
- Automatyczny fallback do **WASM**, jeśli WebGPU nie przejdzie przy ładowaniu lub pierwszym inference.
- Model pracuje asynchronicznie i tylko wtedy, gdy ramka jest aktywna. Tracking dłoni pozostaje niezależny i celuje w 30 FPS.
- Freeform/self-crossing `|><|`, outlier rejection i szybki tracking pozostają z v7.2/v8.
- Face Landmarker został celowo usunięty, żeby nie konkurował z AnimeGAN o GPU.

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

Podczas `npm run build` skrypt pobiera lokalne assety:
- oficjalny MediaPipe Hand Landmarker,
- AnimeGANv2 Face Portrait v2 ONNX,
- pliki WASM MediaPipe i ONNX Runtime z `node_modules`.

Po deploymencie inference odbywa się na urządzeniu użytkownika — nie ma płatnego API ani wysyłania klatek do backendu.

## Wydajność

Tracking dłoni działa niezależnie od AI. AnimeGAN jest modelem 512×512, więc szybkość stylizacji zależy od GPU/CPU użytkownika. Aplikacja nie kolejkuje inferencji — zawsze pokazuje ostatnią gotową klatkę anime i aktualizuje ją tak szybko, jak pozwala urządzenie.

## Źródła / licencje

- Tracking: Google MediaPipe Tasks Vision.
- Runtime: Microsoft ONNX Runtime Web (MIT).
- Model/architektura: AnimeGANv2 / `bryandlee/animegan2-pytorch` Face Portrait v2 oraz ONNX conversion udostępnione publicznie.

Przed komercyjnym wykorzystaniem zweryfikuj osobno prawa do użytych wag modelu i danych treningowych. Ta paczka jest przygotowana do prototypowania/testów.
