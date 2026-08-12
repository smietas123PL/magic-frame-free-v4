# Magic Frame Free v10.2 — Whole-scene Anime

Cel tej wersji: podnieść podobieństwo do referencji bez ponownego wprowadzania laga GAN-a.

## Co zmieniono
- cały obszar ramki dostaje anime treatment, nie tylko twarz,
- edge-preserving smoothing (bilateral-like),
- wieloskalowy Sobel + czystszy ink,
- 4–7 poziomów tonalnych i cel-shading,
- heurystyczne uproszczenie skóry i ciemnych partii włosów,
- mocniejszy face warp oraz stylizowane iris/catch-light,
- clean under-stroke pod line-artem twarzy,
- FX canvas ograniczony do 760 px szerokości dla płynności,
- tracking/freeform/self-crossing `|><|` pozostaje bez zmian,
- CartoonGAN zostaje wyłącznie jako opcjonalne AI Quality.

## Tryby
- Anime Live · Cinematic — domyślny balans,
- Anime Live · Soft — subtelny,
- Anime Live · Strong / Manga — mocniejszy line-art i płaskie cienie,
- AI Quality · Shinkai/Paprika — wolny wariant porównawczy.

## Vercel
`npm install && npm run build`, output `dist`.
