# Magic Frame Free v10.3.2 — Anime Reconstruction

Realtime browser effect for Vercel. v10.3.2 keeps the freeform/self-crossing hand frame and replaces the v10.2 comic-like rendering with a reconstruction-first Anime Live pipeline.

## What changed

- aggressive photographic micro-detail suppression before stylization
- flatter skin reconstruction with fewer tonal bands
- semantic/blurred edge detection to avoid beard/skin noise
- simplified hair masses and anime color grading
- selective face line-art only (eyes, brows, mouth, short nose hint, silhouette)
- anime irises/pupils/catch-lights
- same low-latency hand tracking and self-crossing `|><|` geometry
- optional CartoonGAN remains for comparison only

## Build

```bash
npm install
npm run build
```

Vercel builds the project with `npm run build`.
