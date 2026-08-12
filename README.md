# Magic Frame Free v12.0.1 — Hybrid Anime Renderer

## Architecture
- Hand tracking / self-crossing quad: unchanged low-latency MediaPipe
- Whole scene: realtime WebGL2 anime renderer
- Face: local CartoonGAN Shinkai on a 96x96 face crop
- FaceMesh stabilization: the newest neural face result is translated, scaled and rotated every render frame to follow the live face
- Zero queue: while the worker is busy, old requests are dropped
- v11 synthetic face reconstruction remains the fallback while the neural model loads

The neural model is copied locally from `local-tfjs-models` during build; no paid API is used.

## v12.0.1 fix
- fixes startup TDZ crash (`Cannot access ... before initialization`) by delaying the first diagnostics render until after NeuralFace state declarations.
