# Magic Frame Free v10 — Anime Live

Domyślny tryb `Anime Live` nie używa GAN-a. Efekt wykonywany jest w WebGL2 lokalnie na GPU: smoothing, posterization, edge ink, grading i face-aware warp. CartoonGAN Light pozostaje jako opcjonalny tryb `AI Quality`.

- MediaPipe Hands: freeform/self-crossing quad
- MediaPipe Face Landmarker: lekkie face-aware warp dla Live Anime
- WebGL2: realtime anime renderer
- CartoonGAN TFJS Worker: opcjonalny quality mode
- Runtime Cache Storage: modele/WASM po pierwszym użyciu zostają lokalnie w przeglądarce
