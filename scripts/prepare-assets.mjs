import { mkdir, cp, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

async function exists(file) { try { await access(file); return true; } catch { return false; } }

const mediapipeSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const mediapipeTarget = path.join(root, 'public', 'mediapipe', 'wasm');
await mkdir(mediapipeTarget, { recursive: true });
console.log('[assets] Copying MediaPipe WASM...');
await cp(mediapipeSource, mediapipeTarget, { recursive: true, force: true });

const modelRoot = path.join(root, 'node_modules', 'local-tfjs-models', 'cartoon-GAN');
for (const style of ['shinkai', 'paprika']) {
  const src = path.join(modelRoot, style);
  const dst = path.join(root, 'public', 'models', `cartoongan-${style}`);
  if (!(await exists(path.join(src, 'model.json')))) {
    throw new Error(`Brak modelu CartoonGAN ${style} w local-tfjs-models@0.0.3: ${src}`);
  }
  await mkdir(dst, { recursive: true });
  console.log(`[assets] Copying CartoonGAN Light ${style} TFJS...`);
  await cp(src, dst, { recursive: true, force: true });
}

const handTarget = path.join(root, 'public', 'models', 'hand_landmarker.task');
if (!(await exists(handTarget))) {
  console.log('[assets] Downloading MediaPipe Hand Landmarker...');
  const url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Hand Landmarker download failed: HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  await mkdir(path.dirname(handTarget), { recursive: true });
  const { writeFile } = await import('node:fs/promises');
  await writeFile(handTarget, bytes);
}
