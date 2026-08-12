import { mkdir, cp, access, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
async function exists(file) { try { await access(file); return true; } catch { return false; } }

const wasmSrc = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmDst = path.join(root, 'public', 'mediapipe', 'wasm');
await mkdir(wasmDst, { recursive: true });
await cp(wasmSrc, wasmDst, { recursive: true, force: true });

const cartoonRoot = path.join(root, 'node_modules', 'local-tfjs-models', 'cartoon-GAN');
for (const style of ['shinkai', 'paprika']) {
  const src = path.join(cartoonRoot, style);
  const dst = path.join(root, 'public', 'models', `cartoongan-${style}`);
  if (!(await exists(path.join(src, 'model.json')))) throw new Error(`Brak CartoonGAN ${style}`);
  await mkdir(dst, { recursive: true });
  await cp(src, dst, { recursive: true, force: true });
}

async function ensureModel(target, url, label) {
  if (await exists(target)) return;
  console.log(`[assets] Downloading ${label}...`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${label}: HTTP ${res.status}`);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, new Uint8Array(await res.arrayBuffer()));
}
await ensureModel(path.join(root,'public','models','hand_landmarker.task'), 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'Hand Landmarker');
await ensureModel(path.join(root,'public','models','face_landmarker.task'), 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'Face Landmarker');
