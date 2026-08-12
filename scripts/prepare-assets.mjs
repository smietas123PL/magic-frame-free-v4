import { mkdir, cp, access, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmTarget = path.join(root, 'public', 'mediapipe', 'wasm');

const models = [
  {
    target: path.join(root, 'public', 'models', 'hand_landmarker.task'),
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    label: 'Hand Landmarker'
  },
  {
    target: path.join(root, 'public', 'models', 'face_landmarker.task'),
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    label: 'Face Landmarker'
  }
];

async function exists(file) {
  try { await access(file); return true; } catch { return false; }
}

async function ensureModel(model) {
  await mkdir(path.dirname(model.target), { recursive: true });
  if (await exists(model.target)) {
    console.log(`[assets] ${model.label} already present.`);
    return;
  }
  console.log(`[assets] Downloading official ${model.label} model...`);
  const response = await fetch(model.url);
  if (!response.ok) throw new Error(`${model.label} download failed: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(model.target, bytes);
  console.log(`[assets] ${model.label} saved (${Math.round(bytes.byteLength / 1024 / 1024)} MB).`);
}

await mkdir(wasmTarget, { recursive: true });
console.log('[assets] Copying MediaPipe WASM from node_modules...');
await cp(wasmSource, wasmTarget, { recursive: true, force: true });
for (const model of models) await ensureModel(model);
