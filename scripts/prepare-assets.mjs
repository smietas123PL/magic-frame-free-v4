import { mkdir, cp, access, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmTarget = path.join(root, 'public', 'mediapipe', 'wasm');
const modelTarget = path.join(root, 'public', 'models', 'hand_landmarker.task');
const modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

async function exists(file) {
  try { await access(file); return true; } catch { return false; }
}

await mkdir(wasmTarget, { recursive: true });
await mkdir(path.dirname(modelTarget), { recursive: true });

console.log('[assets] Copying MediaPipe WASM from node_modules...');
await cp(wasmSource, wasmTarget, { recursive: true, force: true });

if (!(await exists(modelTarget))) {
  console.log('[assets] Downloading official Hand Landmarker model...');
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`Model download failed: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(modelTarget, bytes);
  console.log(`[assets] Model saved (${Math.round(bytes.byteLength / 1024 / 1024)} MB).`);
} else {
  console.log('[assets] Hand model already present.');
}
