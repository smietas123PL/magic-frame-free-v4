import { mkdir, cp, access, writeFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const mediapipeSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const mediapipeTarget = path.join(root, 'public', 'mediapipe', 'wasm');
const ortSource = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
const ortTarget = path.join(root, 'public', 'ort');

const assets = [
  {
    target: path.join(root, 'public', 'models', 'hand_landmarker.task'),
    url: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    label: 'MediaPipe Hand Landmarker'
  },
  {
    target: path.join(root, 'public', 'models', 'face_paint_512_v2_0.onnx'),
    url: 'https://huggingface.co/akhaliq/AnimeGANv2-ONNX/resolve/9d1a763dc816409bdf940e6eba51759d79679115/face_paint_512_v2_0.onnx?download=true',
    label: 'AnimeGANv2 Face Portrait v2 ONNX'
  }
];

async function exists(file) {
  try { await access(file); return true; } catch { return false; }
}

async function downloadAsset(asset) {
  await mkdir(path.dirname(asset.target), { recursive: true });
  if (await exists(asset.target)) {
    console.log(`[assets] ${asset.label} already present.`);
    return;
  }
  console.log(`[assets] Downloading ${asset.label}...`);
  const response = await fetch(asset.url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`${asset.label} download failed: HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  await writeFile(asset.target, bytes);
  console.log(`[assets] ${asset.label} saved (${(bytes.byteLength / 1024 / 1024).toFixed(1)} MB).`);
}

await mkdir(mediapipeTarget, { recursive: true });
console.log('[assets] Copying MediaPipe WASM...');
await cp(mediapipeSource, mediapipeTarget, { recursive: true, force: true });

await mkdir(ortTarget, { recursive: true });
console.log('[assets] Copying ONNX Runtime Web runtime files...');
for (const name of await readdir(ortSource)) {
  if (name.startsWith('ort-wasm') || name.endsWith('.wasm')) {
    await cp(path.join(ortSource, name), path.join(ortTarget, name), { force: true });
  }
}

for (const asset of assets) await downloadAsset(asset);
