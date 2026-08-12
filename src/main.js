import './styles.css';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as ort from 'onnxruntime-web/webgpu';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
const animeCanvas = document.getElementById('animeCanvas');
const animeCtx = animeCanvas.getContext('2d', { alpha: false });
const inputCanvas = document.getElementById('inputCanvas');
const inputCtx = inputCanvas.getContext('2d', { alpha: false, willReadFrequently: true });
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const effectSelect = document.getElementById('effectSelect');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');
const recBadge = document.getElementById('recBadge');

let handLandmarker = null;
let animeSession = null;
let animeInputName = null;
let animeOutputName = null;
let stream = null;
let running = false;
let handsReady = false;
let loadingHands = false;
let loadingAnime = false;
let animeReady = false;
let animeBusy = false;
let animeBackend = '—';
let animeState = '—';
let latestHands = [];
let latestHandedness = [];
let smoothedQuad = null;
let previousRawQuad = null;
let previousRawAt = 0;
let lastValidFrameAt = 0;
let lastHandDetect = 0;
let lastAnimeStart = 0;
let lastAnimeDone = 0;
let lastAnimeMs = 0;
let animeInterval = 0;
let selectedProfile = 'fast';
let modelSize = 256;
let modelLayout = 'nhwc';
let aiCompleted = 0;
let aiWindowStart = performance.now();
let aiFps = 0;
let handCompleted = 0;
let handWindowStart = performance.now();
let handFps = 0;
let renderCompleted = 0;
let renderWindowStart = performance.now();
let renderFps = 0;
let skippedBusy = 0;
let animeFrameValid = false;
let animeMapping = null;
let cameraState = '—';
let handState = '—';

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;

const HAND_INTERVAL = 1000 / 30;
const FRAME_HOLD_MS = 90;
const PROFILES = {
  fast: { label: 'FAST 256', size: 256, layout: 'nhwc', model: '/models/Shinkai_53.onnx' },
  quality: { label: 'QUALITY 512', size: 512, layout: 'nchw', model: '/models/face_paint_512_v2_0.onnx' }
};

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() {
  const perf = animeReady && lastAnimeMs ? ` ${Math.round(lastAnimeMs)}ms` : '';
  const fps = animeReady ? ` · AI ${aiFps.toFixed(1)}fps · H ${handFps.toFixed(0)}fps · R ${renderFps.toFixed(0)}fps · ${modelSize}px · skip ${skippedBusy}` : '';
  diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · AnimeGAN: ${animeState}${perf}${fps}`;
}
function showError(text) { errorEl.hidden = !text; errorEl.textContent = text || ''; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function readableCameraError(err) {
  const name = err?.name || 'Error';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Brak zgody na kamerę. Kliknij ikonę kamery przy adresie → Zezwalaj, potem odśwież.';
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'Nie znaleziono kamery.';
  if (name === 'NotReadableError' || name === 'TrackStartError') return 'Kamera jest zajęta przez inną aplikację.';
  return `${name}: ${err?.message || 'Nieznany błąd kamery'}`;
}

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Wymagany jest HTTPS lub localhost.');
  const attempts = [
    { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, max: 60 } }, audio: false },
    { video: { facingMode: 'user' }, audio: false },
    { video: true, audio: false }
  ];
  let lastError;
  for (const constraints of attempts) {
    try { return await navigator.mediaDevices.getUserMedia(constraints); }
    catch (err) { lastError = err; if (err?.name === 'NotAllowedError') break; }
  }
  throw lastError || new Error('Nie udało się uruchomić kamery.');
}

async function initHands() {
  if (loadingHands || handsReady) return;
  loadingHands = true;
  handState = 'WASM…'; updateDiag();
  try {
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    const common = {
      baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
      runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: 0.40,
      minHandPresenceConfidence: 0.40,
      minTrackingConfidence: 0.40
    };
    handState = 'model…'; updateDiag();
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, { ...common, baseOptions: { ...common.baseOptions, delegate: 'GPU' } });
      handState = 'GPU OK';
    } catch {
      handLandmarker = await HandLandmarker.createFromOptions(vision, common);
      handState = 'CPU OK';
    }
    handsReady = true; updateDiag();
  } catch (err) {
    console.error(err); handState = 'BŁĄD'; updateDiag();
    showError(`MediaPipe Hands nie wystartował.\n${err?.name || 'Error'}: ${err?.message || err}`);
  } finally { loadingHands = false; }
}

async function createAnimeSession(executionProviders, label, profile) {
  animeState = `${label} ${profile.label} load…`; updateDiag();
  const session = await ort.InferenceSession.create(profile.model, {
    executionProviders,
    graphOptimizationLevel: 'all'
  });
  animeBackend = label;
  return session;
}

async function initAnime(force = false) {
  const requested = effectSelect.value === 'quality' ? 'quality' : 'fast';
  if (!force && (loadingAnime || (animeReady && selectedProfile === requested))) return;
  if (loadingAnime) return;
  loadingAnime = true;
  animeReady = false;
  animeBusy = false;
  animeFrameValid = false;
  animeMapping = null;
  selectedProfile = requested;
  const profile = PROFILES[selectedProfile];
  modelSize = profile.size;
  modelLayout = profile.layout;
  animeCanvas.width = animeCanvas.height = modelSize;
  inputCanvas.width = inputCanvas.height = modelSize;
  animeState = `${profile.label} runtime…`; updateDiag();
  try {
    try { await animeSession?.release?.(); } catch {}
    animeSession = null;
    ort.env.wasm.wasmPaths = '/ort/';
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;

    if (navigator.gpu) {
      try {
        animeSession = await createAnimeSession(['webgpu'], 'WebGPU', profile);
      } catch (err) {
        console.warn('AnimeGAN WebGPU failed, switching to WASM', err);
      }
    }
    if (!animeSession) animeSession = await createAnimeSession(['wasm'], 'WASM', profile);

    animeInputName = animeSession.inputNames[0];
    animeOutputName = animeSession.outputNames[0];
    animeReady = true;
    animeState = `${animeBackend} ${profile.label} OK`;
    animeInterval = animeBackend === 'WebGPU' ? 0 : (selectedProfile === 'fast' ? 18 : 80);
    lastAnimeStart = 0;
    skippedBusy = 0;
    updateDiag();
  } catch (err) {
    console.error(err);
    animeState = 'BŁĄD'; updateDiag();
    showError(`AnimeGAN nie wystartował. Tracking dłoni nadal działa.\n${err?.name || 'Error'}: ${err?.message || err}`);
  } finally { loadingAnime = false; }
}

effectSelect.addEventListener('change', () => {
  if (effectSelect.value === 'original') {
    animeFrameValid = false;
    animeState = 'wyłączony';
    updateDiag();
    return;
  }
  initAnime(true);
});

async function startCamera() {
  if (running) return;
  showError(''); startBtn.disabled = true; startBtn.textContent = 'Uruchamianie…';
  cameraState = 'prośba…'; updateDiag(); setStatus('Prośba o dostęp do kamery…');
  try {
    stream = await requestCamera();
    video.srcObject = stream; video.muted = true; video.playsInline = true;
    await new Promise(resolve => {
      if (video.readyState >= 1) return resolve();
      video.addEventListener('loadedmetadata', resolve, { once: true });
      setTimeout(resolve, 1800);
    });
    await video.play().catch(() => {});
    running = true; cameraState = 'OK'; updateDiag();
    startBtn.textContent = 'Kamera działa'; recordBtn.disabled = false;
    resizeCanvas(); requestAnimationFrame(renderLoop);
    setTimeout(initHands, 20);
    setTimeout(() => initAnime(false), 80);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
recordBtn.addEventListener('click', toggleRecording);
setStatus('Gotowy • v9.1 Performance'); updateDiag();
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(t => t.stop()));

function resizeCanvas() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
}
function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z || 0 }; }

// ---------- semantic hand identity ----------
let semanticMemory = { leftWrist: null, rightWrist: null };
function semanticHands(hands, handedness) {
  const items = hands.map((pts, i) => ({
    label: handedness?.[i]?.[0]?.categoryName || '',
    pts: pts.map(mirrorPoint)
  }));
  if (items.length < 2) return items;

  let left = items.find(x => x.label.toLowerCase() === 'left');
  let right = items.find(x => x.label.toLowerCase() === 'right');
  if (!left || !right || left === right) {
    if (semanticMemory.leftWrist && semanticMemory.rightWrist) {
      const a = items[0], b = items[1];
      const costAB = dist(a.pts[0], semanticMemory.leftWrist) + dist(b.pts[0], semanticMemory.rightWrist);
      const costBA = dist(b.pts[0], semanticMemory.leftWrist) + dist(a.pts[0], semanticMemory.rightWrist);
      [left, right] = costAB <= costBA ? [a, b] : [b, a];
    } else {
      const byX = [...items].sort((a, b) => a.pts[0].x - b.pts[0].x);
      [left, right] = [byX[0], byX[1]];
    }
  }
  semanticMemory.leftWrist = { ...left.pts[0] };
  semanticMemory.rightWrist = { ...right.pts[0] };
  return [left, right];
}

function measureFreeformQuad(semantic) {
  if (semantic.length < 2) return null;
  const L = semantic[0]?.pts, R = semantic[1]?.pts;
  if (!L?.[4] || !L?.[8] || !R?.[4] || !R?.[8]) return null;
  const q = [{ ...L[8] }, { ...R[8] }, { ...R[4] }, { ...L[4] }];
  const xs = q.map(p => p.x), ys = q.map(p => p.y);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  if (span < 0.055) return null;
  return q;
}

function rejectCornerOutliers(next) {
  if (!previousRawQuad) return next.map(p => ({ ...p }));
  const jumps = next.map((p, i) => dist(p, previousRawQuad[i]));
  const sorted = [...jumps].sort((a, b) => a - b);
  const median = (sorted[1] + sorted[2]) / 2;
  return next.map((p, i) => {
    const isolatedSpike = jumps[i] > Math.max(0.17, median * 2.8 + 0.035);
    const hardTeleport = jumps[i] > 0.29;
    return isolatedSpike || hardTeleport ? { ...previousRawQuad[i] } : { ...p };
  });
}

function smoothFreeformQuad(next, now) {
  if (next) {
    lastValidFrameAt = now;
    const clean = rejectCornerOutliers(next);
    const target = clean.map(p => ({ x: p.x, y: p.y, z: p.z || 0 }));
    if (previousRawQuad && previousRawAt > 0) {
      const dt = Math.max(1, now - previousRawAt);
      const predictionMs = Math.min(10, dt * 0.28);
      const k = predictionMs / dt;
      for (let i = 0; i < 4; i++) {
        target[i].x = clamp(clean[i].x + (clean[i].x - previousRawQuad[i].x) * k, 0, 1);
        target[i].y = clamp(clean[i].y + (clean[i].y - previousRawQuad[i].y) * k, 0, 1);
      }
    }
    previousRawQuad = clean.map(p => ({ ...p })); previousRawAt = now;
    if (!smoothedQuad) { smoothedQuad = target; return smoothedQuad; }
    for (let i = 0; i < 4; i++) {
      const jump = dist(smoothedQuad[i], target[i]);
      const alpha = clamp(0.91 + jump * 2.4, 0.91, 1.0);
      smoothedQuad[i].x = lerp(smoothedQuad[i].x, target[i].x, alpha);
      smoothedQuad[i].y = lerp(smoothedQuad[i].y, target[i].y, alpha);
      smoothedQuad[i].z = target[i].z;
    }
    return smoothedQuad;
  }
  if (smoothedQuad && now - lastValidFrameAt <= FRAME_HOLD_MS) return smoothedQuad;
  smoothedQuad = null; previousRawQuad = null; previousRawAt = 0;
  return null;
}

function triangleSet(q) { return [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]; }
function triangleUnionPath(q) {
  const p = new Path2D();
  for (const tri of triangleSet(q)) {
    const t = tri.map(v => ({ x: v.x * canvas.width, y: v.y * canvas.height }));
    p.moveTo(t[0].x, t[0].y); p.lineTo(t[1].x, t[1].y); p.lineTo(t[2].x, t[2].y); p.closePath();
  }
  return p;
}

function drawVideo() {
  if (!video.videoWidth) return;
  ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, canvas.width, canvas.height); ctx.restore();
}
function canvasQuad(q) { return q.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height })); }
function pathQuad(q) { ctx.beginPath(); ctx.moveTo(q[0].x, q[0].y); for (let i = 1; i < q.length; i++) ctx.lineTo(q[i].x, q[i].y); ctx.closePath(); }

// ---------- AnimeGANv2 Face Portrait v2 ----------
function computeAnimeCrop(q) {
  // Work in displayed/mirrored canvas coordinates so the finished anime patch can be drawn directly.
  const px = q.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
  const minX = Math.min(...px.map(p => p.x)), maxX = Math.max(...px.map(p => p.x));
  const minY = Math.min(...px.map(p => p.y)), maxY = Math.max(...px.map(p => p.y));
  const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5;
  // Portrait model works best when the intended region fills the square, but keep some context.
  let side = Math.max(maxX - minX, maxY - minY) * 1.32;
  side = clamp(side, Math.min(canvas.width, canvas.height) * 0.28, Math.min(canvas.width, canvas.height));
  let dx = clamp(cx - side * 0.5, 0, canvas.width - side);
  let dy = clamp(cy - side * 0.5, 0, canvas.height - side);
  // Display-space square [dx,dy,side] maps to mirrored source-video x.
  const sx = canvas.width - (dx + side);
  return { dx, dy, side, sx, sy: dy };
}

function preprocessAnimeFrame(q) {
  const crop = computeAnimeCrop(q);
  inputCtx.save();
  inputCtx.clearRect(0, 0, modelSize, modelSize);
  inputCtx.translate(modelSize, 0);
  inputCtx.scale(-1, 1);
  inputCtx.drawImage(video, crop.sx, crop.sy, crop.side, crop.side, 0, 0, modelSize, modelSize);
  inputCtx.restore();
  const rgba = inputCtx.getImageData(0, 0, modelSize, modelSize).data;
  const plane = modelSize * modelSize;
  let input;
  let dims;
  if (modelLayout === 'nhwc') {
    input = new Float32Array(plane * 3);
    for (let i = 0, p = 0, o = 0; i < plane; i++, p += 4, o += 3) {
      input[o] = rgba[p] / 127.5 - 1;
      input[o + 1] = rgba[p + 1] / 127.5 - 1;
      input[o + 2] = rgba[p + 2] / 127.5 - 1;
    }
    dims = [1, modelSize, modelSize, 3];
  } else {
    input = new Float32Array(plane * 3);
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      input[i] = rgba[p] / 127.5 - 1;
      input[plane + i] = rgba[p + 1] / 127.5 - 1;
      input[plane * 2 + i] = rgba[p + 2] / 127.5 - 1;
    }
    dims = [1, 3, modelSize, modelSize];
  }
  return { tensor: new ort.Tensor('float32', input, dims), crop };
}

function outputToCanvas(tensor) {
  const data = tensor.data;
  const dims = tensor.dims;
  const outH = Number(dims?.[modelLayout === 'nhwc' ? 1 : 2]) || modelSize;
  const outW = Number(dims?.[modelLayout === 'nhwc' ? 2 : 3]) || modelSize;
  if (animeCanvas.width !== outW || animeCanvas.height !== outH) {
    animeCanvas.width = outW; animeCanvas.height = outH;
  }
  const plane = outW * outH;
  const image = animeCtx.createImageData(outW, outH);
  const out = image.data;
  const nchw = dims?.length === 4 && dims[1] === 3;
  const nhwc = dims?.length === 4 && dims[3] === 3;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    let r, g, b;
    if (nhwc) { r = data[i * 3]; g = data[i * 3 + 1]; b = data[i * 3 + 2]; }
    else if (nchw) { r = data[i]; g = data[plane + i]; b = data[plane * 2 + i]; }
    else { r = data[i]; g = data[plane + i]; b = data[plane * 2 + i]; }
    out[p] = clamp(Math.round((r * 0.5 + 0.5) * 255), 0, 255);
    out[p + 1] = clamp(Math.round((g * 0.5 + 0.5) * 255), 0, 255);
    out[p + 2] = clamp(Math.round((b * 0.5 + 0.5) * 255), 0, 255);
    out[p + 3] = 255;
  }
  animeCtx.putImageData(image, 0, 0);
  animeFrameValid = true;
}

function tickAiFps(now) {
  aiCompleted++;
  const dt = now - aiWindowStart;
  if (dt >= 700) { aiFps = aiCompleted * 1000 / dt; aiCompleted = 0; aiWindowStart = now; }
}
function tickHandFps(now) {
  handCompleted++;
  const dt = now - handWindowStart;
  if (dt >= 700) { handFps = handCompleted * 1000 / dt; handCompleted = 0; handWindowStart = now; }
}
function tickRenderFps(now) {
  renderCompleted++;
  const dt = now - renderWindowStart;
  if (dt >= 700) { renderFps = renderCompleted * 1000 / dt; renderCompleted = 0; renderWindowStart = now; updateDiag(); }
}

async function runAnimeInference(ts, q) {
  if (!animeReady || video.readyState < 2 || effectSelect.value === 'original') return;
  const requested = effectSelect.value === 'quality' ? 'quality' : 'fast';
  if (requested !== selectedProfile) { initAnime(true); return; }
  if (animeBusy) { skippedBusy++; return; }
  if (ts - lastAnimeStart < animeInterval) return;

  // ZERO QUEUE: snapshot the newest available camera frame only when the worker is actually free.
  animeBusy = true;
  lastAnimeStart = ts;
  const start = performance.now();
  try {
    const { tensor: inputTensor, crop } = preprocessAnimeFrame(q);
    const results = await animeSession.run({ [animeInputName]: inputTensor });
    outputToCanvas(results[animeOutputName]);
    animeMapping = { x: crop.dx, y: crop.dy, size: crop.side };
    lastAnimeDone = performance.now();
    lastAnimeMs = lastAnimeDone - start;
    tickAiFps(lastAnimeDone);
    animeState = `${animeBackend} ${PROFILES[selectedProfile].label} OK`;
    // No post-inference backlog. On WebGPU the next RAF may launch the next newest frame immediately.
    animeInterval = animeBackend === 'WebGPU' ? 0 : clamp(lastAnimeMs * 0.05, selectedProfile === 'fast' ? 12 : 45, 120);
    updateDiag();
  } catch (err) {
    console.error('AnimeGAN inference failed', err);
    if (animeBackend === 'WebGPU') {
      animeReady = false; animeSession = null; animeBackend = '—'; animeFrameValid = false; animeMapping = null;
      animeState = 'GPU→WASM…'; updateDiag();
      try {
        const profile = PROFILES[selectedProfile];
        animeSession = await createAnimeSession(['wasm'], 'WASM', profile);
        animeInputName = animeSession.inputNames[0]; animeOutputName = animeSession.outputNames[0];
        animeBackend = 'WASM'; animeReady = true; animeInterval = selectedProfile === 'fast' ? 20 : 90;
        animeState = `WASM ${profile.label} OK`; updateDiag();
      } catch (fallbackError) {
        animeState = 'BŁĄD'; updateDiag();
        showError(`AnimeGAN inference nie działa na WebGPU ani WASM.\n${fallbackError?.message || fallbackError}`);
      }
    } else {
      animeState = 'BŁĄD'; updateDiag();
      showError(`AnimeGAN inference błąd: ${err?.message || err}`);
    }
  } finally { animeBusy = false; }
}

function applyAnimeFx(q) {
  if (effectSelect.value === 'original') return;
  const path = triangleUnionPath(q);
  ctx.save(); ctx.clip(path);
  if (animeFrameValid && animeMapping) {
    // Reproject the latest AI patch into the CURRENT frame position. This removes most visible spatial lag
    // when hands move while inference is still running.
    const current = computeAnimeCrop(q);
    ctx.drawImage(animeCanvas, current.dx, current.dy, current.side, current.side);
  } else {
    // Fast local fallback while the ONNX model is loading / producing its first frame.
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.filter = 'saturate(1.18) contrast(1.12) brightness(1.04)';
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  ctx.restore();
}

function drawFrame(q) {
  const cq = canvasQuad(q);
  ctx.save(); pathQuad(cq); ctx.lineWidth = Math.max(3, canvas.width * .004); ctx.strokeStyle = 'rgba(255,255,255,.98)';
  ctx.shadowBlur = 14; ctx.shadowColor = 'rgba(255,255,255,.48)'; ctx.stroke(); ctx.restore();
}

function drawDebug(semantic, q, rawQuad) {
  if (!debugToggle.checked) return;
  ctx.save(); ctx.fillStyle = 'rgba(0,255,180,.9)';
  for (const h of semantic) for (const p of h.pts) { ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 2.6, 0, Math.PI * 2); ctx.fill(); }
  if (rawQuad) {
    const rq = canvasQuad(rawQuad); ctx.save(); ctx.setLineDash([7,7]); ctx.strokeStyle = 'rgba(0,255,255,.95)'; ctx.lineWidth = 2; pathQuad(rq); ctx.stroke();
    ctx.fillStyle = 'rgba(0,255,255,.95)'; rq.forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#001b20'; ctx.font = '12px sans-serif'; ctx.fillText(`P${i}`, p.x + 9, p.y - 8); ctx.fillStyle = 'rgba(0,255,255,.95)'; }); ctx.restore();
  }
  if (q) { const cq = canvasQuad(q); ctx.strokeStyle = 'rgba(80,180,255,.9)'; ctx.lineWidth = 2; pathQuad(cq); ctx.stroke(); }
  ctx.restore();
}

function preferredMimeType() {
  const candidates = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  return candidates.find(t => window.MediaRecorder?.isTypeSupported?.(t)) || '';
}
function toggleRecording() {
  if (!running || !window.MediaRecorder || !canvas.captureStream) {
    showError('Ta przeglądarka nie obsługuje nagrywania canvas przez MediaRecorder.'); return;
  }
  if (mediaRecorder?.state === 'recording') { mediaRecorder.stop(); return; }
  showError(''); recordedChunks = [];
  const canvasStream = canvas.captureStream(30);
  const mimeType = preferredMimeType();
  try { mediaRecorder = new MediaRecorder(canvasStream, mimeType ? { mimeType, videoBitsPerSecond: 8_000_000 } : { videoBitsPerSecond: 8_000_000 }); }
  catch (err) { showError(`Nagrywanie nie wystartowało: ${err.message}`); return; }
  mediaRecorder.ondataavailable = e => { if (e.data?.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const type = mediaRecorder.mimeType || mimeType || 'video/webm';
    const blob = new Blob(recordedChunks, { type });
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = `magic-frame-animegan-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000);
    recordBtn.textContent = 'Nagraj'; recordBtn.classList.remove('recording'); recBadge.hidden = true;
  };
  mediaRecorder.start(250); recordingStartedAt = performance.now();
  recordBtn.textContent = 'Stop'; recordBtn.classList.add('recording'); recBadge.hidden = false;
}

async function renderLoop(ts) {
  if (!running) return;
  tickRenderFps(ts);
  resizeCanvas(); drawVideo();

  if (handsReady && handLandmarker && video.readyState >= 2 && ts - lastHandDetect > HAND_INTERVAL) {
    lastHandDetect = ts;
    try {
      const r = handLandmarker.detectForVideo(video, ts);
      latestHands = r.landmarks || []; latestHandedness = r.handedness || []; tickHandFps(ts);
    } catch (e) { console.warn(e); }
  }

  const semantic = semanticHands(latestHands, latestHandedness);
  const rawQuad = measureFreeformQuad(semantic);
  const q = smoothFreeformQuad(rawQuad, ts);

  if (q) {
    // Fire-and-forget inference. Hand tracking + frame rendering never await the AI model.
    runAnimeInference(ts, q);
    applyAnimeFx(q); drawFrame(q);
    const age = animeFrameValid ? Math.max(0, performance.now() - lastAnimeDone) : 0;
    const ageText = animeFrameValid ? ` · AI ${Math.round(age)}ms old` : animeReady ? ' · AI pierwsza klatka…' : ' · AI ładowanie…';
    const mode = effectSelect.value === 'quality' ? 'QUALITY' : effectSelect.value === 'fast' ? 'FAST' : 'ORIGINAL';
    setStatus(`2/2 dłonie · ${mode}${ageText}`);
  } else if (handsReady) {
    const count = Math.min(2, semantic.length);
    setStatus(`${count}/2 dłonie${count === 2 ? ' · rozsuń palce' : ''}`);
  }

  if (mediaRecorder?.state === 'recording') {
    const secs = Math.floor((ts - recordingStartedAt) / 1000); recBadge.textContent = `● REC ${secs}s`;
  }
  drawDebug(semantic, q, rawQuad);
  requestAnimationFrame(renderLoop);
}
