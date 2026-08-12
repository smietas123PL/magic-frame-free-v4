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
let animeInterval = 130;
let animeFrameValid = false;
let animeMapping = null;
let cameraState = '—';
let handState = '—';

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;

const HAND_INTERVAL = 1000 / 30;
const FRAME_HOLD_MS = 90;
const MODEL_SIZE = 512;

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() {
  const perf = animeReady && lastAnimeMs ? ` ${Math.round(lastAnimeMs)}ms` : '';
  diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · AnimeGAN: ${animeState}${perf}`;
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

async function createAnimeSession(executionProviders, label) {
  animeState = `${label} load…`; updateDiag();
  const session = await ort.InferenceSession.create('/models/face_paint_512_v2_0.onnx', {
    executionProviders,
    graphOptimizationLevel: 'all'
  });
  animeBackend = label;
  return session;
}

async function initAnime() {
  if (loadingAnime || animeReady) return;
  loadingAnime = true;
  animeState = 'runtime…'; updateDiag();
  try {
    ort.env.wasm.wasmPaths = '/ort/';
    ort.env.wasm.numThreads = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;

    let lastError = null;
    if (navigator.gpu) {
      try {
        animeSession = await createAnimeSession(['webgpu'], 'WebGPU');
      } catch (err) {
        lastError = err;
        console.warn('AnimeGAN WebGPU failed, switching to WASM', err);
      }
    }
    if (!animeSession) {
      animeSession = await createAnimeSession(['wasm'], 'WASM');
    }

    animeInputName = animeSession.inputNames[0];
    animeOutputName = animeSession.outputNames[0];
    animeReady = true;
    animeState = `${animeBackend} OK`;
    animeInterval = animeBackend === 'WebGPU' ? 100 : 300;
    updateDiag();

    // Small warmup is intentionally skipped; first real inference warms the graph while the UI stays responsive.
  } catch (err) {
    console.error(err);
    animeState = 'BŁĄD'; updateDiag();
    showError(`AnimeGAN nie wystartował. Tracking dłoni nadal działa.\n${err?.name || 'Error'}: ${err?.message || err}`);
  } finally { loadingAnime = false; }
}

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
    setTimeout(initAnime, 80);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
recordBtn.addEventListener('click', toggleRecording);
setStatus('Gotowy • v9 AnimeGAN'); updateDiag();
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
  inputCtx.clearRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  // Mirror at preprocessing time. Anime output is therefore already in the same orientation as the UI.
  inputCtx.translate(MODEL_SIZE, 0);
  inputCtx.scale(-1, 1);
  inputCtx.drawImage(video, crop.sx, crop.sy, crop.side, crop.side, 0, 0, MODEL_SIZE, MODEL_SIZE);
  inputCtx.restore();
  const rgba = inputCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(plane * 3);
  // Model expects NCHW RGB in [-1, 1].
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    input[i] = rgba[p] / 127.5 - 1;
    input[plane + i] = rgba[p + 1] / 127.5 - 1;
    input[plane * 2 + i] = rgba[p + 2] / 127.5 - 1;
  }
  return { tensor: new ort.Tensor('float32', input, [1, 3, MODEL_SIZE, MODEL_SIZE]), crop };
}

function outputToCanvas(tensor) {
  const data = tensor.data;
  const dims = tensor.dims;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const image = animeCtx.createImageData(MODEL_SIZE, MODEL_SIZE);
  const out = image.data;

  const nchw = dims?.length === 4 && dims[1] === 3;
  const nhwc = dims?.length === 4 && dims[3] === 3;
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    let r, g, b;
    if (nchw) {
      r = data[i]; g = data[plane + i]; b = data[plane * 2 + i];
    } else if (nhwc) {
      r = data[i * 3]; g = data[i * 3 + 1]; b = data[i * 3 + 2];
    } else {
      // Known model is NCHW; this fallback prevents a hard crash if metadata is absent.
      r = data[i]; g = data[plane + i]; b = data[plane * 2 + i];
    }
    out[p] = clamp(Math.round((r * 0.5 + 0.5) * 255), 0, 255);
    out[p + 1] = clamp(Math.round((g * 0.5 + 0.5) * 255), 0, 255);
    out[p + 2] = clamp(Math.round((b * 0.5 + 0.5) * 255), 0, 255);
    out[p + 3] = 255;
  }
  animeCtx.putImageData(image, 0, 0);
  animeFrameValid = true;
}

async function runAnimeInference(ts, q) {
  if (!animeReady || animeBusy || video.readyState < 2 || effectSelect.value !== 'animegan') return;
  if (ts - lastAnimeStart < animeInterval) return;
  animeBusy = true; lastAnimeStart = ts;
  const start = performance.now();
  try {
    const { tensor: inputTensor, crop } = preprocessAnimeFrame(q);
    const results = await animeSession.run({ [animeInputName]: inputTensor });
    const output = results[animeOutputName];
    outputToCanvas(output);
    animeMapping = { x: crop.dx, y: crop.dy, size: crop.side };
    lastAnimeDone = performance.now();
    lastAnimeMs = lastAnimeDone - start;
    // Adaptive cadence: never queue work. Keep a small breathing gap after each inference.
    animeInterval = clamp(lastAnimeMs * 1.08, animeBackend === 'WebGPU' ? 70 : 220, 900);
    animeState = `${animeBackend} OK`;
    updateDiag();
  } catch (err) {
    console.error('AnimeGAN inference failed', err);
    // Some older models load on WebGPU but fail on a specific kernel at run-time. Retry once on WASM.
    if (animeBackend === 'WebGPU') {
      animeReady = false; animeSession = null; animeBackend = '—'; animeFrameValid = false; animeMapping = null;
      animeState = 'GPU→WASM…'; updateDiag();
      try {
        animeSession = await createAnimeSession(['wasm'], 'WASM');
        animeInputName = animeSession.inputNames[0]; animeOutputName = animeSession.outputNames[0];
        animeBackend = 'WASM'; animeReady = true; animeInterval = 300; animeState = 'WASM OK'; updateDiag();
      } catch (fallbackError) {
        animeState = 'BŁĄD'; updateDiag();
        showError(`AnimeGAN inference nie działa na WebGPU ani WASM.\n${fallbackError?.message || fallbackError}`);
      }
    } else {
      animeState = 'BŁĄD'; updateDiag();
      showError(`AnimeGAN inference błąd: ${err?.message || err}`);
    }
  } finally {
    animeBusy = false;
  }
}

function applyAnimeFx(q) {
  if (effectSelect.value === 'original') return;
  const path = triangleUnionPath(q);
  ctx.save(); ctx.clip(path);
  if (animeFrameValid && animeMapping) {
    // animeCanvas was mirrored during preprocessing and already uses display-space orientation.
    ctx.drawImage(animeCanvas, animeMapping.x, animeMapping.y, animeMapping.size, animeMapping.size);
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
  resizeCanvas(); drawVideo();

  if (handsReady && handLandmarker && video.readyState >= 2 && ts - lastHandDetect > HAND_INTERVAL) {
    lastHandDetect = ts;
    try {
      const r = handLandmarker.detectForVideo(video, ts);
      latestHands = r.landmarks || []; latestHandedness = r.handedness || [];
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
    setStatus(`2/2 dłonie · AnimeGAN${ageText}`);
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
