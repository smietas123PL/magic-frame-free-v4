import './styles.css';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

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
let stream = null;
let running = false;
let handsReady = false;
let loadingHands = false;
let loadingAnime = false;
let latestHands = [];
let latestHandedness = [];
let smoothedQuad = null;
let previousRawQuad = null;
let previousRawAt = 0;
let lastValidFrameAt = 0;
let lastHandDetect = 0;
let handCompleted = 0;
let handWindowStart = performance.now();
let handFps = 0;
let renderCompleted = 0;
let renderWindowStart = performance.now();
let renderFps = 0;
let cameraState = '—';
let handState = '—';

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;

const HAND_INTERVAL = 1000 / 30;
const FRAME_HOLD_MS = 90;
function setStatus(text) { statusEl.textContent = text; }

function showError(text) { errorEl.hidden = !text; errorEl.textContent = text || ''; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function readableCameraError(err) {
  const name = err?.name || 'Error';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Brak zgody na kamerę. Kliknij ikonę kamery przy adresie → Zezwalaj, potem odśwież.';
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'Nie znaleziono kamery.';
  if (name === 'NotReadableError' || name === 'TrackStartError') return 'Kamera jest zajęta przez inną aplikację.';
  if (/timeout/i.test(err?.message || '')) return 'Przeglądarka nie odpowiedziała na prośbę o kamerę w 10 s. Sprawdź ikonę kłódki przy adresie → Kamera → Zezwalaj, zamknij inne aplikacje używające kamery i odśwież stronę.';
  return `${name}: ${err?.message || 'Nieznany błąd kamery'}`;
}

async function cameraPermissionState() {
  try {
    if (!navigator.permissions?.query) return 'unknown';
    const p = await navigator.permissions.query({ name: 'camera' });
    return p?.state || 'unknown';
  } catch { return 'unknown'; }
}

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timeout po ${Math.round(ms/1000)} s`)), ms);
    })
  ]);
}

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Wymagany jest HTTPS lub localhost.');

  const permission = await cameraPermissionState();
  cameraState = `permission:${permission}`; updateDiag();
  if (permission === 'denied') {
    const e = new Error('Dostęp do kamery jest zablokowany dla tej domeny. Kliknij ikonę kłódki przy adresie → Uprawnienia witryny → Kamera → Zezwalaj.');
    e.name = 'NotAllowedError';
    throw e;
  }

  // Najpierw prosimy o najprostszy stream. To najszybciej wywołuje prompt
  // uprawnień w Edge/Chrome. Dopiero kolejne wersje mogą stosować ostrzejsze constraints.
  const attempts = [
    { video: true, audio: false },
    { video: { facingMode: { ideal: 'user' } }, audio: false },
    { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } }, audio: false }
  ];

  let lastError;
  for (let i = 0; i < attempts.length; i++) {
    cameraState = `request ${i+1}/${attempts.length}`; updateDiag();
    try {
      return await withTimeout(navigator.mediaDevices.getUserMedia(attempts[i]), 10000, 'getUserMedia');
    } catch (err) {
      lastError = err;
      console.warn('camera attempt failed', i + 1, err);
      if (err?.name === 'NotAllowedError' || /timeout/i.test(err?.message || '')) break;
    }
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
    setTimeout(() => initAnime(false), 350);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
recordBtn.addEventListener('click', toggleRecording);
setStatus('Gotowy • v9.3 Worker AI');
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


// ---------- CartoonGAN Light / TensorFlow.js WORKER ----------
// v9.3: inference nie wykonuje się już na głównym wątku. Main thread robi tylko
// mały crop 160-224 px i wysyła RGBA do Worker. Dzięki temu MediaPipe + render
// nie czekają na GAN, nawet jeżeli pojedyncza klatka anime trwa setki ms.
function computeAnimeCrop(q) {
  const px = q.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
  const minX = Math.min(...px.map(p => p.x)), maxX = Math.max(...px.map(p => p.x));
  const minY = Math.min(...px.map(p => p.y)), maxY = Math.max(...px.map(p => p.y));
  const cx = (minX + maxX) * 0.5, cy = (minY + maxY) * 0.5;
  let side = Math.max(maxX - minX, maxY - minY) * 1.28;
  side = clamp(side, Math.min(canvas.width, canvas.height) * 0.25, Math.min(canvas.width, canvas.height));
  const dx = clamp(cx - side * 0.5, 0, canvas.width - side);
  const dy = clamp(cy - side * 0.5, 0, canvas.height - side);
  const sx = canvas.width - (dx + side);
  return { dx, dy, side, sx, sy: dy };
}

const AI_SIZES = [160, 192, 224];
let sizeIndex = 0; // start FAST, podnosimy jakość tylko gdy Worker jest naprawdę szybki
let stableFastFrames = 0;
let slowFrames = 0;
let cartoonWorker = null;
let cartoonStyle = 'shinkai';
let cartoonReady = false;
let cartoonBusy = false;
let cartoonBackend = '—';
let cartoonState = '—';
let cartoonFrameValid = false;
let lastCartoonDone = 0;
let lastCartoonMs = 0;
let lastCartoonComputeMs = 0;
let lastCartoonReadMs = 0;
let cartoonCompleted = 0;
let cartoonWindowStart = performance.now();
let cartoonFps = 0;
let skippedBusy = 0;
let latestQuadForAi = null;
let aiKickScheduled = false;
let aiRequestId = 0;


function currentAiSize() { return AI_SIZES[sizeIndex]; }
function setAiCanvasSize(size) {
  if (inputCanvas.width !== size || inputCanvas.height !== size) {
    inputCanvas.width = inputCanvas.height = size;
    animeCanvas.width = animeCanvas.height = size;
  }
}
function tickCartoonFps(now) {
  cartoonCompleted++;
  const dt = now - cartoonWindowStart;
  if (dt >= 700) {
    cartoonFps = cartoonCompleted * 1000 / dt;
    cartoonCompleted = 0;
    cartoonWindowStart = now;
  }
}
function adaptAiSize(ms) {
  if (ms > 180) { slowFrames++; stableFastFrames = 0; }
  else if (ms < 75) { stableFastFrames++; slowFrames = 0; }
  else { slowFrames = 0; stableFastFrames = 0; }
  if (slowFrames >= 2 && sizeIndex > 0) {
    sizeIndex--; slowFrames = 0; stableFastFrames = 0; setAiCanvasSize(currentAiSize());
  } else if (stableFastFrames >= 12 && sizeIndex < AI_SIZES.length - 1) {
    sizeIndex++; slowFrames = 0; stableFastFrames = 0; setAiCanvasSize(currentAiSize());
  }
}

function stopCartoonWorker() {
  if (cartoonWorker) cartoonWorker.terminate();
  cartoonWorker = null;
  cartoonReady = false;
  cartoonBusy = false;
}

function startCartoonWorker(style) {
  stopCartoonWorker();
  cartoonStyle = style;
  cartoonFrameValid = false;
  cartoonState = `${style} worker load…`;
  updateDiag();
  cartoonWorker = new Worker(new URL('./cartoon-worker.js', import.meta.url), { type: 'module' });

  cartoonWorker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === 'ready') {
      cartoonReady = true;
      cartoonBusy = false;
      cartoonBackend = String(msg.backend || 'worker').toUpperCase();
      cartoonState = `${cartoonBackend} ${cartoonStyle} OK`;
      setAiCanvasSize(currentAiSize());
      updateDiag();
      scheduleAiKick();
      return;
    }
    if (msg.type === 'frame') {
      cartoonBusy = false;
      if (msg.requestId !== aiRequestId) {
        scheduleAiKick();
        return;
      }
      const rgba = new Uint8ClampedArray(msg.rgba);
      const image = new ImageData(rgba, msg.size, msg.size);
      if (animeCanvas.width !== msg.size || animeCanvas.height !== msg.size) {
        animeCanvas.width = animeCanvas.height = msg.size;
      }
      animeCtx.putImageData(image, 0, 0);
      lastCartoonDone = performance.now();
      lastCartoonMs = Number(msg.totalMs || 0);
      lastCartoonComputeMs = Number(msg.computeMs || 0);
      lastCartoonReadMs = Number(msg.readMs || 0);
      tickCartoonFps(lastCartoonDone);
      adaptAiSize(lastCartoonMs);
      cartoonFrameValid = true;
      cartoonState = `${cartoonBackend} ${cartoonStyle} OK`;
      updateDiag();
      // Latest-frame-wins: od razu po zakończeniu bierzemy NAJNOWSZY quad/obraz,
      // ale na osobnym ticku event loop, nigdy z requestAnimationFrame.
      scheduleAiKick();
      return;
    }
    if (msg.type === 'error') {
      cartoonBusy = false;
      cartoonReady = false;
      cartoonState = 'BŁĄD';
      updateDiag();
      showError(`CartoonGAN Worker błąd: ${msg.message || 'nieznany błąd'}`);
    }
  };
  cartoonWorker.onerror = (err) => {
    cartoonBusy = false;
    cartoonReady = false;
    cartoonState = 'WORKER BŁĄD';
    updateDiag();
    showError(`CartoonGAN Worker nie wystartował: ${err.message || err}`);
  };
  cartoonWorker.postMessage({
    type: 'init',
    style,
    modelUrl: `/models/cartoongan-${style}/model.json`
  });
}

async function initAnime(force = false) {
  const requested = effectSelect.value;
  if (requested === 'original') {
    cartoonFrameValid = false;
    cartoonState = 'wyłączony';
    stopCartoonWorker();
    updateDiag();
    return;
  }
  if (!force && cartoonReady && cartoonStyle === requested) return;
  if (loadingAnime) return;
  loadingAnime = true;
  try { startCartoonWorker(requested); }
  finally { loadingAnime = false; }
}

effectSelect.addEventListener('change', () => initAnime(true));

function updateDiag() {
  const perf = cartoonReady && lastCartoonMs ? ` ${Math.round(lastCartoonMs)}ms` : '';
  const split = cartoonReady && lastCartoonMs ? ` (GPU ${Math.round(lastCartoonComputeMs)} + read ${Math.round(lastCartoonReadMs)})` : '';
  const fps = cartoonReady ? ` · AI ${cartoonFps.toFixed(1)}fps · H ${handFps.toFixed(0)}fps · R ${renderFps.toFixed(0)}fps · ${currentAiSize()}px · skip ${skippedBusy}` : '';
  diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · CartoonGAN: ${cartoonState}${perf}${split}${fps}`;
}
updateDiag();

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

function scheduleAiKick() {
  if (aiKickScheduled || cartoonBusy || !cartoonReady || !latestQuadForAi || effectSelect.value === 'original' || !running) return;
  aiKickScheduled = true;
  setTimeout(() => {
    aiKickScheduled = false;
    kickCartoonWorker();
  }, 0);
}

function kickCartoonWorker() {
  if (cartoonBusy || !cartoonReady || !cartoonWorker || !latestQuadForAi || video.readyState < 2 || effectSelect.value === 'original') return;
  const q = latestQuadForAi;
  const size = currentAiSize();
  setAiCanvasSize(size);
  const crop = computeAnimeCrop(q);
  cartoonBusy = true;
  aiRequestId++;
  try {
    inputCtx.save();
    inputCtx.clearRect(0, 0, size, size);
    inputCtx.translate(size, 0); inputCtx.scale(-1, 1);
    inputCtx.drawImage(video, crop.sx, crop.sy, crop.side, crop.side, 0, 0, size, size);
    inputCtx.restore();
    // Mały CPU readback (160-224px) jest jedyną operacją wykonywaną na main thread.
    // Cały TensorFlow.js + GAN + output readback działa w Workerze.
    const frame = inputCtx.getImageData(0, 0, size, size);
    cartoonWorker.postMessage({
      type: 'infer', requestId: aiRequestId, size,
      rgba: frame.data.buffer
    }, [frame.data.buffer]);
  } catch (err) {
    cartoonBusy = false;
    console.error('CartoonGAN worker enqueue failed', err);
    scheduleAiKick();
  }
}

function applyAnimeFx(q) {
  if (effectSelect.value === 'original') return;
  const path = triangleUnionPath(q);
  ctx.save(); ctx.clip(path);
  if (cartoonFrameValid) {
    const current = computeAnimeCrop(q);
    ctx.drawImage(animeCanvas, current.dx, current.dy, current.side, current.side);
  } else {
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.filter = 'saturate(1.18) contrast(1.10)';
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
    const a = document.createElement('a'); a.href = url; a.download = `magic-frame-cartoongan-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
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
    latestQuadForAi = q.map(p => ({ ...p }));
    scheduleAiKick();
    applyAnimeFx(q); drawFrame(q);
    const age = cartoonFrameValid ? Math.max(0, performance.now() - lastCartoonDone) : 0;
    const ageText = cartoonFrameValid ? ` · AI ${Math.round(age)}ms old` : cartoonReady ? ' · AI pierwsza klatka…' : ' · AI ładowanie…';
    const mode = effectSelect.value === 'original' ? 'ORIGINAL' : `ANIME ${effectSelect.value.toUpperCase()}`;
    setStatus(`2/2 dłonie · ${mode}${ageText}`);
  } else if (handsReady) {
    latestQuadForAi = null;
    const count = Math.min(2, semantic.length);
    setStatus(`${count}/2 dłonie${count === 2 ? ' · rozsuń palce' : ''}`);
  }

  if (mediaRecorder?.state === 'recording') {
    const secs = Math.floor((ts - recordingStartedAt) / 1000); recBadge.textContent = `● REC ${secs}s`;
  }
  drawDebug(semantic, q, rawQuad);
  requestAnimationFrame(renderLoop);
}
