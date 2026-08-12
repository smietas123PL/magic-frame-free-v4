import './styles.css';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
const startBtn = document.getElementById('startBtn');
const effectSelect = document.getElementById('effectSelect');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');

let handLandmarker = null;
let stream = null;
let running = false;
let trackerReady = false;
let trackerLoading = false;
let lastDetect = 0;
let latestHands = [];
let latestHandedness = [];
let smoothRect = null;
let frameCounter = 0;
let cameraState = '—';
let trackerState = '—';

const DETECT_INTERVAL = 1000 / 20;

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() { diagEl.textContent = `JS: OK · kamera: ${cameraState} · tracker: ${trackerState}`; }
function showError(text) {
  errorEl.hidden = !text;
  errorEl.textContent = text || '';
}

function readableCameraError(err) {
  const name = err?.name || 'Error';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Brak zgody na kamerę. W Edge kliknij ikonę kłódki/kamery przy adresie → Kamera → Zezwalaj, potem odśwież.';
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'Nie znaleziono kamery na tym urządzeniu.';
  if (name === 'NotReadableError' || name === 'TrackStartError') return 'Kamera jest zajęta przez inną aplikację. Zamknij Teams/Zoom/aplikację Aparat i spróbuj ponownie.';
  if (name === 'OverconstrainedError' || name === 'ConstraintNotSatisfiedError') return 'Przeglądarka nie obsługuje żądanych parametrów kamery.';
  return `${name}: ${err?.message || 'Nieznany błąd kamery'}`;
}

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Ta przeglądarka nie udostępnia getUserMedia. Wymagany jest HTTPS lub localhost.');
  const attempts = [
    { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false },
    { video: { facingMode: 'user' }, audio: false },
    { video: true, audio: false }
  ];
  let lastError;
  for (const constraints of attempts) {
    try { return await navigator.mediaDevices.getUserMedia(constraints); }
    catch (err) {
      lastError = err;
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') break;
    }
  }
  throw lastError || new Error('Nie udało się uruchomić kamery.');
}

async function initHands() {
  if (handLandmarker || trackerLoading) return;
  trackerLoading = true;
  trackerState = 'WASM…'; updateDiag();
  setStatus('Kamera działa • uruchamiam tracking…');
  showError('');

  try {
    // WASM jest kopiowany z node_modules do public/ podczas builda.
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    trackerState = 'model…'; updateDiag();

    const common = {
      baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45
    };

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        ...common,
        baseOptions: { ...common.baseOptions, delegate: 'GPU' }
      });
      trackerState = 'GPU OK';
    } catch (gpuError) {
      console.warn('MediaPipe GPU unavailable, switching to CPU', gpuError);
      trackerState = 'CPU…'; updateDiag();
      handLandmarker = await HandLandmarker.createFromOptions(vision, common);
      trackerState = 'CPU OK';
    }

    trackerReady = true;
    updateDiag();
    setStatus('Pokaż obie dłonie');
  } catch (err) {
    console.error('MediaPipe init error', err);
    trackerReady = false;
    trackerState = 'BŁĄD'; updateDiag();
    showError(`Tracking dłoni nie wystartował.\n${err?.name || 'Error'}: ${err?.message || err}\n\nW v4 MediaPipe JS, WASM i model powinny być serwowane z tej samej domeny Vercel.`);
    setStatus('Kamera działa • tracking niedostępny');
  } finally {
    trackerLoading = false;
  }
}

async function startCamera() {
  if (running) return;
  showError('');
  startBtn.disabled = true;
  startBtn.textContent = 'Uruchamianie…';
  cameraState = 'prośba…'; updateDiag();
  setStatus('Prośba o dostęp do kamery…');

  try {
    stream = await requestCamera();
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await new Promise((resolve) => {
      if (video.readyState >= 1) return resolve();
      video.addEventListener('loadedmetadata', resolve, { once: true });
      setTimeout(resolve, 1800);
    });
    await video.play().catch((e) => console.warn('video.play()', e));

    running = true;
    cameraState = 'OK'; updateDiag();
    startBtn.textContent = 'Kamera działa';
    setStatus('Kamera działa');
    resizeCanvas();
    requestAnimationFrame(renderLoop);
    setTimeout(initHands, 50);
  } catch (err) {
    console.error('Camera error', err);
    running = false;
    cameraState = 'BŁĄD'; updateDiag();
    startBtn.disabled = false;
    startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery');
    showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
setStatus('Gotowy • v4');
updateDiag();

window.addEventListener('error', (ev) => {
  console.error('Global JS error', ev.error || ev.message);
  showError(`Błąd JavaScript: ${ev.message || 'nieznany błąd'}`);
});
window.addEventListener('unhandledrejection', (ev) => console.error('Unhandled promise rejection', ev.reason));

function resizeCanvas() {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
}

function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z }; }
function sortHands(hands, handedness) {
  const out = [];
  for (let i = 0; i < hands.length; i++) {
    const name = handedness?.[i]?.[0]?.categoryName || '';
    out.push({ name, pts: hands[i].map(mirrorPoint) });
  }
  out.sort((a, b) => a.pts[0].x - b.pts[0].x);
  return out;
}
function pointDistance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function lGestureScore(pts) {
  const wrist = pts[0], thumb = pts[4], index = pts[8], indexMcp = pts[5], middle = pts[12], ring = pts[16], pinky = pts[20];
  const palm = Math.max(pointDistance(wrist, indexMcp), 0.05);
  const thumbOpen = pointDistance(thumb, indexMcp) / palm;
  const indexOpen = pointDistance(index, wrist) / palm;
  const folded = (pointDistance(middle, wrist) + pointDistance(ring, wrist) + pointDistance(pinky, wrist)) / (3 * palm);
  return thumbOpen > 0.75 && indexOpen > 1.2 && folded < 1.45;
}

function computeRect(sorted) {
  if (sorted.length < 2) return null;
  const a = sorted[0].pts, b = sorted[1].pts;
  if (!lGestureScore(a) || !lGestureScore(b)) return null;
  const xs = [a[4].x, a[8].x, b[4].x, b[8].x];
  const ys = [a[4].y, a[8].y, b[4].y, b[8].y];
  let x = Math.min(...xs), y = Math.min(...ys), r = Math.max(...xs), bot = Math.max(...ys);
  const pad = 0.012;
  x = Math.max(0, x - pad); y = Math.max(0, y - pad); r = Math.min(1, r + pad); bot = Math.min(1, bot + pad);
  if (r - x < 0.12 || bot - y < 0.10) return null;
  return { x, y, w: r - x, h: bot - y };
}

function smoothRectangle(next) {
  if (!next) { smoothRect = null; return null; }
  if (!smoothRect) { smoothRect = { ...next }; return smoothRect; }
  const a = 0.22;
  for (const k of ['x', 'y', 'w', 'h']) smoothRect[k] += (next[k] - smoothRect[k]) * a;
  return smoothRect;
}

function drawVideo() {
  if (!video.videoWidth || !video.videoHeight) return;
  ctx.save();
  ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function applyEffect(rect, effect) {
  const x = rect.x * canvas.width, y = rect.y * canvas.height, w = rect.w * canvas.width, h = rect.h * canvas.height;
  ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
  const srcX = (1 - rect.x - rect.w) * video.videoWidth;
  const srcY = rect.y * video.videoHeight;
  const srcW = rect.w * video.videoWidth, srcH = rect.h * video.videoHeight;
  const dx = canvas.width - (x + w), dy = y;

  switch (effect) {
    case 'bw': ctx.filter = 'grayscale(1) contrast(1.25)'; break;
    case 'comic': ctx.filter = 'contrast(1.65) saturate(1.6) brightness(1.05)'; break;
    case 'anime': ctx.filter = 'saturate(1.9) contrast(1.3) brightness(1.12)'; break;
    case 'cyber': ctx.filter = 'hue-rotate(255deg) saturate(2.2) contrast(1.35)'; break;
    case 'clay': ctx.filter = 'saturate(.75) contrast(1.15) brightness(1.12) sepia(.18)'; break;
    case 'glitch': ctx.filter = 'contrast(1.5) saturate(2) hue-rotate(35deg)'; break;
  }
  ctx.drawImage(video, srcX, srcY, srcW, srcH, dx, dy, w, h);
  ctx.filter = 'none'; ctx.restore();

  if (effect === 'comic') drawComicEdges(x, y, w, h);
  if (effect === 'anime') drawAnimeBloom(x, y, w, h);
  if (effect === 'glitch') drawGlitch(x, y, w, h);
  if (effect === 'clay') drawClay(x, y, w, h);
  if (effect === 'cyber') drawCyber(x, y, w, h);

  ctx.save(); ctx.lineWidth = Math.max(3, canvas.width * .004); ctx.strokeStyle = 'rgba(255,255,255,.95)';
  ctx.shadowBlur = 14; ctx.shadowColor = 'rgba(255,255,255,.55)'; ctx.strokeRect(x, y, w, h); ctx.restore();
}

function drawComicEdges(x, y, w, h) {
  ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip(); ctx.globalCompositeOperation = 'multiply'; ctx.globalAlpha = .22; ctx.fillStyle = '#111';
  const step = Math.max(10, Math.round(w / 28)); for (let yy = y; yy < y + h; yy += step) ctx.fillRect(x, yy, w, 1); ctx.restore();
}
function drawAnimeBloom(x, y, w, h) {
  const g = ctx.createRadialGradient(x + w*.5, y + h*.45, 0, x + w*.5, y + h*.45, Math.max(w,h)*.65);
  g.addColorStop(0, 'rgba(255,255,255,.14)'); g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.save(); ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip(); ctx.fillStyle=g; ctx.fillRect(x,y,w,h); ctx.restore();
}
function drawCyber(x,y,w,h) { ctx.save(); ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip(); ctx.globalCompositeOperation='screen'; ctx.globalAlpha=.16; ctx.fillStyle='cyan'; ctx.fillRect(x-3,y,w,h); ctx.fillStyle='magenta'; ctx.fillRect(x+3,y,w,h); ctx.restore(); }
function drawClay(x,y,w,h) { ctx.save(); ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip(); ctx.globalAlpha=.08; ctx.fillStyle='#f0d0b5'; ctx.fillRect(x,y,w,h); ctx.restore(); }
function drawGlitch(x,y,w,h) {
  if (frameCounter % 3 !== 0) return;
  ctx.save(); ctx.beginPath(); ctx.rect(x,y,w,h); ctx.clip();
  for (let i=0;i<7;i++) { const sh=Math.max(2,Math.random()*h*.035), sy=y+Math.random()*(h-sh), off=(Math.random()-.5)*18; try { const img=ctx.getImageData(x,sy,w,sh); ctx.putImageData(img,x+off,sy); } catch {} }
  ctx.restore();
}
function drawDebug(sorted) {
  if (!debugToggle.checked) return;
  ctx.save(); ctx.fillStyle='rgba(0,255,180,.9)';
  for (const hand of sorted) for (const p of hand.pts) { ctx.beginPath(); ctx.arc(p.x*canvas.width,p.y*canvas.height,3,0,Math.PI*2); ctx.fill(); }
  ctx.restore();
}

async function renderLoop(ts) {
  if (!running) return;
  frameCounter++; resizeCanvas(); drawVideo();

  if (trackerReady && handLandmarker && video.readyState >= 2 && ts - lastDetect > DETECT_INTERVAL) {
    lastDetect = ts;
    try {
      const res = handLandmarker.detectForVideo(video, ts);
      latestHands = res.landmarks || [];
      latestHandedness = res.handedness || [];
    } catch (e) { console.warn('detectForVideo', e); }
  }

  const sorted = sortHands(latestHands, latestHandedness);
  const rect = smoothRectangle(computeRect(sorted));
  if (trackerReady) {
    if (rect) { applyEffect(rect, effectSelect.value); setStatus(`Efekt: ${effectSelect.options[effectSelect.selectedIndex].text}`); }
    else setStatus(sorted.length < 2 ? 'Pokaż obie dłonie' : 'Ułóż dłonie w dwa gesty L');
    drawDebug(sorted);
  }
  requestAnimationFrame(renderLoop);
}

window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(track => track.stop()));
