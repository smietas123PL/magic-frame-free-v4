import './styles.css';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const effectSelect = document.getElementById('effectSelect');
const intensityInput = document.getElementById('intensity');
const intensityValue = document.getElementById('intensityValue');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');
const recBadge = document.getElementById('recBadge');

let stream = null;
let running = false;
let handLandmarker = null;
let handsReady = false;
let loadingHands = false;
let lastDetectAt = 0;
let handState = '—';
let cameraState = '—';
let handFps = 0;
let renderFps = 0;
let handFrames = 0;
let handFpsWindow = performance.now();
let renderFrames = 0;
let renderFpsWindow = performance.now();
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;

const HAND_INTERVAL = 1000 / 30;
const HAND_MODEL_STALL_TIMEOUT_MS = 18000;
const TRACK_HOLD_MS = 150;
const TRACK_DROP_MS = 420;
const MAX_PARTICLES = 1050;
const MAX_SHOCKWAVES = 18;
const TIP_IDS = [4, 8, 12, 16, 20];
const PALM_IDS = [0, 5, 9, 13, 17];
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17]
];

const THEMES = {
  quantum: { a: 188, b: 282, accent: 326, label: 'QUANTUM RIFT' },
  solar: { a: 30, b: 350, accent: 55, label: 'SOLAR FLARE' },
  void: { a: 222, b: 272, accent: 188, label: 'VOID STORM' }
};

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const hsl = (h, s = 100, l = 65, a = 1) => `hsla(${((h % 360) + 360) % 360} ${s}% ${l}% / ${a})`;
const intensity = () => Number(intensityInput.value) / 100;
const currentTheme = () => THEMES[effectSelect.value] || THEMES.quantum;
const setStatus = text => { statusEl.textContent = text; };
const showError = text => { errorEl.hidden = !text; errorEl.textContent = text || ''; };
const mirrorPoint = p => ({ x: 1 - p.x, y: p.y, z: p.z || 0 });
const px = p => ({ x: p.x * canvas.width, y: p.y * canvas.height, z: p.z || 0 });

class OneEuro {
  constructor(minCutoff = 1.7, beta = 0.24, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.value = null;
    this.derivative = 0;
    this.lastTime = 0;
  }
  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / Math.max(0.0001, dt));
  }
  filter(value, now) {
    if (this.value == null || !this.lastTime) {
      this.value = value;
      this.lastTime = now;
      return value;
    }
    const dt = clamp((now - this.lastTime) / 1000, 1 / 120, 0.08);
    const rawD = (value - this.value) / dt;
    const ad = this.alpha(this.dCutoff, dt);
    this.derivative = lerp(this.derivative, rawD, ad);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.derivative);
    const a = this.alpha(cutoff, dt);
    this.value = lerp(this.value, value, a);
    this.lastTime = now;
    return this.value;
  }
}

function newTrack(id, now) {
  return {
    id,
    lastSeen: now,
    points: Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0, vx: 0, vy: 0 })),
    filters: Array.from({ length: 21 }, () => ({ x: new OneEuro(), y: new OneEuro(), z: new OneEuro(1.3, 0.08) })),
    palm: { x: 0, y: 0 },
    previousPalm: null,
    speed: 0,
    pinch: 1,
    pinchDown: false,
    openness: 0,
    visible: 0,
    hueOffset: id * 82,
    fistCharge: 0,
    trails: new Map(TIP_IDS.map(t => [t, []]))
  };
}

const tracks = [newTrack(0, 0), newTrack(1, 0)];
let particles = [];
let shockwaves = [];
let burstEnergy = 0;

function readableCameraError(err) {
  const name = err?.name || 'Error';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Brak zgody na kamerę. Zezwól na dostęp do kamery w ustawieniach przeglądarki.';
  if (name === 'NotFoundError') return 'Nie znaleziono kamery.';
  if (name === 'NotReadableError') return 'Kamera jest zajęta przez inną aplikację.';
  return `${name}: ${err?.message || 'Nieznany błąd kamery'}`;
}

async function requestCamera() {
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('Wymagany HTTPS lub localhost.');
  return navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: 'user' },
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 60, min: 24, max: 60 }
    },
    audio: false
  });
}

async function fetchModelBytes(modelPath, sourceLabel) {
  const controller = new AbortController();
  let stallTimer = null;
  let received = 0;
  let lastUiAt = 0;

  const armStallTimer = () => {
    clearTimeout(stallTimer);
    stallTimer = setTimeout(() => controller.abort(), HAND_MODEL_STALL_TIMEOUT_MS);
  };

  try {
    handState = `${sourceLabel}: model 0 MB…`;
    updateDiag();
    armStallTimer();

    const response = await fetch(modelPath, {
      cache: 'reload',
      signal: controller.signal
    });
    if (!response.ok) throw new Error(`${sourceLabel} model HTTP ${response.status}`);

    const total = Number(response.headers.get('content-length')) || 0;
    if (!response.body?.getReader) {
      // Starsze przeglądarki: bez całkowitego timeoutu. Abortujemy tylko gdy
      // przez długi czas nie ma odpowiedzi z sieci.
      clearTimeout(stallTimer);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength < 1000000) {
        throw new Error(`${sourceLabel} model ma podejrzanie mały rozmiar: ${buffer.byteLength} B`);
      }
      return new Uint8Array(buffer);
    }

    const reader = response.body.getReader();
    const chunks = [];
    while (true) {
      armStallTimer();
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      chunks.push(value);
      received += value.byteLength;

      const now = performance.now();
      if (now - lastUiAt > 180) {
        const got = (received / 1048576).toFixed(1);
        const all = total ? `/${(total / 1048576).toFixed(1)}` : '';
        handState = `${sourceLabel}: model ${got}${all} MB…`;
        updateDiag();
        lastUiAt = now;
      }
    }
    clearTimeout(stallTimer);

    if (received < 1000000) {
      throw new Error(`${sourceLabel} model ma podejrzanie mały rozmiar: ${received} B`);
    }

    const merged = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return merged;
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw new Error(`${sourceLabel} model: brak transferu przez ${Math.round(HAND_MODEL_STALL_TIMEOUT_MS / 1000)}s`);
    }
    throw err;
  } finally {
    clearTimeout(stallTimer);
  }
}

async function createHandsFrom(wasmPath, modelPath, sourceLabel) {
  // Najpierw pobieramy model jawnie. Dzięki modelAssetBuffer MediaPipe nie wykonuje
  // drugiego ukrytego requestu podczas inicjalizacji taska.
  const modelBytes = await fetchModelBytes(modelPath, sourceLabel);

  handState = `${sourceLabel}: WASM…`;
  updateDiag();
  // Nie ubijamy poprawnej inicjalizacji arbitralnym timeoutem. WASM jest
  // mały, a źródła są próbowane sekwencyjnie.
  const vision = await FilesetResolver.forVisionTasks(wasmPath);

  const common = {
    runningMode: 'VIDEO',
    numHands: 2,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.4
  };

  // v13.3: CPU jest celowym trybem startowym. W v13.1 GPU było obejmowane
  // Promise.race z timeoutem; createFromOptions nie daje AbortSignal, więc po
  // timeout GPU mógł nadal inicjalizować się w tle, podczas gdy startował CPU.
  // To potrafiło zostawić tracking na 0 fps. Tutaj inicjalizacja jest pojedyncza.
  handState = `${sourceLabel}: CPU init…`;
  updateDiag();
  const landmarker = await HandLandmarker.createFromOptions(vision, {
    ...common,
    baseOptions: {
      modelAssetBuffer: modelBytes,
      delegate: 'CPU'
    }
  });

  return { landmarker, mode: `${sourceLabel}/CPU` };
}

async function initHands() {
  if (loadingHands || handsReady) return;
  loadingHands = true;
  handState = 'start…';
  updateDiag();
  showError('');

  const googleModel = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
  const sources = [
    {
      label: 'GOOGLE',
      wasm: '/mediapipe/wasm',
      model: googleModel
    },
    {
      label: 'LOCAL',
      wasm: '/mediapipe/wasm',
      model: '/models/hand_landmarker.task?v=13.3'
    },
    {
      label: 'CDN',
      wasm: 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm',
      model: googleModel
    }
  ];

  let lastError = null;
  try {
    for (const source of sources) {
      try {
        const created = await createHandsFrom(source.wasm, source.model, source.label);
        handLandmarker = created.landmarker;
        handState = created.mode;
        handsReady = true;
        setStatus('Tracking dłoni gotowy · pokaż dłoń');
        console.info(`HANDSTORM hand tracking ready: ${created.mode}`);
        return;
      } catch (err) {
        lastError = err;
        console.warn(`Hand tracking init failed via ${source.label}`, err);
      }
    }
    throw lastError || new Error('Nie udało się zainicjalizować MediaPipe HandLandmarker.');
  } catch (err) {
    handState = 'BŁĄD';
    console.error(err);
    showError(`Tracking dłoni nie wystartował. ${err?.message || err}. Otwórz DevTools → Network i sprawdź /models/hand_landmarker.task oraz /mediapipe/wasm.`);
  } finally {
    loadingHands = false;
    updateDiag();
  }
}

async function startCamera() {
  if (running) return;
  showError('');
  startBtn.disabled = true;
  startBtn.textContent = 'Uruchamianie…';
  cameraState = 'prośba…';
  updateDiag();
  try {
    stream = await requestCamera();
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await new Promise(resolve => {
      if (video.readyState >= 1) return resolve();
      video.addEventListener('loadedmetadata', resolve, { once: true });
      setTimeout(resolve, 1400);
    });
    await video.play();
    running = true;
    cameraState = 'OK';
    resizeCanvas();
    startBtn.textContent = 'Kamera działa';
    recordBtn.disabled = false;
    setStatus('Pokaż dłonie · tracking 21 punktów / dłoń');
    requestAnimationFrame(renderLoop);
    if (!handsReady && !loadingHands) setTimeout(initHands, 30);
  } catch (err) {
    running = false;
    cameraState = 'BŁĄD';
    startBtn.disabled = false;
    startBtn.textContent = 'Spróbuj ponownie';
    showError(readableCameraError(err));
    updateDiag();
  }
}

function resizeCanvas() {
  const w = video.videoWidth || 1280;
  const h = video.videoHeight || 720;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function drawVideo() {
  if (!video.videoWidth) return;
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function assignDetections(hands) {
  if (!hands.length) return [];
  const wrists = hands.map((pts, index) => ({ index, p: mirrorPoint(pts[0]) }));
  const active = tracks.filter(t => t.lastSeen > 0);
  if (!active.length) return wrists.map((w, i) => ({ track: tracks[i], handIndex: w.index }));
  if (wrists.length === 1) {
    const w = wrists[0];
    let best = tracks[0];
    if (tracks[1].lastSeen > 0 && dist(w.p, tracks[1].palm) < dist(w.p, tracks[0].palm)) best = tracks[1];
    return [{ track: best, handIndex: w.index }];
  }
  const a = wrists[0].p, b = wrists[1].p;
  const c1 = dist(a, tracks[0].palm) + dist(b, tracks[1].palm);
  const c2 = dist(a, tracks[1].palm) + dist(b, tracks[0].palm);
  return c1 <= c2
    ? [{ track: tracks[0], handIndex: wrists[0].index }, { track: tracks[1], handIndex: wrists[1].index }]
    : [{ track: tracks[1], handIndex: wrists[0].index }, { track: tracks[0], handIndex: wrists[1].index }];
}

function palmCenter(points) {
  const c = PALM_IDS.reduce((acc, id) => ({ x: acc.x + points[id].x, y: acc.y + points[id].y }), { x: 0, y: 0 });
  return { x: c.x / PALM_IDS.length, y: c.y / PALM_IDS.length };
}

function palmScale(points) {
  return Math.max(0.025, (dist(points[0], points[5]) + dist(points[0], points[9]) + dist(points[0], points[13]) + dist(points[0], points[17])) / 4);
}

function openness(points) {
  const wrist = points[0];
  const pairs = [[4,2],[8,6],[12,10],[16,14],[20,18]];
  return pairs.reduce((score, [tip, joint]) => score + (dist(points[tip], wrist) > dist(points[joint], wrist) * 1.12 ? 1 : 0), 0);
}

function updateTrack(track, rawPoints, now) {
  if (!track.lastSeen) {
    track.filters = Array.from({ length: 21 }, () => ({ x: new OneEuro(), y: new OneEuro(), z: new OneEuro(1.3, 0.08) }));
  }
  const mirrored = rawPoints.map(mirrorPoint);
  const dt = track.lastSeen > 0 ? clamp((now - track.lastSeen) / 1000, 1 / 120, 0.1) : 1 / 45;
  const oldPoints = track.points.map(p => ({ ...p }));

  for (let i = 0; i < 21; i++) {
    const f = track.filters[i];
    const x = f.x.filter(mirrored[i].x, now);
    const y = f.y.filter(mirrored[i].y, now);
    const z = f.z.filter(mirrored[i].z || 0, now);
    const old = oldPoints[i];
    const vx = track.lastSeen > 0 ? (x - old.x) / dt : 0;
    const vy = track.lastSeen > 0 ? (y - old.y) / dt : 0;
    track.points[i] = {
      x, y, z,
      vx: lerp(old.vx || 0, vx, 0.55),
      vy: lerp(old.vy || 0, vy, 0.55)
    };
  }

  const palm = palmCenter(track.points);
  const prevPalm = track.previousPalm || palm;
  const normSpeed = Math.hypot(palm.x - prevPalm.x, palm.y - prevPalm.y) / dt;
  track.speed = lerp(track.speed, normSpeed, 0.5);
  track.previousPalm = palm;
  track.palm = palm;
  track.lastSeen = now;
  track.visible = Math.min(1, track.visible + 0.32);
  const previousOpen = track.openness;
  track.openness = openness(track.points);
  if (track.openness <= 1) track.fistCharge = Math.min(1, track.fistCharge + 0.16);
  else if (track.fistCharge > 0.7 && track.openness >= 4 && previousOpen <= 2) {
    triggerPalmBurst(track.palm, track, track.fistCharge);
    track.fistCharge = 0;
  } else if (track.openness >= 3) track.fistCharge = Math.max(0, track.fistCharge - 0.08);

  const scale = palmScale(track.points);
  track.pinch = dist(track.points[4], track.points[8]) / scale;
  const wasPinching = track.pinchDown;
  if (!track.pinchDown && track.pinch < 0.46) track.pinchDown = true;
  else if (track.pinchDown && track.pinch > 0.62) track.pinchDown = false;

  if (!wasPinching && track.pinchDown) {
    const pinchPoint = {
      x: (track.points[4].x + track.points[8].x) * 0.5,
      y: (track.points[4].y + track.points[8].y) * 0.5
    };
    triggerBurst(pinchPoint, track);
  }

  for (const tipId of TIP_IDS) {
    const history = track.trails.get(tipId);
    const p = track.points[tipId];
    history.push({ x: p.x, y: p.y, t: now });
    while (history.length > 24 || (history[0] && now - history[0].t > 520)) history.shift();
  }
}

function updateTracking(result, now) {
  const hands = result?.landmarks || [];
  const assignments = assignDetections(hands);
  const seen = new Set();
  for (const { track, handIndex } of assignments) {
    updateTrack(track, hands[handIndex], now);
    seen.add(track.id);
  }
  for (const track of tracks) {
    if (!seen.has(track.id)) {
      const age = now - track.lastSeen;
      if (age > TRACK_HOLD_MS) track.visible = Math.max(0, track.visible - 0.12);
      if (age > TRACK_DROP_MS) {
        track.lastSeen = 0;
        track.visible = 0;
        track.previousPalm = null;
        track.speed = 0;
        track.pinchDown = false;
        for (const history of track.trails.values()) history.length = 0;
      }
    }
  }
}

function displayPoint(track, id, now) {
  const p = track.points[id];
  const age = track.lastSeen ? clamp((now - track.lastSeen) / 1000, 0, 0.028) : 0;
  return {
    x: clamp(p.x + p.vx * age * 0.65, 0, 1),
    y: clamp(p.y + p.vy * age * 0.65, 0, 1),
    z: p.z,
    vx: p.vx,
    vy: p.vy
  };
}

function activeTracks(now) {
  return tracks.filter(t => t.lastSeen && now - t.lastSeen < TRACK_DROP_MS && t.visible > 0.02);
}

function triggerPalmBurst(point, track, charge = 1) {
  const theme = currentTheme();
  const p = px(point);
  const power = intensity() * clamp(charge, 0.4, 1);
  burstEnergy = Math.min(1.7, burstEnergy + 1.05 * power);
  shockwaves.push({ x: p.x, y: p.y, born: performance.now(), hue: theme.b + track.hueOffset * 0.15, life: 1100, power: 1.55 * power });
  const count = Math.round(86 * power);
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const speed = 160 + Math.random() * 620;
    particles.push({ x: p.x, y: p.y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, born: performance.now(), life: 520 + Math.random() * 900, size: 1.2 + Math.random() * 5.6, hue: lerp(theme.a, theme.b, Math.random()), drag: 0.968 + Math.random() * 0.02, kind: Math.random() < 0.48 ? 'spark' : 'orb' });
  }
}

function triggerBurst(point, track) {
  const p = px(point);
  const theme = currentTheme();
  const power = intensity();
  burstEnergy = Math.min(1.5, burstEnergy + 0.78 * power);
  shockwaves.push({ x: p.x, y: p.y, born: performance.now(), hue: theme.accent + track.hueOffset, life: 850, power: 1.1 * power });
  if (shockwaves.length > MAX_SHOCKWAVES) shockwaves.shift();

  const count = Math.round(52 * power);
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2 + Math.random() * 0.18;
    const speed = (110 + Math.random() * 430) * power;
    particles.push({
      x: p.x, y: p.y,
      vx: Math.cos(a) * speed,
      vy: Math.sin(a) * speed,
      born: performance.now(),
      life: 420 + Math.random() * 650,
      size: 1.2 + Math.random() * 4.8,
      hue: lerp(theme.a, theme.b, Math.random()) + track.hueOffset * 0.22,
      drag: 0.965 + Math.random() * 0.025,
      kind: Math.random() < 0.24 ? 'spark' : 'orb'
    });
  }
}

function spawnHandParticles(track, now, dt) {
  const theme = currentTheme();
  const power = intensity();
  const frameFactor = clamp(dt / 16.67, 0.45, 2.2);
  const speedBoost = clamp(track.speed * 1.8, 0, 2.8);

  for (const tipId of TIP_IDS) {
    const tp = displayPoint(track, tipId, now);
    const p = px(tp);
    const speed = Math.hypot(tp.vx * canvas.width, tp.vy * canvas.height);
    const chance = clamp((0.4 + speed / 1100 + speedBoost * 0.18) * power * frameFactor, 0.15, 2.8);
    const count = Math.floor(chance) + (Math.random() < chance % 1 ? 1 : 0);
    for (let i = 0; i < count; i++) {
      const hueMix = (tipId / 20 + Math.random() * 0.22) % 1;
      particles.push({
        x: p.x + (Math.random() - 0.5) * 8,
        y: p.y + (Math.random() - 0.5) * 8,
        vx: -tp.vx * canvas.width * (0.09 + Math.random() * 0.12) + (Math.random() - 0.5) * 36,
        vy: -tp.vy * canvas.height * (0.09 + Math.random() * 0.12) + (Math.random() - 0.5) * 36,
        born: now,
        life: 320 + Math.random() * 720,
        size: 0.8 + Math.random() * (2.8 + speedBoost),
        hue: lerp(theme.a, theme.b, hueMix) + track.hueOffset * 0.18,
        drag: 0.975,
        kind: speed > 520 ? 'spark' : 'orb'
      });
    }
  }

  if (track.pinchDown) {
    const a = displayPoint(track, 4, now), b = displayPoint(track, 8, now);
    const c = px({ x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 });
    const count = Math.max(1, Math.round(2 * power * frameFactor));
    for (let i = 0; i < count; i++) {
      const angle = now * 0.007 + i * 2.4 + Math.random();
      const radius = 18 + Math.random() * 52;
      particles.push({
        x: c.x + Math.cos(angle) * radius,
        y: c.y + Math.sin(angle) * radius,
        vx: -Math.sin(angle) * 90,
        vy: Math.cos(angle) * 90,
        born: now,
        life: 360 + Math.random() * 300,
        size: 1.4 + Math.random() * 3.5,
        hue: theme.accent + Math.random() * 24,
        drag: 0.985,
        kind: 'orb'
      });
    }
  }

  if (particles.length > MAX_PARTICLES) particles.splice(0, particles.length - MAX_PARTICLES);
}

function updateParticles(now, dt) {
  const step = clamp(dt / 1000, 0, 0.05);
  const dragPow = step * 60;
  particles = particles.filter(p => {
    const age = now - p.born;
    if (age >= p.life) return false;
    const drag = Math.pow(p.drag, dragPow);
    p.vx *= drag;
    p.vy *= drag;
    p.x += p.vx * step;
    p.y += p.vy * step;
    return true;
  });
  shockwaves = shockwaves.filter(w => now - w.born < w.life);
  burstEnergy *= Math.pow(0.90, dragPow);
}

function drawSoftGlow(x, y, radius, hue, alpha) {
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, hsl(hue, 100, 72, alpha));
  gradient.addColorStop(0.18, hsl(hue + 20, 100, 62, alpha * 0.58));
  gradient.addColorStop(0.58, hsl(hue + 48, 95, 48, alpha * 0.16));
  gradient.addColorStop(1, hsl(hue, 100, 40, 0));
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawFingerTrails(track, now) {
  const theme = currentTheme();
  const power = intensity();
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let tipIndex = 0; tipIndex < TIP_IDS.length; tipIndex++) {
    const tipId = TIP_IDS[tipIndex];
    const hist = track.trails.get(tipId);
    if (!hist || hist.length < 2) continue;
    const hue = lerp(theme.a, theme.b, tipIndex / 4) + track.hueOffset * 0.15;
    for (let pass = 0; pass < 3; pass++) {
      ctx.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const p = hist[i];
        const q = { x: p.x * canvas.width, y: p.y * canvas.height };
        if (i === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y);
      }
      const base = (pass === 0 ? 10 : pass === 1 ? 4.2 : 1.5) * power;
      ctx.lineWidth = base;
      ctx.strokeStyle = hsl(hue + pass * 8, 100, pass === 2 ? 88 : 62, (pass === 0 ? 0.08 : pass === 1 ? 0.24 : 0.84) * track.visible);
      ctx.shadowBlur = pass === 2 ? 14 : 24;
      ctx.shadowColor = hsl(hue, 100, 65, 0.8);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawHandEnergy(track, now) {
  const theme = currentTheme();
  const power = intensity();
  const points = track.points.map((_, i) => px(displayPoint(track, i, now)));
  const palm = px(track.palm);
  const scale = Math.max(34, palmScale(track.points) * Math.min(canvas.width, canvas.height) * 2.4);
  const baseHue = theme.a + track.hueOffset * 0.15;

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  drawSoftGlow(palm.x, palm.y, scale * (1.05 + track.speed * 0.1) * power, baseHue, 0.16 * track.visible);

  for (let pass = 0; pass < 3; pass++) {
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
    }
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.lineWidth = (pass === 0 ? 10 : pass === 1 ? 4.4 : 1.45) * power;
    ctx.strokeStyle = hsl(baseHue + pass * 11, 100, pass === 2 ? 92 : 64, (pass === 0 ? 0.07 : pass === 1 ? 0.25 : 0.9) * track.visible);
    ctx.shadowBlur = pass === 2 ? 15 : 28;
    ctx.shadowColor = hsl(baseHue + 22, 100, 68, 0.85);
    ctx.stroke();
  }

  for (let i = 0; i < TIP_IDS.length; i++) {
    const tip = points[TIP_IDS[i]];
    const velocity = Math.hypot(track.points[TIP_IDS[i]].vx * canvas.width, track.points[TIP_IDS[i]].vy * canvas.height);
    const pulse = 0.5 + 0.5 * Math.sin(now * 0.009 + i * 1.7 + track.id);
    const radius = (6 + pulse * 3 + clamp(velocity / 260, 0, 7)) * power;
    const hue = lerp(theme.a, theme.b, i / 4) + track.hueOffset * 0.15;
    drawSoftGlow(tip.x, tip.y, radius * 5.4, hue, 0.13 * track.visible);
    ctx.beginPath();
    ctx.arc(tip.x, tip.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = hsl(hue, 100, 88, 0.95 * track.visible);
    ctx.shadowBlur = 24;
    ctx.shadowColor = hsl(hue, 100, 68, 1);
    ctx.fill();
  }

  if (track.fistCharge > 0.05) {
    const chargeR = scale * (0.3 + track.fistCharge * 0.45);
    ctx.beginPath();
    ctx.arc(palm.x, palm.y, chargeR, 0, Math.PI * 2);
    ctx.lineWidth = (1.5 + track.fistCharge * 4) * power;
    ctx.strokeStyle = hsl(theme.accent, 100, 82, 0.18 + track.fistCharge * 0.62);
    ctx.shadowBlur = 34;
    ctx.shadowColor = hsl(theme.accent, 100, 65, 0.9);
    ctx.stroke();
  }
  if (track.pinchDown) drawSingularity(track, now, points);
  ctx.restore();
}

function drawSingularity(track, now, points) {
  const theme = currentTheme();
  const power = intensity();
  const thumb = points[4], index = points[8];
  const x = (thumb.x + index.x) * 0.5, y = (thumb.y + index.y) * 0.5;
  const pulse = 0.5 + 0.5 * Math.sin(now * 0.018);
  const r = (15 + pulse * 7) * power;

  drawSoftGlow(x, y, r * 6.5, theme.accent, 0.28 * track.visible);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(now * 0.004 * (track.id ? -1 : 1));
  for (let i = 0; i < 3; i++) {
    ctx.beginPath();
    ctx.ellipse(0, 0, r * (1.2 + i * 0.55), r * (0.42 + i * 0.12), i * 0.7, 0, Math.PI * 2);
    ctx.lineWidth = (2.4 - i * 0.45) * power;
    ctx.strokeStyle = hsl(theme.accent + i * 28, 100, 76, 0.88 - i * 0.18);
    ctx.shadowBlur = 20;
    ctx.shadowColor = hsl(theme.accent + 20, 100, 68, 1);
    ctx.stroke();
  }
  ctx.restore();
}

function drawPortal(a, b, now) {
  const theme = currentTheme();
  const power = intensity();
  const A = px(a.palm), B = px(b.palm);
  const cx = (A.x + B.x) * 0.5, cy = (A.y + B.y) * 0.5;
  const d = Math.hypot(B.x - A.x, B.y - A.y);
  if (d < 90) return;
  const angle = Math.atan2(B.y - A.y, B.x - A.x);
  const openness = (a.openness + b.openness) / 10;
  const energy = clamp(0.45 + openness * 0.6 + (a.speed + b.speed) * 0.12, 0.45, 1.4) * power;
  const rx = clamp(d * 0.32, 48, canvas.width * 0.34);
  const ry = clamp(rx * 0.34, 20, canvas.height * 0.15);

  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  drawSoftGlow(cx, cy, rx * 1.55, theme.b, 0.10 * energy);
  ctx.translate(cx, cy);
  ctx.rotate(angle);

  for (let ring = 0; ring < 6; ring++) {
    const phase = now * (0.0015 + ring * 0.00013) * (ring % 2 ? -1 : 1);
    ctx.save();
    ctx.rotate(phase);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx * (0.72 + ring * 0.082), ry * (0.68 + ring * 0.1), ring * 0.18, 0, Math.PI * 2);
    ctx.setLineDash([10 + ring * 3, 14 + ring * 5]);
    ctx.lineDashOffset = -now * (0.02 + ring * 0.007);
    ctx.lineWidth = Math.max(1.2, (4.6 - ring * 0.46) * energy);
    ctx.strokeStyle = hsl(lerp(theme.a, theme.b, ring / 5) + ring * 8, 100, 69 + ring * 3, 0.42 + ring * 0.06);
    ctx.shadowBlur = 24;
    ctx.shadowColor = hsl(theme.b, 100, 65, 0.95);
    ctx.stroke();
    ctx.restore();
  }

  for (let i = 0; i < 18; i++) {
    const t = (i / 18 + now * 0.00022) % 1;
    const theta = t * Math.PI * 2;
    const r = rx * (0.66 + 0.3 * Math.sin(i * 2.7 + now * 0.003));
    const x = Math.cos(theta) * r;
    const y = Math.sin(theta) * ry * (0.8 + 0.24 * Math.cos(i));
    ctx.beginPath();
    ctx.arc(x, y, (1.5 + (i % 4)) * energy, 0, Math.PI * 2);
    ctx.fillStyle = hsl(lerp(theme.a, theme.b, i / 18), 100, 82, 0.72);
    ctx.shadowBlur = 15;
    ctx.shadowColor = hsl(theme.accent, 100, 66, 0.8);
    ctx.fill();
  }
  ctx.restore();

  // Curved lightning from matching fingertips between both hands.
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < TIP_IDS.length; i++) {
    const p1 = px(displayPoint(a, TIP_IDS[i], now));
    const p2 = px(displayPoint(b, TIP_IDS[i], now));
    const mx = (p1.x + p2.x) * 0.5;
    const my = (p1.y + p2.y) * 0.5;
    const bend = Math.sin(now * 0.006 + i * 1.9) * d * 0.06;
    const nx = -(p2.y - p1.y) / d;
    const ny = (p2.x - p1.x) / d;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.quadraticCurveTo(mx + nx * bend, my + ny * bend, p2.x, p2.y);
    ctx.lineWidth = (0.8 + (i % 2) * 0.7) * energy;
    ctx.strokeStyle = hsl(lerp(theme.a, theme.b, i / 4), 100, 88, 0.42);
    ctx.shadowBlur = 15;
    ctx.shadowColor = hsl(theme.accent, 100, 70, 0.8);
    ctx.stroke();
  }
  ctx.restore();
}

function drawParticles(now) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const p of particles) {
    const age = now - p.born;
    const t = clamp(age / p.life, 0, 1);
    const alpha = (1 - t) * (p.kind === 'spark' ? 0.95 : 0.72);
    if (p.kind === 'spark') {
      const len = Math.min(26, Math.hypot(p.vx, p.vy) * 0.028 + 4);
      const m = Math.max(1, Math.hypot(p.vx, p.vy));
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.vx / m * len, p.y - p.vy / m * len);
      ctx.lineWidth = Math.max(0.7, p.size * (1 - t * 0.5));
      ctx.strokeStyle = hsl(p.hue, 100, 82, alpha);
      ctx.shadowBlur = 12;
      ctx.shadowColor = hsl(p.hue, 100, 65, alpha);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(0.5, p.size * (1 - t * 0.55)), 0, Math.PI * 2);
      ctx.fillStyle = hsl(p.hue, 100, 78, alpha);
      ctx.shadowBlur = 14;
      ctx.shadowColor = hsl(p.hue, 100, 64, alpha);
      ctx.fill();
    }
  }
  ctx.restore();
}

function drawShockwaves(now) {
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (const w of shockwaves) {
    const t = clamp((now - w.born) / w.life, 0, 1);
    const e = 1 - Math.pow(1 - t, 3);
    const radius = (20 + e * 210 * w.power);
    ctx.beginPath();
    ctx.arc(w.x, w.y, radius, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(0.8, 5 * (1 - t) * w.power);
    ctx.strokeStyle = hsl(w.hue, 100, 75, (1 - t) * 0.72);
    ctx.shadowBlur = 24;
    ctx.shadowColor = hsl(w.hue + 20, 100, 66, 0.9);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(w.x, w.y, radius * 0.63, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(0.5, 2.2 * (1 - t) * w.power);
    ctx.strokeStyle = hsl(w.hue + 40, 100, 88, (1 - t) * 0.45);
    ctx.stroke();
  }
  ctx.restore();
}

function drawScreenPulse() {
  if (burstEnergy < 0.02) return;
  const theme = currentTheme();
  ctx.save();
  ctx.globalCompositeOperation = 'screen';
  const g = ctx.createRadialGradient(canvas.width * 0.5, canvas.height * 0.5, 0, canvas.width * 0.5, canvas.height * 0.5, Math.max(canvas.width, canvas.height) * 0.72);
  g.addColorStop(0, hsl(theme.accent, 100, 70, burstEnergy * 0.035));
  g.addColorStop(0.6, hsl(theme.b, 100, 55, burstEnergy * 0.018));
  g.addColorStop(1, hsl(theme.a, 100, 40, 0));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function drawDebug(active, now) {
  if (!debugToggle.checked) return;
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.font = `${Math.max(11, canvas.width * 0.012)}px ui-monospace, monospace`;
  for (const track of active) {
    const points = track.points.map((_, i) => px(displayPoint(track, i, now)));
    ctx.strokeStyle = 'rgba(0,255,210,.7)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (const [a, b] of CONNECTIONS) {
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
    }
    ctx.stroke();
    ctx.fillStyle = '#00ffd5';
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.7, 0, Math.PI * 2);
      ctx.fill();
    }
    const palm = px(track.palm);
    ctx.fillStyle = 'rgba(0,0,0,.68)';
    ctx.fillRect(palm.x + 10, palm.y - 30, 170, 46);
    ctx.fillStyle = '#d8fff9';
    ctx.fillText(`H${track.id + 1} pinch ${track.pinch.toFixed(2)}`, palm.x + 16, palm.y - 11);
    ctx.fillText(`open ${track.openness}/5 speed ${track.speed.toFixed(2)}`, palm.x + 16, palm.y + 6);
  }
  ctx.restore();
}

function updateDiag() {
  diagEl.textContent = `JS: OK · kamera: ${cameraState} · tracking: ${handState} ${handFps.toFixed(0)}fps · render: ${renderFps.toFixed(0)}fps · particles: ${particles.length}`;
}

function tickHand(now) {
  handFrames++;
  const dt = now - handFpsWindow;
  if (dt >= 700) {
    handFps = handFrames * 1000 / dt;
    handFrames = 0;
    handFpsWindow = now;
  }
}

function tickRender(now) {
  renderFrames++;
  const dt = now - renderFpsWindow;
  if (dt >= 700) {
    renderFps = renderFrames * 1000 / dt;
    renderFrames = 0;
    renderFpsWindow = now;
    updateDiag();
  }
}

let lastRenderAt = performance.now();
function renderLoop(now) {
  if (!running) return;
  const dt = clamp(now - lastRenderAt, 4, 50);
  lastRenderAt = now;
  tickRender(now);
  resizeCanvas();
  drawVideo();

  if (handsReady && video.readyState >= 2 && now - lastDetectAt >= HAND_INTERVAL) {
    lastDetectAt = now;
    try {
      const result = handLandmarker.detectForVideo(video, now);
      updateTracking(result, now);
      tickHand(now);
    } catch (err) {
      console.warn('Hand tracking frame failed', err);
    }
  }

  const active = activeTracks(now);
  for (const track of active) {
    spawnHandParticles(track, now, dt);
    drawFingerTrails(track, now);
  }
  updateParticles(now, dt);
  drawScreenPulse();
  drawParticles(now);
  if (active.length >= 2) drawPortal(active[0], active[1], now);
  for (const track of active) drawHandEnergy(track, now);
  drawShockwaves(now);
  drawDebug(active, now);

  if (!handsReady) setStatus('Ładowanie precyzyjnego trackingu dłoni…');
  else if (!active.length) setStatus('Pokaż dłoń · 21 punktów + gesty');
  else {
    const pinches = active.filter(t => t.pinchDown).length;
    const portal = active.length >= 2 ? ' · PORTAL AKTYWNY' : '';
    const pinch = pinches ? ` · PINCH ×${pinches}` : '';
    setStatus(`${active.length}/2 dłonie · ${currentTheme().label}${portal}${pinch}`);
  }

  if (mediaRecorder?.state === 'recording') recBadge.textContent = `● REC ${Math.floor((now - recordingStartedAt) / 1000)}s`;
  requestAnimationFrame(renderLoop);
}

function preferredMime() {
  return ['video/mp4;codecs=avc1.42E01E','video/mp4','video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm']
    .find(type => MediaRecorder?.isTypeSupported?.(type)) || '';
}

function toggleRecording() {
  if (!running || !window.MediaRecorder || !canvas.captureStream) {
    showError('Nagrywanie nie jest dostępne w tej przeglądarce.');
    return;
  }
  if (mediaRecorder?.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  recordedChunks = [];
  const mime = preferredMime();
  const outStream = canvas.captureStream(30);
  mediaRecorder = new MediaRecorder(outStream, mime ? { mimeType: mime, videoBitsPerSecond: 9_000_000 } : { videoBitsPerSecond: 9_000_000 });
  mediaRecorder.ondataavailable = e => { if (e.data?.size) recordedChunks.push(e.data); };
  mediaRecorder.onstop = () => {
    const type = mediaRecorder.mimeType || mime || 'video/webm';
    const blob = new Blob(recordedChunks, { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `handstorm-v13-3-${Date.now()}.${type.includes('mp4') ? 'mp4' : 'webm'}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
    recordBtn.textContent = 'Nagraj';
    recBadge.hidden = true;
  };
  mediaRecorder.start(250);
  recordingStartedAt = performance.now();
  recordBtn.textContent = 'Stop';
  recBadge.hidden = false;
}

startBtn.addEventListener('click', startCamera);
recordBtn.addEventListener('click', toggleRecording);
intensityInput.addEventListener('input', () => { intensityValue.textContent = `${intensityInput.value}%`; });
effectSelect.addEventListener('change', () => {
  particles.length = 0;
  shockwaves.length = 0;
  burstEnergy = 0;
});
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(track => track.stop()));
async function removeLegacyServiceWorker() {
  if (!('serviceWorker' in navigator)) return false;
  try {
    const hadController = !!navigator.serviceWorker.controller;
    const regs = await navigator.serviceWorker.getRegistrations();
    await Promise.all(regs.map(reg => reg.unregister()));
    if ('caches' in window) {
      const keys = await caches.keys();
      await Promise.all(keys.filter(key => key.startsWith('magic-frame')).map(key => caches.delete(key)));
    }
    if (hadController && !sessionStorage.getItem('handstorm-sw-cleared-v13.3')) {
      sessionStorage.setItem('handstorm-sw-cleared-v13.3', '1');
      location.reload();
      return true;
    }
    sessionStorage.removeItem('handstorm-sw-cleared-v13.3');
  } catch (err) {
    console.warn('Legacy Service Worker cleanup failed', err);
  }
  return false;
}

// Ładuj MediaPipe od razu, ale najpierw usuń cache starego Service Workera.
(async () => {
  const reloading = await removeLegacyServiceWorker();
  if (!reloading) setTimeout(initHands, 10);
})();
updateDiag();
setStatus('Gotowy · HANDSTORM v13.3 · pobieranie modelu z kontrolą transferu…');
