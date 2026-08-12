import './styles.css';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const fxCanvas = document.getElementById('fxCanvas');
const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const effectSelect = document.getElementById('effectSelect');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');
const recBadge = document.getElementById('recBadge');

let handLandmarker = null;
let faceLandmarker = null;
let stream = null;
let running = false;
let handsReady = false;
let faceReady = false;
let loadingVision = false;
let lastHandDetect = 0;
let lastFaceDetect = 0;
let latestHands = [];
let latestHandedness = [];
let latestFace = null;
let smoothedQuad = null;
let previousRawQuad = null;
let previousRawAt = 0;
let lastValidFrameAt = 0;
let frameCounter = 0;
let cameraState = '—';
let handState = '—';
let faceState = '—';
let fxState = 'init…';

let mediaRecorder = null;
let recordedChunks = [];
let recordingStartedAt = 0;

const HAND_INTERVAL = 1000 / 30;
const FACE_INTERVAL = 1000 / 24;
const FRAME_HOLD_MS = 90;

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() { diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · twarz: ${faceState} · FX: ${fxState}`; }
function showError(text) { errorEl.hidden = !text; errorEl.textContent = text || ''; }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function avgPoint(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;
  return {
    x: valid.reduce((s, p) => s + p.x, 0) / valid.length,
    y: valid.reduce((s, p) => s + p.y, 0) / valid.length
  };
}

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

async function initVision() {
  if (loadingVision || (handsReady && faceReady)) return;
  loadingVision = true;
  handState = 'WASM…'; faceState = 'WASM…'; updateDiag();
  try {
    const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm');
    const handCommon = {
      baseOptions: { modelAssetPath: '/models/hand_landmarker.task' },
      runningMode: 'VIDEO', numHands: 2,
      minHandDetectionConfidence: 0.40,
      minHandPresenceConfidence: 0.40,
      minTrackingConfidence: 0.40
    };
    handState = 'model…'; updateDiag();
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, { ...handCommon, baseOptions: { ...handCommon.baseOptions, delegate: 'GPU' } });
      handState = 'GPU OK';
    } catch {
      handLandmarker = await HandLandmarker.createFromOptions(vision, handCommon);
      handState = 'CPU OK';
    }
    handsReady = true; updateDiag();

    const faceCommon = {
      baseOptions: { modelAssetPath: '/models/face_landmarker.task' },
      runningMode: 'VIDEO', numFaces: 1,
      minFaceDetectionConfidence: 0.42,
      minFacePresenceConfidence: 0.42,
      minTrackingConfidence: 0.42,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    };
    faceState = 'model…'; updateDiag();
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, { ...faceCommon, baseOptions: { ...faceCommon.baseOptions, delegate: 'GPU' } });
      faceState = 'GPU OK';
    } catch {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, faceCommon);
      faceState = 'CPU OK';
    }
    faceReady = true; updateDiag();
    setStatus('0/2 dłonie');
  } catch (err) {
    console.error(err);
    if (!handsReady) handState = 'BŁĄD';
    if (!faceReady) faceState = 'BŁĄD';
    updateDiag();
    showError(`MediaPipe nie wystartował.\n${err?.name || 'Error'}: ${err?.message || err}`);
    setStatus('Kamera działa • tracker niedostępny');
  } finally { loadingVision = false; }
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
    startBtn.textContent = 'Kamera działa';
    recordBtn.disabled = false;
    resizeCanvas(); initFxRenderer(); requestAnimationFrame(renderLoop); setTimeout(initVision, 50);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
recordBtn.addEventListener('click', toggleRecording);
setStatus('Gotowy • v8.1 Anime 2D'); updateDiag();
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(t => t.stop()));

function resizeCanvas() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
  if (fxCanvas.width !== vw || fxCanvas.height !== vh) { fxCanvas.width = vw; fxCanvas.height = vh; resizeFx(vw, vh); }
}
function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z || 0 }; }

// ---------- semantic hand identity ----------
let semanticMemory = { leftWrist: null, rightWrist: null };
function semanticHands(hands, handedness) {
  const items = hands.map((pts, i) => ({
    label: handedness?.[i]?.[0]?.categoryName || '',
    score: handedness?.[i]?.[0]?.score || 0,
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
    if (isolatedSpike || hardTeleport) return { ...previousRawQuad[i] };
    return { ...p };
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

    previousRawQuad = clean.map(p => ({ ...p }));
    previousRawAt = now;
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
function pointInTriangle(p, a, b, c) {
  const s = (p1, p2, p3) => (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
  const d1 = s(p, a, b), d2 = s(p, b, c), d3 = s(p, c, a);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0, hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}
function pointInFreeform(p, q) { return triangleSet(q).some(t => pointInTriangle(p, t[0], t[1], t[2])); }

function drawVideo() {
  if (!video.videoWidth) return;
  ctx.save(); ctx.translate(canvas.width, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, canvas.width, canvas.height); ctx.restore();
}
function canvasQuad(q) { return q.map(p => ({ x: p.x * canvas.width, y: p.y * canvas.height })); }
function pathQuad(q) { ctx.beginPath(); ctx.moveTo(q[0].x, q[0].y); for (let i = 1; i < q.length; i++) ctx.lineTo(q[i].x, q[i].y); ctx.closePath(); }
function triangleUnionPath(q) {
  const p = new Path2D();
  for (const tri of triangleSet(q)) {
    const t = tri.map(v => ({ x: v.x * canvas.width, y: v.y * canvas.height }));
    p.moveTo(t[0].x, t[0].y); p.lineTo(t[1].x, t[1].y); p.lineTo(t[2].x, t[2].y); p.closePath();
  }
  return p;
}

// ---------- WebGL2 anime renderer ----------
let gl = null, glProgram = null, glTexture = null;
let uTexel = null, uFaceOn = null, uLeftEye = null, uRightEye = null, uFaceCenter = null, uFaceSize = null, uMouth = null;
const VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv; out vec2 v_uv;
void main(){gl_Position=vec4(a_pos,0.0,1.0);v_uv=a_uv;}`;
const FS = `#version 300 es
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_texel;
uniform int u_faceOn;
uniform vec2 u_leftEye;
uniform vec2 u_rightEye;
uniform vec2 u_faceCenter;
uniform vec2 u_faceSize;
uniform vec2 u_mouth;
in vec2 v_uv; out vec4 outColor;

float lum(vec3 c){return dot(c,vec3(.299,.587,.114));}
vec3 sat(vec3 c,float s){float l=lum(c);return mix(vec3(l),c,s);}
vec3 poster(vec3 c,float n){return floor(c*n+.5)/n;}
float ellipseMask(vec2 uv, vec2 c, vec2 r){vec2 d=(uv-c)/max(r,vec2(.0001));return 1.0-smoothstep(.68,1.03,dot(d,d));}

vec2 eyeWarp(vec2 uv, vec2 c, vec2 radius, float strength){
  vec2 d=(uv-c)/max(radius,vec2(.0001));
  float r=length(d);
  if(r>=1.0) return uv;
  float t=1.0-r;
  float s=1.0-strength*t*t*(2.0-t);
  return c+(uv-c)*s;
}

vec2 animeWarp(vec2 uv){
  if(u_faceOn==0) return uv;
  vec2 x=uv;
  vec2 eyeR=vec2(max(u_faceSize.x*.205,u_faceSize.y*.12), max(u_faceSize.y*.135,u_faceSize.x*.09));
  x=eyeWarp(x,u_leftEye,eyeR,.38);
  x=eyeWarp(x,u_rightEye,eyeR,.38);

  vec2 d=(x-u_faceCenter)/max(u_faceSize,vec2(.001));
  float faceInside=1.0-smoothstep(.66,1.02,length(vec2(d.x*1.05,d.y*.84)));
  float lower=smoothstep(-.06,.62,d.y)*faceInside;
  float chin=smoothstep(.20,.72,d.y)*faceInside;
  // sample wider source coordinates -> visually narrower anime jaw
  x.x=u_faceCenter.x+(x.x-u_faceCenter.x)*(1.0+.30*lower);
  // slightly shorter lower face / smaller chin
  x.y=u_faceCenter.y+(x.y-u_faceCenter.y)*(1.0+.14*chin);
  return clamp(x,vec2(.001),vec2(.999));
}

vec3 sampleWarp(vec2 uv){return texture(u_tex,animeWarp(uv)).rgb;}

void main(){
  vec2 uv=animeWarp(v_uv);
  vec3 src=texture(u_tex,uv).rgb;

  float faceM=0.0;
  if(u_faceOn==1) faceM=ellipseMask(v_uv,u_faceCenter,u_faceSize*vec2(.60,.72));

  // Strong local smoothing: deliberately removes photographic skin/beard microtexture.
  vec3 b1=(sampleWarp(v_uv+u_texel*vec2(-2.,0.))+sampleWarp(v_uv+u_texel*vec2(2.,0.))+
           sampleWarp(v_uv+u_texel*vec2(0.,-2.))+sampleWarp(v_uv+u_texel*vec2(0.,2.))+
           sampleWarp(v_uv+u_texel*vec2(-1.,-1.))+sampleWarp(v_uv+u_texel*vec2(1.,-1.))+
           sampleWarp(v_uv+u_texel*vec2(-1.,1.))+sampleWarp(v_uv+u_texel*vec2(1.,1.))+src*4.0)/12.0;
  vec3 b2=(sampleWarp(v_uv+u_texel*vec2(-4.,0.))+sampleWarp(v_uv+u_texel*vec2(4.,0.))+
           sampleWarp(v_uv+u_texel*vec2(0.,-4.))+sampleWarp(v_uv+u_texel*vec2(0.,4.))+b1*4.0)/8.0;
  float photoDetail=clamp(length(src-b1)*4.0,0.0,1.0);
  vec3 c=mix(src,b1,.52+.28*faceM);
  c=mix(c,b2,(.20+.34*faceM)*(1.0-photoDetail*.28));

  // Sobel line extraction after smoothing.
  float tl=lum(sampleWarp(v_uv+u_texel*vec2(-2.,-2.)));
  float tc=lum(sampleWarp(v_uv+u_texel*vec2(0.,-2.)));
  float tr=lum(sampleWarp(v_uv+u_texel*vec2(2.,-2.)));
  float ml=lum(sampleWarp(v_uv+u_texel*vec2(-2.,0.)));
  float mr=lum(sampleWarp(v_uv+u_texel*vec2(2.,0.)));
  float bl=lum(sampleWarp(v_uv+u_texel*vec2(-2.,2.)));
  float bc=lum(sampleWarp(v_uv+u_texel*vec2(0.,2.)));
  float br=lum(sampleWarp(v_uv+u_texel*vec2(2.,2.)));
  float gx=-tl-2.*ml-bl+tr+2.*mr+br;
  float gy=-tl-2.*tc-tr+bl+2.*bc+br;
  float edge=clamp(length(vec2(gx,gy))*2.35,0.,1.);

  // Reduced palette + brighter anime midtones.
  c=pow(max(c,vec3(0.0)),vec3(.88));
  c=sat(c,1.20);
  vec3 outc=poster(c,5.0);
  float l=lum(outc);

  // Skin/face treatment: warm paper-like highlights, cool lavender shadows.
  vec3 animeHi=vec3(1.00,.91,.88);
  vec3 animeShadow=vec3(.56,.49,.66);
  float highlight=smoothstep(.55,.88,l)*faceM;
  float shadow=(1.0-smoothstep(.28,.58,l))*faceM;
  outc=mix(outc,mix(outc,animeHi,.19),highlight*.64);
  outc=mix(outc,mix(outc,animeShadow,.20),shadow*.55);

  // Compress beard/hair microtexture into larger dark graphic masses.
  float darkRegion=1.0-smoothstep(.18,.44,l);
  outc=mix(outc,poster(b2,4.0)*vec3(.88,.90,.98),darkRegion*.34);

  // Thin dark-purple linework, suppressing tiny photographic noise.
  float line=smoothstep(.21,.52,edge)*(1.0-smoothstep(.48,.84,photoDetail));
  vec3 ink=vec3(.055,.040,.085);
  outc=mix(outc,ink,line*(.76+.10*faceM));

  if(u_faceOn==1){
    float le=ellipseMask(v_uv,u_leftEye,vec2(u_faceSize.x*.145,u_faceSize.y*.086));
    float re=ellipseMask(v_uv,u_rightEye,vec2(u_faceSize.x*.145,u_faceSize.y*.086));
    float eyes=max(le,re);
    float eyeLum=lum(sampleWarp(v_uv));
    // cleaner, brighter anime eye whites
    outc=mix(outc,vec3(.98,.97,1.0),eyes*smoothstep(.46,.73,eyeLum)*.50);
    // deep iris / lash impression
    outc=mix(outc,vec3(.045,.035,.075),eyes*(1.0-smoothstep(.24,.50,eyeLum))*.58);
    // subtle cool eye tint
    outc=mix(outc,vec3(.22,.32,.48),eyes*smoothstep(.26,.54,eyeLum)*(1.0-smoothstep(.66,.88,eyeLum))*.18);

    float mouth=ellipseMask(v_uv,u_mouth,vec2(u_faceSize.x*.17,u_faceSize.y*.057));
    outc=mix(outc,vec3(.58,.20,.29),mouth*(1.0-smoothstep(.46,.70,eyeLum))*.16);
  }

  outColor=vec4(clamp(outc,0.,1.),1.);
}`;

function compileShader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function initFxRenderer() {
  try {
    gl = fxCanvas.getContext('webgl2', { alpha: false, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 niedostępny');
    glProgram = gl.createProgram(); gl.attachShader(glProgram, compileShader(gl.VERTEX_SHADER, VS)); gl.attachShader(glProgram, compileShader(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(glProgram);
    if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(glProgram));
    gl.useProgram(glProgram);
    const verts = new Float32Array([-1,-1,0,1, 1,-1,1,1, -1,1,0,0, -1,1,0,0, 1,-1,1,1, 1,1,1,0]);
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(glProgram, 'a_pos'), aUv = gl.getAttribLocation(glProgram, 'a_uv');
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(aUv); gl.vertexAttribPointer(aUv, 2, gl.FLOAT, false, 16, 8);
    glTexture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, glTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    uTexel = gl.getUniformLocation(glProgram, 'u_texel');
    uFaceOn = gl.getUniformLocation(glProgram, 'u_faceOn');
    uLeftEye = gl.getUniformLocation(glProgram, 'u_leftEye');
    uRightEye = gl.getUniformLocation(glProgram, 'u_rightEye');
    uFaceCenter = gl.getUniformLocation(glProgram, 'u_faceCenter');
    uFaceSize = gl.getUniformLocation(glProgram, 'u_faceSize');
    uMouth = gl.getUniformLocation(glProgram, 'u_mouth');
    fxState = 'WebGL2 Anime OK'; updateDiag();
  } catch (e) {
    console.warn(e); fxState = 'Canvas fallback'; updateDiag(); gl = null;
  }
}
function resizeFx(w, h) { if (gl) gl.viewport(0, 0, w, h); }

function rawFaceData(q) {
  if (!latestFace?.length) return null;
  const raw = latestFace;
  const display = raw.map(mirrorPoint);
  const minX = Math.min(...display.map(p => p.x)), maxX = Math.max(...display.map(p => p.x));
  const minY = Math.min(...display.map(p => p.y)), maxY = Math.max(...display.map(p => p.y));
  const displayCenter = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  if (!pointInFreeform(displayCenter, q)) return null;

  const leftEye = avgPoint([raw[33], raw[133], raw[159], raw[145]]);
  const rightEye = avgPoint([raw[362], raw[263], raw[386], raw[374]]);
  const mouth = avgPoint([raw[13], raw[14], raw[61], raw[291]]);
  const minRX = Math.min(...raw.map(p => p.x)), maxRX = Math.max(...raw.map(p => p.x));
  const minRY = Math.min(...raw.map(p => p.y)), maxRY = Math.max(...raw.map(p => p.y));
  return {
    leftEye, rightEye, mouth,
    center: { x: (minRX + maxRX) / 2, y: (minRY + maxRY) / 2 },
    size: { x: maxRX - minRX, y: maxRY - minRY }
  };
}

function renderFx(q) {
  if (!gl || video.readyState < 2) return false;
  gl.useProgram(glProgram); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, glTexture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video); } catch { return false; }
  gl.uniform2f(uTexel, 1 / Math.max(1, fxCanvas.width), 1 / Math.max(1, fxCanvas.height));
  const face = rawFaceData(q);
  gl.uniform1i(uFaceOn, face ? 1 : 0);
  if (face) {
    gl.uniform2f(uLeftEye, face.leftEye.x, face.leftEye.y);
    gl.uniform2f(uRightEye, face.rightEye.x, face.rightEye.y);
    gl.uniform2f(uFaceCenter, face.center.x, face.center.y);
    gl.uniform2f(uFaceSize, Math.max(.001, face.size.x), Math.max(.001, face.size.y));
    gl.uniform2f(uMouth, face.mouth.x, face.mouth.y);
  } else {
    gl.uniform2f(uLeftEye, 0, 0); gl.uniform2f(uRightEye, 0, 0); gl.uniform2f(uFaceCenter, 0, 0); gl.uniform2f(uFaceSize, 1, 1); gl.uniform2f(uMouth, 0, 0);
  }
  gl.drawArrays(gl.TRIANGLES, 0, 6); return true;
}

function applyAnimeFx(q) {
  if (effectSelect.value === 'original') return;
  const ok = renderFx(q);
  const path = triangleUnionPath(q);
  ctx.save(); ctx.clip(path); ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
  if (ok) {
    ctx.drawImage(fxCanvas, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.filter = 'saturate(1.5) contrast(1.2) brightness(1.06)';
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }
  ctx.restore();
}


const LEFT_EYE_LINE = [33,160,158,133,153,144,33];
const RIGHT_EYE_LINE = [362,385,387,263,373,380,362];
const LEFT_BROW = [70,63,105,66,107];
const RIGHT_BROW = [336,296,334,293,300];
const UPPER_LIP = [61,185,40,39,37,0,267,269,270,409,291];
const LOWER_LIP = [61,146,91,181,84,17,314,405,321,375,291];
const NOSE_LINE = [168,6,197,195,5,4];

function drawLandmarkLine(pts, indices, width, color, close = false) {
  const selected = indices.map(i => pts[i]).filter(Boolean);
  if (selected.length < 2) return;
  ctx.beginPath();
  selected.forEach((p, i) => {
    const x = p.x * canvas.width, y = p.y * canvas.height;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  if (close) ctx.closePath();
  ctx.lineWidth = width;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = color; ctx.stroke();
}

function drawAnimeLineArt(q) {
  if (!latestFace?.length || !rawFaceData(q)) return;
  const pts = latestFace.map(mirrorPoint);
  const mask = triangleUnionPath(q);
  const w = Math.max(1.6, canvas.width * .00225);
  const ink = 'rgba(31,18,43,.82)';
  ctx.save(); ctx.clip(mask);

  // clean face silhouette and graphic facial features
  drawLandmarkLine(pts, FACE_OVAL, w * 1.05, 'rgba(39,24,48,.64)', true);
  drawLandmarkLine(pts, LEFT_BROW, w * 1.65, ink);
  drawLandmarkLine(pts, RIGHT_BROW, w * 1.65, ink);
  drawLandmarkLine(pts, LEFT_EYE_LINE, w * 1.48, 'rgba(24,15,40,.92)', true);
  drawLandmarkLine(pts, RIGHT_EYE_LINE, w * 1.48, 'rgba(24,15,40,.92)', true);
  drawLandmarkLine(pts, NOSE_LINE, w * .55, 'rgba(70,43,62,.42)');
  drawLandmarkLine(pts, UPPER_LIP, w * .88, 'rgba(91,35,55,.72)');
  drawLandmarkLine(pts, LOWER_LIP, w * .68, 'rgba(125,51,71,.52)');

  // iris / catch-light marks based on eye centers
  const face = rawFaceData(q);
  if (face) {
    for (const e of [mirrorPoint(face.leftEye), mirrorPoint(face.rightEye)]) {
      const ex = e.x * canvas.width, ey = e.y * canvas.height;
      const r = Math.max(3.5, canvas.width * Math.max(face.size.x, .08) * .018);
      ctx.fillStyle = 'rgba(32,34,65,.80)'; ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,.88)'; ctx.beginPath(); ctx.arc(ex-r*.28, ey-r*.30, r*.28, 0, Math.PI * 2); ctx.fill();
    }
  }
  ctx.restore();
}

function drawFrame(q) {
  const cq = canvasQuad(q);
  ctx.save(); pathQuad(cq); ctx.lineWidth = Math.max(3, canvas.width * .004); ctx.strokeStyle = 'rgba(255,255,255,.98)';
  ctx.shadowBlur = 14; ctx.shadowColor = 'rgba(255,255,255,.48)'; ctx.stroke(); ctx.restore();
}

const FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
function drawDebug(semantic, q, rawQuad) {
  if (!debugToggle.checked) return;
  ctx.save(); ctx.fillStyle = 'rgba(0,255,180,.9)';
  for (const h of semantic) for (const p of h.pts) { ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 2.6, 0, Math.PI * 2); ctx.fill(); }
  if (rawQuad) {
    const rq = canvasQuad(rawQuad); ctx.save(); ctx.setLineDash([7,7]); ctx.strokeStyle = 'rgba(0,255,255,.95)'; ctx.lineWidth = 2; pathQuad(rq); ctx.stroke();
    ctx.fillStyle = 'rgba(0,255,255,.95)'; rq.forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#001b20'; ctx.font = '12px sans-serif'; ctx.fillText(`P${i}`, p.x + 9, p.y - 8); ctx.fillStyle = 'rgba(0,255,255,.95)'; }); ctx.restore();
  }
  if (latestFace) {
    const pts = latestFace.map(mirrorPoint); ctx.fillStyle = 'rgba(255,220,0,.58)';
    for (let i = 0; i < pts.length; i += 8) { const p = pts[i]; ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 1.5, 0, Math.PI * 2); ctx.fill(); }
    ctx.strokeStyle = 'rgba(255,220,0,.7)'; ctx.lineWidth = 1.2; ctx.beginPath(); FACE_OVAL.forEach((i, k) => { const p = pts[i]; if (!p) return; k ? ctx.lineTo(p.x * canvas.width, p.y * canvas.height) : ctx.moveTo(p.x * canvas.width, p.y * canvas.height); }); ctx.closePath(); ctx.stroke();
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
    showError('Ta przeglądarka nie obsługuje nagrywania canvas przez MediaRecorder.');
    return;
  }
  if (mediaRecorder?.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
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
    const a = document.createElement('a'); a.href = url; a.download = `magic-frame-anime-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000);
    recordBtn.textContent = 'Nagraj'; recordBtn.classList.remove('recording'); recBadge.hidden = true;
  };
  mediaRecorder.start(250); recordingStartedAt = performance.now();
  recordBtn.textContent = 'Stop'; recordBtn.classList.add('recording'); recBadge.hidden = false;
}

async function renderLoop(ts) {
  if (!running) return;
  frameCounter++; resizeCanvas(); drawVideo();
  if (handsReady && handLandmarker && video.readyState >= 2 && ts - lastHandDetect > HAND_INTERVAL) {
    lastHandDetect = ts;
    try { const r = handLandmarker.detectForVideo(video, ts); latestHands = r.landmarks || []; latestHandedness = r.handedness || []; } catch (e) { console.warn(e); }
  }
  if (faceReady && faceLandmarker && video.readyState >= 2 && ts - lastFaceDetect > FACE_INTERVAL) {
    lastFaceDetect = ts;
    try { const r = faceLandmarker.detectForVideo(video, ts); latestFace = r.faceLandmarks?.[0] || null; } catch (e) { console.warn(e); }
  }
  const semantic = semanticHands(latestHands, latestHandedness);
  const rawQuad = measureFreeformQuad(semantic);
  const q = smoothFreeformQuad(rawQuad, ts);
  if (q) {
    applyAnimeFx(q); drawAnimeLineArt(q); drawFrame(q);
    const faceInside = !!rawFaceData(q);
    setStatus(`2/2 dłonie · Anime 2D v8.1${faceInside ? ' · face warp' : ''}`);
  } else if (handsReady) {
    const count = Math.min(2, semantic.length); setStatus(`${count}/2 dłonie${count === 2 ? ' · rozsuń palce' : ''}`);
  }
  if (mediaRecorder?.state === 'recording') {
    const secs = Math.floor((ts - recordingStartedAt) / 1000); recBadge.textContent = `● REC ${secs}s`;
  }
  drawDebug(semantic, q, rawQuad);
  requestAnimationFrame(renderLoop);
}
