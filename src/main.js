import './styles.css';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const fxCanvas = document.getElementById('fxCanvas');
const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
const startBtn = document.getElementById('startBtn');
const effectSelect = document.getElementById('effectSelect');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');

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
let virtualFrame = null;
let lastValidFrameAt = 0;
let frameCounter = 0;
let previousMeasurement = null;
let previousMeasurementAt = 0;
let cameraState = '—';
let handState = '—';
let faceState = '—';
let fxState = 'init…';

const HAND_INTERVAL = 1000 / 30;
const FACE_INTERVAL = 1000 / 18;
const FRAME_HOLD_MS = 90;

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() { diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · twarz: ${faceState} · FX: ${fxState}`; }
function showError(text) { errorEl.hidden = !text; errorEl.textContent = text || ''; }

function readableCameraError(err) {
  const name = err?.name || 'Error';
  if (name === 'NotAllowedError' || name === 'PermissionDeniedError') return 'Brak zgody na kamerę. W Edge kliknij ikonę kamery przy adresie → Zezwalaj, potem odśwież.';
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
    running = true; cameraState = 'OK'; updateDiag(); startBtn.textContent = 'Kamera działa';
    resizeCanvas(); initFxRenderer(); requestAnimationFrame(renderLoop); setTimeout(initVision, 50);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
setStatus('Gotowy • v7.2 freeform quad'); updateDiag();
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(t => t.stop()));

function resizeCanvas() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
  if (fxCanvas.width !== vw || fxCanvas.height !== vh) { fxCanvas.width = vw; fxCanvas.height = vh; resizeFx(vw, vh); }
}
function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z || 0 }; }
function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function dist(a,b){ return Math.hypot(a.x-b.x,a.y-b.y); }

// v7.2: zachowujemy tożsamość dłoni. Nie sortujemy ich co klatkę po X,
// bo przy skrzyżowaniu rąk zamieniłoby to narożniki i "naprawiło" twist.
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

  // Jeżeli model nie da dwóch unikalnych etykiet, przypisujemy dłonie do
  // poprzednich nadgarstków. Dopiero bez pamięci używamy pozycji X.
  if (!left || !right || left === right) {
    if (semanticMemory.leftWrist && semanticMemory.rightWrist) {
      const a = items[0], b = items[1];
      const costAB = dist(a.pts[0], semanticMemory.leftWrist) + dist(b.pts[0], semanticMemory.rightWrist);
      const costBA = dist(b.pts[0], semanticMemory.leftWrist) + dist(a.pts[0], semanticMemory.rightWrist);
      [left, right] = costAB <= costBA ? [a,b] : [b,a];
    } else {
      const byX = [...items].sort((a,b)=>a.pts[0].x-b.pts[0].x);
      [left, right] = [byX[0], byX[1]];
    }
  }
  semanticMemory.leftWrist = { ...left.pts[0] };
  semanticMemory.rightWrist = { ...right.pts[0] };
  return [left, right];
}

// P0 = lewy index, P1 = prawy index, P2 = prawy thumb, P3 = lewy thumb.
// Kolejność jest SEMANTYCZNA i nigdy nie jest sortowana po obwodzie.
// Dzięki temu quad może być wypukły, wklęsły albo samoprzecinający (bow-tie / |><|).
function measureFreeformQuad(semantic) {
  if (semantic.length < 2) return null;
  const L = semantic[0]?.pts, R = semantic[1]?.pts;
  if (!L?.[4] || !L?.[8] || !R?.[4] || !R?.[8]) return null;
  const q = [
    { ...L[8] },
    { ...R[8] },
    { ...R[4] },
    { ...L[4] }
  ];

  // Tylko minimalne zabezpieczenie przed przypadkiem, gdy dłonie leżą niemal w jednym punkcie.
  const xs=q.map(p=>p.x), ys=q.map(p=>p.y);
  const span=Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
  if (span < 0.055) return null;
  return q;
}

let smoothedQuad = null;
let previousRawQuad = null;
let previousRawAt = 0;
function smoothFreeformQuad(next, now) {
  if (next) {
    lastValidFrameAt = now;
    const target = next.map(p=>({x:p.x,y:p.y,z:p.z||0}));

    // Delikatna predykcja każdego narożnika z osobna (~część klatki), bez center/angle/aspect ratio.
    if (previousRawQuad && previousRawAt > 0) {
      const dt=Math.max(1,now-previousRawAt);
      const predictionMs=Math.min(12,dt*0.32);
      const k=predictionMs/dt;
      for(let i=0;i<4;i++){
        target[i].x=clamp(next[i].x+(next[i].x-previousRawQuad[i].x)*k,0,1);
        target[i].y=clamp(next[i].y+(next[i].y-previousRawQuad[i].y)*k,0,1);
      }
    }
    previousRawQuad = next.map(p=>({...p}));
    previousRawAt = now;

    if (!smoothedQuad) {
      smoothedQuad = target;
      return smoothedQuad;
    }

    for(let i=0;i<4;i++){
      const jump=dist(smoothedQuad[i],target[i]);
      // 88% nowej próbki przy spokojnym ruchu, do 100% przy szybkim ruchu.
      const alpha=clamp(0.88+jump*2.6,0.88,1.0);
      smoothedQuad[i].x=lerp(smoothedQuad[i].x,target[i].x,alpha);
      smoothedQuad[i].y=lerp(smoothedQuad[i].y,target[i].y,alpha);
      smoothedQuad[i].z=target[i].z;
    }
    return smoothedQuad;
  }

  if (smoothedQuad && now-lastValidFrameAt <= FRAME_HOLD_MS) return smoothedQuad;
  smoothedQuad=null; previousRawQuad=null; previousRawAt=0;
  return null;
}

function triangleSet(q){
  // Stała triangulacja semantyczna; nie zależy od convexity ani kolejności ekranowej.
  return [[q[0],q[1],q[2]],[q[0],q[2],q[3]]];
}
function pointInTriangle(p,a,b,c){
  const s=(p1,p2,p3)=>(p1.x-p3.x)*(p2.y-p3.y)-(p2.x-p3.x)*(p1.y-p3.y);
  const d1=s(p,a,b), d2=s(p,b,c), d3=s(p,c,a);
  const hasNeg=(d1<0)||(d2<0)||(d3<0), hasPos=(d1>0)||(d2>0)||(d3>0);
  return !(hasNeg&&hasPos);
}
function pointInFreeform(p,q){ return triangleSet(q).some(t=>pointInTriangle(p,t[0],t[1],t[2])); }

function drawVideo() {
  if(!video.videoWidth)return;
  ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
}
function canvasQuad(q){ return q.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height})); }
function pathQuad(q){ctx.beginPath();ctx.moveTo(q[0].x,q[0].y);for(let i=1;i<q.length;i++)ctx.lineTo(q[i].x,q[i].y);ctx.closePath();}
function pathTriangle(t){ctx.beginPath();ctx.moveTo(t[0].x,t[0].y);ctx.lineTo(t[1].x,t[1].y);ctx.lineTo(t[2].x,t[2].y);ctx.closePath();}
function forEachCanvasTriangle(q,fn){for(const tri of triangleSet(q)){fn(tri.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height})));}}

// ---------- WebGL local toon renderer ----------
let gl=null, glProgram=null, glTexture=null, glPos=null, glUv=null, uMode=null, uTexel=null;
const VS=`#version 300 es
in vec2 a_pos; in vec2 a_uv; out vec2 v_uv;
void main(){gl_Position=vec4(a_pos,0.0,1.0);v_uv=a_uv;}`;
const FS=`#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_texel; uniform int u_mode; in vec2 v_uv; out vec4 outColor;
float lum(vec3 c){return dot(c,vec3(.299,.587,.114));}
vec3 sat(vec3 c,float s){float l=lum(c);return mix(vec3(l),c,s);}
vec3 poster(vec3 c,float n){return floor(c*n+.5)/n;}
void main(){
  vec3 c=texture(u_tex,v_uv).rgb;
  vec3 b=(texture(u_tex,v_uv+vec2(u_texel.x,0)).rgb+texture(u_tex,v_uv-vec2(u_texel.x,0)).rgb+texture(u_tex,v_uv+vec2(0,u_texel.y)).rgb+texture(u_tex,v_uv-vec2(0,u_texel.y)).rgb+c*2.0)/6.0;
  float tl=lum(texture(u_tex,v_uv+u_texel*vec2(-1.,-1.)).rgb), tc=lum(texture(u_tex,v_uv+u_texel*vec2(0.,-1.)).rgb), tr=lum(texture(u_tex,v_uv+u_texel*vec2(1.,-1.)).rgb);
  float ml=lum(texture(u_tex,v_uv+u_texel*vec2(-1.,0.)).rgb), mr=lum(texture(u_tex,v_uv+u_texel*vec2(1.,0.)).rgb);
  float bl=lum(texture(u_tex,v_uv+u_texel*vec2(-1.,1.)).rgb), bc=lum(texture(u_tex,v_uv+u_texel*vec2(0.,1.)).rgb), br=lum(texture(u_tex,v_uv+u_texel*vec2(1.,1.)).rgb);
  float gx=-tl-2.*ml-bl+tr+2.*mr+br, gy=-tl-2.*tc-tr+bl+2.*bc+br;
  float edge=clamp(length(vec2(gx,gy))*2.5,0.,1.);
  vec3 outc=c;
  if(u_mode==0){ outc=poster(sat(mix(c,b,.58),1.45),7.0); outc=mix(outc,vec3(.10,.08,.14),smoothstep(.18,.52,edge)*.78); outc+=vec3(.035,.015,.025); }
  else if(u_mode==1){ outc=poster(sat(mix(c,b,.35),1.28),5.0); outc*=1.0-smoothstep(.13,.43,edge)*.68; outc=pow(outc,vec3(.92)); }
  else if(u_mode==2){ outc=sat(mix(c,b,.78),.72); outc=poster(outc,9.0); outc=pow(outc,vec3(.88)); }
  else if(u_mode==3){ vec3 x=sat(c,1.8); outc=vec3(x.b*.85+x.r*.15,x.g*.45+x.b*.55,x.r*.25+x.b*.95); outc=poster(outc,8.0); outc+=vec3(0.,.08,.11)*(1.-lum(outc)); }
  else if(u_mode==4){ float l=lum(c); l=poster(vec3(l),5.0).r; l*=1.0-smoothstep(.12,.38,edge)*.9; outc=vec3(l); }
  else { outc=poster(sat(c,1.7),6.0); outc=mix(outc,vec3(outc.r*.7,outc.g*.9,outc.b*1.25),.35); }
  outColor=vec4(clamp(outc,0.,1.),1.);
}`;
function compileShader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function initFxRenderer(){
  try{
    gl=fxCanvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:true});
    if(!gl)throw new Error('WebGL2 niedostępny');
    glProgram=gl.createProgram();gl.attachShader(glProgram,compileShader(gl.VERTEX_SHADER,VS));gl.attachShader(glProgram,compileShader(gl.FRAGMENT_SHADER,FS));gl.linkProgram(glProgram);if(!gl.getProgramParameter(glProgram,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(glProgram));gl.useProgram(glProgram);
    const verts=new Float32Array([-1,-1,0,1, 1,-1,1,1, -1,1,0,0, -1,1,0,0, 1,-1,1,1, 1,1,1,0]);
    const buf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buf);gl.bufferData(gl.ARRAY_BUFFER,verts,gl.STATIC_DRAW);
    glPos=gl.getAttribLocation(glProgram,'a_pos');glUv=gl.getAttribLocation(glProgram,'a_uv');gl.enableVertexAttribArray(glPos);gl.vertexAttribPointer(glPos,2,gl.FLOAT,false,16,0);gl.enableVertexAttribArray(glUv);gl.vertexAttribPointer(glUv,2,gl.FLOAT,false,16,8);
    glTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,glTexture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    uMode=gl.getUniformLocation(glProgram,'u_mode');uTexel=gl.getUniformLocation(glProgram,'u_texel');fxState='WebGL2 OK';updateDiag();
  }catch(e){console.warn(e);fxState='Canvas fallback';updateDiag();gl=null;}
}
function resizeFx(w,h){if(gl){gl.viewport(0,0,w,h);}}
function modeNumber(effect){return ({anime:0,comic:1,clay:2,cyber:3,bw:4,glitch:5})[effect]??0;}
function renderFx(effect){
  if(!gl||video.readyState<2)return false;
  gl.useProgram(glProgram);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,glTexture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
  try{gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,video);}catch{return false;}
  gl.uniform1i(uMode,modeNumber(effect));gl.uniform2f(uTexel,1/Math.max(1,fxCanvas.width),1/Math.max(1,fxCanvas.height));gl.drawArrays(gl.TRIANGLES,0,6);return true;
}
function drawFallbackEffect(q,effect){
  const filters={anime:'saturate(1.6) contrast(1.25) brightness(1.08)',comic:'contrast(1.65) saturate(1.4)',clay:'saturate(.7) contrast(1.1) brightness(1.12)',cyber:'hue-rotate(250deg) saturate(2.1) contrast(1.35)',glitch:'saturate(2) contrast(1.5)',bw:'grayscale(1) contrast(1.35)'};
  forEachCanvasTriangle(q, tri=>{
    ctx.save();pathTriangle(tri);ctx.clip();ctx.translate(canvas.width,0);ctx.scale(-1,1);
    ctx.filter=filters[effect]||'none';ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
  });
}
function applyFx(q,effect){
  const ok=renderFx(effect);
  if(ok){
    // Dwa niezależne trójkąty: self-intersection jest legalna i nie jest "naprawiana".
    forEachCanvasTriangle(q, tri=>{
      ctx.save();pathTriangle(tri);ctx.clip();ctx.translate(canvas.width,0);ctx.scale(-1,1);
      ctx.drawImage(fxCanvas,0,0,canvas.width,canvas.height);ctx.restore();
    });
  } else drawFallbackEffect(q,effect);
  if(effect==='glitch')drawGlitch(canvasQuad(q));
}

function faceBounds(face){
  if(!face?.length)return null;const pts=face.map(mirrorPoint);let minX=1,minY=1,maxX=0,maxY=0;
  for(const p of pts){minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y);}return {pts,minX,minY,maxX,maxY,w:maxX-minX,h:maxY-minY,cx:(minX+maxX)/2,cy:(minY+maxY)/2};
}
function drawEyePatch(facePts, indices, scale=1.16){
  const pts=indices.map(i=>facePts[i]).filter(Boolean);if(!pts.length)return;
  let minX=Math.min(...pts.map(p=>p.x)),maxX=Math.max(...pts.map(p=>p.x)),minY=Math.min(...pts.map(p=>p.y)),maxY=Math.max(...pts.map(p=>p.y));
  const padX=(maxX-minX)*1.25,padY=(maxY-minY)*1.6;minX-=padX;maxX+=padX;minY-=padY;maxY+=padY;
  const sx=(1-maxX)*video.videoWidth,sy=minY*video.videoHeight,sw=(maxX-minX)*video.videoWidth,sh=(maxY-minY)*video.videoHeight;
  const dx=minX*canvas.width,dy=minY*canvas.height,dw=(maxX-minX)*canvas.width,dh=(maxY-minY)*canvas.height,cx=dx+dw/2,cy=dy+dh/2;
  ctx.save();ctx.beginPath();ctx.ellipse(cx,cy,dw*.50,dh*.55,0,0,Math.PI*2);ctx.clip();ctx.translate(cx,cy);ctx.scale(scale,scale);ctx.translate(-cx,-cy);ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,sx,sy,sw,sh,canvas.width-(dx+dw),dy,dw,dh);ctx.restore();
}
const FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
function drawFaceEnhancement(q,effect){
  const fb=faceBounds(latestFace);if(!fb)return;const c={x:fb.cx,y:fb.cy};if(!pointInFreeform(c,q))return;
  const pts=fb.pts;ctx.save();ctx.beginPath();for(const tri of triangleSet(q)){const t=tri.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height}));ctx.moveTo(t[0].x,t[0].y);ctx.lineTo(t[1].x,t[1].y);ctx.lineTo(t[2].x,t[2].y);ctx.closePath();}ctx.clip();
  if(effect==='anime'){
    drawEyePatch(pts,[33,133,159,145],1.18);drawEyePatch(pts,[362,263,386,374],1.18);
    ctx.strokeStyle='rgba(28,20,38,.55)';ctx.lineWidth=Math.max(1.4,canvas.width*.002);for(const loop of [[33,160,158,133,153,144],[362,385,387,263,373,380]]){ctx.beginPath();loop.forEach((i,k)=>{const p=pts[i];if(!p)return;k?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height)});ctx.stroke();}
  }
  if(effect==='comic'){
    ctx.strokeStyle='rgba(10,10,10,.52)';ctx.lineWidth=Math.max(1.5,canvas.width*.0025);ctx.beginPath();FACE_OVAL.forEach((i,k)=>{const p=pts[i];if(!p)return;k?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height)});ctx.closePath();ctx.stroke();
  }
  if(effect==='clay'){
    const x=fb.minX*canvas.width,y=fb.minY*canvas.height,w=fb.w*canvas.width,h=fb.h*canvas.height;const g=ctx.createRadialGradient(x+w*.5,y+h*.45,0,x+w*.5,y+h*.45,Math.max(w,h)*.6);g.addColorStop(0,'rgba(255,225,205,.16)');g.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=g;ctx.fillRect(x,y,w,h);
  }
  ctx.restore();
}
function drawGlitch(cq){if(frameCounter%3!==0)return;let minX=Math.min(...cq.map(p=>p.x)),maxX=Math.max(...cq.map(p=>p.x)),minY=Math.min(...cq.map(p=>p.y)),maxY=Math.max(...cq.map(p=>p.y));for(const tri of [[cq[0],cq[1],cq[2]],[cq[0],cq[2],cq[3]]]){ctx.save();pathTriangle(tri);ctx.clip();for(let i=0;i<3;i++){const h=Math.max(2,Math.random()*(maxY-minY)*.03),y=minY+Math.random()*Math.max(1,maxY-minY-h),off=(Math.random()-.5)*16;try{const img=ctx.getImageData(minX,y,maxX-minX,h);ctx.putImageData(img,minX+off,y)}catch{}}ctx.restore();}}
function drawFrame(q){const cq=canvasQuad(q);ctx.save();pathQuad(cq);ctx.lineWidth=Math.max(3,canvas.width*.004);ctx.strokeStyle='rgba(255,255,255,.98)';ctx.shadowBlur=16;ctx.shadowColor='rgba(255,255,255,.55)';ctx.stroke();ctx.restore();}
function drawDebug(semantic,q,rawQuad){
  if(!debugToggle.checked)return;ctx.save();ctx.fillStyle='rgba(0,255,180,.9)';
  for(const h of semantic)for(const p of h.pts){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,2.6,0,Math.PI*2);ctx.fill();}
  if(rawQuad){const rq=canvasQuad(rawQuad);ctx.save();ctx.setLineDash([7,7]);ctx.strokeStyle='rgba(0,255,255,.95)';ctx.lineWidth=2;pathQuad(rq);ctx.stroke();ctx.fillStyle='rgba(0,255,255,.95)';rq.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='#001b20';ctx.font='12px sans-serif';ctx.fillText(`P${i}`,p.x+9,p.y-8);ctx.fillStyle='rgba(0,255,255,.95)';});ctx.restore();}
  if(latestFace){const pts=latestFace.map(mirrorPoint);ctx.fillStyle='rgba(255,220,0,.58)';for(let i=0;i<pts.length;i+=8){const p=pts[i];ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,1.5,0,Math.PI*2);ctx.fill();}ctx.fillStyle='rgba(255,220,0,.08)';ctx.beginPath();FACE_OVAL.forEach((i,k)=>{const p=pts[i];if(!p)return;k?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height)});ctx.closePath();ctx.fill();}
  if(q){const cq=canvasQuad(q);ctx.strokeStyle='rgba(80,180,255,.9)';ctx.lineWidth=2;pathQuad(cq);ctx.stroke();}ctx.restore();
}

async function renderLoop(ts){
  if(!running)return;frameCounter++;resizeCanvas();drawVideo();
  if(handsReady&&handLandmarker&&video.readyState>=2&&ts-lastHandDetect>HAND_INTERVAL){lastHandDetect=ts;try{const r=handLandmarker.detectForVideo(video,ts);latestHands=r.landmarks||[];latestHandedness=r.handedness||[];}catch(e){console.warn(e)}}
  if(faceReady&&faceLandmarker&&video.readyState>=2&&ts-lastFaceDetect>FACE_INTERVAL){lastFaceDetect=ts;try{const r=faceLandmarker.detectForVideo(video,ts);latestFace=r.faceLandmarks?.[0]||null;}catch(e){console.warn(e)}}
  const semantic=semanticHands(latestHands,latestHandedness);const rawQuad=measureFreeformQuad(semantic);const q=smoothFreeformQuad(rawQuad,ts);
  if(q){applyFx(q,effectSelect.value);drawFaceEnhancement(q,effectSelect.value);drawFrame(q);setStatus(`2/2 dłonie · Freeform Quad · ${effectSelect.options[effectSelect.selectedIndex].text}`);}
  else if(handsReady){const count=Math.min(2,semantic.length);setStatus(`${count}/2 dłonie${count===2?' · rozsuń palce':''}`);}
  drawDebug(semantic,q,rawQuad);requestAnimationFrame(renderLoop);
}
