import './styles.css';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';

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
let smoothQuad = null;
let lastValidQuadAt = 0;
let frameCounter = 0;
let cameraState = '—';
let handState = '—';
let faceState = '—';

const HAND_INTERVAL = 1000 / 24;
const FACE_INTERVAL = 1000 / 18;
const QUAD_HOLD_MS = 550;

function setStatus(text) { statusEl.textContent = text; }
function updateDiag() { diagEl.textContent = `JS: OK · kamera: ${cameraState} · dłonie: ${handState} · twarz: ${faceState}`; }
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
    { video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false },
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
      minHandDetectionConfidence: 0.42,
      minHandPresenceConfidence: 0.42,
      minTrackingConfidence: 0.42
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
      minFaceDetectionConfidence: 0.45,
      minFacePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
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
    resizeCanvas(); requestAnimationFrame(renderLoop); setTimeout(initVision, 50);
  } catch (err) {
    running = false; cameraState = 'BŁĄD'; updateDiag(); startBtn.disabled = false; startBtn.textContent = 'Spróbuj ponownie';
    setStatus('Nie udało się uruchomić kamery'); showError(readableCameraError(err));
  }
}

startBtn.addEventListener('click', startCamera);
setStatus('Gotowy • v6.1'); updateDiag();
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(t => t.stop()));

function resizeCanvas() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
}
function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z || 0 }; }
function sortHands(hands, handedness) {
  const out = hands.map((pts, i) => ({ name: handedness?.[i]?.[0]?.categoryName || '', pts: pts.map(mirrorPoint) }));
  out.sort((a,b) => a.pts[0].x - b.pts[0].x); return out;
}
function dist(a,b){ return Math.hypot(a.x-b.x,a.y-b.y); }
function orderClockwise(points) {
  const cx=points.reduce((s,p)=>s+p.x,0)/points.length, cy=points.reduce((s,p)=>s+p.y,0)/points.length;
  const ordered=[...points].sort((a,b)=>Math.atan2(a.y-cy,a.x-cx)-Math.atan2(b.y-cy,b.x-cx));
  let idx=0, best=Infinity;
  ordered.forEach((p,i)=>{ const v=p.x+p.y; if(v<best){best=v;idx=i;} });
  return [...ordered.slice(idx),...ordered.slice(0,idx)];
}
function polygonArea(q){ let a=0; for(let i=0;i<q.length;i++){const p=q[i],n=q[(i+1)%q.length];a+=p.x*n.y-n.x*p.y;} return Math.abs(a)/2; }
function computeQuad(sorted) {
  // v6.1: dwie poprawnie wykryte dłonie od razu aktywują ramkę.
  // Nie wymagamy już pozycji L ani wyprostowanych palców.
  if(sorted.length<2) return null;
  const a=sorted[0].pts,b=sorted[1].pts;
  if(!a?.[4] || !a?.[8] || !b?.[4] || !b?.[8]) return null;
  const q=orderClockwise([a[4],a[8],b[4],b[8]].map(p=>({
    x:Math.min(1,Math.max(0,p.x)),
    y:Math.min(1,Math.max(0,p.y))
  })));
  // Odrzucamy tylko ramki praktycznie zerowej wielkości.
  if(polygonArea(q)<.0045) return null;
  return q;
}
function smoothQuadrilateral(next, now) {
  if(next){
    lastValidQuadAt=now;
    if(!smoothQuad){ smoothQuad=next.map(p=>({...p})); return smoothQuad; }
    const oldC=smoothQuad.reduce((s,p)=>({x:s.x+p.x/4,y:s.y+p.y/4}),{x:0,y:0});
    const newC=next.reduce((s,p)=>({x:s.x+p.x/4,y:s.y+p.y/4}),{x:0,y:0});
    const jump=dist(oldC,newC), alpha=jump>.18?.34:.16;
    for(let i=0;i<4;i++){smoothQuad[i].x+=(next[i].x-smoothQuad[i].x)*alpha;smoothQuad[i].y+=(next[i].y-smoothQuad[i].y)*alpha;}
    return smoothQuad;
  }
  if(smoothQuad&&now-lastValidQuadAt<=QUAD_HOLD_MS)return smoothQuad;
  smoothQuad=null; return null;
}
function drawVideo() {
  if(!video.videoWidth)return;
  ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
}
function canvasQuad(q){ return q.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height})); }
function pathQuad(q){ctx.beginPath();ctx.moveTo(q[0].x,q[0].y);for(let i=1;i<q.length;i++)ctx.lineTo(q[i].x,q[i].y);ctx.closePath();}
function pointInPoly(p, poly){let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){if(((poly[i].y>p.y)!=(poly[j].y>p.y))&&(p.x<(poly[j].x-poly[i].x)*(p.y-poly[i].y)/(poly[j].y-poly[i].y)+poly[i].x))c=!c;}return c;}

function applyBaseEffect(q, effect){
  const cq=canvasQuad(q);
  ctx.save(); pathQuad(cq); ctx.clip();
  ctx.translate(canvas.width,0);ctx.scale(-1,1);
  switch(effect){
    case 'bw':ctx.filter='grayscale(1) contrast(1.28)';break;
    case 'comic':ctx.filter='contrast(1.72) saturate(1.55) brightness(1.05)';break;
    case 'anime':ctx.filter='saturate(1.75) contrast(1.28) brightness(1.12)';break;
    case 'cyber':ctx.filter='hue-rotate(250deg) saturate(2.25) contrast(1.38)';break;
    case 'clay':ctx.filter='saturate(.72) contrast(1.16) brightness(1.12) sepia(.2)';break;
    case 'glitch':ctx.filter='contrast(1.55) saturate(2.1) hue-rotate(30deg)';break;
  }
  ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
  if(effect==='glitch') drawGlitch(cq);
  if(effect==='cyber'){ctx.save();pathQuad(cq);ctx.clip();ctx.globalCompositeOperation='screen';ctx.globalAlpha=.12;ctx.fillStyle='cyan';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.restore();}
}

function faceBounds(face){
  if(!face?.length)return null;
  const pts=face.map(mirrorPoint); let minX=1,minY=1,maxX=0,maxY=0;
  for(const p of pts){minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y);}
  return {pts,minX,minY,maxX,maxY,w:maxX-minX,h:maxY-minY,cx:(minX+maxX)/2,cy:(minY+maxY)/2};
}
function drawEyePatch(facePts, indices, scale=1.18){
  const pts=indices.map(i=>facePts[i]).filter(Boolean); if(!pts.length)return;
  let minX=Math.min(...pts.map(p=>p.x)),maxX=Math.max(...pts.map(p=>p.x)),minY=Math.min(...pts.map(p=>p.y)),maxY=Math.max(...pts.map(p=>p.y));
  const padX=(maxX-minX)*1.35,padY=(maxY-minY)*1.8;minX-=padX;maxX+=padX;minY-=padY;maxY+=padY;
  const sx=(1-maxX)*video.videoWidth, sy=minY*video.videoHeight, sw=(maxX-minX)*video.videoWidth, sh=(maxY-minY)*video.videoHeight;
  const dx=minX*canvas.width,dy=minY*canvas.height,dw=(maxX-minX)*canvas.width,dh=(maxY-minY)*canvas.height;
  const cx=dx+dw/2,cy=dy+dh/2;
  ctx.save();ctx.beginPath();ctx.ellipse(cx,cy,dw*.52,dh*.58,0,0,Math.PI*2);ctx.clip();ctx.translate(cx,cy);ctx.scale(scale,scale);ctx.translate(-cx,-cy);
  ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,sx,sy,sw,sh,canvas.width-(dx+dw),dy,dw,dh);ctx.restore();
}
function drawFaceStyle(q,effect){
  const fb=faceBounds(latestFace); if(!fb)return;
  const c={x:fb.cx,y:fb.cy}; if(!pointInPoly(c,q))return;
  const pts=fb.pts,cq=canvasQuad(q);
  ctx.save();pathQuad(cq);ctx.clip();

  if(effect==='anime'||effect==='clay'||effect==='comic'){
    const x=fb.minX*canvas.width,y=fb.minY*canvas.height,w=fb.w*canvas.width,h=fb.h*canvas.height;
    const g=ctx.createRadialGradient(x+w*.5,y+h*.48,0,x+w*.5,y+h*.48,Math.max(w,h)*.62);
    if(effect==='anime'){g.addColorStop(0,'rgba(255,225,235,.13)');g.addColorStop(1,'rgba(255,255,255,0)');}
    else if(effect==='clay'){g.addColorStop(0,'rgba(245,205,175,.18)');g.addColorStop(1,'rgba(255,255,255,0)');}
    else {g.addColorStop(0,'rgba(255,255,255,.07)');g.addColorStop(1,'rgba(0,0,0,0)');}
    ctx.fillStyle=g;ctx.fillRect(x,y,w,h);
  }

  if(effect==='anime'){
    drawEyePatch(pts,[33,133,159,145],1.22);drawEyePatch(pts,[362,263,386,374],1.22);
    ctx.strokeStyle='rgba(20,20,30,.55)';ctx.lineWidth=Math.max(1.5,canvas.width*.0022);
    for(const loop of [[33,160,158,133,153,144],[362,385,387,263,373,380]]){ctx.beginPath();loop.forEach((i,k)=>{const p=pts[i];if(!p)return;const X=p.x*canvas.width,Y=p.y*canvas.height;k?ctx.lineTo(X,Y):ctx.moveTo(X,Y)});ctx.stroke();}
  }
  if(effect==='comic'){
    ctx.strokeStyle='rgba(10,10,10,.45)';ctx.lineWidth=Math.max(1.5,canvas.width*.0025);
    const jaw=[234,93,132,58,172,136,150,149,176,148,152,377,400,378,379,365,397,288,361,323,454];
    ctx.beginPath();jaw.forEach((i,k)=>{const p=pts[i];if(!p)return;k?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height)});ctx.stroke();
  }
  ctx.restore();
}
function drawGlitch(cq){if(frameCounter%3!==0)return;let minX=Math.min(...cq.map(p=>p.x)),maxX=Math.max(...cq.map(p=>p.x)),minY=Math.min(...cq.map(p=>p.y)),maxY=Math.max(...cq.map(p=>p.y));ctx.save();pathQuad(cq);ctx.clip();for(let i=0;i<6;i++){const h=Math.max(2,Math.random()*(maxY-minY)*.035),y=minY+Math.random()*Math.max(1,maxY-minY-h),off=(Math.random()-.5)*18;try{const img=ctx.getImageData(minX,y,maxX-minX,h);ctx.putImageData(img,minX+off,y)}catch{}}ctx.restore();}
function drawFrame(q){const cq=canvasQuad(q);ctx.save();pathQuad(cq);ctx.lineWidth=Math.max(3,canvas.width*.004);ctx.strokeStyle='rgba(255,255,255,.97)';ctx.shadowBlur=14;ctx.shadowColor='rgba(255,255,255,.55)';ctx.stroke();ctx.restore();}
function drawDebug(sorted){if(!debugToggle.checked)return;ctx.save();ctx.fillStyle='rgba(0,255,180,.9)';for(const h of sorted)for(const p of h.pts){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,3,0,Math.PI*2);ctx.fill();}if(latestFace){ctx.fillStyle='rgba(255,220,0,.65)';for(let i=0;i<latestFace.length;i+=8){const p=mirrorPoint(latestFace[i]);ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,1.6,0,Math.PI*2);ctx.fill();}}ctx.restore();}

async function renderLoop(ts){
  if(!running)return; frameCounter++; resizeCanvas(); drawVideo();
  if(handsReady&&handLandmarker&&video.readyState>=2&&ts-lastHandDetect>HAND_INTERVAL){lastHandDetect=ts;try{const r=handLandmarker.detectForVideo(video,ts);latestHands=r.landmarks||[];latestHandedness=r.handedness||[];}catch(e){console.warn(e)}}
  if(faceReady&&faceLandmarker&&video.readyState>=2&&ts-lastFaceDetect>FACE_INTERVAL){lastFaceDetect=ts;try{const r=faceLandmarker.detectForVideo(video,ts);latestFace=r.faceLandmarks?.[0]||null;}catch(e){console.warn(e)}}
  const sorted=sortHands(latestHands,latestHandedness),raw=computeQuad(sorted),q=smoothQuadrilateral(raw,ts);
  if(q){
    applyBaseEffect(q,effectSelect.value);drawFaceStyle(q,effectSelect.value);drawFrame(q);
    setStatus(`2/2 dłonie · ${effectSelect.options[effectSelect.selectedIndex].text}`);
  }
  else if(handsReady){
    const count=Math.min(2,sorted.length);
    setStatus(`${count}/2 dłonie${count===2?' · ustaw palce szerzej':''}`);
  }
  drawDebug(sorted); requestAnimationFrame(renderLoop);
}
