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
let cameraState = '—';
let handState = '—';
let faceState = '—';
let fxState = 'init…';

const HAND_INTERVAL = 1000 / 24;
const FACE_INTERVAL = 1000 / 18;
const FRAME_HOLD_MS = 700;

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
setStatus('Gotowy • v7'); updateDiag();
window.addEventListener('pagehide', () => stream?.getTracks?.().forEach(t => t.stop()));

function resizeCanvas() {
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  if (canvas.width !== vw || canvas.height !== vh) { canvas.width = vw; canvas.height = vh; }
  if (fxCanvas.width !== vw || fxCanvas.height !== vh) { fxCanvas.width = vw; fxCanvas.height = vh; resizeFx(vw, vh); }
}
function mirrorPoint(p) { return { x: 1 - p.x, y: p.y, z: p.z || 0 }; }
function sortHands(hands, handedness) {
  const out = hands.map((pts, i) => ({ name: handedness?.[i]?.[0]?.categoryName || '', pts: pts.map(mirrorPoint) }));
  out.sort((a,b) => a.pts[0].x - b.pts[0].x); return out;
}
function dist(a,b){ return Math.hypot(a.x-b.x,a.y-b.y); }
function clamp(v,a,b){ return Math.max(a,Math.min(b,v)); }
function lerp(a,b,t){ return a+(b-a)*t; }
function angleLerp(a,b,t){ let d=((b-a+Math.PI*3)%(Math.PI*2))-Math.PI; return a+d*t; }

// v7 Virtual Frame: dłonie sterują ramką jak uchwytami, nie są literalnymi narożnikami.
function measureVirtualFrame(sorted) {
  if (sorted.length < 2) return null;
  const L = sorted[0].pts, R = sorted[1].pts;
  if (!L?.[4] || !L?.[8] || !R?.[4] || !R?.[8]) return null;

  const lGrip = { x:(L[4].x+L[8].x)/2, y:(L[4].y+L[8].y)/2 };
  const rGrip = { x:(R[4].x+R[8].x)/2, y:(R[4].y+R[8].y)/2 };
  const dx = rGrip.x-lGrip.x, dy = rGrip.y-lGrip.y;
  const gripDistance = Math.hypot(dx,dy);
  if (gripDistance < 0.12) return null;

  const angle = Math.atan2(dy,dx);
  const center = { x:(lGrip.x+rGrip.x)/2, y:(lGrip.y+rGrip.y)/2 };
  const lSpan = dist(L[4],L[8]), rSpan = dist(R[4],R[8]);
  const fingerSpan = (lSpan+rSpan)/2;

  // Szerokość głównie z rozstawu dłoni. Wysokość adaptacyjna, ale z bezpiecznym aspect ratio.
  const width = clamp(gripDistance * 0.92, 0.20, 0.82);
  const targetFromRatio = width / 1.72;
  const targetFromHands = fingerSpan * 2.15;
  const height = clamp(targetFromRatio * 0.72 + targetFromHands * 0.28, width/2.25, width/1.28);

  return { center, width, height, angle, lGrip, rGrip };
}

function smoothVirtualFrame(next, now) {
  if (next) {
    lastValidFrameAt = now;
    if (!virtualFrame) {
      virtualFrame = JSON.parse(JSON.stringify(next));
      return virtualFrame;
    }
    const centerJump = dist(virtualFrame.center,next.center);
    const alphaPos = centerJump > .16 ? .34 : .18;
    const alphaSize = .14;
    const alphaAngle = .16;
    virtualFrame.center.x = lerp(virtualFrame.center.x,next.center.x,alphaPos);
    virtualFrame.center.y = lerp(virtualFrame.center.y,next.center.y,alphaPos);
    virtualFrame.width = lerp(virtualFrame.width,next.width,alphaSize);
    virtualFrame.height = lerp(virtualFrame.height,next.height,alphaSize);
    virtualFrame.angle = angleLerp(virtualFrame.angle,next.angle,alphaAngle);
    virtualFrame.lGrip.x = lerp(virtualFrame.lGrip.x,next.lGrip.x,.28);
    virtualFrame.lGrip.y = lerp(virtualFrame.lGrip.y,next.lGrip.y,.28);
    virtualFrame.rGrip.x = lerp(virtualFrame.rGrip.x,next.rGrip.x,.28);
    virtualFrame.rGrip.y = lerp(virtualFrame.rGrip.y,next.rGrip.y,.28);
    return virtualFrame;
  }
  if (virtualFrame && now-lastValidFrameAt <= FRAME_HOLD_MS) return virtualFrame;
  virtualFrame = null; return null;
}

function frameToQuad(f) {
  const ux={x:Math.cos(f.angle),y:Math.sin(f.angle)};
  const uy={x:-Math.sin(f.angle),y:Math.cos(f.angle)};
  const hw=f.width/2, hh=f.height/2;
  const p=(sx,sy)=>({x:clamp(f.center.x+ux.x*hw*sx+uy.x*hh*sy,0,1),y:clamp(f.center.y+ux.y*hw*sx+uy.y*hh*sy,0,1)});
  return [p(-1,-1),p(1,-1),p(1,1),p(-1,1)];
}

function drawVideo() {
  if(!video.videoWidth)return;
  ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
}
function canvasQuad(q){ return q.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height})); }
function pathQuad(q){ctx.beginPath();ctx.moveTo(q[0].x,q[0].y);for(let i=1;i<q.length;i++)ctx.lineTo(q[i].x,q[i].y);ctx.closePath();}
function pointInPoly(p, poly){let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){if(((poly[i].y>p.y)!=(poly[j].y>p.y))&&(p.x<(poly[j].x-poly[i].x)*(p.y-poly[i].y)/(poly[j].y-poly[i].y)+poly[i].x))c=!c;}return c;}

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
  const cq=canvasQuad(q);ctx.save();pathQuad(cq);ctx.clip();ctx.translate(canvas.width,0);ctx.scale(-1,1);
  const filters={anime:'saturate(1.6) contrast(1.25) brightness(1.08)',comic:'contrast(1.65) saturate(1.4)',clay:'saturate(.7) contrast(1.1) brightness(1.12)',cyber:'hue-rotate(250deg) saturate(2.1) contrast(1.35)',glitch:'saturate(2) contrast(1.5)',bw:'grayscale(1) contrast(1.35)'};
  ctx.filter=filters[effect]||'none';ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();
}
function applyFx(q,effect){
  const cq=canvasQuad(q);const ok=renderFx(effect);
  if(ok){ctx.save();pathQuad(cq);ctx.clip();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(fxCanvas,0,0,canvas.width,canvas.height);ctx.restore();}
  else drawFallbackEffect(q,effect);
  if(effect==='glitch')drawGlitch(cq);
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
  const fb=faceBounds(latestFace);if(!fb)return;const c={x:fb.cx,y:fb.cy};if(!pointInPoly(c,q))return;
  const pts=fb.pts,cq=canvasQuad(q);ctx.save();pathQuad(cq);ctx.clip();
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
function drawGlitch(cq){if(frameCounter%3!==0)return;let minX=Math.min(...cq.map(p=>p.x)),maxX=Math.max(...cq.map(p=>p.x)),minY=Math.min(...cq.map(p=>p.y)),maxY=Math.max(...cq.map(p=>p.y));ctx.save();pathQuad(cq);ctx.clip();for(let i=0;i<5;i++){const h=Math.max(2,Math.random()*(maxY-minY)*.03),y=minY+Math.random()*Math.max(1,maxY-minY-h),off=(Math.random()-.5)*16;try{const img=ctx.getImageData(minX,y,maxX-minX,h);ctx.putImageData(img,minX+off,y)}catch{}}ctx.restore();}
function drawFrame(q){const cq=canvasQuad(q);ctx.save();pathQuad(cq);ctx.lineWidth=Math.max(3,canvas.width*.004);ctx.strokeStyle='rgba(255,255,255,.98)';ctx.shadowBlur=16;ctx.shadowColor='rgba(255,255,255,.55)';ctx.stroke();ctx.restore();}
function drawDebug(sorted,f,q){
  if(!debugToggle.checked)return;ctx.save();ctx.fillStyle='rgba(0,255,180,.9)';for(const h of sorted)for(const p of h.pts){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,2.6,0,Math.PI*2);ctx.fill();}
  if(f){ctx.fillStyle='#ff3bd4';for(const p of [f.lGrip,f.rGrip]){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,8,0,Math.PI*2);ctx.fill();}ctx.strokeStyle='rgba(255,59,212,.8)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(f.lGrip.x*canvas.width,f.lGrip.y*canvas.height);ctx.lineTo(f.rGrip.x*canvas.width,f.rGrip.y*canvas.height);ctx.stroke();}
  if(latestFace){const pts=latestFace.map(mirrorPoint);ctx.fillStyle='rgba(255,220,0,.58)';for(let i=0;i<pts.length;i+=8){const p=pts[i];ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,1.5,0,Math.PI*2);ctx.fill();}ctx.fillStyle='rgba(255,220,0,.08)';ctx.beginPath();FACE_OVAL.forEach((i,k)=>{const p=pts[i];if(!p)return;k?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height)});ctx.closePath();ctx.fill();}
  if(q){const cq=canvasQuad(q);ctx.strokeStyle='rgba(80,180,255,.9)';ctx.lineWidth=2;pathQuad(cq);ctx.stroke();}ctx.restore();
}

async function renderLoop(ts){
  if(!running)return;frameCounter++;resizeCanvas();drawVideo();
  if(handsReady&&handLandmarker&&video.readyState>=2&&ts-lastHandDetect>HAND_INTERVAL){lastHandDetect=ts;try{const r=handLandmarker.detectForVideo(video,ts);latestHands=r.landmarks||[];latestHandedness=r.handedness||[];}catch(e){console.warn(e)}}
  if(faceReady&&faceLandmarker&&video.readyState>=2&&ts-lastFaceDetect>FACE_INTERVAL){lastFaceDetect=ts;try{const r=faceLandmarker.detectForVideo(video,ts);latestFace=r.faceLandmarks?.[0]||null;}catch(e){console.warn(e)}}
  const sorted=sortHands(latestHands,latestHandedness);const measured=measureVirtualFrame(sorted);const vf=smoothVirtualFrame(measured,ts);const q=vf?frameToQuad(vf):null;
  if(q){applyFx(q,effectSelect.value);drawFaceEnhancement(q,effectSelect.value);drawFrame(q);setStatus(`2/2 dłonie · Virtual Frame · ${effectSelect.options[effectSelect.selectedIndex].text}`);}
  else if(handsReady){const count=Math.min(2,sorted.length);setStatus(`${count}/2 dłonie${count===2?' · rozsuń dłonie':''}`);}
  drawDebug(sorted,vf,q);requestAnimationFrame(renderLoop);
}
