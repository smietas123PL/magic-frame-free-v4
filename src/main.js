import './styles.css';
import { FilesetResolver, HandLandmarker, FaceLandmarker } from '@mediapipe/tasks-vision';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha:false, desynchronized:true });
const fxCanvas = document.getElementById('fxCanvas');
const animeCanvas = document.getElementById('animeCanvas');
const animeCtx = animeCanvas.getContext('2d', { alpha:false });
const inputCanvas = document.getElementById('inputCanvas');
const inputCtx = inputCanvas.getContext('2d', { alpha:false, willReadFrequently:true });
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const effectSelect = document.getElementById('effectSelect');
const debugToggle = document.getElementById('debugToggle');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('errorDetails');
const diagEl = document.getElementById('diag');
const recBadge = document.getElementById('recBadge');

let stream=null, running=false;
let handLandmarker=null, faceLandmarker=null;
let handsReady=false, faceReady=false, loadingHands=false, loadingFace=false;
let latestHands=[], latestHandedness=[], latestFace=null;
let smoothedQuad=null, previousRawQuad=null, previousRawAt=0, lastValidFrameAt=0;
let lastHandDetect=0, lastFaceDetect=0;
let cameraState='—', handState='—', faceState='—', fxState='—';
let handCompleted=0, handWindowStart=performance.now(), handFps=0;
let renderCompleted=0, renderWindowStart=performance.now(), renderFps=0;
let mediaRecorder=null, recordedChunks=[], recordingStartedAt=0;

const HAND_INTERVAL=1000/30, FACE_INTERVAL=1000/15, FRAME_HOLD_MS=90;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const lerp=(a,b,t)=>a+(b-a)*t;
const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
const setStatus=t=>statusEl.textContent=t;
function showError(t){errorEl.hidden=!t;errorEl.textContent=t||'';}

// ---------- diagnostics ----------
function updateDiag(){
  const ai = isAiMode() ? ` · AI: ${cartoonState}${lastCartoonMs?` ${Math.round(lastCartoonMs)}ms/${cartoonFps.toFixed(1)}fps`:''}` : '';
  diagEl.textContent=`JS: OK · kamera: ${cameraState} · dłonie: ${handState} ${handFps.toFixed(0)}fps · twarz: ${faceState} · FX: ${fxState} · R ${renderFps.toFixed(0)}fps${ai}`;
}
updateDiag();

function readableCameraError(err){
  const n=err?.name||'Error';
  if(n==='NotAllowedError'||n==='PermissionDeniedError') return 'Brak zgody na kamerę. Kliknij ikonę kamery/kłódki przy adresie → Zezwalaj.';
  if(n==='NotFoundError') return 'Nie znaleziono kamery.';
  if(n==='NotReadableError') return 'Kamera jest zajęta przez inną aplikację.';
  return `${n}: ${err?.message||'Nieznany błąd kamery'}`;
}
async function requestCamera(){
  if(!navigator.mediaDevices?.getUserMedia) throw new Error('Wymagany HTTPS lub localhost.');
  return navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'user'},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30,max:60}},audio:false});
}

async function getVision(){ return FilesetResolver.forVisionTasks('/mediapipe/wasm'); }
async function initHands(){
  if(loadingHands||handsReady)return; loadingHands=true; handState='load…'; updateDiag();
  try{
    const vision=await getVision();
    const base={modelAssetPath:'/models/hand_landmarker.task'};
    const common={runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:.4,minHandPresenceConfidence:.4,minTrackingConfidence:.4};
    try{handLandmarker=await HandLandmarker.createFromOptions(vision,{...common,baseOptions:{...base,delegate:'GPU'}});handState='GPU OK';}
    catch{handLandmarker=await HandLandmarker.createFromOptions(vision,{...common,baseOptions:base});handState='CPU OK';}
    handsReady=true;
  }catch(e){handState='BŁĄD';console.error(e);}
  finally{loadingHands=false;updateDiag();}
}
async function initFace(){
  if(loadingFace||faceReady)return; loadingFace=true; faceState='load…'; updateDiag();
  try{
    const vision=await getVision();
    const base={modelAssetPath:'/models/face_landmarker.task'};
    const common={runningMode:'VIDEO',numFaces:1,minFaceDetectionConfidence:.35,minFacePresenceConfidence:.35,minTrackingConfidence:.35,outputFaceBlendshapes:false,outputFacialTransformationMatrixes:false};
    try{faceLandmarker=await FaceLandmarker.createFromOptions(vision,{...common,baseOptions:{...base,delegate:'GPU'}});faceState='GPU OK';}
    catch{faceLandmarker=await FaceLandmarker.createFromOptions(vision,{...common,baseOptions:base});faceState='CPU OK';}
    faceReady=true;
  }catch(e){faceState='BŁĄD';console.warn(e);}
  finally{loadingFace=false;updateDiag();}
}

async function startCamera(){
  if(running)return; showError(''); startBtn.disabled=true; startBtn.textContent='Uruchamianie…'; cameraState='prośba…'; updateDiag();
  try{
    stream=await requestCamera(); video.srcObject=stream; video.muted=true; video.playsInline=true;
    await new Promise(r=>{if(video.readyState>=1)return r();video.addEventListener('loadedmetadata',r,{once:true});setTimeout(r,1500);});
    await video.play(); running=true; cameraState='OK'; resizeCanvas(); initLiveFx();
    startBtn.textContent='Kamera działa'; recordBtn.disabled=false; setStatus('Kamera działa · pokaż 2 dłonie'); updateDiag();
    requestAnimationFrame(renderLoop); setTimeout(initHands,20); setTimeout(initFace,120);
  }catch(e){running=false;cameraState='BŁĄD';startBtn.disabled=false;startBtn.textContent='Spróbuj ponownie';showError(readableCameraError(e));updateDiag();}
}
startBtn.addEventListener('click',startCamera);
recordBtn.addEventListener('click',toggleRecording);
window.addEventListener('pagehide',()=>stream?.getTracks?.().forEach(t=>t.stop()));
if('serviceWorker' in navigator) navigator.serviceWorker.register('/sw.js').catch(()=>{});

function resizeCanvas(){
  const w=video.videoWidth||1280,h=video.videoHeight||720;
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;resizeFx(w,h);}
}
function drawVideo(){if(!video.videoWidth)return;ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();}
function mirrorPoint(p){return{x:1-p.x,y:p.y,z:p.z||0};}

// ---------- hand identity + freeform quad ----------
let semanticMemory={leftWrist:null,rightWrist:null};
function semanticHands(hands,handedness){
  const items=hands.map((pts,i)=>({label:handedness?.[i]?.[0]?.categoryName||'',pts:pts.map(mirrorPoint)}));
  if(items.length<2)return items;
  let left=items.find(x=>x.label.toLowerCase()==='left'),right=items.find(x=>x.label.toLowerCase()==='right');
  if(!left||!right||left===right){
    if(semanticMemory.leftWrist&&semanticMemory.rightWrist){
      const[a,b]=items,c1=dist(a.pts[0],semanticMemory.leftWrist)+dist(b.pts[0],semanticMemory.rightWrist),c2=dist(b.pts[0],semanticMemory.leftWrist)+dist(a.pts[0],semanticMemory.rightWrist);[left,right]=c1<=c2?[a,b]:[b,a];
    }else [left,right]=[...items].sort((a,b)=>a.pts[0].x-b.pts[0].x);
  }
  semanticMemory.leftWrist={...left.pts[0]};semanticMemory.rightWrist={...right.pts[0]};return[left,right];
}
function measureFreeformQuad(s){
  if(s.length<2)return null;const L=s[0]?.pts,R=s[1]?.pts;if(!L?.[4]||!L?.[8]||!R?.[4]||!R?.[8])return null;
  const q=[{...L[8]},{...R[8]},{...R[4]},{...L[4]}],xs=q.map(p=>p.x),ys=q.map(p=>p.y);if(Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))<.055)return null;return q;
}
function rejectCornerOutliers(next){
  if(!previousRawQuad)return next.map(p=>({...p}));const jumps=next.map((p,i)=>dist(p,previousRawQuad[i])),sorted=[...jumps].sort((a,b)=>a-b),med=(sorted[1]+sorted[2])/2;
  return next.map((p,i)=>(jumps[i]>Math.max(.17,med*2.8+.035)||jumps[i]>.29)?{...previousRawQuad[i]}:{...p});
}
function smoothQuad(next,now){
  if(next){lastValidFrameAt=now;const clean=rejectCornerOutliers(next),target=clean.map(p=>({...p}));if(previousRawQuad&&previousRawAt){const dt=Math.max(1,now-previousRawAt),k=Math.min(10,dt*.28)/dt;for(let i=0;i<4;i++){target[i].x=clamp(clean[i].x+(clean[i].x-previousRawQuad[i].x)*k,0,1);target[i].y=clamp(clean[i].y+(clean[i].y-previousRawQuad[i].y)*k,0,1);}}previousRawQuad=clean.map(p=>({...p}));previousRawAt=now;if(!smoothedQuad){smoothedQuad=target;return smoothedQuad;}for(let i=0;i<4;i++){const a=clamp(.91+dist(smoothedQuad[i],target[i])*2.4,.91,1);smoothedQuad[i].x=lerp(smoothedQuad[i].x,target[i].x,a);smoothedQuad[i].y=lerp(smoothedQuad[i].y,target[i].y,a);}return smoothedQuad;}
  if(smoothedQuad&&now-lastValidFrameAt<=FRAME_HOLD_MS)return smoothedQuad;smoothedQuad=null;previousRawQuad=null;previousRawAt=0;return null;
}
function triangleUnionPath(q){const p=new Path2D();for(const tri of [[q[0],q[1],q[2]],[q[0],q[2],q[3]]]){const t=tri.map(v=>({x:v.x*canvas.width,y:v.y*canvas.height}));p.moveTo(t[0].x,t[0].y);p.lineTo(t[1].x,t[1].y);p.lineTo(t[2].x,t[2].y);p.closePath();}return p;}
function drawFrame(q){const t=q.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height}));ctx.save();ctx.beginPath();ctx.moveTo(t[0].x,t[0].y);for(let i=1;i<4;i++)ctx.lineTo(t[i].x,t[i].y);ctx.closePath();ctx.lineWidth=Math.max(3,canvas.width*.004);ctx.strokeStyle='white';ctx.shadowBlur=12;ctx.shadowColor='rgba(255,255,255,.45)';ctx.stroke();ctx.restore();}

// ---------- face uniforms ----------
function faceUniforms(){
  const f=latestFace;if(!f||f.length<455)return null;
  const m=i=>mirrorPoint(f[i]);
  const eyeL={x:(m(33).x+m(133).x)*.5,y:(m(33).y+m(133).y)*.5};
  const eyeR={x:(m(362).x+m(263).x)*.5,y:(m(362).y+m(263).y)*.5};
  const cheekL=m(234),cheekR=m(454),nose=m(1),jaw=m(152);
  return{eyeL,eyeR,center:nose,jaw,width:Math.max(.08,dist(cheekL,cheekR))};
}

// ---------- realtime WebGL2 anime ----------
let gl=null,program=null,tex=null,vao=null;
let u={};
function shader(gl,type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function initLiveFx(){
  try{
    gl=fxCanvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});if(!gl)throw new Error('WebGL2 niedostępny');
    const vs=`#version 300 es\nprecision highp float;out vec2 vUv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUv=p;gl_Position=vec4(p*2.0-1.0,0,1);}`;
    const fs=`#version 300 es
precision highp float;in vec2 vUv;out vec4 outColor;uniform sampler2D t;uniform vec2 res;uniform int mode;uniform int hasFace;uniform vec2 eyeL;uniform vec2 eyeR;uniform vec2 faceC;uniform vec2 jaw;uniform float faceW;
vec2 magnify(vec2 uv,vec2 c,float r,float k){vec2 d=uv-c;float x=length(d)/r;if(x<1.0){float s=mix(1.0-k,1.0,smoothstep(0.0,1.0,x));uv=c+d*s;}return uv;}
vec2 warpFace(vec2 uv){if(hasFace==0)return uv;uv=magnify(uv,eyeL,faceW*.18,.18);uv=magnify(uv,eyeR,faceW*.18,.18);float y0=faceC.y+faceW*.05;float y1=jaw.y+faceW*.08;if(uv.y>y0&&uv.y<y1){float t0=smoothstep(y0,y1,uv.y);uv.x=faceC.x+(uv.x-faceC.x)*(1.0+.12*t0);}return uv;}
float lum(vec3 c){return dot(c,vec3(.299,.587,.114));}
void main(){vec2 uv=vec2(1.0-vUv.x,vUv.y);uv=warpFace(uv);vec2 px=1.0/res;vec3 c=texture(t,uv).rgb;vec3 b=(c*4.0+texture(t,uv+vec2(px.x,0)).rgb+texture(t,uv-vec2(px.x,0)).rgb+texture(t,uv+vec2(0,px.y)).rgb+texture(t,uv-vec2(0,px.y)).rgb)/8.0;float l=lum(c);float gx=lum(texture(t,uv+vec2(px.x,0)).rgb)-lum(texture(t,uv-vec2(px.x,0)).rgb);float gy=lum(texture(t,uv+vec2(0,px.y)).rgb)-lum(texture(t,uv-vec2(0,px.y)).rgb);float edge=smoothstep(.055,.16,length(vec2(gx,gy)));float levels=mode==2?8.0:6.0;vec3 q=floor(b*levels+.5)/levels;q=pow(q,vec3(.92));q=mix(vec3(l),q,1.18);q.r*=1.035;q.b*=1.025;float ink=mode==2?.42:.62;q*=1.0-edge*ink;q=mix(q,vec3(.08,.07,.09),edge*(mode==2?.18:.34));outColor=vec4(clamp(q,0.0,1.0),1.0);}`;
    program=gl.createProgram();gl.attachShader(program,shader(gl,gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    vao=gl.createVertexArray();u.res=gl.getUniformLocation(program,'res');u.mode=gl.getUniformLocation(program,'mode');u.hasFace=gl.getUniformLocation(program,'hasFace');u.eyeL=gl.getUniformLocation(program,'eyeL');u.eyeR=gl.getUniformLocation(program,'eyeR');u.faceC=gl.getUniformLocation(program,'faceC');u.jaw=gl.getUniformLocation(program,'jaw');u.faceW=gl.getUniformLocation(program,'faceW');fxState='WebGL2 Live OK';
  }catch(e){gl=null;fxState='Canvas fallback';console.warn(e);}updateDiag();
}
function resizeFx(w,h){
  // half-res is enough for live anime and keeps 30-60 FPS on modest GPUs
  const scale=Math.min(1,960/w);fxCanvas.width=Math.max(2,Math.round(w*scale));fxCanvas.height=Math.max(2,Math.round(h*scale));
}
function renderLiveFx(mode){
  if(!gl){const c=fxCanvas.getContext('2d');c.save();c.translate(fxCanvas.width,0);c.scale(-1,1);c.filter=mode==='live-soft'?'saturate(1.15) contrast(1.08)':'saturate(1.3) contrast(1.2)';c.drawImage(video,0,0,fxCanvas.width,fxCanvas.height);c.restore();return;}
  gl.viewport(0,0,fxCanvas.width,fxCanvas.height);gl.useProgram(program);gl.bindVertexArray(vao);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,video);gl.uniform2f(u.res,fxCanvas.width,fxCanvas.height);gl.uniform1i(u.mode,mode==='live-soft'?2:1);
  const f=faceUniforms();gl.uniform1i(u.hasFace,f?1:0);if(f){gl.uniform2f(u.eyeL,f.eyeL.x,1-f.eyeL.y);gl.uniform2f(u.eyeR,f.eyeR.x,1-f.eyeR.y);gl.uniform2f(u.faceC,f.center.x,1-f.center.y);gl.uniform2f(u.jaw,f.jaw.x,1-f.jaw.y);gl.uniform1f(u.faceW,f.width);}gl.drawArrays(gl.TRIANGLES,0,3);
}
function applyLiveFx(q,mode){renderLiveFx(mode);ctx.save();ctx.clip(triangleUnionPath(q));ctx.drawImage(fxCanvas,0,0,canvas.width,canvas.height);ctx.restore();}

// ---------- optional CartoonGAN quality worker ----------
let cartoonWorker=null,cartoonReady=false,cartoonBusy=false,cartoonState='wyłączony',cartoonStyle='shinkai',lastCartoonMs=0,lastCartoonDone=0,cartoonCompleted=0,cartoonWindowStart=performance.now(),cartoonFps=0,aiRequestId=0,latestQuadForAi=null;
const AI_SIZE=96;
function isAiMode(){return effectSelect.value.startsWith('ai-');}
function stopCartoon(){cartoonWorker?.terminate();cartoonWorker=null;cartoonReady=false;cartoonBusy=false;cartoonState='wyłączony';}
function initCartoon(){
  if(!isAiMode()){stopCartoon();updateDiag();return;}const style=effectSelect.value.replace('ai-','');if(cartoonReady&&style===cartoonStyle)return;stopCartoon();cartoonStyle=style;cartoonState='load…';updateDiag();
  cartoonWorker=new Worker(new URL('./cartoon-worker.js',import.meta.url),{type:'module'});cartoonWorker.onmessage=e=>{const m=e.data||{};if(m.type==='ready'){cartoonReady=true;cartoonState=`${String(m.backend).toUpperCase()} ${style} OK`;kickAi();}else if(m.type==='frame'){cartoonBusy=false;const img=new ImageData(new Uint8ClampedArray(m.rgba),m.size,m.size);animeCanvas.width=animeCanvas.height=m.size;animeCtx.putImageData(img,0,0);lastCartoonMs=m.totalMs||0;lastCartoonDone=performance.now();cartoonCompleted++;const dt=lastCartoonDone-cartoonWindowStart;if(dt>700){cartoonFps=cartoonCompleted*1000/dt;cartoonCompleted=0;cartoonWindowStart=lastCartoonDone;}kickAi();updateDiag();}else if(m.type==='error'){cartoonBusy=false;cartoonState='BŁĄD';updateDiag();}};
  cartoonWorker.postMessage({type:'init',style,preferredBackend:/Windows/i.test(navigator.userAgent)?'webgl':'auto',modelUrl:`/models/cartoongan-${style}/model.json`});
}
function computeCrop(q){const p=q.map(v=>({x:v.x*canvas.width,y:v.y*canvas.height})),minX=Math.min(...p.map(v=>v.x)),maxX=Math.max(...p.map(v=>v.x)),minY=Math.min(...p.map(v=>v.y)),maxY=Math.max(...p.map(v=>v.y)),cx=(minX+maxX)/2,cy=(minY+maxY)/2;let side=clamp(Math.max(maxX-minX,maxY-minY)*1.28,Math.min(canvas.width,canvas.height)*.25,Math.min(canvas.width,canvas.height));const dx=clamp(cx-side/2,0,canvas.width-side),dy=clamp(cy-side/2,0,canvas.height-side);return{dx,dy,side,sx:canvas.width-(dx+side),sy:dy};}
function kickAi(){if(!cartoonReady||cartoonBusy||!latestQuadForAi||!isAiMode()||video.readyState<2)return;cartoonBusy=true;const crop=computeCrop(latestQuadForAi);inputCanvas.width=inputCanvas.height=AI_SIZE;inputCtx.save();inputCtx.translate(AI_SIZE,0);inputCtx.scale(-1,1);inputCtx.drawImage(video,crop.sx,crop.sy,crop.side,crop.side,0,0,AI_SIZE,AI_SIZE);inputCtx.restore();const fr=inputCtx.getImageData(0,0,AI_SIZE,AI_SIZE);cartoonWorker.postMessage({type:'infer',requestId:++aiRequestId,size:AI_SIZE,rgba:fr.data.buffer},[fr.data.buffer]);}
function applyAi(q){if(animeCanvas.width&&lastCartoonDone){const c=computeCrop(q);ctx.save();ctx.clip(triangleUnionPath(q));ctx.drawImage(animeCanvas,c.dx,c.dy,c.side,c.side);ctx.restore();}}

effectSelect.addEventListener('change',()=>{if(isAiMode())initCartoon();else stopCartoon();updateDiag();});

// ---------- recording/debug ----------
function preferredMime(){return['video/mp4;codecs=avc1.42E01E','video/mp4','video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(t=>MediaRecorder?.isTypeSupported?.(t))||'';}
function toggleRecording(){if(!running||!MediaRecorder||!canvas.captureStream){showError('Nagrywanie niedostępne.');return;}if(mediaRecorder?.state==='recording'){mediaRecorder.stop();return;}recordedChunks=[];const mime=preferredMime(),s=canvas.captureStream(30);mediaRecorder=new MediaRecorder(s,mime?{mimeType:mime,videoBitsPerSecond:8_000_000}:{videoBitsPerSecond:8_000_000});mediaRecorder.ondataavailable=e=>{if(e.data?.size)recordedChunks.push(e.data)};mediaRecorder.onstop=()=>{const type=mediaRecorder.mimeType||mime||'video/webm',blob=new Blob(recordedChunks,{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`magic-frame-v10-${Date.now()}.${type.includes('mp4')?'mp4':'webm'}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);recordBtn.textContent='Nagraj';recBadge.hidden=true;};mediaRecorder.start(250);recordingStartedAt=performance.now();recordBtn.textContent='Stop';recBadge.hidden=false;}
function drawDebug(s,q,raw){if(!debugToggle.checked)return;ctx.save();ctx.fillStyle='#00ffb4';for(const h of s)for(const p of h.pts){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,2.5,0,Math.PI*2);ctx.fill();}if(raw){const t=raw.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height}));ctx.setLineDash([6,6]);ctx.strokeStyle='#00ffff';ctx.beginPath();ctx.moveTo(t[0].x,t[0].y);for(let i=1;i<4;i++)ctx.lineTo(t[i].x,t[i].y);ctx.closePath();ctx.stroke();}ctx.restore();}
function tickHand(now){handCompleted++;const dt=now-handWindowStart;if(dt>=700){handFps=handCompleted*1000/dt;handCompleted=0;handWindowStart=now;}}
function tickRender(now){renderCompleted++;const dt=now-renderWindowStart;if(dt>=700){renderFps=renderCompleted*1000/dt;renderCompleted=0;renderWindowStart=now;updateDiag();}}

async function renderLoop(ts){
  if(!running)return;tickRender(ts);resizeCanvas();drawVideo();
  if(handsReady&&video.readyState>=2&&ts-lastHandDetect>HAND_INTERVAL){lastHandDetect=ts;try{const r=handLandmarker.detectForVideo(video,ts);latestHands=r.landmarks||[];latestHandedness=r.handedness||[];tickHand(ts);}catch{}}
  if(faceReady&&video.readyState>=2&&ts-lastFaceDetect>FACE_INTERVAL){lastFaceDetect=ts;try{latestFace=faceLandmarker.detectForVideo(video,ts).faceLandmarks?.[0]||null;}catch{}}
  const semantic=semanticHands(latestHands,latestHandedness),raw=measureFreeformQuad(semantic),q=smoothQuad(raw,ts),mode=effectSelect.value;
  if(q){
    if(mode==='live'||mode==='live-soft')applyLiveFx(q,mode);
    else if(isAiMode()){latestQuadForAi=q.map(p=>({...p}));kickAi();applyAi(q);}
    drawFrame(q);const extra=isAiMode()&&lastCartoonDone?` · AI ${Math.round(performance.now()-lastCartoonDone)}ms old`:'';setStatus(`2/2 dłonie · ${mode==='live'?'ANIME LIVE':mode==='live-soft'?'ANIME SOFT':mode==='original'?'ORIGINAL':'AI QUALITY'}${extra}`);
  }else{latestQuadForAi=null;if(handsReady)setStatus(`${Math.min(2,semantic.length)}/2 dłonie`);}
  if(mediaRecorder?.state==='recording')recBadge.textContent=`● REC ${Math.floor((ts-recordingStartedAt)/1000)}s`;
  drawDebug(semantic,q,raw);requestAnimationFrame(renderLoop);
}

setStatus('Gotowy • v10 Anime Live');
