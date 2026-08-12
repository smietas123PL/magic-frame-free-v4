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

const HAND_INTERVAL=1000/30, FACE_INTERVAL=1000/20, FRAME_HOLD_MS=90;
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
  const cheekL=m(234),cheekR=m(454),nose=m(1),jaw=m(152),forehead=m(10);
  const mouth={x:(m(13).x+m(14).x)*.5,y:(m(13).y+m(14).y)*.5};
  return{eyeL,eyeR,center:nose,jaw,forehead,mouth,width:Math.max(.08,dist(cheekL,cheekR)),height:Math.max(.1,dist(forehead,jaw))};
}

// ---------- realtime WebGL2 whole-scene anime v10.2 ----------
let gl=null,program=null,tex=null,vao=null;
let u={};
function shader(gl,type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;}
function initLiveFx(){
  try{
    gl=fxCanvas.getContext('webgl2',{alpha:false,antialias:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});if(!gl)throw new Error('WebGL2 niedostępny');
    const vs=`#version 300 es\nprecision highp float;out vec2 vUv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);vUv=p;gl_Position=vec4(p*2.0-1.0,0,1);}`;
    const fs=`#version 300 es
precision highp float;
in vec2 vUv;out vec4 outColor;
uniform sampler2D t;uniform vec2 res;uniform int mode;uniform int hasFace;
uniform vec2 eyeL;uniform vec2 eyeR;uniform vec2 faceC;uniform vec2 jaw;uniform vec2 forehead;uniform float faceW;uniform float faceH;
float lum(vec3 c){return dot(c,vec3(.299,.587,.114));}
vec2 magnify(vec2 uv,vec2 c,float r,float k){vec2 d=uv-c;float x=length(d)/max(r,.0001);if(x<1.0){float s=mix(1.0-k,1.0,smoothstep(0.0,1.0,x));uv=c+d*s;}return uv;}
vec2 warpFace(vec2 uv){
  if(hasFace==0)return uv;
  float eyeK=mode==1?.075:(mode==2?.17:.23);
  float jawK=mode==1?.030:(mode==2?.10:.15);
  float chinK=mode==1?.010:(mode==2?.025:.038);
  uv=magnify(uv,eyeL,faceW*.215,eyeK);
  uv=magnify(uv,eyeR,faceW*.215,eyeK);
  float y0=faceC.y+faceH*.03;
  float y1=jaw.y+faceH*.05;
  if(uv.y>y0&&uv.y<y1){
    float tt=smoothstep(y0,y1,uv.y);
    uv.x=faceC.x+(uv.x-faceC.x)*(1.0+jawK*tt);
    uv.y-=chinK*tt*faceH;
  }
  return uv;
}
float faceMask(vec2 uv){
  if(hasFace==0)return 0.0;
  vec2 d=vec2((uv.x-faceC.x)/(faceW*.61),(uv.y-(faceC.y+faceH*.045))/(faceH*.62));
  return 1.0-smoothstep(.76,1.02,length(d));
}
float upperHeadMask(vec2 uv){
  if(hasFace==0)return 0.0;
  float x=1.0-smoothstep(faceW*.50,faceW*.76,abs(uv.x-faceC.x));
  float y=1.0-smoothstep(forehead.y-faceH*.34,faceC.y+faceH*.15,uv.y);
  return clamp(x*y,0.0,1.0);
}
float lowerFaceMask(vec2 uv){
  if(hasFace==0)return 0.0;
  float fm=faceMask(uv);
  float y=smoothstep(faceC.y+faceH*.07,jaw.y+faceH*.03,uv.y);
  return fm*y;
}
float skinMask(vec3 c){
  float mx=max(c.r,max(c.g,c.b)),mn=min(c.r,min(c.g,c.b));
  float sat=(mx-mn)/max(mx,.001);
  float warm=smoothstep(-.055,.13,c.r-c.b)*smoothstep(-.09,.10,c.g-c.b);
  float rg=1.0-smoothstep(.045,.25,abs(c.r-c.g));
  float br=smoothstep(.18,.96,mx)*smoothstep(.03,.74,1.0-sat);
  return clamp(warm*rg*br,0.0,1.0);
}
vec3 blur9(vec2 uv,vec2 px,float r){
  vec2 d=px*r;
  vec3 s=texture(t,uv).rgb*4.0;
  s+=texture(t,uv+vec2(d.x,0)).rgb+texture(t,uv-vec2(d.x,0)).rgb;
  s+=texture(t,uv+vec2(0,d.y)).rgb+texture(t,uv-vec2(0,d.y)).rgb;
  s+=.65*(texture(t,uv+d).rgb+texture(t,uv-d).rgb+texture(t,uv+vec2(d.x,-d.y)).rgb+texture(t,uv+vec2(-d.x,d.y)).rgb);
  return s/10.6;
}
vec3 simplify(vec2 uv,vec2 px,float fm,float sm){
  vec3 src=texture(t,uv).rgb;
  vec3 b1=blur9(uv,px,1.5);
  vec3 b2=blur9(uv,px,3.1);
  vec3 b3=blur9(uv,px,4.5);
  float micro=abs(lum(src)-lum(b1));
  float preserve=smoothstep(.075,.19,micro);
  float base=mode==1?.52:(mode==2?.76:.86);
  float faceExtra=fm*sm*(mode==1?.20:(mode==2?.34:.42));
  vec3 smoothColor=mix(mix(b1,b2,.56),b3,fm*sm*.28);
  return mix(src,smoothColor,clamp(base+faceExtra-preserve*.08,0.0,.96));
}
vec3 faceShading(vec3 c,float fm,float sm){
  float w=fm*sm;
  if(w<=.01)return c;
  float L=lum(c);
  float sh=smoothstep(.26,.42,L);
  float hi=smoothstep(.68,.84,L);
  vec3 shadowTone=vec3(.72,.60,.62);
  vec3 midTone=vec3(.93,.80,.75);
  vec3 highTone=vec3(1.00,.92,.86);
  vec3 skinTone=mix(shadowTone,midTone,sh);
  skinTone=mix(skinTone,highTone,hi);
  float amount=mode==1?.40:(mode==2?.68:.78);
  return mix(c,skinTone,w*amount);
}
vec3 scenePalette(vec3 c){
  float levels=mode==1?7.0:(mode==2?6.0:5.0);
  float L=max(lum(c),.004);
  float q=floor(L*levels+.52)/levels;
  vec3 chroma=c/L;
  chroma=mix(vec3(1.0),chroma,mode==1?1.02:1.08);
  vec3 o=chroma*q;
  float shadow=1.0-smoothstep(.25,.42,q);
  float hi=smoothstep(.72,.90,q);
  o*=1.0-shadow*(mode==1?.05:(mode==2?.10:.14));
  o=mix(o,vec3(1.0,.97,.94),hi*(mode==1?.02:.045));
  return pow(clamp(o,0.0,1.0),vec3(.96));
}
float cleanEdge(vec2 uv,vec2 px){
  float lR=lum(blur9(uv+vec2(px.x*3.2,0),px,1.2));
  float lL=lum(blur9(uv-vec2(px.x*3.2,0),px,1.2));
  float lU=lum(blur9(uv+vec2(0,px.y*3.2),px,1.2));
  float lD=lum(blur9(uv-vec2(0,px.y*3.2),px,1.2));
  float grad=length(vec2(lR-lL,lU-lD));
  float threshold=mode==1?.15:(mode==2?.125:.105);
  return smoothstep(threshold,threshold+.085,grad);
}
void main(){
  vec2 uv=vec2(1.0-vUv.x,vUv.y);
  uv=warpFace(uv);
  vec2 px=1.0/res;
  vec3 src=texture(t,uv).rgb;
  float fm=faceMask(uv);
  float sm=skinMask(src);
  vec3 simplified=simplify(uv,px,fm,sm);
  vec3 q=scenePalette(simplified);

  q=faceShading(q,fm,skinMask(simplified));

  float lower=lowerFaceMask(uv);
  float beard=lower*(1.0-smoothstep(.19,.43,lum(simplified)))*(1.0-sm*.55);
  vec3 beardTone=mix(vec3(.105,.090,.125),vec3(.16,.14,.18),smoothstep(.18,.40,lum(simplified)));
  q=mix(q,beardTone,beard*(mode==1?.46:(mode==2?.66:.74)));

  float head=upperHeadMask(uv);
  float dark=1.0-smoothstep(.24,.50,lum(simplified));
  float hair=head*dark*(1.0-fm*sm*.70);
  float band=.5+.5*sin((uv.x-faceC.x)/max(faceW,.001)*9.0 + (uv.y-forehead.y)/max(faceH,.001)*4.0);
  float hairHi=smoothstep(.74,.96,band)*smoothstep(.16,.48,lum(simplified));
  vec3 hairBase=mix(vec3(.075,.070,.105),vec3(.15,.13,.19),smoothstep(.18,.42,lum(simplified)));
  vec3 hairLight=hairBase+vec3(.085,.070,.12);
  vec3 hairTone=mix(hairBase,hairLight,hairHi*.55);
  q=mix(q,hairTone,hair*(mode==1?.46:(mode==2?.70:.78)));

  float edge=cleanEdge(uv,px);
  float faceSkin=fm*sm;
  edge*=1.0-faceSkin*(mode==1?.72:(mode==2?.86:.90));
  edge*=1.0-beard*.52;
  vec3 ink=mode==3?vec3(.055,.045,.075):vec3(.085,.060,.095);
  float inkAmount=mode==1?.32:(mode==2?.54:.66);
  q=mix(q,ink,edge*inkAmount);

  outColor=vec4(clamp(q,0.0,1.0),1.0);
}`
    program=gl.createProgram();gl.attachShader(program,shader(gl,gl.VERTEX_SHADER,vs));gl.attachShader(program,shader(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
    tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    vao=gl.createVertexArray();
    for(const name of ['res','mode','hasFace','eyeL','eyeR','faceC','jaw','forehead','faceW','faceH'])u[name]=gl.getUniformLocation(program,name);
    fxState='WebGL2 Clean Anime v10.4 OK';
  }catch(e){gl=null;fxState='Canvas fallback';console.warn(e);}updateDiag();
}
function resizeFx(w,h){const scale=Math.min(1,760/w);fxCanvas.width=Math.max(2,Math.round(w*scale));fxCanvas.height=Math.max(2,Math.round(h*scale));}
function liveModeNumber(mode){return mode==='live-soft'?1:mode==='live-strong'?3:2;}
function renderLiveFx(mode){
  const mn=liveModeNumber(mode);
  if(!gl){const c=fxCanvas.getContext('2d');if(!c){fxState='FX fallback unavailable';return;}c.save();c.translate(fxCanvas.width,0);c.scale(-1,1);c.filter=mn===1?'saturate(1.12) contrast(1.08)':mn===3?'saturate(1.45) contrast(1.30)':'saturate(1.32) contrast(1.20)';c.drawImage(video,0,0,fxCanvas.width,fxCanvas.height);c.restore();return;}
  gl.viewport(0,0,fxCanvas.width,fxCanvas.height);gl.useProgram(program);gl.bindVertexArray(vao);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,tex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,video);gl.uniform2f(u.res,fxCanvas.width,fxCanvas.height);gl.uniform1i(u.mode,mn);
  const f=faceUniforms();gl.uniform1i(u.hasFace,f?1:0);if(f){gl.uniform2f(u.eyeL,f.eyeL.x,1-f.eyeL.y);gl.uniform2f(u.eyeR,f.eyeR.x,1-f.eyeR.y);gl.uniform2f(u.faceC,f.center.x,1-f.center.y);gl.uniform2f(u.jaw,f.jaw.x,1-f.jaw.y);gl.uniform2f(u.forehead,f.forehead.x,1-f.forehead.y);gl.uniform1f(u.faceW,f.width);gl.uniform1f(u.faceH,f.height);}gl.drawArrays(gl.TRIANGLES,0,3);
}
function drawAnimeLineArt(q,mode){
  const f=latestFace;if(!f||f.length<455||mode==='live-soft')return;
  const p=i=>{const m=mirrorPoint(f[i]);return{x:m.x*canvas.width,y:m.y*canvas.height}};
  const paths=[
    // Intentional anime lines only: eyes, brows, mouth, short nose hint and face silhouette.
    [33,160,158,133,153,144,163,7,33],[362,385,387,263,373,380,390,249,362],
    [70,63,105,66,107],[336,296,334,293,300],
    [61,146,91,181,84,17,314,405,321,375,291],
    [168,6,197],
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
  ];
  const strength=mode==='live-strong'?.86:.68;ctx.save();ctx.clip(triangleUnionPath(q));ctx.lineCap='round';ctx.lineJoin='round';
  // soft white under-stroke cleans photographic texture below the important anime lines
  ctx.strokeStyle=`rgba(255,246,241,${.26*strength})`;ctx.lineWidth=Math.max(2.0,canvas.width*(mode==='live-strong'?.0038:.0030));
  for(const ids of paths){ctx.beginPath();const a=p(ids[0]);ctx.moveTo(a.x,a.y);for(let i=1;i<ids.length;i++){const b=p(ids[i]);ctx.lineTo(b.x,b.y);}ctx.stroke();}
  ctx.strokeStyle=`rgba(45,31,58,${.78*strength})`;ctx.lineWidth=Math.max(1.15,canvas.width*(mode==='live-strong'?.00215:.00175));
  for(const ids of paths){ctx.beginPath();const a=p(ids[0]);ctx.moveTo(a.x,a.y);for(let i=1;i<ids.length;i++){const b=p(ids[i]);ctx.lineTo(b.x,b.y);}ctx.stroke();}
  const fu=faceUniforms();if(fu){
    const eyeRadius=canvas.width*fu.width*(mode==='live-strong'?.027:.023);
    for(const [idx,e] of [[0,fu.eyeL],[1,fu.eyeR]]){
      const ex=e.x*canvas.width,ey=e.y*canvas.height;
      const grad=ctx.createRadialGradient(ex-eyeRadius*.22,ey-eyeRadius*.25,eyeRadius*.10,ex,ey,eyeRadius);
      grad.addColorStop(0,'rgba(250,252,255,.96)');grad.addColorStop(.25,idx===0?'rgba(104,126,190,.94)':'rgba(104,126,190,.94)');grad.addColorStop(.72,'rgba(50,54,92,.96)');grad.addColorStop(1,'rgba(20,18,35,.98)');
      ctx.fillStyle=grad;ctx.beginPath();ctx.ellipse(ex,ey+eyeRadius*.08,eyeRadius*.62,eyeRadius,0,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='rgba(10,8,20,.96)';ctx.beginPath();ctx.arc(ex,ey+eyeRadius*.10,eyeRadius*.34,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='rgba(255,255,255,.92)';ctx.beginPath();ctx.arc(ex-eyeRadius*.20,ey-eyeRadius*.20,eyeRadius*.16,0,Math.PI*2);ctx.fill();
    }
  }
  ctx.restore();
}
function applyLiveFx(q,mode){renderLiveFx(mode);ctx.save();ctx.clip(triangleUnionPath(q));ctx.drawImage(fxCanvas,0,0,canvas.width,canvas.height);ctx.restore();drawAnimeLineArt(q,mode);}

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
function toggleRecording(){if(!running||!MediaRecorder||!canvas.captureStream){showError('Nagrywanie niedostępne.');return;}if(mediaRecorder?.state==='recording'){mediaRecorder.stop();return;}recordedChunks=[];const mime=preferredMime(),s=canvas.captureStream(30);mediaRecorder=new MediaRecorder(s,mime?{mimeType:mime,videoBitsPerSecond:8_000_000}:{videoBitsPerSecond:8_000_000});mediaRecorder.ondataavailable=e=>{if(e.data?.size)recordedChunks.push(e.data)};mediaRecorder.onstop=()=>{const type=mediaRecorder.mimeType||mime||'video/webm',blob=new Blob(recordedChunks,{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`magic-frame-v10-4-${Date.now()}.${type.includes('mp4')?'mp4':'webm'}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);recordBtn.textContent='Nagraj';recBadge.hidden=true;};mediaRecorder.start(250);recordingStartedAt=performance.now();recordBtn.textContent='Stop';recBadge.hidden=false;}
function drawDebug(s,q,raw){if(!debugToggle.checked)return;ctx.save();ctx.fillStyle='#00ffb4';for(const h of s)for(const p of h.pts){ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,2.5,0,Math.PI*2);ctx.fill();}if(raw){const t=raw.map(p=>({x:p.x*canvas.width,y:p.y*canvas.height}));ctx.setLineDash([6,6]);ctx.strokeStyle='#00ffff';ctx.beginPath();ctx.moveTo(t[0].x,t[0].y);for(let i=1;i<4;i++)ctx.lineTo(t[i].x,t[i].y);ctx.closePath();ctx.stroke();}ctx.restore();}
function tickHand(now){handCompleted++;const dt=now-handWindowStart;if(dt>=700){handFps=handCompleted*1000/dt;handCompleted=0;handWindowStart=now;}}
function tickRender(now){renderCompleted++;const dt=now-renderWindowStart;if(dt>=700){renderFps=renderCompleted*1000/dt;renderCompleted=0;renderWindowStart=now;updateDiag();}}

async function renderLoop(ts){
  if(!running)return;tickRender(ts);resizeCanvas();drawVideo();
  if(handsReady&&video.readyState>=2&&ts-lastHandDetect>HAND_INTERVAL){lastHandDetect=ts;try{const r=handLandmarker.detectForVideo(video,ts);latestHands=r.landmarks||[];latestHandedness=r.handedness||[];tickHand(ts);}catch{}}
  if(faceReady&&video.readyState>=2&&ts-lastFaceDetect>FACE_INTERVAL){lastFaceDetect=ts;try{latestFace=faceLandmarker.detectForVideo(video,ts).faceLandmarks?.[0]||null;}catch{}}
  const semantic=semanticHands(latestHands,latestHandedness),raw=measureFreeformQuad(semantic),q=smoothQuad(raw,ts),mode=effectSelect.value;
  if(q){
    if(mode==='live'||mode==='live-soft'||mode==='live-strong')applyLiveFx(q,mode);
    else if(isAiMode()){latestQuadForAi=q.map(p=>({...p}));kickAi();applyAi(q);}
    drawFrame(q);const extra=isAiMode()&&lastCartoonDone?` · AI ${Math.round(performance.now()-lastCartoonDone)}ms old`:'';setStatus(`2/2 dłonie · ${mode==='live'?'ANIME':mode==='live-soft'?'ANIME SOFT':mode==='live-strong'?'STRONG ANIME':mode==='original'?'ORIGINAL':'AI QUALITY'}${extra}`);
  }else{latestQuadForAi=null;if(handsReady)setStatus(`${Math.min(2,semantic.length)}/2 dłonie`);}
  if(mediaRecorder?.state==='recording')recBadge.textContent=`● REC ${Math.floor((ts-recordingStartedAt)/1000)}s`;
  drawDebug(semantic,q,raw);requestAnimationFrame(renderLoop);
}

setStatus('Gotowy • v10.4 Clean Anime');
