import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-webgl';

let model = null;
let backend = 'cpu';
let style = 'shinkai';
let busy = false;

function postError(err) {
  self.postMessage({ type: 'error', message: `${err?.name || 'Error'}: ${err?.message || err}` });
}

async function setPreferredBackend(preferred = 'auto') {
  const candidates = [];
  if (preferred === 'webgl') candidates.push('webgl', 'webgpu', 'cpu');
  else if (preferred === 'webgpu') candidates.push('webgpu', 'webgl', 'cpu');
  else candidates.push('webgl', 'webgpu', 'cpu');

  for (const candidate of candidates) {
    if (candidate === 'webgpu' && !self.navigator?.gpu) continue;
    try {
      const ok = await tf.setBackend(candidate);
      if (!ok) continue;
      await tf.ready();
      backend = candidate;
      return;
    } catch (err) {
      console.warn(`Cartoon worker backend ${candidate} failed`, err);
    }
  }
  throw new Error('Brak działającego backendu TensorFlow.js');
}

async function warmup(size = 96) {
  const warm = tf.zeros([1, size, size, 3]);
  let out = null;
  try {
    // Ten GraphModel nie ma control-flow/dynamic output. execute() usuwa narzut
    // executeAsync(); rzeczywiste GPU zakończy pracę przy await out.data().
    const t = performance.now();
    out = model.execute(warm);
    if (Array.isArray(out)) out = out[0];
    await out.data();
    return performance.now() - t;
  } finally {
    warm.dispose();
    out?.dispose?.();
  }
}

async function init(msg) {
  try {
    tf.enableProdMode();
    style = msg.style || 'shinkai';
    await setPreferredBackend(msg.preferredBackend || 'auto');
    model = await tf.loadGraphModel(msg.modelUrl);
    const first = await warmup(96);
    const second = await warmup(96);
    console.log(`Cartoon worker warmup ${backend}: ${Math.round(first)}ms -> ${Math.round(second)}ms`);
    self.postMessage({ type: 'ready', backend, style, warmupMs: second });
  } catch (err) { postError(err); }
}

async function infer(msg) {
  if (!model || busy) return;
  busy = true;
  const totalStart = performance.now();
  let input = null;
  let output = null;
  let rendered = null;
  try {
    const size = msg.size;
    const rgba = new Uint8ClampedArray(msg.rgba);
    const bgr = new Float32Array(size * size * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
      bgr[j] = rgba[i + 2];
      bgr[j + 1] = rgba[i + 1];
      bgr[j + 2] = rgba[i];
    }
    input = tf.tensor4d(bgr, [1, size, size, 3], 'float32');
    const computeStart = performance.now();
    output = model.execute(input);
    if (Array.isArray(output)) output = output[0];
    rendered = tf.tidy(() => output.squeeze([0]).reverse(2).mul(0.5).add(0.5).clipByValue(0, 1));
    const readStart = performance.now();
    const pixels = await rendered.data();
    const readMs = performance.now() - readStart;
    // GPU jest leniwe: computeMs do readStart mierzy enqueue, a pełna inferencja
    // kończy się podczas data(). Raportujemy więc inferMs jako czas execute+read.
    const inferMs = performance.now() - computeStart;
    const outRgba = new Uint8ClampedArray(size * size * 4);
    for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
      outRgba[j] = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
      outRgba[j + 1] = Math.max(0, Math.min(255, Math.round(pixels[i + 1] * 255)));
      outRgba[j + 2] = Math.max(0, Math.min(255, Math.round(pixels[i + 2] * 255)));
      outRgba[j + 3] = 255;
    }
    self.postMessage({
      type: 'frame', requestId: msg.requestId, size,
      rgba: outRgba.buffer,
      totalMs: performance.now() - totalStart,
      computeMs: inferMs,
      readMs,
      backend
    }, [outRgba.buffer]);
  } catch (err) {
    postError(err);
  } finally {
    input?.dispose?.();
    output?.dispose?.();
    rendered?.dispose?.();
    busy = false;
  }
}

self.onmessage = (event) => {
  const msg = event.data || {};
  if (msg.type === 'init') init(msg);
  else if (msg.type === 'infer') infer(msg);
};
