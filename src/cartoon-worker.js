import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

let model = null;
let backend = 'cpu';
let style = 'shinkai';
let busy = false;

function postError(err) {
  self.postMessage({ type: 'error', message: `${err?.name || 'Error'}: ${err?.message || err}` });
}

async function chooseBackend() {
  if (self.navigator?.gpu) {
    try {
      await tf.setBackend('webgpu');
      await tf.ready();
      backend = 'webgpu';
      return;
    } catch (err) {
      console.warn('Worker WebGPU failed, fallback CPU', err);
    }
  }
  await tf.setBackend('cpu');
  await tf.ready();
  backend = 'cpu';
}

async function init(msg) {
  try {
    tf.enableProdMode();
    style = msg.style || 'shinkai';
    await chooseBackend();
    model = await tf.loadGraphModel(msg.modelUrl);
    const warm = tf.zeros([1, 160, 160, 3]);
    let out = null;
    try {
      const t = performance.now();
      out = await model.executeAsync(warm);
      if (Array.isArray(out)) out = out[0];
      await out.data();
      console.log(`Cartoon worker warmup ${backend}: ${Math.round(performance.now() - t)}ms`);
    } finally {
      warm.dispose();
      out?.dispose?.();
    }
    self.postMessage({ type: 'ready', backend, style });
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
    output = await model.executeAsync(input);
    if (Array.isArray(output)) output = output[0];
    rendered = tf.tidy(() => output.squeeze([0]).reverse(2).mul(0.5).add(0.5).clipByValue(0, 1));
    const readStart = performance.now();
    const pixels = await rendered.data(); // async GPU->CPU readback, ale WYŁĄCZNIE w Workerze
    const readMs = performance.now() - readStart;
    const computeMs = readStart - computeStart;
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
      computeMs, readMs,
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
