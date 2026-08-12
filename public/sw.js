const CACHE = 'magic-frame-v10-3-runtime-v1';
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  const cacheable = /\.(task|wasm|json|bin|js|css)$/i.test(url.pathname) || url.pathname.includes('/models/') || url.pathname.includes('/mediapipe/');
  if (!cacheable) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(req);
    if (cached) return cached;
    const response = await fetch(req);
    if (response.ok) cache.put(req, response.clone());
    return response;
  })());
});
