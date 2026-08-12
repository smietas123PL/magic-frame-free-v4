self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil((async () => {
  try { await self.registration.unregister(); } catch {}
  try {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k.startsWith('magic-frame')).map(k => caches.delete(k)));
  } catch {}
  await self.clients.claim();
})()));
