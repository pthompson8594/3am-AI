// No-op service worker — prevents 404 on browser's speculative fetch
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
