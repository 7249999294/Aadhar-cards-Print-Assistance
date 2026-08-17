const CACHE_NAME = 'aadhaar-print-v2';
const ASSETS = [
  './aadhaar-print-assistant.html',
  './manifest.json',
  './icons/icon-192.png',
  './icons/icon-512.png',
];

self.addEventListener('install', function(event){
  event.waitUntil(
    caches.open(CACHE_NAME).then(function(cache){ return cache.addAll(ASSETS); })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function(event){
  event.waitUntil(
    caches.keys().then(function(keys){
      return Promise.all(keys.filter(function(k){ return k !== CACHE_NAME; }).map(function(k){ return caches.delete(k); }));
    })
  );
  self.clients.claim();
});

// Network-first: always try to fetch the latest deployed version when
// online, only falling back to the cached copy if the request fails
// (offline). This app gets updated often, so caching-first would leave
// installed phones permanently stuck showing whatever version they first
// loaded, no matter how many times the page is refreshed.
self.addEventListener('fetch', function(event){
  event.respondWith(
    fetch(event.request).then(function(response){
      var copy = response.clone();
      caches.open(CACHE_NAME).then(function(cache){ cache.put(event.request, copy); });
      return response;
    }).catch(function(){
      return caches.match(event.request);
    })
  );
});
