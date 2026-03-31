import urllib.request, json, time
t = time.time()
req = urllib.request.Request("http://127.0.0.1:8000/admin/stores")
req.add_header("X-Admin-Key", "test")
r = urllib.request.urlopen(req, timeout=15)
d = json.loads(r.read())
elapsed = time.time() - t
print(f"Loaded {len(d['stores'])} stores in {elapsed:.1f}s")
for s in d["stores"]:
    print(f"  - {s['display_name']} ({s['doc_count']} docs)")
