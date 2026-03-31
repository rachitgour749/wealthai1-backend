"""Quick check with longer timeout."""
import urllib.request, json

BASE = "http://127.0.0.1:8000"

tests = [
    ("Backend Health", f"{BASE}/api/auth/health", {"Accept": "application/json"}, 10),
    ("ChatAI Health", f"{BASE}/api/health", {"Accept": "application/json"}, 10),
    ("Admin Stores", f"{BASE}/admin/stores", {"X-Admin-Key": "test"}, 60),
    ("MFD Profile", f"{BASE}/api/mfd/profile", {"x-user-email": "iamshourya007@gmail.com"}, 30),
    ("Admin Access", f"{BASE}/admin/access", {"X-Admin-Key": "test"}, 30),
]

print("=" * 60)
print("ENDPOINT CHECK (after exempt paths fix)")
print("=" * 60)

for name, url, headers, timeout in tests:
    try:
        req = urllib.request.Request(url)
        for k, v in headers.items():
            req.add_header(k, v)
        r = urllib.request.urlopen(req, timeout=timeout)
        data = json.loads(r.read())
        print(f"  [OK] {name}: {json.dumps(data)[:150]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"  [HTTP {e.code}] {name}: {body}")
    except Exception as e:
        print(f"  [ERR] {name}: {str(e)[:150]}")

print("=" * 60)
