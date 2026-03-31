"""Exchange Zoho auth code - INTERACTIVE with CORRECT client credentials."""
import httpx, json, sys, os

CLIENT_ID = "1000.GYYZ1XVI1TCHHIUAHT95Q6HUBWQ5LQ"
CLIENT_SECRET = "df6d07e2db46aa7ad5b6ed956cf0561c573f1f785f"
DOMAIN = "https://accounts.zoho.in"

code = input("Paste your Zoho auth code now: ").strip()

print("Exchanging immediately...")
resp = httpx.post(
    f"{DOMAIN}/oauth/v2/token",
    data={
        "grant_type": "authorization_code",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
    },
    timeout=15.0
)

data = resp.json()
print(f"Response: {json.dumps(data, indent=2)}")

if "refresh_token" in data:
    with open("Services/ChatAI/data/api_keys.json", "r") as f:
        keys = json.load(f)
    k = keys["cust_699df139"]
    k["zoho_access_token"] = data["access_token"]
    k["zoho_refresh_token"] = data["refresh_token"]
    k["zoho_client_id"] = CLIENT_ID
    k["zoho_client_secret"] = CLIENT_SECRET
    keys["cust_699df139"] = k
    with open("Services/ChatAI/data/api_keys.json", "w") as f:
        json.dump(keys, f, indent=2)
    print("SUCCESS - New credentials and refresh token saved!")
else:
    print(f"FAILED: {data}")
