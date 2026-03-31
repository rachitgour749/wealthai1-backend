"""Quick token refresh."""
import httpx, json
keys = json.load(open("Services/ChatAI/data/api_keys.json"))
k = keys["cust_699df139"]
r = httpx.post(
    f"{k['zoho_domain']}/oauth/v2/token",
    data={
        "grant_type": "refresh_token",
        "client_id": k["zoho_client_id"],
        "client_secret": k["zoho_client_secret"],
        "refresh_token": k["zoho_refresh_token"],
    },
    timeout=15.0
)
d = r.json()
print(d)
if "access_token" in d:
    k["zoho_access_token"] = d["access_token"]
    keys["cust_699df139"] = k
    json.dump(keys, open("Services/ChatAI/data/api_keys.json", "w"), indent=2)
    print("Token refreshed OK")
