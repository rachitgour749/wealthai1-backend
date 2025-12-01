import requests
import json

url = "http://localhost:8000/api/subscription/activate-trial"
payload = {
    "user_email": "rachit.gour749@gmail.com",
    "user_name": "Rachit Gour",
    "plan_code": 1
}
headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
