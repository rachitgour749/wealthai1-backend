# AngelOne Broker Login - API Parameters

## Endpoint
```
POST /api/broker/broker_login
```

## Required Parameters for AngelOne

### Request Body (JSON)

```json
{
  "broker_name": "angelone",
  "user_email": "your_app_user_email@example.com",
  "api_key": "YOUR_ANGELONE_API_KEY",
  "client_code": "YOUR_CLIENT_CODE",
  "password": "YOUR_PASSWORD",
  "totp_secret": "YOUR_TOTP_SECRET"
}
```

### Parameter Details

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `broker_name` | string | ✅ | Must be `"angelone"` (case-insensitive) |
| `user_email` | string | ✅ | App user's email for session tracking |
| `api_key` | string | ✅ | AngelOne API key from developer portal |
| `client_code` | string | ✅ | Your AngelOne client code (e.g., "A12345") |
| `password` | string | ✅ | Your AngelOne account password |
| `totp_secret` | string | ✅ | TOTP secret key for 2FA authentication |

### Optional Header

```
X-Static-IP: 192.168.1.100
```

- **Purpose**: For SEBI compliance
- **Description**: Your static IP address that will be used for order placement
- **Optional**: If not provided, orders will be placed without IP

---

## Example Request

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/broker/broker_login" \
  -H "Content-Type: application/json" \
  -H "X-Static-IP: 192.168.1.100" \
  -d '{
    "broker_name": "angelone",
    "user_email": "user@example.com",
    "api_key": "YOUR_API_KEY",
    "client_code": "A12345",
    "password": "your_password",
    "totp_secret": "ABCD1234EFGH5678"
  }'
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/broker/broker_login"

headers = {
    "Content-Type": "application/json",
    "X-Static-IP": "192.168.1.100"  # Optional
}

payload = {
    "broker_name": "angelone",
    "user_email": "user@example.com",
    "api_key": "YOUR_API_KEY",
    "client_code": "A12345",
    "password": "your_password",
    "totp_secret": "ABCD1234EFGH5678"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Using JavaScript (Fetch)

```javascript
const url = "http://localhost:8000/api/broker/broker_login";

const headers = {
  "Content-Type": "application/json",
  "X-Static-IP": "192.168.1.100"  // Optional
};

const payload = {
  broker_name: "angelone",
  user_email: "user@example.com",
  api_key: "YOUR_API_KEY",
  client_code: "A12345",
  password: "your_password",
  totp_secret: "ABCD1234EFGH5678"
};

fetch(url, {
  method: "POST",
  headers: headers,
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error("Error:", error));
```

---

## Success Response

```json
{
  "status": "success",
  "message": "angelone login successful",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expire": "2026-02-04 11:00:00",
  "broker_name": "angelone",
  "user_email": "user@example.com",
  "client_id": "A12345"
}
```

## Error Response

```json
{
  "status": "error",
  "message": "Invalid TOTP or credentials"
}
```

---

## Comparison: Zerodha vs AngelOne Parameters

| Parameter | Zerodha | AngelOne |
|-----------|---------|----------|
| `broker_name` | `"zerodha"` | `"angelone"` |
| `user_email` | ✅ Required | ✅ Required |
| `api_key` | ✅ Required | ✅ Required |
| `api_secret` | ✅ Required | ❌ Not used |
| `username` | ✅ Required | ❌ Not used |
| `client_code` | ❌ Not used | ✅ Required |
| `password` | ✅ Required | ✅ Required |
| `totp_secret` | ✅ Required | ✅ Required |

---

## Notes

1. **TOTP Secret**: This is the secret key used to generate TOTP codes, NOT the 6-digit TOTP code itself. The system will automatically generate the TOTP code.

2. **Client Code**: This is your AngelOne client ID (e.g., "A12345"), not to be confused with the API key.

3. **Session Storage**: After successful login, the session is stored in the database and linked to your `user_email`. You don't need to login again for subsequent order placements.

4. **Static IP**: If you provide a static IP during login, it will be stored and automatically used for all order placements for that user.

5. **Token Expiry**: The access token typically expires after 24 hours. You'll need to login again after expiry.

---

## Testing

You can test the login using the Swagger UI at:
```
http://localhost:8000/docs
```

Look for the `/api/broker/broker_login` endpoint under "Broker Integration" section.
