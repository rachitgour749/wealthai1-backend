import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyAPIs")

BASE_URL = "http://localhost:8000/api/broker"
TEST_EMAIL = "rachit.gour749@gmail.com"
TEST_CLIENT = "CTU900"

def test_api():
    try:
        # 1. Test Fetch Account Details
        logger.info(f"Testing GET /account_details?user_email={TEST_EMAIL}...")
        resp = requests.get(f"{BASE_URL}/account_details", params={"user_email": TEST_EMAIL})
        logger.info(f"Response Status: {resp.status_code}")
        logger.info(f"Response Body: {json.dumps(resp.json(), indent=2)}")

        if resp.status_code == 200:
            logger.info("✅ SUCCESS: Fetch Account Details works.")
        else:
            logger.error("❌ FAILURE: Fetch Account Details failed.")

        # 2. Test Update Credentials
        update_payload = {
            "user_email": TEST_EMAIL,
            "broker_name": "zerodha",
            "username": TEST_CLIENT,
            "password": "NewSecretPassword@123"
        }
        logger.info(f"Testing POST /update_credentials with payload: {json.dumps(update_payload)}...")
        resp = requests.post(f"{BASE_URL}/update_credentials", json=update_payload)
        logger.info(f"Response Status: {resp.status_code}")
        logger.info(f"Response Body: {json.dumps(resp.json(), indent=2)}")

        if resp.status_code == 200:
            logger.info("✅ SUCCESS: Update Credentials works.")
        else:
            logger.error("❌ FAILURE: Update Credentials failed.")

        # 3. Test Fetch again to verify update
        logger.info("Verifying update via /account_details...")
        resp = requests.get(f"{BASE_URL}/account_details", params={"user_email": TEST_EMAIL})
        details = resp.json().get("data", {})
        creds = details.get("broker_credentials", {})
        if creds.get("password") == "NewSecretPassword@123":
            logger.info("✅ SUCCESS: Password updated in DB.")
        else:
            logger.error(f"❌ FAILURE: Password NOT updated. Got: {creds.get('password')}")

        # 4. Test Delete Account
        logger.info(f"Testing DELETE /delete_account?user_email={TEST_EMAIL}&client_id={TEST_CLIENT}...")
        resp = requests.delete(f"{BASE_URL}/delete_account", params={"user_email": TEST_EMAIL, "client_id": TEST_CLIENT})
        logger.info(f"Response Status: {resp.status_code}")
        logger.info(f"Response Body: {json.dumps(resp.json(), indent=2)}")

        if resp.status_code == 200:
            logger.info("✅ SUCCESS: Delete Account works.")
        else:
            logger.error("❌ FAILURE: Delete Account failed.")

    except Exception as e:
        logger.error(f"Error during API verification: {e}")

if __name__ == "__main__":
    test_api()
