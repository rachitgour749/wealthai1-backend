import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Services.broker_manager import login_broker

class TestKotakLogin(unittest.TestCase):
    
    @patch('Broker.Kotak.KOTAK.requests.Session')
    @patch('Broker.Kotak.KOTAK.pyotp.TOTP')
    def test_kotak_login_success(self, mock_totp, mock_session):
        # Mock TOTP
        mock_totp_instance = MagicMock()
        mock_totp_instance.now.return_value = "123456"
        mock_totp.return_value = mock_totp_instance
        
        # Mock Session
        mock_sess = MagicMock()
        mock_session.return_value = mock_sess
        
        # Mock Step 1 Response
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {
            "data": {
                "token": "token_step1",
                "sid": "sid_step1",
                "kType": "View",
                "status": "success"
            }
        }
        
        # Mock Step 2 Response
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.json.return_value = {
            "data": {
                "token": "final_access_token",
                "sid": "final_sid",
                "baseUrl": "https://cis.kotaksecurities.com",
                "kType": "Trade",
                "status": "success"
            }
        }
        
        # Configure side_effect for post calls (first call returns resp1, second returns resp2)
        mock_sess.post.side_effect = [mock_resp1, mock_resp2]
        
        credentials = {
            "mobileNumber": "+919876543210",
            "ucc": "CLIENT123",
            "totp_secret": "JBSWY3DPEHPK3PXP",
            "mpin": "123456",
            "access_token": "initial_auth_token"
        }
        
        result = login_broker("kotak", credentials)
        
        print("Login Result:", result)
        
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['access_token'], 'final_access_token')
        self.assertEqual(result['sid'], 'final_sid')
        self.assertEqual(result['client_id'], 'CLIENT123')
        
    @patch('Broker.Kotak.KOTAK.requests.Session')
    def test_kotak_login_failure_step1(self, mock_session):
         # Mock Session
        mock_sess = MagicMock()
        mock_session.return_value = mock_sess
        
        # Mock Step 1 Response - Failure
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 401
        mock_resp1.text = '{"message": "Invalid TOTP"}'
        mock_resp1.json.return_value = {"message": "Invalid TOTP"}
        
        mock_sess.post.return_value = mock_resp1
        
        credentials = {
            "mobileNumber": "+919876543210",
            "ucc": "CLIENT123",
            "totp_secret": "JBSWY3DPEHPK3PXP",
            "mpin": "123456",
            "access_token": "initial_auth_token"
        }
        
        result = login_broker("kotak", credentials)
        
        print("Login Failure Result:", result)
        self.assertEqual(result['status'], 'error')
        self.assertIn("Login Step 1 Failed", result['message'])

if __name__ == '__main__':
    unittest.main()
