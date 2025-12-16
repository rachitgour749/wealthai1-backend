
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import hashlib
import logging
import sys # Added missing import
from typing import List

from Services.subscription.database import subscription_manager
try:
    from jose import jwt
except ImportError:
    import jwt # Fallback to PyJWT if jose not present

logger = logging.getLogger(__name__)

class SingleSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exempt_paths: List[str] = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or []
        # Normalizing paths for easier matching (strip trailing slashes)
        self.exempt_paths = [p.rstrip("/") for p in self.exempt_paths]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path.rstrip("/")
        
        # Check if path is exempt
        for exempt_path in self.exempt_paths:
            if path == exempt_path:
                 return await call_next(request)
            if exempt_path and path.startswith(f"{exempt_path}/"):
                 return await call_next(request)
        
        # Exclude OPTIONS (CORS preflight) and Doc paths if not in exempt list
        if request.method == "OPTIONS":
            return await call_next(request)
            
        if any(path.startswith(p) for p in ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]):
             return await call_next(request)

        # 1. Extract Token
        auth_header = request.headers.get("Authorization")
        if not auth_header:
             return JSONResponse(
                status_code=401,
                content={"detail": "Missing mandatory header: Authorization"}
            )
        
        token = auth_header.replace("Bearer ", "").strip()

        # 2. Extract User Email (from Header OR Token)
        user_email = request.headers.get("user-email") or request.query_params.get("user_email")
        
        if not user_email:
            # Try to extract from JWT
            try:
                # Decode without verification just to get the email (verification is done via DB lookup of the token string)
                if 'jose' in sys.modules:
                    claims = jwt.get_unverified_claims(token)
                else:
                    claims = jwt.decode(token, options={"verify_signature": False})
                
                user_email = claims.get("email")
            except Exception as e:
                logger.warning(f"Failed to decode token for email extraction: {e}")
        
        if not user_email:
            return JSONResponse(
                status_code=401,
                content={
                    "status_code": 401,
                    "detail": "Could not identify user. Missing 'user-email' header and invalid token claims."
                }
            )
        
        token = auth_header.replace("Bearer ", "").strip()
        
        # 2. Compute Hash - DISABLED (Using Raw Token Comparison)
        # token_hash = hashlib.sha256(token.encode()).hexdigest()

        # 3. Check DB
        try:
            user = subscription_manager.get_user_details(user_email)
            
            if not user:
                 return JSONResponse(
                    status_code=401,
                    content={"detail": "User not found or session invalid"}
                )
            
            if not user.active_token_hash:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Session invalid. Please login again."}
                )
                
            # Compare Raw Token directly
            # Note: user.active_token_hash stores the RAW token now
            if user.active_token_hash != token:
                logger.warning(f"Session invalidated for {user_email}. Token hash mismatch.")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Session expired or logged in from another device."}
                )

        except Exception as e:
            logger.error(f"Middleware session check failed: {e}")
            return await call_next(request)

        return await call_next(request)
