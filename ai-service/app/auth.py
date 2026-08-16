import os
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import HTTPException, Request

SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = "HS256"


def create_service_token() -> str:
    """Token dipakai ai-service sendiri untuk memanggil endpoint backend yang
    dilindungi (fetch cameras, post detections/events, dll) — bukan token user."""
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    return jwt.encode({"sub": "ai-service", "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _extract_token(request: Request) -> str | None:
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.query_params.get("token")  # fallback for <img src> usage


async def get_current_user(request: Request) -> str:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired",
                             headers={"WWW-Authenticate": "Bearer"})
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token",
                             headers={"WWW-Authenticate": "Bearer"})
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    return username
