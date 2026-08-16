from fastapi import APIRouter, HTTPException, Request

from app.auth import create_access_token, verify_password
from app.schemas import LoginRequest, LoginResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request) -> LoginResponse:
    pool = request.app.state.pool
    row = await pool.fetchrow(
        "SELECT id, password_hash FROM users WHERE username = $1", req.username
    )
    if row is None or not verify_password(req.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    await pool.execute("UPDATE users SET last_login = NOW() WHERE id = $1", row["id"])
    return LoginResponse(access_token=create_access_token(req.username))
