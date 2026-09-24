"""Login tokens (JWT) and role checks for the PayMind API.

Flow:
  1. POST /api/auth/login with email + password → the password is checked
     against its scrypt hash → a signed JWT comes back.
  2. Every other request sends `Authorization: Bearer <token>`.
  3. The server verifies the signature and expiry, then loads the user from
     the database. The role used for permissions comes from the DATABASE, not
     from the token, so a changed or removed user loses access immediately.

Token contents (signed with HS256 and JWT_SECRET from .env, not encrypted:
anyone holding it can read it, nobody can change it without the secret):
  sub (user_id), role, name, iat (issued at), exp (expires), iss ("paymind")
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Callable

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..app.database import AppDatabase, User

ALGORITHM = "HS256"
ISSUER = "paymind"
TOKEN_TTL = timedelta(hours=8)

bearer = HTTPBearer(auto_error=False)


def secret_key() -> str:
    key = os.getenv("JWT_SECRET")
    if not key or len(key) < 32:
        raise RuntimeError("JWT_SECRET must be set in .env (at least 32 characters)")
    return key


def create_token(user: User, secret: str, ttl: timedelta = TOKEN_TTL, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    payload = {
        "sub": user.user_id,
        "role": user.role,
        "name": user.name,
        "iss": ISSUER,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> dict:
    """Raises jwt.InvalidTokenError (bad signature, expired, wrong issuer, missing claims)."""
    return jwt.decode(token, secret, algorithms=[ALGORITHM], issuer=ISSUER,
                      options={"require": ["sub", "exp", "iat", "iss"]})


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(request: Request, creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> User:
    """FastAPI dependency: the logged-in user, or 401."""
    if creds is None or creds.scheme.lower() != "bearer":
        raise unauthorized("Log in first: missing bearer token.")
    try:
        claims = decode_token(creds.credentials, request.app.state.jwt_secret)
    except jwt.ExpiredSignatureError:
        raise unauthorized("Your session has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise unauthorized("Invalid token.")
    appdb: AppDatabase = request.app.state.services.appdb
    user = appdb.get_user(claims["sub"])
    if user is None:
        raise unauthorized("This account no longer exists.")
    return user


def require_role(*roles: str) -> Callable[..., User]:
    """FastAPI dependency: the logged-in user if their role is allowed, else 403."""

    def check(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail=f"Only {' or '.join(roles)} users can do this.")
        return user

    return check
