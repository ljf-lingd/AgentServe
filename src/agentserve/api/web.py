"""V6 browser access and local account provisioning."""

import argparse
import asyncio
import hashlib
import os
import re
import secrets
import sqlite3
import time
from getpass import getpass
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from filelock import FileLock
from pydantic import BaseModel, Field

from agentserve.config import DEFAULT_APP_CONFIG, Settings
from agentserve.storage.repository import TaskRepository

COOKIE = "agentserve_login"
SESSION_SECONDS = 8 * 60 * 60
LOOPBACK = {"127.0.0.1", "::1", "localhost"}
USERNAME = r"^[a-z0-9][a-z0-9_.-]{2,63}$"


def web_settings(mode: str) -> Settings:
    config = Settings.load(Path(os.environ.get("AGENTSERVE_CONFIG", str(DEFAULT_APP_CONFIG))))
    state_dir = Path(os.environ.get("AGENTSERVE_WEB_STATE_DIR", str(config.state_database.parent)))
    config.state_database = state_dir.resolve() / f"v6-{mode}.sqlite"
    config.session_sam_all_phases = os.environ.get(
        "AGENTSERVE_WEB_SAM_ALL_PHASES", "1" if config.session_sam else "0",
    ) == "1"
    if model_port := os.environ.get("AGENTSERVE_WEB_MODEL_PORT"):
        config.base_url = f"http://127.0.0.1:{int(model_port)}/v1"
    if resource_port := os.environ.get("AGENTSERVE_WEB_RESOURCE_PORT"):
        if config.memory is None:
            raise ValueError("V6 requires a memory configuration")
        config.memory.resources_url = f"http://127.0.0.1:{int(resource_port)}"
    return config


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def denied(status: int, code: str):
    return JSONResponse(status_code=status, content={"error": {"code": code}})


class LoginInput(BaseModel):
    username: str = Field(pattern=USERNAME)
    password: str = Field(min_length=1, max_length=128)


class Accounts:
    def __init__(self):
        from argon2 import PasswordHasher

        self.hasher = PasswordHasher()
        self.dummy = self.hasher.hash(secrets.token_urlsafe(32))
        self.window_end = 0.0
        self.attempts: dict[str, int] = {}
        self.total = 0

    def rate_allowed(self, username: str) -> bool:
        now = time.monotonic()
        if now >= self.window_end:
            self.window_end = now + 300
            self.attempts.clear()
            self.total = 0
        if self.total >= 30 or self.attempts.get(username, 0) >= 6:
            return False
        self.total += 1
        self.attempts[username] = self.attempts.get(username, 0) + 1
        return True

    async def authenticate(self, repo, body):
        from argon2.exceptions import VerificationError

        async with repo.lock:
            rows = await repo.connection.execute_fetchall(
                "SELECT user_id,username,role,password_hash FROM web_users WHERE username=?",
                (body.username,),
            )
        user = dict(rows[0]) if rows else None
        hashed = user.pop("password_hash") if user else self.dummy
        try:
            await asyncio.to_thread(self.hasher.verify, hashed, body.password)
        except VerificationError:
            return None
        return user


def mount_access(app: FastAPI, mode: str) -> None:
    accounts = Accounts() if mode == "auth" else None

    @app.middleware("http")
    async def browser_access(request: Request, call_next):
        if (
            request.client is None or request.client.host not in LOOPBACK
            or request.url.hostname not in LOOPBACK
        ):
            return denied(403, "LOCAL_ACCESS_REQUIRED")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = f"{request.url.scheme}://{request.url.netloc}"
            if request.headers.get("origin") != origin:
                return denied(403, "ORIGIN_DENIED")
        path = request.url.path
        protected = path.split("/")[1] in {"tasks", "sessions", "admin", "catalog"}
        request.state.user = None
        if accounts is not None and (protected or path == "/auth/me"):
            token = request.cookies.get(COOKIE)
            if token:
                repo = request.app.state.tasks.repository
                async with repo.lock:
                    rows = await repo.connection.execute_fetchall(
                        """SELECT u.user_id,u.username,u.role
                        FROM web_logins l JOIN web_users u USING(user_id)
                        WHERE l.token_hash=? AND l.expires_at>?""",
                        (token_hash(token), time.time()),
                    )
                if rows:
                    request.state.user = dict(rows[0])
            if protected and request.state.user is None:
                return denied(401, "LOGIN_REQUIRED")
            if path.startswith("/admin/") and request.state.user["role"] != "admin":
                return denied(403, "ADMIN_REQUIRED")
        response = await call_next(request)
        if protected or path.startswith("/auth/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/auth/me")
    async def whoami(request: Request):
        user = request.state.user
        return {"mode": mode, "user": (
            {"username": user["username"], "role": user["role"]} if user else None
        )}

    if accounts is None:
        return

    @app.post("/auth/login")
    async def login(body: LoginInput, request: Request, response: Response):
        if not accounts.rate_allowed(body.username):
            return denied(429, "LOGIN_RATE_LIMITED")
        repo = request.app.state.tasks.repository
        user = await accounts.authenticate(repo, body)
        if user is None:
            return denied(401, "INVALID_CREDENTIALS")
        token = secrets.token_urlsafe(32)
        now = time.time()
        async with repo.transaction() as conn:
            await conn.execute("DELETE FROM web_logins WHERE expires_at<=?", (now,))
            if previous := request.cookies.get(COOKIE):
                await conn.execute("DELETE FROM web_logins WHERE token_hash=?", (token_hash(previous),))
            await conn.execute(
                "INSERT INTO web_logins VALUES (?,?,?)",
                (token_hash(token), user["user_id"], now + SESSION_SECONDS),
            )
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return {"username": user["username"], "role": user["role"]}

    @app.post("/auth/logout", status_code=204)
    async def logout(request: Request):
        if token := request.cookies.get(COOKIE):
            async with request.app.state.tasks.repository.transaction() as conn:
                await conn.execute("DELETE FROM web_logins WHERE token_hash=?", (token_hash(token),))
        response = Response(status_code=204)
        response.delete_cookie(COOKIE, httponly=True, samesite="strict",
                               secure=request.url.scheme == "https")
        return response


async def create_account(username: str, password: str, role: str, path: Path) -> None:
    from argon2 import PasswordHasher

    if not re.fullmatch(USERNAME, username) or not 12 <= len(password) <= 128:
        raise ValueError("Username must be 3–64 lowercase characters. Password must be 12–128 characters.")
    hashed = PasswordHasher().hash(password)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        repo = await TaskRepository.open(path, web=True)
        try:
            async with repo.transaction() as conn:
                await conn.execute(
                    "INSERT INTO web_users VALUES (?,?,?,?)", (uuid4().hex, username, hashed, role)
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists") from exc
        finally:
            await repo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision a V6 account with the API stopped.")
    parser.add_argument("username")
    parser.add_argument("--role", choices=("user", "admin"), default="user")
    args = parser.parse_args()
    password = getpass("Password: ")
    if password != getpass("Confirm password: "):
        parser.error("Passwords differ")
    try:
        asyncio.run(create_account(args.username, password, args.role, web_settings("auth").state_database))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
