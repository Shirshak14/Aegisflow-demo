"""Access control, rate limits and request-size limits for the AegisFlow API.

Two modes, chosen by environment variables when the app is built (``create_app``):

* **Local (default, ``AEGISFLOW_AUTH`` unset or ``off``).** No sign-in. Only requests from this machine
  (a loopback client address with no proxy forwarding headers) are served; anything else gets 403. So
  running ``uvicorn --host 0.0.0.0`` or putting the app behind a reverse proxy does not expose it by accident.
* **Authenticated (``AEGISFLOW_AUTH=on``, or ``AEGISFLOW_ENV=production``).** Every API request needs a
  signed token, sent as ``Authorization: Bearer <token>`` or as the session cookie set by ``POST /auth/session``
  (the dashboard's sign-in page). Tokens are HMAC-SHA256 signed with ``AEGISFLOW_AUTH_SECRET`` and carry the
  user name, a role and an expiry; issue one with
  ``python -m backend.app.security issue-token --user asha --role analyst``.
  Rate limits apply per user.

Roles, each including the ones before it:
  viewer   read everything (dashboard pages, ``/live``, exports, chain verification)
  analyst  + analyst decisions on alerts, capture upload, pushing flows to the streaming scorer
  admin    + replay start / stop and stream reset (shared state every viewer sees)

``AEGISFLOW_ENV=production`` fails closed: the app refuses to start unless authentication can be enforced
(``AEGISFLOW_AUTH_SECRET`` of at least 32 characters, ``AEGISFLOW_AUTH`` not ``off``) and the ledger head anchor
is signed (``AEGISFLOW_LEDGER_KEY``). ``POST /alerts`` keeps its own ``X-API-Key`` check for machine producers.

Not covered: single sign-on, a user store, or revoking a token before it expires (keep tokens short-lived and
rotate ``AEGISFLOW_AUTH_SECRET`` to revoke all of them).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

ROLES = ("viewer", "analyst", "admin")
RANK = {r: i for i, r in enumerate(ROLES)}

AUTH_ENV = "AEGISFLOW_AUTH"
SECRET_ENV = "AEGISFLOW_AUTH_SECRET"
MODE_ENV = "AEGISFLOW_ENV"
LIMITS_ENV = "AEGISFLOW_RATE_LIMITS"
UPLOAD_TIMEOUT_ENV = "AEGISFLOW_UPLOAD_TIMEOUT_S"
INFERENCE_TIMEOUT_ENV = "AEGISFLOW_INFERENCE_TIMEOUT_S"
LEDGER_KEY_ENV = "AEGISFLOW_LEDGER_KEY"  # same variable audit.Ledger reads
COOKIE = "aegisflow_session"
MIN_SECRET_CHARS = 32
MAX_TOKEN_HOURS = 24 * 7

# Requests per minute, per user, in authenticated mode. Local mode is unlimited (one person on one machine).
DEFAULT_LIMITS = {"upload": 6, "inference": 60, "write": 120, "login": 10}

# Paths served without a token in authenticated mode: the dashboard shell (static code, no data), the sign-in
# flow, and a liveness probe. POST /alerts is checked by its own X-API-Key dependency instead.
PUBLIC_PATHS = {"/", "/classic", "/login", "/health", "/auth/session", "/auth/logout", "/auth/me"}
PUBLIC_PREFIXES = ("/static/",)


class ConfigError(RuntimeError):
    """The security settings are incomplete or contradictory; the app must not start."""


@dataclass(frozen=True)
class Principal:
    user: str | None  # None in local mode: nobody signed in, the request came from this machine
    role: str

    def at_least(self, role: str) -> bool:
        return RANK[self.role] >= RANK[role]


LOCAL = Principal(user=None, role="admin")


@dataclass(frozen=True)
class Settings:
    auth: bool = False
    production: bool = False
    secret: bytes | None = None
    limits: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    upload_timeout_s: float = 300.0  # parsing one capture (PCAP / NetFlow) into flows and scoring it
    inference_timeout_s: float = 120.0  # one explanation / attention / forecast request

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        mode = env.get(MODE_ENV, "").strip().lower()
        production = mode in {"production", "prod"}
        raw = env.get(AUTH_ENV, "").strip().lower()
        if raw not in {"", "on", "off"}:
            raise ConfigError(f"{AUTH_ENV} must be 'on' or 'off', got {raw!r}")
        secret = env.get(SECRET_ENV, "")
        problems = []
        if production and raw == "off":
            problems.append(f"{AUTH_ENV}=off is not allowed with {MODE_ENV}=production")
        auth = production or raw == "on"
        if auth and len(secret) < MIN_SECRET_CHARS:
            problems.append(f"{SECRET_ENV} must be set to at least {MIN_SECRET_CHARS} random characters "
                            f"(python -m backend.app.security generate-secret)")
        if production and not env.get(LEDGER_KEY_ENV):
            problems.append(f"{LEDGER_KEY_ENV} must be set in production so the ledger head anchor is signed")
        if problems:
            raise ConfigError("refusing to start: " + "; ".join(problems))
        return cls(auth=auth, production=production, secret=secret.encode("utf-8") if auth else None,
                   limits=parse_limits(env.get(LIMITS_ENV, "")),
                   upload_timeout_s=_seconds(env, UPLOAD_TIMEOUT_ENV, 300.0),
                   inference_timeout_s=_seconds(env, INFERENCE_TIMEOUT_ENV, 120.0))


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        raise ConfigError(f"{name} must be a number of seconds, got {raw!r}")
    if value <= 0:
        raise ConfigError(f"{name} must be positive")
    return value


def parse_limits(spec: str) -> dict[str, int]:
    """``"upload=6,inference=60"`` -> per-minute limits, defaults for the rest. 0 disables one bucket."""
    out = dict(DEFAULT_LIMITS)
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, value = part.partition("=")
        if name.strip() not in out or not value.strip().isdigit():
            raise ConfigError(f"bad {LIMITS_ENV} entry {part!r}; expected name=N with name in {sorted(out)}")
        out[name.strip()] = int(value)
    return out


# ---------------------------------------------------------------------------------------------- tokens
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(secret: bytes, user: str, role: str, hours: float = 12, now: float | None = None) -> str:
    user = user.strip()
    if not user or len(user) > 64:
        raise ValueError("user must be 1-64 characters")
    if role not in RANK:
        raise ValueError(f"role must be one of {ROLES}")
    if not 0 < hours <= MAX_TOKEN_HOURS:
        raise ValueError(f"hours must be in (0, {MAX_TOKEN_HOURS}]")
    iat = int(time.time() if now is None else now)
    body = _b64(json.dumps({"sub": user, "role": role, "iat": iat, "exp": iat + int(hours * 3600)},
                           separators=(",", ":"), sort_keys=True).encode("utf-8"))
    sig = _b64(hmac.new(secret, f"v1.{body}".encode("ascii"), hashlib.sha256).digest())
    return f"v1.{body}.{sig}"


def verify_token(secret: bytes, token: str, now: float | None = None) -> tuple[Principal, int]:
    """Return (principal, expiry) or raise HTTPException(401)."""
    try:
        version, body, sig = token.split(".")
        if version != "v1":
            raise ValueError
        good = hmac.new(secret, f"v1.{body}".encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_unb64(sig), good):
            raise ValueError
        claims = json.loads(_unb64(body))
        user, role, exp = claims["sub"], claims["role"], int(claims["exp"])
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise HTTPException(401, "invalid token")
    if role not in RANK or not isinstance(user, str) or not user:
        raise HTTPException(401, "invalid token")
    if exp <= (time.time() if now is None else now):
        raise HTTPException(401, "token expired; ask an admin for a new one")
    return Principal(user=user, role=role), exp


# ---------------------------------------------------------------------------------------------- requests
FORWARD_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def is_local_request(request: Request) -> bool:
    """A loopback client with no proxy forwarding headers. A reverse proxy on this machine connects from
    loopback too, so forwarded requests are treated as remote."""
    if any(h in request.headers for h in FORWARD_HEADERS):
        return False
    host = request.client.host if request.client else ""
    if host == "testclient":  # Starlette's TestClient
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin or origin == "null":
        return origin is None
    return origin.split("://", 1)[-1].rstrip("/") == request.headers.get("host", "")


class Security:
    """Per-app access control: authentication middleware, role checks, rate limits, bounded workers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._buckets_lock = threading.Lock()

    # --- authentication
    def authenticate(self, request: Request) -> Principal | None:
        """The verified caller, or None when no credentials were sent (authenticated mode only)."""
        header = request.headers.get("authorization", "")
        if header:
            scheme, _, token = header.partition(" ")
            if scheme.lower() != "bearer" or not token.strip():
                raise HTTPException(401, "Authorization header must be 'Bearer <token>'")
            return verify_token(self.settings.secret, token.strip())[0]
        cookie = request.cookies.get(COOKIE)
        if cookie:
            principal = verify_token(self.settings.secret, cookie)[0]
            # Cookies are sent by the browser automatically, so a state-changing request authenticated by one
            # must come from this site (SameSite=Strict already blocks most cross-site sends).
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not _same_origin(request):
                raise HTTPException(403, "cross-origin request refused")
            return principal
        return None

    async def middleware(self, request: Request, call_next: Callable) -> Any:
        try:
            if not self.settings.auth:
                if not is_local_request(request):
                    return JSONResponse(status_code=403, content={"detail": (
                        "AegisFlow is running in local mode and only serves this machine. To serve other machines, "
                        f"set {AUTH_ENV}=on and {SECRET_ENV} (see docs/api.md, 'Access control').")})
                request.state.principal = LOCAL
                return await call_next(request)
            principal = self.authenticate(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        path = request.url.path
        public = path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES) or (path == "/alerts" and request.method == "POST")
        if principal is None and not public:
            return JSONResponse(status_code=401, content={"detail": "sign in required"},
                                headers={"WWW-Authenticate": "Bearer"})
        request.state.principal = principal
        return await call_next(request)

    # --- rate limiting (token bucket per user and bucket name)
    def check_rate(self, bucket: str, key: str, now: float | None = None) -> None:
        per_min = self.settings.limits.get(bucket, 0)
        if not self.settings.auth or per_min <= 0:
            return
        now = time.monotonic() if now is None else now
        with self._buckets_lock:
            tokens, last = self._buckets.get((bucket, key), (float(per_min), now))
            tokens = min(float(per_min), tokens + (now - last) * per_min / 60.0)
            if tokens < 1.0:
                self._buckets[(bucket, key)] = (tokens, now)
                wait = max(1, int((1.0 - tokens) * 60.0 / per_min) + 1)
                raise HTTPException(429, f"rate limit: {per_min} {bucket} requests per minute; retry in {wait} s",
                                    headers={"Retry-After": str(wait)})
            self._buckets[(bucket, key)] = (tokens - 1.0, now)


def principal_of(request: Request) -> Principal:
    p = getattr(request.state, "principal", None)
    if p is None:
        raise HTTPException(401, "sign in required", headers={"WWW-Authenticate": "Bearer"})
    return p


def require(role: str, bucket: str | None = None) -> Callable[[Request], Principal]:
    """FastAPI dependency: the caller must hold ``role`` (or higher); optionally count against a rate bucket."""
    def dependency(request: Request) -> Principal:
        p = principal_of(request)
        if not p.at_least(role):
            raise HTTPException(403, f"needs the {role} role; signed in as {p.user} ({p.role})")
        if bucket:
            client = request.client.host if request.client else "?"
            request.app.state.security.check_rate(bucket, p.user or client)
        return p
    return dependency


# ---------------------------------------------------------------------------------------------- bounded work
T = TypeVar("T")


class BoundedRunner:
    """Run blocking work on a small dedicated pool with a deadline.

    At most ``workers`` jobs run at once; a call made while all are busy gets 429 instead of queueing, so slow
    uploads cannot pile up. On timeout the caller gets 504; the worker thread cannot be killed and finishes in
    the background, still holding its slot, so the bound on concurrent work holds even then.
    """

    def __init__(self, name: str, workers: int, timeout_s: float):
        self.name, self.timeout_s = name, timeout_s
        self._slots = threading.BoundedSemaphore(workers)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"aegisflow-{name}")

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not self._slots.acquire(blocking=False):
            raise HTTPException(429, f"the server is busy with other {self.name} requests; try again shortly",
                                headers={"Retry-After": "5"})
        try:
            future = asyncio.get_running_loop().run_in_executor(self._pool, lambda: fn(*args, **kwargs))
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.timeout_s)
        except asyncio.TimeoutError:
            raise HTTPException(504, f"{self.name} took longer than {self.timeout_s:.0f} s and was abandoned")


# ---------------------------------------------------------------------------------------------- body size
class BodySizeLimit:
    """ASGI middleware: 413 for request bodies over the path's limit, by Content-Length or by counting the
    bytes actually received (chunked uploads)."""

    def __init__(self, app: Any, limit_for: Callable[[str], int]):
        self.app, self.limit_for = app, limit_for

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.limit_for(scope["path"])
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await JSONResponse(status_code=413, content={"detail": f"request body larger than {limit} bytes"})(
                scope, receive, send)
            return
        received = 0

        async def counted() -> dict:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:  # FastAPI re-raises HTTPException from body parsing as-is
                    raise HTTPException(413, f"request body larger than {limit} bytes")
            return message

        await self.app(scope, counted, send)


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.app.security", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-secret", help=f"print a random value for {SECRET_ENV} (or {LEDGER_KEY_ENV})")
    tok = sub.add_parser("issue-token", help=f"print a signed token; reads {SECRET_ENV} from the environment")
    tok.add_argument("--user", required=True)
    tok.add_argument("--role", required=True, choices=ROLES)
    tok.add_argument("--hours", type=float, default=12)
    args = ap.parse_args(argv)
    if args.cmd == "generate-secret":
        print(secrets.token_urlsafe(48))
        return 0
    secret = os.environ.get(SECRET_ENV, "")
    if len(secret) < MIN_SECRET_CHARS:
        ap.error(f"set {SECRET_ENV} (at least {MIN_SECRET_CHARS} characters) to the server's value first")
    print(issue_token(secret.encode("utf-8"), args.user, args.role, args.hours))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
