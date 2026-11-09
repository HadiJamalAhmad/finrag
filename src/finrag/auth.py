"""Who is asking: a Clerk session token, verified here, turned into the user's groups.

The browser signs in with Clerk and sends its short-lived session token (one minute)
as `Authorization: Bearer <token>`; the Vercel proxy forwards it untouched. Here:

1. The token is verified locally (RS256) against Clerk's public keys (JWKS, fetched
   from the Backend API with the secret key, cached): signature, `exp` / `nbf` with a
   small leeway, the issuer (the instance's Frontend API, derived from the publishable
   key) and `azp` (the origin that obtained the token must be one of ours).
2. The user's groups are read from Clerk itself (`GET /v1/users/{id}`, public_metadata
   `groups`, cached a minute), the way JobHack reads its users: an admin sets them in
   the Clerk dashboard, never the browser.

No token: no identity, public documents only. A token that fails any check: 401, never
"public by fallback". Without Clerk keys on the server, sign-in is simply off.
"""

from __future__ import annotations

import base64
import re
import time
from dataclasses import dataclass, field

import httpx
import jwt

API = "https://api.clerk.com/v1"
GROUP_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
JWKS_TTL = 3600.0
GROUPS_TTL = 60.0
LEEWAY = 5  # seconds of clock skew allowed on exp / nbf


class AuthError(Exception):
    """The request carried a token that cannot be trusted (-> 401)."""


@dataclass(frozen=True)
class Identity:
    user_id: str
    groups: tuple[str, ...]


def frontend_api(publishable_key: str) -> str:
    """pk_test_<base64("<instance>.clerk.accounts.dev$")> -> https://<instance>... (= iss)."""
    try:
        encoded = publishable_key.split("_", 2)[2]
        host = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except Exception as exc:
        raise ValueError("not a Clerk publishable key") from exc
    if not host.endswith("$"):
        raise ValueError("not a Clerk publishable key")
    return "https://" + host[:-1]


@dataclass
class ClerkAuth:
    publishable_key: str
    secret_key: str
    authorized_parties: tuple[str, ...]  # origins allowed to obtain tokens (the site)
    transport: httpx.BaseTransport | None = None
    _jwks: tuple[float, dict] | None = field(default=None, repr=False)
    _groups: dict[str, tuple[float, tuple[str, ...]]] = field(default_factory=dict, repr=False)

    @property
    def issuer(self) -> str:
        return frontend_api(self.publishable_key)

    def _get(self, path: str) -> httpx.Response:
        with httpx.Client(transport=self.transport, timeout=10,
                          headers={"Authorization": f"Bearer {self.secret_key}"}) as c:
            r = c.get(f"{API}{path}")
        r.raise_for_status()
        return r

    def _key(self, kid: str | None):
        if self._jwks is None or time.time() - self._jwks[0] > JWKS_TTL:
            self._jwks = (time.time(), self._get("/jwks").json())
        keys = self._jwks[1].get("keys", [])
        for k in keys:
            if k.get("kid") == kid:
                return jwt.PyJWK(k, algorithm="RS256").key
        raise AuthError("token signed by an unknown key")

    def verify(self, token: str) -> dict:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise AuthError("unexpected token algorithm")
            claims = jwt.decode(token, self._key(header.get("kid")), algorithms=["RS256"],
                                issuer=self.issuer, leeway=LEEWAY,
                                options={"require": ["exp", "nbf", "sub", "iss"]})
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid session token: {type(exc).__name__}") from None
        if claims.get("azp") and claims["azp"] not in self.authorized_parties:
            raise AuthError("token obtained from an unknown origin")
        return claims

    def groups(self, user_id: str) -> tuple[str, ...]:
        hit = self._groups.get(user_id)
        if hit and time.time() - hit[0] < GROUPS_TTL:
            return hit[1]
        meta = self._get(f"/users/{user_id}").json().get("public_metadata") or {}
        raw = meta.get("groups") or []
        groups = tuple(g for g in raw if isinstance(g, str) and GROUP_RE.fullmatch(g))
        self._groups[user_id] = (time.time(), groups)
        return groups

    def identify(self, authorization: str) -> Identity | None:
        """`Authorization` header -> Identity; empty -> None (public); bad -> AuthError."""
        if not authorization:
            return None
        token = authorization.removeprefix("Bearer ").strip()
        claims = self.verify(token)
        return Identity(claims["sub"], self.groups(claims["sub"]))
