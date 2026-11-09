"""Clerk session tokens: verified locally, groups read from Clerk. Offline: an RSA key made
here signs real RS256 tokens; Clerk's JWKS and user endpoints are a MockTransport."""

import base64
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from finrag import api
from finrag.auth import AuthError, ClerkAuth, frontend_api

HOST = "finrag-test.clerk.accounts.dev"
PK = "pk_test_" + base64.b64encode(f"{HOST}$".encode()).decode().rstrip("=")
ISS = f"https://{HOST}"
SITE = "https://finrag-ten.vercel.app"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWK = {**jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key(), as_dict=True),
       "kid": "k1", "alg": "RS256", "use": "sig"}
USERS = {"user_risk": {"groups": ["risques"]}, "user_branch": {"groups": ["agence"]},
         "user_odd": {"groups": ["risques", "Bad Group!", 42]}}


def token(sub="user_risk", **over) -> str:
    now = int(time.time())
    claims = {"sub": sub, "iss": ISS, "azp": SITE, "iat": now, "nbf": now - 1,
              "exp": now + 60, **over}
    return jwt.encode(claims, KEY, algorithm="RS256", headers={"kid": over.pop("kid", "k1")})


def clerk(calls: list):
    def handler(request: httpx.Request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer sk_test_x"  # secret key used
        if request.url.path == "/v1/jwks":
            return httpx.Response(200, json={"keys": [JWK]})
        uid = request.url.path.rsplit("/", 1)[-1]
        if uid in USERS:
            return httpx.Response(200, json={"id": uid, "public_metadata": USERS[uid]})
        return httpx.Response(404, json={})
    return ClerkAuth(PK, "sk_test_x", (SITE,), transport=httpx.MockTransport(handler))


def test_issuer_comes_from_the_publishable_key():
    assert frontend_api(PK) == ISS
    with pytest.raises(ValueError):
        frontend_api("pk_test_bm90LWEta2V5")  # base64 of "not-a-key": no trailing $


def test_a_valid_token_gives_the_user_and_the_groups_set_in_clerk():
    calls = []
    who = clerk(calls).identify(f"Bearer {token()}")
    assert (who.user_id, who.groups) == ("user_risk", ("risques",))


def test_groups_are_sanitised_and_cached():
    calls = []
    auth = clerk(calls)
    assert auth.groups("user_odd") == ("risques",)  # malformed entries dropped
    n = len(calls)
    auth.groups("user_odd")
    assert len(calls) == n  # second read from the cache


@pytest.mark.parametrize("bad,why", [
    (lambda: token(exp=int(time.time()) - 120), "expired"),
    (lambda: token(nbf=int(time.time()) + 120), "not yet valid"),
    (lambda: token(iss="https://evil.clerk.accounts.dev"), "other instance"),
    (lambda: token(azp="https://evil.example"), "other origin"),
    (lambda: jwt.encode({"sub": "user_risk", "iss": ISS, "exp": int(time.time()) + 60,
                         "nbf": int(time.time())},
                        rsa.generate_private_key(public_exponent=65537, key_size=2048),
                        algorithm="RS256", headers={"kid": "k1"}), "forged signature"),
    (lambda: jwt.encode({"sub": "user_risk", "iss": ISS, "exp": int(time.time()) + 60,
                         "nbf": int(time.time())}, "s" * 40, algorithm="HS256",
                        headers={"kid": "k1"}), "wrong algorithm"),
    (lambda: "not.a.jwt", "garbage"),
])
def test_a_token_that_fails_any_check_is_refused(bad, why):
    with pytest.raises(AuthError):
        clerk([]).identify(f"Bearer {bad()}")


def test_no_token_is_no_identity_not_an_error():
    assert clerk([]).identify("") is None


# ---------------------------------------------------------------- through the API
@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_auth", lambda: clerk([]))
    return TestClient(api.app)


def test_me_says_who_is_asking(client):
    assert client.get("/me").json() == {"signed_in": False, "groups": []}
    r = client.get("/me", headers={"authorization": f"Bearer {token('user_branch')}"})
    assert r.json() == {"signed_in": True, "user_id": "user_branch", "groups": ["agence"]}


def test_a_bad_token_is_401_never_public_by_fallback(client):
    r = client.get("/me", headers={"authorization": f"Bearer {token(azp='https://evil.example')}"})
    assert r.status_code == 401


def test_a_token_without_clerk_configured_is_refused(monkeypatch):
    monkeypatch.setattr(api, "_auth", lambda: None)
    r = TestClient(api.app).get("/me", headers={"authorization": "Bearer x"})
    assert r.status_code == 401 and "not enabled" in r.json()["detail"]
