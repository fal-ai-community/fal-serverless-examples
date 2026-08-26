"""Signature-verifying fal webhook receiver - reference code for a product backend."""

import argparse
import base64
import hashlib
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

JWKS_URL = "https://rest.fal.ai/.well-known/jwks.json"
JWKS_TTL_S = 24 * 3600  # keys rotate; do not cache longer
TIMESTAMP_LEEWAY_S = 300
MAX_BODY_BYTES = 2 * 1024 * 1024  # deliveries are small JSON and the port is public
DEDUPE_CAPACITY = 10_000  # bounded; a long-lived backend cannot keep every id forever

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}


def public_keys() -> list[Ed25519PublicKey]:
    now = time.time()
    if _jwks_cache["keys"] is None or now - _jwks_cache["fetched_at"] > JWKS_TTL_S:
        response = httpx.get(JWKS_URL, timeout=10)
        response.raise_for_status()
        jwks = response.json()
        keys: list[Ed25519PublicKey] = []
        for entry in jwks.get("keys", []):
            # the JWKS may carry keys of other types; one unusable entry must not
            # take the whole key set down
            if not isinstance(entry, dict):
                continue
            if entry.get("kty") != "OKP" or entry.get("crv") != "Ed25519":
                continue
            x = entry.get("x")
            if not isinstance(x, str):
                continue
            try:
                raw = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
                keys.append(Ed25519PublicKey.from_public_bytes(raw))
            except ValueError:
                continue
        if not keys:
            raise RuntimeError(
                "JWKS returned no usable Ed25519 keys - refusing to cache an empty "
                "key set"
            )
        _jwks_cache.update(keys=keys, fetched_at=now)
    return _jwks_cache["keys"]


def verify(
    headers: dict[str, str], body: bytes, keys: list[Ed25519PublicKey]
) -> tuple[bool, str]:
    """(ok, reason); `headers` lowercase-keyed, `body` raw bytes, not re-serialized."""
    try:
        request_id = headers["x-fal-webhook-request-id"]
        user_id = headers["x-fal-webhook-user-id"]
        timestamp = headers["x-fal-webhook-timestamp"]
        signature = bytes.fromhex(headers["x-fal-webhook-signature"])
    except KeyError as missing:
        return False, f"missing header {missing}"
    except ValueError:
        return False, "signature is not valid hex"
    try:
        skew = abs(time.time() - int(timestamp))
    except (ValueError, OverflowError):
        return False, "timestamp is not a plausible unix time"
    if skew > TIMESTAMP_LEEWAY_S:
        return False, f"timestamp outside +/-{TIMESTAMP_LEEWAY_S}s (skew {skew:.0f}s)"
    message = "\n".join(
        [request_id, user_id, timestamp, hashlib.sha256(body).hexdigest()]
    ).encode()
    for key in keys:
        try:
            key.verify(signature, message)
            return True, "signature valid"
        except InvalidSignature:
            continue
    return False, "signature does not match any current JWKS key"


class _Dedupe:
    """Bounded request-id store; every handler thread shares this one instance."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        self._done: OrderedDict[str, None] = OrderedDict()
        self._in_flight: set[str] = set()

    def claim(self, request_id: str) -> Literal["done", "in_flight", "claimed"]:
        # check and claim under one lock, or two concurrent retries both pass
        with self._lock:
            if request_id in self._done:
                return "done"
            if request_id in self._in_flight:
                return "in_flight"
            self._in_flight.add(request_id)
            return "claimed"

    def release(self, request_id: str, *, processed: bool) -> None:
        with self._lock:
            self._in_flight.discard(request_id)
            if not processed:
                return
            self._done[request_id] = None
            while len(self._done) > self._capacity:
                self._done.popitem(last=False)


_dedupe = _Dedupe(DEDUPE_CAPACITY)


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self._read_body()
        if body is None:
            return
        headers = {k.lower(): v for k, v in self.headers.items()}
        ok, reason = verify(headers, body, public_keys())
        if not ok:
            self._reject(401, reason)
            return
        try:
            event = json.loads(body)
        except ValueError:
            self._reject(400, "signed body is not valid JSON")
            return
        if not isinstance(event, dict):
            self._reject(400, "signed body is not a JSON object")
            return
        # fall back to the signed header id; "?" for every id-less body would collide
        request_id = str(event.get("request_id") or headers["x-fal-webhook-request-id"])
        claim = _dedupe.claim(request_id)
        if claim == "done":
            print(
                f"VERIFIED status={event.get('status')} request_id={request_id} "
                "(duplicate delivery - fal retries until it sees a 2xx; "
                "already processed, acking again)"
            )
            self._respond(200, {"ok": True, "duplicate": True})
            return
        if claim == "in_flight":
            # a 2xx here would let fal stop retrying while the other thread is
            # still able to crash, losing the event
            print(
                f"VERIFIED status={event.get('status')} request_id={request_id} "
                "(concurrent duplicate delivery - asking fal to retry)"
            )
            self._respond(503, {"ok": False, "duplicate": True})
            return
        processed = False
        try:
            print(f"VERIFIED status={event.get('status')} request_id={request_id}")
            if event.get("status") == "OK":
                for rendition in (event.get("payload") or {}).get("renditions", []):
                    print(f"  {rendition['label']}: {rendition['video']['url']}")
            else:
                print(f"  error: {event.get('error')}")
                print(f"  payload: {json.dumps(event.get('payload'))[:300]}")
            processed = True
        finally:
            # mark only after successful processing - a crash must let fal's
            # retry re-process
            _dedupe.release(request_id, processed=processed)
        self._respond(200, {"ok": True})

    def _read_body(self) -> bytes | None:
        """Raw body under MAX_BODY_BYTES, or None once the request is rejected."""
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._reject(400, "missing or non-numeric Content-Length", close=True)
            return None
        if not 0 <= length <= MAX_BODY_BYTES:
            self._reject(
                400, f"Content-Length {length} outside 0..{MAX_BODY_BYTES}", close=True
            )
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._reject(400, "body shorter than Content-Length", close=True)
            return None
        return body

    def _reject(self, code: int, reason: str, *, close: bool = False) -> None:
        if close:
            self.close_connection = True  # an unread body poisons the next request
        print(f"REJECTED: {reason}")
        self._respond(code, {"ok": False, "reason": reason})

    def _respond(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass  # our own prints above are the log


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()
    print(f"listening on http://localhost:{args.port} - POST path can be anything")
    print(f"expose it, e.g.: cloudflared tunnel --url http://localhost:{args.port}")
    ThreadingHTTPServer(("", args.port), Receiver).serve_forever()


if __name__ == "__main__":
    main()
