"""Verified fal webhook receiver - the piece every product backend needs.

fal signs every webhook delivery with Ed25519. Verifying means nobody can
forge "your render is done" calls into your backend. The scheme (verified
against fal's production signer):

    message   = f"{request_id}\\n{user_id}\\n{timestamp}\\n{sha256(raw_body).hexdigest()}"
    signature = X-Fal-Webhook-Signature header, hex-encoded Ed25519 over message
    keys      = https://rest.fal.ai/.well-known/jwks.json (cache <= 24h)

Hash the RAW bytes you received - never re-parse and re-serialize the JSON.
Deliveries repeat until fal sees a 2xx, so handle duplicates (this receiver
tracks request_ids and marks replays).

Run:
    pip install httpx==0.28.1 cryptography==50.0.0
    python webhook_receiver.py --port 8321
    # expose it publicly, e.g.:  cloudflared tunnel --url http://localhost:8321

Then submit with:  python client.py --app <owner>/<app> --webhook https://<tunnel-host>/fal-webhook
"""
import argparse
import base64
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

JWKS_URL = "https://rest.fal.ai/.well-known/jwks.json"
JWKS_TTL_S = 24 * 3600  # keys rotate; do not cache longer
TIMESTAMP_LEEWAY_S = 300

_jwks_cache: dict = {"keys": None, "fetched_at": 0.0}


def public_keys() -> list[Ed25519PublicKey]:
    now = time.time()
    if _jwks_cache["keys"] is None or now - _jwks_cache["fetched_at"] > JWKS_TTL_S:
        response = httpx.get(JWKS_URL, timeout=10)
        response.raise_for_status()
        jwks = response.json()
        keys = []
        for entry in jwks.get("keys", []):
            x = entry["x"]
            raw = base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))
            keys.append(Ed25519PublicKey.from_public_bytes(raw))
        if not keys:
            raise RuntimeError("JWKS returned no keys - refusing to cache an empty key set")
        _jwks_cache.update(keys=keys, fetched_at=now)
    return _jwks_cache["keys"]


def verify(headers: dict[str, str], body: bytes, keys: list[Ed25519PublicKey]) -> tuple[bool, str]:
    """(ok, reason) for one delivery. `headers` must be lowercase-keyed."""
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


class Receiver(BaseHTTPRequestHandler):
    seen: set[str] = set()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        headers = {k.lower(): v for k, v in self.headers.items()}
        ok, reason = verify(headers, body, public_keys())
        if not ok:
            print(f"REJECTED: {reason}")
            self._respond(401, {"ok": False, "reason": reason})
            return
        event = json.loads(body)
        request_id = event.get("request_id", "?")
        if request_id in self.seen:
            print(f"VERIFIED status={event.get('status')} request_id={request_id} "
                  "(duplicate delivery - fal retries until it sees a 2xx; already processed, acking again)")
            self._respond(200, {"ok": True, "duplicate": True})
            return
        print(f"VERIFIED status={event.get('status')} request_id={request_id}")
        if event.get("status") == "OK":
            for rendition in (event.get("payload") or {}).get("renditions", []):
                print(f"  {rendition['label']}: {rendition['video']['url']}")
        else:
            print(f"  error: {event.get('error')}")
            print(f"  payload: {json.dumps(event.get('payload'))[:300]}")
        # mark only after successful processing - a crash must let fal's retry re-process
        self.seen.add(request_id)
        self._respond(200, {"ok": True})

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()
    print(f"listening on http://localhost:{args.port} - POST path can be anything")
    print(f"expose it, e.g.: cloudflared tunnel --url http://localhost:{args.port}")
    ThreadingHTTPServer(("", args.port), Receiver).serve_forever()


if __name__ == "__main__":
    main()
