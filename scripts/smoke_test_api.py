"""Black-box check of a running API (used by CI against the Docker container).

    python scripts/smoke_test_api.py http://localhost:8000
Standard library only, so it can run anywhere. Exit code 0 = healthy.
"""
import json
import sys
import urllib.error
import urllib.request

from sentinelml.api import SAMPLE_FLOW


def call(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def main(base: str) -> int:
    base = base.rstrip("/")
    status, body = call(f"{base}/health")
    assert status == 200 and body["status"] == "ok", f"/health failed: {status} {body}"

    status, body = call(f"{base}/predict", SAMPLE_FLOW)
    assert status == 200, f"/predict failed: {status}"
    assert 0 <= body["score"] <= 1 and isinstance(body["is_suspicious"], bool)
    assert body["top_features"], "expected explanations"

    status, _ = call(f"{base}/predict", {**SAMPLE_FLOW, "sbytes": -1})
    assert status == 422, f"invalid input should be rejected with 422, got {status}"

    print(f"OK: model v{body['model_version']} score={body['score']:.4f} "
          f"suspicious={body['is_suspicious']} latency={body['latency_ms']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
