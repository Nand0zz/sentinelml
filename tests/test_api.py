import json

import pytest
from fastapi.testclient import TestClient

from sentinelml import config as cfg
from sentinelml.api import SAMPLE_FLOW, create_app
from tests.conftest import MODEL, flow_dict


@pytest.fixture(scope="module")
def client(promoted_registry, tmp_path_factory):
    log_path = tmp_path_factory.mktemp("logs") / "predictions.jsonl"
    app = create_app(tracking_uri=promoted_registry["uri"], model_name=MODEL, log_path=str(log_path))
    with TestClient(app) as c:
        c.log_path, c.reg = log_path, promoted_registry
        yield c


def test_health_and_model_info(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["model_name"] == MODEL and info["alias"] == "champion"
    assert info["explanation_method"] == "shap_tree"
    assert 0 < info["threshold"] < 1 and len(info["input_features"]) == 40


def test_predict_returns_score_flag_and_top_features(client):
    flow = flow_dict(client.reg["test"], 0)
    r = client.post("/predict", json=flow)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["is_suspicious"] == (body["score"] >= body["threshold"])
    assert 1 <= len(body["top_features"]) <= cfg.TOP_K_EXPLANATIONS
    assert "not proof" in body["note"]
    mags = [abs(f["contribution"]) for f in body["top_features"]]
    assert mags == sorted(mags, reverse=True)


def test_explain_false_skips_explanations(client):
    body = client.post("/predict?explain=false", json=flow_dict(client.reg["test"], 1)).json()
    assert body["top_features"] == []


def test_categorical_case_and_whitespace_do_not_change_score(client):
    flow = flow_dict(client.reg["test"], 2)
    a = client.post("/predict", json=flow).json()["score"]
    b = client.post("/predict", json={**flow, "proto": f"  {flow['proto'].upper()} "}).json()["score"]
    assert a == pytest.approx(b)


def test_unseen_category_and_missing_numeric_are_handled(client):
    flow = {**flow_dict(client.reg["test"], 3), "proto": "brand-new-proto", "dur": None}
    assert client.post("/predict", json=flow).status_code == 200


@pytest.mark.parametrize("patch", [{"sbytes": -5}, {"unknown_field": 1}, {"dur": "abc"}])
def test_invalid_payloads_are_rejected(client, patch):
    assert client.post("/predict", json={**flow_dict(client.reg["test"], 0), **patch}).status_code == 422


def test_missing_required_categorical_is_rejected(client):
    flow = flow_dict(client.reg["test"], 0)
    flow.pop("proto")
    assert client.post("/predict", json=flow).status_code == 422


def test_batch_endpoint_and_limit(client):
    flows = [flow_dict(client.reg["test"], i) for i in range(5)]
    body = client.post("/predict/batch", json={"flows": flows}).json()
    assert len(body["predictions"]) == 5
    assert client.post("/predict/batch", json={"flows": []}).status_code == 422
    too_many = {"flows": [SAMPLE_FLOW] * (cfg.MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_every_prediction_is_logged_with_features(client):
    n_before = len(client.log_path.read_text().splitlines()) if client.log_path.exists() else 0
    client.post("/predict", json=flow_dict(client.reg["test"], 4))
    lines = client.log_path.read_text().splitlines()
    assert len(lines) == n_before + 1
    rec = json.loads(lines[-1])
    assert {"ts", "model_version", "score", "is_suspicious", "request_latency_ms", "features"} <= rec.keys()
    assert set(rec["features"]) == set(cfg.FEATURES)


def test_sample_flow_from_docs_is_valid(client):
    assert client.post("/predict", json=SAMPLE_FLOW).status_code == 200


def test_status_counts_requests_and_predictions(client):
    before = client.get("/status").json()
    client.post("/predict", json=SAMPLE_FLOW)
    after = client.get("/status").json()
    assert after["predictions"] == before["predictions"] + 1
    assert after["requests"] >= before["requests"] + 2
    assert after["model_name"] == MODEL and after["uptime_seconds"] >= 0


def test_recent_predictions_returns_logged_flows(client):
    client.post("/predict", json=SAMPLE_FLOW)
    body = client.get("/predictions/recent?limit=3").json()
    assert 1 <= body["count"] <= 3 and len(body["records"]) == body["count"]
    assert set(body["records"][-1]["features"]) == set(cfg.FEATURES)
    assert client.get("/predictions/recent?limit=0").status_code == 422
