"""FastAPI inference service.

    uvicorn sentinelml.api:app --port 8000          # loads the `champion` from the MLflow registry
    MODEL_DIR=exported_model uvicorn sentinelml.api:app   # or from an exported folder (Docker)

Predictions are indicators for analyst review, never proof of an attack.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from pydantic import BaseModel, ConfigDict, Field, create_model

from sentinelml import config as cfg
from sentinelml.data import normalize_features
from sentinelml.explain import Explainer, build_explainer, contributions
from sentinelml.registry import environment_versions, load_exported, load_from_registry

log = logging.getLogger("sentinelml.api")

NOTE = "Indicator requiring analyst review, not proof of an attack."

# Illustrative flow, shown in the /docs "Try it out" box and used by the smoke test.
SAMPLE_FLOW = {
    "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT", "spkts": 2, "dpkts": 0,
    "sbytes": 496, "dbytes": 0, "rate": 90909.0902, "sttl": 254, "dttl": 0, "sload": 180363632.0,
    "dload": 0, "sloss": 0, "dloss": 0, "sinpkt": 0.011, "dinpkt": 0, "sjit": 0, "djit": 0,
    "swin": 0, "dwin": 0, "tcprtt": 0, "synack": 0, "ackdat": 0, "smean": 248, "dmean": 0,
    "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 2, "ct_state_ttl": 2, "ct_dst_ltm": 1,
    "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 2, "is_ftp_login": 0,
    "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1, "ct_srv_dst": 2, "is_sm_ips_ports": 0,
}


def _build_flow_model():
    """Request schema generated from config, so it can never drift from the model's features."""
    fields = {c: (str, Field(..., min_length=1, max_length=32)) for c in cfg.CATEGORICAL}
    fields.update(
        {c: (Optional[float], Field(None, ge=0, allow_inf_nan=False)) for c in cfg.NUMERIC}
    )
    return create_model(
        "FlowFeatures",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [SAMPLE_FLOW]}),
        **fields,
    )


FlowFeatures = _build_flow_model()


class BatchRequest(BaseModel):
    flows: list[FlowFeatures] = Field(..., min_length=1, max_length=cfg.MAX_BATCH)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float
    direction: str
    value: Optional[float] = None


class Item(BaseModel):
    score: float
    is_suspicious: bool
    top_features: list[FeatureContribution]


class Meta(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    explanation_method: str
    latency_ms: float
    note: str = NOTE


class PredictResponse(Meta, Item):
    pass


class BatchResponse(Meta):
    predictions: list[Item]


@dataclass
class Bundle:
    pipe: object
    explainer: Explainer
    meta: dict

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])


class PredictionLogger:
    """Append-only JSONL log of every scored flow; Day 3 drift monitoring reads it."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, records: list[dict]) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tail(self, n: int) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock, open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in deque(f, maxlen=n) if line.strip()]


def _clean_value(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def load_bundle(model_dir: str | None, tracking_uri: str, model_name: str, alias: str) -> Bundle:
    if model_dir:
        pipe, meta = load_exported(model_dir)
        trained_env = meta.get("environment", {})
        minor = lambda v: ".".join(str(v).split(".")[:2])  # patch-level differences are harmless
        for pkg, ver in environment_versions().items():
            if trained_env.get(pkg) is not None and minor(trained_env[pkg]) != minor(ver):
                log.warning("Environment mismatch for %s: model exported with %s, running %s. "
                            "Rebuild the image if predictions look wrong.", pkg, trained_env[pkg], ver)
    else:
        pipe, meta = load_from_registry(model_name, alias, tracking_uri)
    return Bundle(pipe, build_explainer(pipe), meta)


def run_inference(bundle: Bundle, flows: list[dict], explain: bool = True):
    """-> (items, latency_ms, normalised feature frame)."""
    df = normalize_features(pd.DataFrame(flows).reindex(columns=cfg.FEATURES))
    t0 = time.perf_counter()
    scores = bundle.pipe.predict_proba(df)[:, 1]
    contrib = contributions(bundle.pipe, bundle.explainer, df) if explain else None
    latency_ms = (time.perf_counter() - t0) * 1000

    eng = bundle.pipe.named_steps["engineer"].transform(df) if contrib is not None else None
    names = bundle.explainer.names
    items = []
    for i, score in enumerate(scores):
        top = []
        if contrib is not None:
            for j in np.argsort(-np.abs(contrib[i]))[: cfg.TOP_K_EXPLANATIONS]:
                name, c = names[j], float(contrib[i, j])
                raw = eng.iloc[i][name] if name in eng.columns else None
                top.append({
                    "feature": name,
                    "contribution": round(c, 6),
                    "direction": "toward_suspicious" if c > 0 else "toward_benign",
                    "value": None if raw is None or pd.isna(raw) else float(raw),
                })
        items.append({"score": float(score), "is_suspicious": bool(score >= bundle.threshold),
                      "top_features": top})
    return items, latency_ms, df


def create_app(model_dir: str | None = None, tracking_uri: str | None = None,
               model_name: str | None = None, alias: str | None = None,
               log_path: str | None = None) -> FastAPI:
    # Explicit arguments win, then environment variables, then defaults.
    model_dir = model_dir or os.getenv("MODEL_DIR") or None
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", cfg.DEFAULT_TRACKING_URI)
    model_name = model_name or os.getenv("MODEL_NAME", cfg.DEFAULT_MODEL_NAME)
    alias = alias or os.getenv("MODEL_ALIAS", cfg.ALIAS_CHAMPION)
    log_path = log_path or os.getenv("PREDICTION_LOG", str(cfg.PREDICTION_LOG))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_bundle(model_dir, tracking_uri, model_name, alias)
        app.state.logger = PredictionLogger(log_path)
        log.info("Loaded %s v%s (threshold %.4f)", model_name,
                 app.state.bundle.meta["version"], app.state.bundle.threshold)
        yield

    app = FastAPI(
        title="SentinelML API",
        description="Scores network flows as potentially malicious. " + NOTE,
        version="0.3.0",
        lifespan=lifespan,
    )
    started = time.time()
    counters = {"requests": 0, "errors": 0, "predictions": 0}  # per worker process

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        counters["requests"] += 1
        response = await call_next(request)
        if response.status_code >= 400:
            counters["errors"] += 1
        return response

    def _meta(b: Bundle, latency_ms: float) -> dict:
        return {"model_name": b.meta["model_name"], "model_version": str(b.meta["version"]),
                "threshold": b.threshold, "explanation_method": b.explainer.kind,
                "latency_ms": round(latency_ms, 3)}

    def _log(request: Request, b: Bundle, items, df, latency_ms):
        now = datetime.now(timezone.utc).isoformat()
        rows = df.astype(object).to_dict(orient="records")
        request.app.state.logger.write([
            {"ts": now, "model_version": str(b.meta["version"]), "score": it["score"],
             "is_suspicious": it["is_suspicious"], "request_latency_ms": round(latency_ms, 3),
             "batch_size": len(items), "features": {k: _clean_value(v) for k, v in row.items()}}
            for it, row in zip(items, rows)
        ])
        counters["predictions"] += len(items)

    @app.get("/health")
    def health(request: Request):
        b: Bundle = request.app.state.bundle
        return {"status": "ok", "model_loaded": True, "model_version": str(b.meta["version"])}

    @app.get("/model-info")
    def model_info(request: Request):
        b: Bundle = request.app.state.bundle
        return {
            "model_name": b.meta["model_name"], "model_version": str(b.meta["version"]),
            "alias": b.meta.get("alias"), "source": b.meta.get("source"),
            "threshold": b.threshold, "explanation_method": b.explainer.kind,
            "registry_metrics": b.meta.get("metrics", {}), "input_features": cfg.FEATURES,
            "environment": environment_versions(), "note": NOTE,
            "run_name": b.meta.get("run_name"), "run_id": b.meta.get("run_id"),
        }

    @app.get("/status")
    def status(request: Request):
        """Lightweight monitoring: uptime, request/prediction counters, which model is served."""
        b: Bundle = request.app.state.bundle
        return {"status": "ok", "uptime_seconds": round(time.time() - started, 1), **counters,
                "model_name": b.meta["model_name"], "model_version": str(b.meta["version"]),
                "run_name": b.meta.get("run_name"), "model_source": b.meta.get("source"),
                "prediction_log": str(log_path)}

    @app.get("/predictions/recent")
    def recent_predictions(request: Request, limit: int = Query(1000, ge=1, le=10_000)):
        """Last `limit` entries of the prediction log, for monitoring and drift checks."""
        records = request.app.state.logger.tail(limit)
        return {"count": len(records), "records": records}

    @app.post("/predict", response_model=PredictResponse)
    def predict(flow: FlowFeatures, request: Request, explain: bool = Query(True)):
        b: Bundle = request.app.state.bundle
        items, latency_ms, df = run_inference(b, [flow.model_dump()], explain)
        _log(request, b, items, df, latency_ms)
        return {**_meta(b, latency_ms), **items[0]}

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(body: BatchRequest, request: Request, explain: bool = Query(True)):
        b: Bundle = request.app.state.bundle
        items, latency_ms, df = run_inference(b, [f.model_dump() for f in body.flows], explain)
        _log(request, b, items, df, latency_ms)
        return {**_meta(b, latency_ms), "predictions": items}

    return app


app = create_app()
