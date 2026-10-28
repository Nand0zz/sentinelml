"""SentinelML operations dashboard.

    streamlit run dashboard/app.py                                        # from the repo root
    SENTINELML_API=http://<ec2-ip>:8000 streamlit run dashboard/app.py    # point it at the deployment

Reads pipeline artifacts from the repo (models/, reports/, data/processed/ after `dvc pull`) and talks to a
running API for live status, scoring, generated traffic and the prediction log.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentinelml import config as cfg
from sentinelml.monitoring import PSI_MODERATE, PSI_SIGNIFICANT, drift_report, load_prediction_log

st.set_page_config(page_title="SentinelML", layout="wide")

NAVY, RED, AMBER, SLATE, GRID, INK = "#1f4e79", "#b03a2e", "#c27c0e", "#8a94a3", "#eceff3", "#1c2430"
STATUS_COLORS = {"stable": SLATE, "moderate": AMBER, "significant": RED}
MODEL_LABELS = {"lightgbm": "LightGBM", "random_forest": "Random forest", "logreg": "Logistic regression"}
REPORTS, MODELS, PROCESSED = cfg.ROOT / "reports", cfg.MODELS_DIR, cfg.PROCESSED_DIR

st.markdown(
    """<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
    [data-testid="stMetricLabel"] p {font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: #5b6573;}
    [data-testid="stMetricValue"] {font-size: 1.45rem; font-weight: 600;}
    h1 {font-weight: 650; letter-spacing: -.015em; margin-bottom: 0;}
    [data-testid="stMarkdownContainer"] h3 {font-weight: 600; font-size: 1.12rem; padding: .9rem 0 .2rem 0;}
    </style>""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers

def style(fig, height: int = 300, title: str | None = None):
    fig.update_layout(
        template="plotly_white", height=height, title=title, title_font=dict(size=13, color=INK),
        margin=dict(l=8, r=8, t=44 if title else 8, b=8), font=dict(size=12, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def chart(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def call_api(base: str, path: str, payload=None, timeout: float = 20):
    """-> (json, None) or (None, error message)."""
    req = urllib.request.Request(
        base.rstrip("/") + path, headers={"Content-Type": "application/json"},
        data=None if payload is None else json.dumps(payload).encode(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # refused, timeout, DNS
        return None, str(getattr(e, "reason", e))


@st.cache_data(show_spinner=False)
def read_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(path: str):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Loading evaluation data")
def eval_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(test split, 20k-row training reference, source). Synthetic stand-in if `dvc pull` hasn't run."""
    tr, te = PROCESSED / cfg.PROCESSED_TRAIN, PROCESSED / cfg.PROCESSED_TEST
    if tr.exists() and te.exists():
        train, test, source = pd.read_parquet(tr), pd.read_parquet(te), "UNSW-NB15"
    else:
        from sentinelml.data import load_synthetic_splits
        train, test = load_synthetic_splits()
        source = "synthetic"
    return test, train.sample(min(20_000, len(train)), random_state=0), source


@st.cache_resource(show_spinner="Loading champion model")
def local_bundle(model_dir: str):
    from sentinelml.api import load_bundle
    return load_bundle(model_dir, cfg.DEFAULT_TRACKING_URI, cfg.DEFAULT_MODEL_NAME, cfg.ALIAS_CHAMPION)


def to_flow(row: pd.Series) -> dict:
    flow = {c: str(row[c]) for c in cfg.CATEGORICAL}
    flow |= {c: (None if pd.isna(row[c]) else float(row[c])) for c in cfg.NUMERIC}
    return flow


def sample_rows(test: pd.DataFrame, kind: str, n: int = 1) -> pd.DataFrame:
    pool = {"Malicious": test[test[cfg.TARGET] == 1], "Benign": test[test[cfg.TARGET] == 0]}.get(kind, test)
    return pool.sample(min(n, len(pool)), random_state=np.random.randint(1_000_000))


def contribution_chart(names, values, top: int = 10, title: str | None = None):
    df = pd.DataFrame({"feature": names, "contribution": values})
    df = df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["contribution"], y=df["feature"], orientation="h",
        marker_color=[RED if v > 0 else NAVY for v in df["contribution"]],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=SLATE, line_width=1)
    return style(fig, height=34 * len(df) + 70, title=title).update_xaxes(title="contribution (log-odds)")


def score_gauge(score: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"valueformat": ".3f", "font": {"size": 34}},
        gauge={"axis": {"range": [0, 1], "tickvals": [0, .25, .5, .75, 1], "ticks": "", "tickfont": {"size": 10, "color": SLATE}},
               "bar": {"color": RED if score >= threshold else NAVY},
               "bgcolor": "#f4f6f8", "borderwidth": 0,
               "threshold": {"line": {"color": INK, "width": 2}, "thickness": .9, "value": threshold}},
    ))
    return style(fig, height=190).update_layout(margin=dict(l=24, r=24, t=16, b=0))


def verdict(score: float, threshold: float) -> str:
    return ":red[**Suspicious**]" if score >= threshold else ":blue[**Benign**]"


# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("**SentinelML**")
    st.caption("Network intrusion detection · model operations")
    api_base = st.text_input("API endpoint", os.getenv("SENTINELML_API", "http://localhost:8000"))
    with st.expander("Local artifacts"):
        model_dir = st.text_input("Model folder", str(MODELS / cfg.ALIAS_CHAMPION))
    health, health_err = call_api(api_base, "/health", timeout=5)
    if health:
        st.markdown(f":green[●] Service online · model v{health['model_version']}")
    else:
        st.markdown(f":red[●] Service unreachable  \n<small>{health_err}</small>", unsafe_allow_html=True)
    if st.button("Reload artifacts", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption("Scores are indicators for analyst review, not proof of an attack.")

manifest = read_json(str(MODELS / cfg.MODEL_MANIFEST)) or {}
meta = read_json(str(Path(model_dir) / cfg.MODEL_META)) or {}
test, reference, data_source = eval_data()
threshold = float(meta.get("threshold", manifest.get("threshold", 0.5)))

st.title("SentinelML")
st.caption("UNSW-NB15 intrusion detection · DVC pipeline · MLflow tracking · gated releases · EC2 serving")
tab_overview, tab_live, tab_whatif, tab_monitor = st.tabs(["Overview", "Live service", "What-if analysis", "Monitoring"])


# ----------------------------------------------------------------------------- overview

with tab_overview:
    if not manifest:
        st.warning("models/model_manifest.json not found. Run `dvc pull` in the repo first.")
    m = manifest.get("metrics", {})
    rm = meta.get("run_metrics", {})
    cols = st.columns(6)
    cols[0].metric("Champion", f"v{manifest.get('champion_version', '-')}",
                   help=MODEL_LABELS.get(manifest.get("model_type"), manifest.get("model_type")))
    cols[1].metric("Model", MODEL_LABELS.get(manifest.get("model_type"), "-"))
    cols[2].metric("Test PR-AUC", f"{m.get('test_pr_auc', float('nan')):.4f}")
    cols[3].metric("Test recall", f"{m.get('test_recall', float('nan')):.4f}")
    cols[4].metric("Test FPR", f"{m.get('test_fpr', float('nan')):.4f}", help="Budget 5% + 1 pt slack")
    cols[5].metric("Latency (1 row)", f"{rm.get('single_row_latency_ms', float('nan')):.1f} ms")
    if manifest:
        st.caption(f"Run `{manifest.get('run_name')}` · promoted {str(manifest.get('promoted_at', ''))[:19].replace('T', ' ')} UTC"
                   f" · alert threshold {threshold:.4f} · DVC `{str(manifest.get('dvc_md5'))[:12]}` · artifact `{manifest.get('artifact_path')}`")

    st.markdown("### Release pipeline")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.28;
      node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#c9d1db", fontname="Helvetica", fontsize=15, fontcolor="#1c2430", margin="0.16,0.08"];
      edge [color="#8a94a3", arrowsize=0.6];
      raw [label="Raw data\\nDVC · S3"]; pre [label="Preprocess"]; train [label="Train\\nMLflow"];
      gate [label="Quality gate", fillcolor="#e6edf5", color="#1f4e79", fontcolor="#1f4e79"];
      champ [label="Champion\\nDVC · S3"]; ci [label="CI build\\nGHCR image"];
      ec2 [label="EC2 · FastAPI", fillcolor="#e6edf5", color="#1f4e79", fontcolor="#1f4e79"];
      raw -> pre -> train -> gate -> champ -> ci -> ec2;
      champ -> ec2 [style=dashed, constraint=false, label="  model fetched at start-up", fontsize=12, fontcolor="#5b6573", fontname="Helvetica"];
    }""", width="stretch")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("### Quality gate")
        gate = read_json(str(REPORTS / "gate_report.json"))
        if gate:
            res = gate["result"]
            checks = pd.DataFrame(res["checks"]).rename(columns={"name": "Check", "passed": "Result", "detail": "Detail"})
            checks["Result"] = checks["Result"].map({True: "PASS", False: "FAIL"})
            st.dataframe(
                checks.style.map(lambda v: f"color: {NAVY if v == 'PASS' else RED}; font-weight: 600", subset=["Result"]),
                hide_index=True, width="stretch",
            )
            st.caption(f"{res['message']} · {gate['timestamp'][:19].replace('T', ' ')} UTC")
        else:
            st.info("No gate report yet. Run `python -m sentinelml.gate promote`.")
    with right:
        st.markdown("### Confusion matrix (test set)")
        cm = read_json(str(REPORTS / "confusion_matrix.json"))
        if cm:
            t = cm["test"]
            z = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
            pct = z / z.sum(axis=1, keepdims=True)
            text = [[f"{z[i, j]:,}<br>{pct[i, j]:.1%}" for j in range(2)] for i in range(2)]
            fig = go.Figure(go.Heatmap(
                z=pct, x=["Predicted benign", "Predicted malicious"], y=["Actual benign", "Actual malicious"],
                text=text, texttemplate="%{text}", colorscale=[[0, "#f4f6f8"], [1, NAVY]], showscale=False,
                hovertemplate="%{y} / %{x}<extra></extra>",
            ))
            fig.update_yaxes(autorange="reversed")
            chart(style(fig, height=280))
            st.caption(f"Threshold {cm['threshold']:.4f}, chosen on validation (max recall with FPR ≤ 5%).")

    st.markdown("### Model comparison")
    comp = read_csv(str(REPORTS / "model_comparison.csv"))
    if comp is not None:
        comp = comp.assign(Model=comp["model"].map(MODEL_LABELS).fillna(comp["model"]))
        c1, c2 = st.columns(2, gap="large")
        with c1:
            long = comp.melt(id_vars="Model", value_vars=["test_pr_auc", "test_recall", "test_precision"],
                             var_name="metric", value_name="value")
            long["metric"] = long["metric"].map({"test_pr_auc": "PR-AUC", "test_recall": "Recall", "test_precision": "Precision"})
            fig = px.bar(long, x="Model", y="value", color="metric", barmode="group",
                         color_discrete_sequence=[NAVY, "#5b8db8", "#b7c9db"])
            chart(style(fig, height=290, title="Test-set quality").update_yaxes(range=[0.75, 1.0], title=None)
                  .update_xaxes(title=None))
        with c2:
            fig = go.Figure(go.Bar(x=comp["Model"], y=comp["latency_ms"],
                                   marker_color=[RED if v > 100 else NAVY for v in comp["latency_ms"]],
                                   hovertemplate="%{x}: %{y:.1f} ms<extra></extra>"))
            fig.add_hline(y=100, line_dash="dot", line_color=INK, annotation_text="gate: 100 ms",
                          annotation_position="top left", annotation_font_size=11)
            chart(style(fig, height=290, title="Single-row latency (ms)"))
        table = comp[["Model", "val_pr_auc", "test_pr_auc", "test_precision", "test_recall", "test_fpr", "latency_ms"]]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "val_pr_auc": st.column_config.NumberColumn("Val PR-AUC", format="%.4f"),
            "test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.4f"),
            "test_precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "test_recall": st.column_config.NumberColumn("Recall", format="%.4f"),
            "test_fpr": st.column_config.NumberColumn("FPR", format="%.4f"),
            "latency_ms": st.column_config.NumberColumn("Latency (ms)", format="%.1f"),
        })
        st.caption("Selection by validation PR-AUC; the official test set is evaluated once at the validation threshold.")

    imp = read_csv(str(REPORTS / f"feature_importance_{manifest.get('model_type', 'lightgbm')}.csv"))
    if imp is not None:
        st.markdown("### Feature importance (champion)")
        top = imp.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color=NAVY,
                               hovertemplate="%{y}: %{x}<extra></extra>"))
        chart(style(fig, height=420).update_xaxes(title="split importance"))


# ----------------------------------------------------------------------------- live service

with tab_live:
    status, status_err = call_api(api_base, "/status", timeout=5)
    if status is None:
        st.error(f"Cannot reach {api_base} ({status_err}). Set the endpoint in the sidebar.")
    else:
        cols = st.columns(6)
        cols[0].metric("Status", status["status"].upper())
        cols[1].metric("Model", f"v{status['model_version']}", help=status.get("run_name"))
        cols[2].metric("Requests", f"{status['requests']:,}")
        cols[3].metric("Flows scored", f"{status['predictions']:,}")
        cols[4].metric("Errors (4xx/5xx)", f"{status['errors']:,}")
        up = status["uptime_seconds"]
        cols[5].metric("Uptime", f"{int(up // 3600)}h {int(up % 3600 // 60)}m" if up >= 3600 else f"{int(up // 60)}m {int(up % 60)}s")
        st.caption(f"{api_base} · run `{status.get('run_name')}` · model source: {status.get('model_source')}")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### Score a flow")
            kind = st.segmented_control("Sample from the test set", ["Malicious", "Benign", "Any"], default="Malicious",
                                        key="live_kind")
            if st.button("Score random flow", type="primary"):
                row = sample_rows(test, kind or "Any").iloc[0]
                resp, err = call_api(api_base, "/predict", to_flow(row))
                st.session_state["live"] = (to_flow(row), int(row[cfg.TARGET]), resp, err)
            if "live" in st.session_state:
                flow, label, resp, err = st.session_state["live"]
                if err:
                    st.error(f"/predict failed: {err}")
                else:
                    g, info = st.columns([1.1, 1])
                    with g:
                        chart(score_gauge(resp["score"], resp["threshold"]))
                    with info:
                        st.markdown(f"Verdict  \n{verdict(resp['score'], resp['threshold'])}")
                        st.markdown(f"Ground truth  \n**{'Malicious' if label else 'Benign'}**")
                        st.caption(f"Threshold {resp['threshold']:.4f} · {resp['latency_ms']:.1f} ms · "
                                   f"{resp['explanation_method'].replace('_', ' ')}")
                    tf = resp["top_features"]
                    chart(contribution_chart([f["feature"] for f in tf], [f["contribution"] for f in tf],
                                             title="Top contributing features"))
                    with st.expander("Request payload"):
                        st.json(flow)
        with right:
            st.markdown("### Generate traffic")
            n = st.select_slider("Flows per batch", [25, 50, 100, 200, 300, 500], value=200)
            mix = st.segmented_control("Mix", ["Test-set mix", "Malicious", "Benign"], default="Test-set mix", key="mix")
            if st.button("Send batch"):
                rows = sample_rows(test, "Any" if mix in (None, "Test-set mix") else mix, n)
                resp, err = call_api(api_base, "/predict/batch?explain=false",
                                     {"flows": [to_flow(r) for _, r in rows.iterrows()]}, timeout=60)
                st.session_state["batch"] = (rows[cfg.TARGET].to_numpy(), resp, err)
            if "batch" in st.session_state:
                y, resp, err = st.session_state["batch"]
                if err:
                    st.error(f"/predict/batch failed: {err}")
                else:
                    pred = np.array([p["is_suspicious"] for p in resp["predictions"]])
                    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
                    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
                    c = st.columns(4)
                    c[0].metric("Flows", f"{len(y):,}")
                    c[1].metric("Flagged", f"{pred.mean():.1%}")
                    c[2].metric("Recall", f"{tp / (tp + fn):.3f}" if tp + fn else "-")
                    c[3].metric("FPR", f"{fp / (fp + tn):.3f}" if fp + tn else "-")
                    scores = pd.DataFrame({"score": [p["score"] for p in resp["predictions"]],
                                           "truth": np.where(y == 1, "Malicious", "Benign")})
                    fig = px.histogram(scores, x="score", color="truth", nbins=40, barmode="overlay", opacity=.75,
                                       color_discrete_map={"Malicious": RED, "Benign": NAVY})
                    fig.add_vline(x=resp["threshold"], line_dash="dot", line_color=INK)
                    chart(style(fig, height=260, title="Score distribution").update_xaxes(range=[0, 1]))
                    st.caption(f"Batch latency {resp['latency_ms']:.0f} ms. Every flow is appended to the service's "
                               "prediction log (Monitoring tab).")


# ----------------------------------------------------------------------------- what-if

with tab_whatif:
    try:
        bundle = local_bundle(model_dir)
    except Exception as e:  # no model folder yet
        bundle = None
        st.warning(f"Champion model not available locally ({e}). Run `dvc pull` first.")
    if bundle is not None:
        from sentinelml.data import normalize_features
        from sentinelml.explain import contributions

        schema = read_json(str(PROCESSED / cfg.FEATURE_SCHEMA)) or {}
        top_numeric = [f for f in (imp["feature"] if imp is not None else []) if f in cfg.NUMERIC][:8] or \
            ["sttl", "ct_state_ttl", "sbytes", "dbytes", "smean", "dmean", "ct_srv_src", "dload"]

        head = st.columns([1, 1, 3])
        start = head[0].segmented_control("Start from", ["Malicious", "Benign"], default="Malicious", key="wi_kind")
        if head[1].button("Load sample") or "wi_base" not in st.session_state:
            row = sample_rows(test, start or "Malicious").iloc[0]
            st.session_state["wi_base"] = (to_flow(row), int(row[cfg.TARGET]))
            st.session_state["wi_id"] = st.session_state.get("wi_id", 0) + 1
        base, base_label = st.session_state["wi_base"]
        wid = st.session_state["wi_id"]

        def score_and_contrib(flow: dict):
            df = normalize_features(pd.DataFrame([flow]).reindex(columns=cfg.FEATURES))
            return float(bundle.pipe.predict_proba(df)[0, 1]), contributions(bundle.pipe, bundle.explainer, df)[0]

        edit_col, out_col = st.columns([1, 1.25], gap="large")
        with edit_col:
            st.markdown("### Flow features")
            st.caption(f"Sample from the test set, ground truth **{'Malicious' if base_label else 'Benign'}**. "
                       "Edit the most influential features; the rest stay as sampled.")
            flow = dict(base)
            cc = st.columns(3)
            for i, c in enumerate(cfg.CATEGORICAL):
                options = list((schema.get("categorical", {}).get(c) or {}).get("values", [])) or [base[c]]
                if base[c] not in options:
                    options = [base[c]] + options
                flow[c] = cc[i].selectbox(c, options, index=options.index(base[c]), key=f"wi_{c}_{wid}")
            nc = st.columns(2)
            for i, c in enumerate(top_numeric):
                v = base[c] or 0.0
                step = 10.0 ** math.floor(math.log10(abs(v))) if v else 1.0
                flow[c] = nc[i % 2].number_input(c, min_value=0.0, value=float(v), step=step, format="%g",
                                                 key=f"wi_{c}_{wid}")
        with out_col:
            base_score, _ = score_and_contrib(base)
            score, contrib = score_and_contrib(flow)
            c = st.columns(3)
            c[0].metric("Score", f"{score:.3f}", delta=f"{score - base_score:+.3f}" if abs(score - base_score) > 5e-4 else None,
                        delta_color="inverse")
            c[1].metric("Threshold", f"{threshold:.3f}")
            c[2].metric("Verdict", "Suspicious" if score >= threshold else "Benign")
            chart(contribution_chart(bundle.explainer.names, contrib, top=12,
                                     title="Feature contributions (SHAP, log-odds)"))
            st.caption("Red pushes the flow toward suspicious, blue toward benign. Contributions explain the model, "
                       "not causality in the network.")


# ----------------------------------------------------------------------------- monitoring

with tab_monitor:
    h = st.columns([1.3, 1.3, 3])
    src = h[0].segmented_control("Source", ["Live service", "Local log"], default="Live service", key="mon_src")
    window = h[1].select_slider("Window (flows)", [200, 500, 1000, 2000, 5000], value=1000)
    log_err = None
    if src == "Local log":
        log = load_prediction_log(cfg.PREDICTION_LOG, limit=window)
        st.caption(f"{cfg.PREDICTION_LOG}")
    else:
        data, log_err = call_api(api_base, f"/predictions/recent?limit={window}", timeout=30)
        if log_err:
            hint = " The service is running an older build without this endpoint." if log_err == "HTTP 404" else ""
            st.error(f"Cannot read the prediction log from {api_base} ({log_err}).{hint}")
        log = load_prediction_log(data["records"] if data else [])

    if log_err:
        pass
    elif log.empty:
        st.info("No predictions logged yet. Send traffic from the Live service tab.")
    else:
        requests = log.drop_duplicates(["ts", "request_latency_ms"])
        c = st.columns(5)
        c[0].metric("Flows in window", f"{len(log):,}")
        c[1].metric("Flagged suspicious", f"{log['is_suspicious'].mean():.1%}")
        c[2].metric("Median score", f"{log['score'].median():.3f}")
        c[3].metric("Request latency p95", f"{requests['request_latency_ms'].quantile(.95):.1f} ms",
                    help="Server-side inference time per request (single flows and batches)")
        c[4].metric("Model versions", ", ".join(f"v{v}" for v in sorted(log["model_version"].astype(str).unique())))
        st.caption(f"{log['ts'].min():%Y-%m-%d %H:%M:%S} to {log['ts'].max():%Y-%m-%d %H:%M:%S} UTC")

        left, right = st.columns(2, gap="large")
        with left:
            fig = px.histogram(log, x="score", nbins=40, color_discrete_sequence=[NAVY])
            fig.add_vline(x=threshold, line_dash="dot", line_color=INK, annotation_text="threshold",
                          annotation_font_size=11)
            chart(style(fig, height=280, title="Score distribution").update_xaxes(range=[0, 1]).update_yaxes(title="flows"))
        with right:
            span = (log["ts"].max() - log["ts"].min()).total_seconds()
            freq = "10s" if span <= 900 else "1min" if span <= 6 * 3600 else "15min"
            ts = (log.assign(verdict=np.where(log["is_suspicious"], "Suspicious", "Benign"))
                  .groupby([pd.Grouper(key="ts", freq=freq), "verdict"]).size().reset_index(name="flows"))
            fig = px.bar(ts, x="ts", y="flows", color="verdict", color_discrete_map={"Suspicious": RED, "Benign": NAVY})
            chart(style(fig, height=280, title=f"Flows over time ({freq} buckets)").update_xaxes(title=None))

        st.markdown("### Feature drift vs. training data")
        if len(log) < 50:
            st.caption("At least 50 logged flows are needed for a meaningful drift estimate.")
        else:
            rep = drift_report(reference, log)
            counts = rep["status"].value_counts()
            c = st.columns(4)
            c[0].metric("Features checked", len(rep))
            c[1].metric("Significant drift", int(counts.get("significant", 0)), help=f"PSI > {PSI_SIGNIFICANT}")
            c[2].metric("Moderate drift", int(counts.get("moderate", 0)), help=f"{PSI_MODERATE} < PSI ≤ {PSI_SIGNIFICANT}")
            c[3].metric("Stable", int(counts.get("stable", 0)), help=f"PSI ≤ {PSI_MODERATE}")
            left, right = st.columns([1.2, 1], gap="large")
            with left:
                top = rep.head(15).iloc[::-1]
                fig = go.Figure(go.Bar(x=top["psi"], y=top["feature"], orientation="h",
                                       marker_color=[STATUS_COLORS[s] for s in top["status"]],
                                       hovertemplate="%{y}: PSI %{x:.3f}<extra></extra>"))
                for x in (PSI_MODERATE, PSI_SIGNIFICANT):
                    fig.add_vline(x=x, line_dash="dot", line_color=SLATE)
                chart(style(fig, height=470, title="Population stability index, top 15").update_xaxes(title="PSI"))
            with right:
                shown = rep[["feature", "psi", "ks_stat", "status"]].assign(
                    ks_stat=rep["ks_stat"].map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"))
                st.dataframe(
                    shown.style.map(lambda s: f"color: {STATUS_COLORS.get(s, INK)}; font-weight: 600", subset=["status"]),
                    hide_index=True, width="stretch", height=470,
                    column_config={
                        "feature": "Feature", "status": "Status",
                        "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                        "ks_stat": st.column_config.TextColumn("KS statistic"),
                    },
                )
            st.caption(f"Reference: {len(reference):,} rows of the {data_source} training split. PSI ≤ {PSI_MODERATE} stable, "
                       f"≤ {PSI_SIGNIFICANT} moderate, above that significant. The official UNSW-NB15 test split is "
                       "known to be shifted from the training split, so test-set traffic shows drift by design.")
