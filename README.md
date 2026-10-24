# SentinelML — Drift-Aware Network Intrusion Detection MLOps Platform

An end-to-end, **defensive** ML system on the public UNSW-NB15 dataset: train, evaluate,
track, register, gate, serve and explain (Day 2), then monitor and explore (Day 3).
Predictions are *indicators requiring analyst review*, never proof of an attack.

**Status: Day 2 — registry quality gate, FastAPI service with explanations, Docker, CI.**

## Quick start
```bash
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows (Mac/Linux: .venv/bin/activate)
pip install -r requirements.txt
pytest                                                   # 31 tests, ~1-2 min

# data: put UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv in data/raw/
python -m sentinelml.train                # trains 3 models, registers the winner as `candidate`
python -m sentinelml.gate status          # which version holds which alias
python -m sentinelml.gate promote         # quality gate: candidate -> champion if it passes
uvicorn sentinelml.api:app --port 8000    # then open http://127.0.0.1:8000/docs
```
`--synthetic` (train, scripts) runs everything on generated data with no download.

## The release flow (what makes this MLOps)
```
train  ->  candidate  --[quality gate]-->  champion  -->  API (+ prediction log)
                              |                  ^
                              v                  |
                         rejected          rollback (previous_champion)
```
| Alias | Meaning |
|---|---|
| `candidate` | newest model, not yet approved |
| `champion` | approved model the API serves |
| `previous_champion` | the one before, for `python -m sentinelml.gate rollback` |

### Quality gate rules (`python -m sentinelml.gate promote`)
All must pass, otherwise the candidate is rejected (exit code 1) and the champion is untouched:
1. test FPR <= budget (5%) + 1 pt slack
2. test PR-AUC >= 0.95 and test recall >= 0.80
3. single-row latency <= 100 ms (this is what rejects a slow Random Forest)
4. no regression vs. the champion: PR-AUC drop <= 0.002, recall drop <= 0.01 (skipped for the first model)

Every run writes `reports/gate_report.json`. Thresholds are CLI flags (`--help`). `--dry-run` shows the verdict without promoting.

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model version |
| `GET /model-info` | version, alias, threshold, registry metrics, input features |
| `POST /predict` | one flow -> score, `is_suspicious`, top-5 contributing features, latency |
| `POST /predict/batch` | up to 500 flows |
`?explain=false` skips explanations. Inputs are validated (unknown fields and negative numbers -> 422); categoricals are normalised exactly as in training. Every scored flow is appended to `logs/predictions.jsonl` (Day 3 drift monitoring reads it).

**Explanations:** SHAP `TreeExplainer` for tree models, exact coefficient contributions for logistic regression. Values are in log-odds: positive pushes toward "suspicious". They explain the model, not causality.
Global importance: `python -m sentinelml.explain` -> `reports/shap_global.html`.

## Docker
The registry stores absolute artifact paths from the training machine, so containers use an **exported** copy:
```bash
python -m sentinelml.export                                   # -> exported_model/
docker build -t sentinelml-api .
docker run -p 8000:8000 -v "$(pwd)/exported_model:/models/champion:ro" sentinelml-api
python scripts/smoke_test_api.py http://localhost:8000
```
(Git Bash on Windows: use `-v "/$(pwd)/exported_model:/models/champion:ro"` or run from PowerShell with `${PWD}`.)

## CI (`.github/workflows/ci.yml`)
1. `test`: pytest on every push/PR.
2. `gate-and-docker`: synthetic train -> quality gate -> export -> docker build -> run container -> API smoke test. Gate thresholds are relaxed there because synthetic data only checks the plumbing.

## Analysis scripts
- `python scripts/fpr_sensitivity.py` — recall at FPR budgets of 0.5/1/2/5/10% (does not retrain).
- `python scripts/eda.py` — EDA report.

## Methodology and honest limits
- Threshold: max recall subject to FPR <= budget, chosen on validation, applied unchanged to test. The 5% budget is an assumption; on real traffic it would be far too high (5% of 1M benign flows = 50,000 false alerts/day).
- Model selection on validation PR-AUC, never on test. The gate's metrics come from the held-out test evaluation; in production re-score both models on a fresh labelled set.
- UNSW-NB15 is lab-generated (2015) and attack-heavy: precision/PR-AUC are optimistic; scores are benchmark results, not real-world detection performance.
- No cross-validation, resampling or calibration; hyperparameters are defaults.

## Roadmap
Day 3: PSI/KS drift + latency monitoring from the prediction log, Streamlit dashboard, limitations write-up.
