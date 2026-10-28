# SentinelML — Drift-Aware Network Intrusion Detection MLOps Platform

An end-to-end, **defensive** ML system on the public UNSW-NB15 dataset: train, evaluate,
track, register, gate, serve and explain (Day 2), then monitor and explore (Day 3).
Predictions are *indicators requiring analyst review*, never proof of an attack.

**Status: Day 2 + MLOps lab — DVC-versioned data and models, reproducible `dvc repro` pipeline, MLflow tracking,
quality-gated promotion, runtime model fetch in Docker, CI/CD to GHCR, EC2 deployment with a restart policy.**

## Quick start
```bash
python -m venv .venv && source .venv/Scripts/activate   # Git Bash on Windows (Mac/Linux: .venv/bin/activate)
pip install -r requirements.txt                          # project + dev + DVC (S3)
pytest                                                   # ~2 min

dvc pull                                  # raw data + current models from the DVC remote (needs AWS creds, see below)
dvc repro                                 # preprocess -> train: writes models/candidate + reports/metrics.json
python -m sentinelml.gate promote         # quality gate: candidate -> champion if it passes
dvc push                                  # upload the new champion; then commit models/*.dvc + models/model_manifest.json
uvicorn sentinelml.api:app --port 8000    # then open http://127.0.0.1:8000/docs
```
`--synthetic` (preprocess, train, scripts) runs everything on generated data with no download.
Without DVC the original commands still work: put the two CSVs in `data/raw/` and run `python -m sentinelml.train`.

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
When `models/candidate/` exists (the DVC pipeline output) the gate works on the model **folders**, so it runs the same
on a fresh clone after `dvc pull`; the MLflow aliases are kept in sync when the local registry has the model.

## API
| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model version |
| `GET /model-info` | version, alias, run name, threshold, registry metrics, input features |
| `GET /status` | uptime, request / error / prediction counters, which model is served |
| `GET /predictions/recent?limit=N` | last N prediction-log entries (monitoring dashboard) |
| `POST /predict` | one flow -> score, `is_suspicious`, top-5 contributing features, latency |
| `POST /predict/batch` | up to 500 flows |
`?explain=false` skips explanations. Inputs are validated (unknown fields and negative numbers -> 422); categoricals are normalised exactly as in training. Every scored flow is appended to `logs/predictions.jsonl` (Day 3 drift monitoring reads it).

**Explanations:** SHAP `TreeExplainer` for tree models, exact coefficient contributions for logistic regression. Values are in log-odds: positive pushes toward "suspicious". They explain the model, not causality.
Global importance: `python -m sentinelml.explain` -> `reports/shap_global.html`.

## Data versioning and the reproducible pipeline (DVC)
| Path | Tracked by | What |
|---|---|---|
| `data/raw/*.csv` | DVC (`data/raw/*.csv.dvc`) | the two official UNSW-NB15 CSVs, never committed to git |
| `data/processed/{train,test}.parquet`, `feature_schema.json` | DVC stage `preprocess` | cleaned splits + input schema (column groups, dtypes, categorical vocabularies, numeric ranges) |
| `models/candidate/` | DVC stage `train` | newest model (MLflow sklearn format + `export_meta.json`: metrics, threshold, params, run id) |
| `models/champion/`, `models/previous_champion/` | DVC (`models/*.dvc`, written by the gate) | served model and one-step rollback |
| `models/model_manifest.json` | git | current champion: version, run, artifact path, DVC md5, promotion record |
| `reports/metrics.json`, `reports/confusion_matrix.json` | DVC metrics (in git) | winner's val/test metrics + confusion matrix |
| `params.yaml` | git | pipeline parameters (experiment name, FPR budget, seed, paths) |

**Stages** (`dvc.yaml`): `preprocess` (raw CSV -> parquet + schema, deterministic) and `train` (parquet -> candidate,
metrics). Preprocessing only *cleans* (types, `inf` -> NaN, lower-cased categoricals, leakage columns dropped, train
de-duplicated); imputation, scaling and one-hot encoding stay inside the sklearn Pipeline so serving can never drift
from training. Promotion is deliberately **not** a stage: it depends on the current champion, so it is an explicit
release step.

**Experiment tracking (MLflow):** experiment `sentinelml-training`; each training session is a parent run named
`run-YYYYMMDD-HHMMSS` (UTC) with one nested run per model (params, val/test PR-AUC, recall, FPR, threshold, latency,
confusion matrix, model). The parent run records the winner, its registered version, its metrics and a timestamp.
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Reproducible pipeline
```bash
dvc pull                              # data + models for the current commit
dvc repro                             # re-runs only stages whose inputs/params/code changed
dvc status                            # what is out of date
dvc metrics show                      # reports/metrics.json + confusion matrix
python -m sentinelml.gate promote     # gate the candidate; on PASS: champion -> previous_champion, candidate -> champion,
                                      #   dvc add models/{champion,previous_champion}, write models/model_manifest.json
dvc push                              # (or: gate promote --push) upload data + models to the remote
git add dvc.lock models/*.dvc models/model_manifest.json reports/ && git commit -m "Promote model"
python -m sentinelml.gate status      # which version holds each alias; `gate rollback` swaps champion <-> previous
```

### DVC remote setup (S3)
The remote is configured in `.dvc/config` (`storage` -> `s3://<bucket>/dvcstore`). Credentials are **never** in the repo:
DVC reads the standard AWS variables (or `~/.aws/credentials`, or an EC2 instance role):
```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=ap-south-1
dvc remote list && dvc pull
# use your own bucket:   dvc remote modify storage url s3://<your-bucket>/dvcstore
# offline/lab fallback:  dvc remote add -d --local localdev /some/folder   (kept in .dvc/config.local, not committed)
```
Fresh clone: `git clone ... && pip install -r requirements.txt && dvc pull && dvc repro` reproduces everything.

## Docker (model fetched at runtime)
The image contains **no model**. `scripts/start_api.sh` (the entrypoint) prepares `MODEL_DIR` and then runs uvicorn:
- `DVC_REPO_URL` set: reads `models/model_manifest.json` at `DVC_REV` (default `main`); if `MODEL_DIR` is empty or holds
  a different run, it runs `dvc get <repo> models/champion` from the DVC remote into `MODEL_DIR` (needs AWS creds).
  If the refresh fails but a model is already there, it keeps serving it.
- `DVC_REPO_URL` unset: serves whatever is mounted at `MODEL_DIR` (default `/models/champion`).
```bash
docker build -t sentinelml-api .
# a) mount a local model folder
docker run -p 8000:8000 -v "$(pwd)/models/champion:/models/champion:ro" sentinelml-api
# b) fetch the latest champion from the DVC remote at start-up
docker run -p 8000:8000 -e DVC_REPO_URL=https://github.com/Nand0zz/sentinelml \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION -v sentinelml-model:/models/champion sentinelml-api
python scripts/smoke_test_api.py http://localhost:8000     # /health, /model-info, /predict, 422 check, /status
```
`python -m sentinelml.export` still writes the registry champion to `exported_model/` for option (a).
(Git Bash on Windows: use `-v "/$(pwd)/models/champion:/models/champion:ro"` or run from PowerShell with `${PWD}`.)

## CI/CD (`.github/workflows/ci.yml`)
1. `test`: pytest on every push/PR.
2. `pipeline-and-image`: synthetic preprocess -> train (same commands as the DVC stages) -> quality gate -> docker build
   -> run with the champion mounted -> smoke test. On pushes (not PRs) the image is pushed to **GHCR** as
   `ghcr.io/<owner>/sentinelml:<commit-sha>`, `:<branch>` and `:latest` (default branch). Uses the built-in
   `GITHUB_TOKEN`, no extra secret. Gate thresholds are relaxed because synthetic data only checks the plumbing.
3. `deploy-ec2` (default branch, on push or a manual "Run workflow"): runs `scripts/deploy_ec2.sh` when these repository secrets exist:
   `EC2_HOST`, `EC2_SSH_KEY` (private key), `EC2_USER` (optional), `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_DEFAULT_REGION`. Without them the job is skipped.

## EC2 deployment
One-time AWS setup: an S3 bucket for the DVC remote, an IAM user with access to that bucket only (its keys go into
your shell, the GitHub secrets and the server's `.env`), and an EC2 instance (Amazon Linux 2023, t3.small or larger)
whose security group allows SSH (22) from your IP and TCP 8000 from the clients. Make the GHCR package public
(GitHub -> Packages -> sentinelml -> settings) or pass `GHCR_USER`/`GHCR_TOKEN` to the deploy script.
```bash
EC2_HOST=<public-ip> EC2_KEY=~/.ssh/sentinelml.pem \
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-south-1 \
  scripts/deploy_ec2.sh
```
The script SSHes in, installs Docker + the compose plugin if missing, writes `/opt/sentinelml/docker-compose.yml`
and a `chmod 600` `.env` with the AWS credentials, pulls the image, starts it with `docker compose up -d`, waits for
`/health` and sends `/model-info` + a sample `/predict`.
**Restart policy:** `restart: unless-stopped` restarts the container if it crashes and after an instance reboot
(Docker starts on boot), and only stays down after an explicit `docker compose down`/`stop`. The model folder
`/opt/sentinelml/models/champion` is a volume, so restarts reuse the downloaded model and only fetch when the
manifest names a newer champion. Client call: `curl http://<public-ip>:8000/predict -H 'Content-Type: application/json' -d @flow.json`.

## Production flow
```
dvc repro (train) -> gate promote (champion) -> dvc push (model to S3) -> git push (manifest + .dvc files)
   -> CI: tests, pipeline check, docker build, push image to GHCR -> deploy-ec2: pull image
   -> container start: fetch champion from S3 if stale -> serve on :8000 (restart: unless-stopped)
```

## Monitoring
Every prediction is appended to `logs/predictions.jsonl` (mounted to `/opt/sentinelml/logs` on EC2); `GET /status`
returns uptime and request/error/prediction counters; Docker's health check polls `/health`; uvicorn logs every request
(`docker logs sentinelml-api`).

## Operations dashboard (Streamlit)
```bash
streamlit run dashboard/app.py                                         # from the repo root -> http://localhost:8501
SENTINELML_API=http://<ec2-public-ip>:8000 streamlit run dashboard/app.py   # point it at the deployment
```
| Tab | Shows |
|---|---|
| Overview | champion metrics and manifest, release pipeline, quality-gate checks, test confusion matrix, model comparison, feature importance |
| Live service | `/status` of the running API; score a random test-set flow (score, verdict, top SHAP features); send batches of real flows |
| What-if analysis | edit a flow's most influential features and watch the score and SHAP contributions change (local champion model) |
| Monitoring | prediction log from the service (`GET /predictions/recent`) or the local file: score distribution, flagged rate, request latency, and feature drift vs. the training data (PSI + KS per feature) |

Needs `dvc pull` for the local artifacts; without them it falls back to synthetic data. Drift uses PSI
(<= 0.10 stable, <= 0.25 moderate, above that significant). Traffic sampled from the official UNSW-NB15 test
split shows drift on purpose: that split is distribution-shifted from the training split.

## Analysis scripts
- `python scripts/fpr_sensitivity.py` — recall at FPR budgets of 0.5/1/2/5/10% (does not retrain).
- `python scripts/eda.py` — EDA report.

## Methodology and honest limits
- Threshold: max recall subject to FPR <= budget, chosen on validation, applied unchanged to test. The 5% budget is an assumption; on real traffic it would be far too high (5% of 1M benign flows = 50,000 false alerts/day).
- Model selection on validation PR-AUC, never on test. The gate's metrics come from the held-out test evaluation; in production re-score both models on a fresh labelled set.
- UNSW-NB15 is lab-generated (2015) and attack-heavy: precision/PR-AUC are optimistic; scores are benchmark results, not real-world detection performance.
- No cross-validation, resampling or calibration; hyperparameters are defaults.

## Roadmap
Done: PSI/KS drift + latency monitoring from the prediction log and the Streamlit dashboard. Next: alerting on drift, scheduled retraining.
