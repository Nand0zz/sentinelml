#!/bin/sh
# Container entrypoint: make sure the champion model is present and current, then start the API.
#
#   MODEL_DIR     where the model lives (a mounted volume), default /models/champion
#   DVC_REPO_URL  git URL of this project. When set, the champion is fetched from the DVC remote if
#                 MODEL_DIR is empty or stale (models/model_manifest.json at DVC_REV decides "latest").
#                 When unset, the model must already be mounted at MODEL_DIR.
#   DVC_REV       branch/tag/commit to follow (default: main)
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION   S3 remote credentials
#   PORT, WORKERS uvicorn settings (default 8000, 1)
set -eu
export MODEL_DIR="${MODEL_DIR:-/models/champion}"
log() { echo "[start_api] $*"; }

if [ -n "${DVC_REPO_URL:-}" ]; then
    log "checking for the latest champion: $DVC_REPO_URL@${DVC_REV:-main}"
    if ! python -m sentinelml.fetch_model --repo "$DVC_REPO_URL" --rev "${DVC_REV:-main}" --dest "$MODEL_DIR"; then
        if [ -f "$MODEL_DIR/export_meta.json" ]; then
            log "WARNING: could not refresh the model, serving the copy already in $MODEL_DIR"
        else
            log "ERROR: no model in $MODEL_DIR and the fetch failed"
            exit 1
        fi
    fi
else
    log "DVC_REPO_URL not set, using the model mounted at $MODEL_DIR"
fi

if [ ! -f "$MODEL_DIR/export_meta.json" ]; then
    log "ERROR: $MODEL_DIR/export_meta.json not found (mount a model folder or set DVC_REPO_URL)"
    exit 1
fi
log "serving $(python -c "import json,os; m=json.load(open(os.path.join(os.environ['MODEL_DIR'],'export_meta.json'))); print(m['model_name'], 'v' + str(m['version']), m.get('run_name') or '')")"
exec uvicorn sentinelml.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers "${WORKERS:-1}"
