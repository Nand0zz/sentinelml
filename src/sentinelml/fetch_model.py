"""Fetch the current champion from the DVC remote into MODEL_DIR (run at container start).

    python -m sentinelml.fetch_model --repo https://github.com/<user>/sentinelml --rev main --dest /models/champion

`models/model_manifest.json` at <repo>@<rev> says which champion is current (its MLflow run id).
If MODEL_DIR already holds that model nothing is downloaded; otherwise `dvc get` pulls
models/champion from the DVC remote (S3 credentials come from the usual AWS_* env vars or an
instance role). The image never contains a model. Exit codes: 0 = model ready, 1 = failed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sentinelml import config as cfg

MANIFEST = f"models/{cfg.MODEL_MANIFEST}"
CHAMPION = f"models/{cfg.ALIAS_CHAMPION}"


def local_run_id(dest: Path) -> str | None:
    f = dest / cfg.MODEL_META
    return json.loads(f.read_text(encoding="utf-8")).get("run_id") if f.exists() else None


def remote_manifest(repo: str, rev: str) -> dict:
    import dvc.api  # heavy import, only needed here

    return json.loads(dvc.api.read(MANIFEST, repo=repo, rev=rev, encoding="utf-8"))


def download(repo: str, rev: str, dest: Path) -> None:
    """dvc get into a temp dir, then swap the files in (dest may be a mount point, so keep the dir)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent if os.access(dest.parent, os.W_OK) else None) as tmp:
        out = Path(tmp) / "champion"
        subprocess.run([sys.executable, "-m", "dvc", "get", repo, CHAMPION, "--rev", rev, "-o", str(out)],
                       check=True)
        for child in dest.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        for child in out.iterdir():
            shutil.move(str(child), dest / child.name)


def ensure_model(repo: str, rev: str, dest: Path) -> str:
    """-> 'up-to-date' | 'downloaded'. Raises on failure."""
    manifest = remote_manifest(repo, rev)
    want, have = manifest.get("run_id"), local_run_id(dest)
    print(f"[fetch_model] {repo}@{rev}: champion v{manifest.get('champion_version')} "
          f"({manifest.get('run_name')}), local: {have or 'none'}", flush=True)
    if have is not None and have == want:
        return "up-to-date"
    download(repo, rev, dest)
    if local_run_id(dest) != want:
        raise RuntimeError("downloaded model does not match the manifest (was the champion dvc-pushed?)")
    return "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=os.getenv("DVC_REPO_URL"), help="git URL of this project (env DVC_REPO_URL)")
    ap.add_argument("--rev", default=os.getenv("DVC_REV", "main"), help="branch/tag/commit (env DVC_REV)")
    ap.add_argument("--dest", default=os.getenv("MODEL_DIR", "/models/champion"))
    a = ap.parse_args()
    if not a.repo:
        print("[fetch_model] DVC_REPO_URL not set; nothing to fetch", file=sys.stderr)
        return 1
    dest = Path(a.dest)
    try:
        print(f"[fetch_model] {ensure_model(a.repo, a.rev, dest)}: {dest}", flush=True)
        return 0
    except Exception as e:  # network/credentials/remote problems all end up here
        print(f"[fetch_model] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
