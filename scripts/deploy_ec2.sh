#!/usr/bin/env bash
# Deploy the API to an EC2 instance (Amazon Linux 2023 or Ubuntu) over SSH.
#
#   EC2_HOST=1.2.3.4 EC2_KEY=~/.ssh/sentinelml.pem \
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-south-1 \
#   scripts/deploy_ec2.sh
#
# Optional: EC2_USER (default ec2-user; use ubuntu on Ubuntu), IMAGE (default ghcr.io/nand0zz/sentinelml:latest),
#           DVC_REPO_URL, DVC_REV (default main), GHCR_USER + GHCR_TOKEN (only if the image is private).
# Steps: install Docker + compose if missing -> write /opt/sentinelml/{docker-compose.yml,.env} -> pull image ->
#        compose up (the container pulls the champion from the DVC remote) -> wait for health -> smoke test.
set -euo pipefail

: "${EC2_HOST:?set EC2_HOST}" "${EC2_KEY:?set EC2_KEY (path to the .pem key)}"
: "${AWS_ACCESS_KEY_ID:?}" "${AWS_SECRET_ACCESS_KEY:?}"
EC2_USER="${EC2_USER:-ec2-user}"
IMAGE="${IMAGE:-ghcr.io/nand0zz/sentinelml:latest}"
DVC_REPO_URL="${DVC_REPO_URL:-https://github.com/Nand0zz/sentinelml}"
DVC_REV="${DVC_REV:-main}"
APP_DIR=/opt/sentinelml
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH=(ssh -i "$EC2_KEY" -o StrictHostKeyChecking=accept-new "$EC2_USER@$EC2_HOST")

echo ">> [1/4] Docker on $EC2_HOST"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null; then
  if command -v dnf >/dev/null; then sudo dnf install -y docker; else curl -fsSL https://get.docker.com | sudo sh; fi
fi
sudo systemctl enable --now docker
if ! sudo docker compose version >/dev/null 2>&1; then   # Amazon Linux's docker package has no compose plugin
  sudo mkdir -p /usr/local/lib/docker/cli-plugins
  sudo curl -fsSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-$(uname -m)" \
       -o /usr/local/lib/docker/cli-plugins/docker-compose
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi
sudo docker --version && sudo docker compose version
REMOTE

echo ">> [2/4] compose file + secrets -> $APP_DIR"
"${SSH[@]}" "sudo mkdir -p $APP_DIR/models/champion $APP_DIR/logs && sudo chown -R 1000:1000 $APP_DIR/models $APP_DIR/logs"
scp -i "$EC2_KEY" -o StrictHostKeyChecking=accept-new "$HERE/docker-compose.yml" "$EC2_USER@$EC2_HOST:/tmp/docker-compose.yml"
"${SSH[@]}" "sudo mv /tmp/docker-compose.yml $APP_DIR/ && sudo install -m 600 /dev/stdin $APP_DIR/.env" <<EOF
IMAGE=$IMAGE
DVC_REPO_URL=$DVC_REPO_URL
DVC_REV=$DVC_REV
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-ap-south-1}
EOF

echo ">> [3/4] pull image + start (restart: unless-stopped)"
if [ -n "${GHCR_TOKEN:-}" ]; then
  "${SSH[@]}" "sudo docker login ghcr.io -u '${GHCR_USER:-nand0zz}' --password-stdin" <<<"$GHCR_TOKEN"
fi
"${SSH[@]}" "cd $APP_DIR && sudo docker compose pull && sudo docker compose up -d --force-recreate"

echo ">> [4/4] health + smoke test"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
for i in $(seq 1 60); do
  curl -sf http://localhost:8000/health >/dev/null && break
  [ "$i" = 60 ] && { sudo docker logs --tail 50 sentinelml-api; exit 1; }
  sleep 5
done
echo "health:     $(curl -s http://localhost:8000/health)"
echo "model-info: $(curl -s http://localhost:8000/model-info | head -c 300)..."
curl -s -X POST http://localhost:8000/predict?explain=false -H 'Content-Type: application/json' -d '{
  "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT", "spkts": 2, "dpkts": 0, "sbytes": 496,
  "dbytes": 0, "rate": 90909.09, "sttl": 254, "dttl": 0, "sload": 180363632.0, "dload": 0, "smean": 248}'
echo
REMOTE
echo ">> deployed: http://$EC2_HOST:8000/docs"
