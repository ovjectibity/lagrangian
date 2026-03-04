#!/bin/bash
set -euxo pipefail

if command -v yum >/dev/null 2>&1; then
  yum update -y
  yum install -y docker awscli
  systemctl enable docker
  systemctl start docker
elif command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y docker.io awscli
  systemctl enable docker
  systemctl start docker
else
  echo "Unsupported package manager" >&2
  exit 1
fi

__ECR_LOGIN__
docker pull __IMAGE_URI__
docker rm -f __CONTAINER_NAME__ || true
cat >/etc/lagrangian-worker.env <<'ENV_EOF'
WS_HOST=0.0.0.0
WS_PORT=__WORKER_PORT__
WORKER_SHARED_SECRET=__WORKER_SHARED_SECRET__
ENV_EOF

docker run -d --restart unless-stopped \
  --name __CONTAINER_NAME__ \
  --env-file /etc/lagrangian-worker.env \
  -p __WORKER_PORT__:__WORKER_PORT__ \
  __IMAGE_URI__
