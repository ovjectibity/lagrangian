#!/usr/bin/env bash
set -euo pipefail

ONLY_BOOTSTRAP=0
PUBLIC_IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-bootstrap)
      ONLY_BOOTSTRAP=1
      shift
      if [[ $# -gt 0 && "$1" != --* ]]; then
        PUBLIC_IP="$1"
        shift
      fi
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      echo "usage: $0 [--only-bootstrap <ip>]" >&2
      exit 1
      ;;
  esac
done

if [[ "${ONLY_BOOTSTRAP}" -eq 1 && -z "${PUBLIC_IP}" ]]; then
  echo "error: --only-bootstrap requires an IP address argument." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${AWS_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${AWS_ENV_FILE}"
  set +a
fi

# ---- Validate required env vars ----
for var in REMOTE_WORKER_AMI_ID REMOTE_WORKER_SECURITY_GROUP_IDS \
           REMOTE_WORKER_IAM_INSTANCE_PROFILE_ARN REMOTE_WORKER_IMAGE_URI \
           REMOTE_WORKER_CONTAINER_NAME REMOTE_WORKER_PORT \
           REMOTE_WORKER_SHARED_SECRET; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: ${var} is not set." >&2
    exit 1
  fi
done

REGION="${REMOTE_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
AMI_ID="${REMOTE_WORKER_AMI_ID}"
SECURITY_GROUP_IDS="${REMOTE_WORKER_SECURITY_GROUP_IDS}"
IAM_INSTANCE_PROFILE_ARN="${REMOTE_WORKER_IAM_INSTANCE_PROFILE_ARN}"
KEY_NAME="${REMOTE_WORKER_KEY_NAME:-}"
KEY_FILE="${REMOTE_WORKER_KEY_FILE:-}"
SSH_USER="${REMOTE_WORKER_SSH_USER:-ec2-user}"

if [[ -n "${KEY_FILE}" && "${KEY_FILE}" != /* ]]; then
  KEY_FILE="${REPO_ROOT}/${KEY_FILE}"
fi
INSTANCE_NAME="${REMOTE_WORKER_INSTANCE_NAME_TAG:-lagrangian-remote-worker}"
INSTANCE_TYPE="g4dn.xlarge"
VOLUME_SIZE=60

BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/bootstrap_remote_worker.sh"
if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "error: bootstrap script not found at ${BOOTSTRAP_SCRIPT}" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH." >&2
  exit 1
fi

if [[ "${ONLY_BOOTSTRAP}" -eq 0 ]]; then
  # ---- Launch instance ----
  echo "Launching ${INSTANCE_TYPE} instance in ${REGION} ..."

  LAUNCH_ARGS=(
    --region "${REGION}"
    --image-id "${AMI_ID}"
    --instance-type "${INSTANCE_TYPE}"
    --security-group-ids "${SECURITY_GROUP_IDS}"
    --iam-instance-profile "Arn=${IAM_INSTANCE_PROFILE_ARN}"
    --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]"
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}}]"
    --count 1
    --query 'Instances[0].InstanceId'
    --output text
  )

  if [[ -n "${KEY_NAME}" ]]; then
    LAUNCH_ARGS+=(--key-name "${KEY_NAME}")
  fi

  INSTANCE_ID="$(aws ec2 run-instances "${LAUNCH_ARGS[@]}")"
  echo "Launched instance: ${INSTANCE_ID}"

  # ---- Wait for instance to be running ----
  echo "Waiting for instance to enter 'running' state ..."
  aws ec2 wait instance-running --region "${REGION}" --instance-ids "${INSTANCE_ID}"

  # ---- Get public IP ----
  PUBLIC_IP="$(
    aws ec2 describe-instances \
      --region "${REGION}" \
      --instance-ids "${INSTANCE_ID}" \
      --query 'Reservations[0].Instances[0].PublicIpAddress' \
      --output text
  )"
  echo "Instance public IP: ${PUBLIC_IP}"
fi

SSH_ARGS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
if [[ -n "${KEY_FILE}" ]]; then
  SSH_ARGS+=(-i "${KEY_FILE}")
fi

# ---- Wait for SSH to become available ----
echo "Waiting for SSH to become available ..."
for i in $(seq 1 30); do
  if nc -z -w 5 "${PUBLIC_IP}" 22 2>/dev/null; then
    break
  fi
  if [[ "${i}" -eq 30 ]]; then
    echo "error: SSH did not become available after 30 attempts." >&2
    exit 1
  fi
  echo "  attempt ${i}/30 ..."
  sleep 10
done

# ---- Transfer scripts ----
echo "Transferring bootstrap script ..."
scp "${SSH_ARGS[@]}" \
  "${BOOTSTRAP_SCRIPT}" \
  "${SSH_USER}@${PUBLIC_IP}:/tmp/bootstrap_remote_worker.sh"
echo "Transferring model download script ..."
scp "${SSH_ARGS[@]}" \
  "${SCRIPT_DIR}/download_pi0_model.sh" \
  "${SSH_USER}@${PUBLIC_IP}:/tmp/download_pi0_model.sh"

# ---- Run bootstrap on instance ----
echo "Running bootstrap on instance ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${PUBLIC_IP}" bash -s <<EOF
set -euo pipefail

AWS_REGION="${REGION}"
IMAGE_URI="${REMOTE_WORKER_IMAGE_URI}"
CONTAINER_NAME="${REMOTE_WORKER_CONTAINER_NAME}"
WORKER_PORT="${REMOTE_WORKER_PORT}"
WORKER_SHARED_SECRET="${REMOTE_WORKER_SHARED_SECRET}"
CUDA_DEBUG_MODE="${ENABLE_WORKER_CUDA_DEBUG_MODE:-0}"
HF_TOKEN="${HF_TOKEN:-}"

REGISTRY="\${IMAGE_URI%%/*}"
if [[ "\${REGISTRY}" == *".dkr.ecr."* ]]; then
  ECR_LOGIN="aws ecr get-login-password --region \${AWS_REGION} | docker login --username AWS --password-stdin \${REGISTRY}"
else
  ECR_LOGIN="true"
fi

perl -pe "
s#__ECR_LOGIN__#\${ECR_LOGIN}#g;
s#__IMAGE_URI__#\${IMAGE_URI}#g;
s#__CONTAINER_NAME__#\${CONTAINER_NAME}#g;
s#__WORKER_PORT__#\${WORKER_PORT}#g;
s#__WORKER_SHARED_SECRET__#\${WORKER_SHARED_SECRET}#g;
s#__CUDA_DEBUG_MODE__#\${CUDA_DEBUG_MODE}#g;
s#__HF_TOKEN__#\${HF_TOKEN}#g;
" /tmp/bootstrap_remote_worker.sh > /tmp/bootstrap_remote_worker.rendered.sh

chmod +x /tmp/bootstrap_remote_worker.rendered.sh
sudo nohup /tmp/bootstrap_remote_worker.rendered.sh > /tmp/bootstrap.log 2>&1 &
echo "Bootstrap running in background (PID \$!). Tail: sudo tail -f /tmp/bootstrap.log"
wait
EOF

echo ""
echo "=== [1/2] Bootstrap: system setup, Docker pull, Pi-0 download ==="
echo ""

ssh "${SSH_ARGS[@]}" "${SSH_USER}@${PUBLIC_IP}" "
  until [ -f /tmp/bootstrap.log ]; do sleep 1; done
  tail -n +1 -f /tmp/bootstrap.log | while IFS= read -r line; do
    echo \"\$line\"
    echo \"\$line\" | grep -qF '### BOOTSTRAP_DONE ###' && break
  done
" || true

echo ""
echo "=== [2/2] Container startup ==="
echo ""

ssh "${SSH_ARGS[@]}" "${SSH_USER}@${PUBLIC_IP}" "
  for i in \$(seq 1 60); do
    sudo docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^${REMOTE_WORKER_CONTAINER_NAME}\$' && break
    [ \"\$i\" -eq 60 ] && echo 'error: container did not start within 2 minutes' >&2 && exit 1
    sleep 2
  done
  sudo docker logs -f '${REMOTE_WORKER_CONTAINER_NAME}' 2>&1 | while IFS= read -r line; do
    echo \"\$line\"
    echo \"\$line\" | grep -q 'server listening on' && break
  done
" || true

echo ""
echo "=== Worker ready at ws://${PUBLIC_IP}:${REMOTE_WORKER_PORT} ==="
