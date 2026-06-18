#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${AWS_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${AWS_ENV_FILE}"
  set +a
fi

PURGE_MODEL=0
KILL_INSTANCE=0
PUBLIC_IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge-model)
      PURGE_MODEL=1
      ;;
    --kill-instance)
      KILL_INSTANCE=1
      ;;
    -*)
      echo "error: unknown flag '$1'" >&2
      echo "usage: $0 [--purge-model] [--kill-instance] <instance-ip>" >&2
      exit 1
      ;;
    *)
      if [[ -n "${PUBLIC_IP}" ]]; then
        echo "error: unexpected argument '$1'" >&2
        exit 1
      fi
      PUBLIC_IP="$1"
      ;;
  esac
  shift
done

if [[ -z "${PUBLIC_IP}" ]]; then
  echo "error: instance IP required." >&2
  echo "usage: $0 [--purge-model] [--kill-instance] <instance-ip>" >&2
  exit 1
fi

KEY_FILE="${REMOTE_WORKER_KEY_FILE:-}"
SSH_USER="${REMOTE_WORKER_SSH_USER:-ec2-user}"

if [[ -n "${KEY_FILE}" && "${KEY_FILE}" != /* ]]; then
  KEY_FILE="${REPO_ROOT}/${KEY_FILE}"
fi

SSH_ARGS=(-o StrictHostKeyChecking=no)
if [[ -n "${KEY_FILE}" ]]; then
  SSH_ARGS+=(-i "${KEY_FILE}")
fi

if [[ "${KILL_INSTANCE}" -eq 1 ]]; then
  REGION="${REMOTE_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
  echo "Looking up instance ID for ${PUBLIC_IP} ..."
  INSTANCE_ID="$(
    aws ec2 describe-instances \
      --region "${REGION}" \
      --filters "Name=ip-address,Values=${PUBLIC_IP}" \
      --query 'Reservations[0].Instances[0].InstanceId' \
      --output text
  )"
  if [[ -z "${INSTANCE_ID}" || "${INSTANCE_ID}" == "None" ]]; then
    echo "error: could not find instance with IP ${PUBLIC_IP}" >&2
    exit 1
  fi
  echo "Terminating instance ${INSTANCE_ID} ..."
  aws ec2 terminate-instances --region "${REGION}" --instance-ids "${INSTANCE_ID}" --output text
  echo "Terminated ${INSTANCE_ID}."
  exit 0
fi

echo "Cleaning instance ${PUBLIC_IP} ..."

ssh "${SSH_ARGS[@]}" "${SSH_USER}@${PUBLIC_IP}" bash -s -- "${PURGE_MODEL}" <<'REMOTE'
set -euo pipefail
PURGE_MODEL="$1"

echo "--- Stopping and removing all containers ..."
CONTAINERS="$(sudo docker ps -aq 2>/dev/null || true)"
if [[ -n "${CONTAINERS}" ]]; then
  sudo docker stop ${CONTAINERS} 2>/dev/null || true
  sudo docker rm -f ${CONTAINERS} 2>/dev/null || true
  echo "Removed containers: ${CONTAINERS}"
else
  echo "No containers found."
fi

echo "--- Removing all Docker images ..."
IMAGES="$(sudo docker images -q 2>/dev/null || true)"
if [[ -n "${IMAGES}" ]]; then
  sudo docker rmi -f ${IMAGES} 2>/dev/null || true
  echo "Removed images."
else
  echo "No images found."
fi

echo "--- Pruning build cache and volumes ..."
sudo docker system prune -af --volumes 2>/dev/null || true

if [[ "${PURGE_MODEL}" -eq 1 ]]; then
  echo "--- Removing Pi-0 model files ..."
  sudo rm -rf /opt/lagrangian/pi0
  echo "Removed /opt/lagrangian/pi0"
else
  echo "--- Skipping Pi-0 model (pass --purge-model to remove)"
fi

echo "Done."
REMOTE

echo "Instance ${PUBLIC_IP} cleaned."
