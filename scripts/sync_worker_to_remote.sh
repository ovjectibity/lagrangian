#!/usr/bin/env bash
# Syncs locally changed worker files (staged or unstaged) to a remote EC2 instance
# and restarts the worker container. Useful for iterating without a full ECR push.
#
# Usage: sync_worker_to_remote.sh <remote-ip>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <remote-ip>" >&2
  exit 1
fi

REMOTE_IP="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AWS_ENV_FILE="${REPO_ROOT}/.envrc"

if [[ -f "${AWS_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AWS_ENV_FILE}"
  set +a
fi

SSH_USER="${REMOTE_WORKER_SSH_USER:-ec2-user}"
KEY_FILE="${REMOTE_WORKER_KEY_FILE:-}"
CONTAINER_NAME="${REMOTE_WORKER_CONTAINER_NAME:-lagrangian-worker-py}"

if [[ -n "${KEY_FILE}" && "${KEY_FILE}" != /* ]]; then
  KEY_FILE="${REPO_ROOT}/${KEY_FILE}"
fi

SSH_ARGS=(-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=10)
if [[ -n "${KEY_FILE}" ]]; then
  SSH_ARGS+=(-i "${KEY_FILE}")
fi

# Collect changed files under worker/worker-py/ (staged, unstaged, and untracked)
cd "${REPO_ROOT}"
CHANGED_FILES=$(
  {
    git diff --name-only HEAD 2>/dev/null
    git ls-files --others --exclude-standard 2>/dev/null
  } | sort -u | grep '^worker/worker-py/' || true
)

if [[ -z "${CHANGED_FILES}" ]]; then
  echo "No changed files under worker/worker-py/. Nothing to sync."
  exit 0
fi

echo "Files to sync to ${REMOTE_IP}:"
echo "${CHANGED_FILES}" | sed 's/^/  /'
echo ""

# Local: worker/worker-py/<rel>  →  Container: /opt/lagrangian/worker/<rel>
while IFS= read -r local_path <&3; do
  rel="${local_path#worker/worker-py/}"
  container_path="/opt/lagrangian/worker/${rel}"
  container_dir="$(dirname "${container_path}")"
  remote_tmp="/tmp/_worker_sync_$(echo "${rel}" | tr '/' '__')"

  echo "  ${local_path} → container:${container_path}"

  scp "${SSH_ARGS[@]}" \
    "${REPO_ROOT}/${local_path}" \
    "${SSH_USER}@${REMOTE_IP}:${remote_tmp}"

  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker exec ${CONTAINER_NAME} mkdir -p ${container_dir} \
     && sudo docker cp ${remote_tmp} ${CONTAINER_NAME}:${container_path} \
     && rm -f ${remote_tmp}" </dev/null
done 3<<< "${CHANGED_FILES}"
# ^ Both changes matter: `ssh` inside the loop must not read stdin (</dev/null),
# and the loop must read from fd 3 (not 0), so even a stdin-consuming command
# that slips in later can't silently truncate the file list. Without these, ssh
# eats the remaining lines of CHANGED_FILES via stdin, causing the loop to
# process only the first file and exit early with no error.

echo ""
echo "Restarting container ${CONTAINER_NAME} ..."
ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" "sudo docker restart ${CONTAINER_NAME}"
echo "Done."
