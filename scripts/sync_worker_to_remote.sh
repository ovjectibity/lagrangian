#!/usr/bin/env bash
# Syncs locally changed worker and vendored simulator files to a remote EC2
# instance and restarts the worker container. Useful for iterating without a
# full ECR push. Changes are lost when the container is recreated from its image.
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

WORKER_ROOT="/opt/lagrangian/worker"
THIRD_PARTY_ROOT="/opt/lagrangian/third-party"
SITE_PACKAGES_ROOT="${WORKER_ROOT}/.venv/lib/python3.14/site-packages"
NEEDS_UV_SYNC=0
SYNCED_FILES=0

changed_files() {
  local repository="$1"

  {
    git -C "${repository}" diff --name-only HEAD
    git -C "${repository}" ls-files --others --exclude-standard
  } | sort -u
}

copy_file() {
  local local_path="$1"
  local container_path="$2"
  local remote_tmp

  remote_tmp="/tmp/_worker_sync_$(echo "${container_path}" | tr '/' '__')"

  scp "${SSH_ARGS[@]}" \
    "${local_path}" \
    "${SSH_USER}@${REMOTE_IP}:${remote_tmp}"

  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker cp ${remote_tmp} ${CONTAINER_NAME}:${container_path} && rm -f ${remote_tmp}" </dev/null
}

restart_and_wait() {
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker restart ${CONTAINER_NAME} >/dev/null \
     && for attempt in \$(seq 1 30); do \
          if [ \"\$(sudo docker container inspect --format '{{.State.Running}}' ${CONTAINER_NAME})\" = true ]; then exit 0; fi; \
          sleep 1; \
        done; \
        echo \"error: container '${CONTAINER_NAME}' did not stay running after the file sync.\" >&2; \
        exit 1" </dev/null
}

sync_worker_file() {
  local relative_path="$1"
  local local_path="${REPO_ROOT}/worker/worker-py/${relative_path}"

  if [[ ! -f "${local_path}" ]]; then
    echo "  skipping deleted worker file: ${relative_path}"
    return
  fi

  echo "  worker/worker-py/${relative_path}"
  copy_file "${local_path}" "${WORKER_ROOT}/${relative_path}"
  SYNCED_FILES=1

  if [[ "${relative_path}" == "pyproject.toml" || "${relative_path}" == "uv.lock" ]]; then
    NEEDS_UV_SYNC=1
  fi
}

sync_third_party_file() {
  local package_name="$1"
  local relative_path="$2"
  local local_path="${REPO_ROOT}/worker/third-party/${package_name}/${relative_path}"
  local installed_path=""

  if [[ ! -f "${local_path}" ]]; then
    echo "  skipping deleted ${package_name} file: ${relative_path}"
    return
  fi

  echo "  third-party/${package_name}/${relative_path}"
  copy_file "${local_path}" "${THIRD_PARTY_ROOT}/${package_name}/${relative_path}"
  SYNCED_FILES=1

  case "${package_name}:${relative_path}" in
    libero:libero/libero/*)
      installed_path="${SITE_PACKAGES_ROOT}/libero/libero/${relative_path#libero/libero/}"
      ;;
    robosuite:robosuite/*)
      installed_path="${SITE_PACKAGES_ROOT}/robosuite/${relative_path#robosuite/}"
      ;;
    libero:setup.py|libero:pyproject.toml|robosuite:setup.py|robosuite:pyproject.toml)
      NEEDS_UV_SYNC=1
      ;;
  esac

  if [[ -n "${installed_path}" ]]; then
    copy_file "${local_path}" "${installed_path}"
  fi
}

echo "Files to sync to ${REMOTE_IP}:"
while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_worker_file "${relative_path}"
done < <(changed_files "${REPO_ROOT}" | sed -n 's#^worker/worker-py/##p')

while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_third_party_file "libero" "${relative_path}"
done < <(changed_files "${REPO_ROOT}/worker/third-party/libero")

while IFS= read -r relative_path; do
  [[ -n "${relative_path}" ]] && sync_third_party_file "robosuite" "${relative_path}"
done < <(changed_files "${REPO_ROOT}/worker/third-party/robosuite")

if [[ "${SYNCED_FILES}" -eq 0 ]]; then
  echo "No changed worker, LIBERO, or robosuite files to sync."
  exit 0
fi

if [[ "${NEEDS_UV_SYNC}" -eq 1 ]]; then
  echo "Restarting container ${CONTAINER_NAME} before dependency synchronization ..."
  restart_and_wait
  echo "Synchronizing changed package metadata and dependencies ..."
  ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" \
    "sudo docker exec ${CONTAINER_NAME} sh -lc 'cd ${WORKER_ROOT} && CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked --no-dev --no-install-project'" </dev/null
fi

echo ""
echo "Restarting container ${CONTAINER_NAME} ..."
restart_and_wait
# ssh "${SSH_ARGS[@]}" "${SSH_USER}@${REMOTE_IP}" "sudo docker stop ${CONTAINER_NAME}" && "sudo docker rm ${CONTAINER_NAME}" && "sudo docker ${CONTAINER_NAME}"
echo "Done."
