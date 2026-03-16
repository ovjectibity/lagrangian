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

# ---- Validate required env vars ----
for var in TRAINING_WORKER_AMI_ID TRAINING_WORKER_SECURITY_GROUP_IDS \
           TRAINING_WORKER_IAM_INSTANCE_PROFILE_ARN TRAINING_WORKER_IMAGE_URI \
           TRAINING_WORKER_CONTAINER_NAME TRAINING_WORKER_PORT \
           TRAINING_WORKER_SHARED_SECRET; do
  if [[ -z "${!var:-}" ]]; then
    echo "error: ${var} is not set." >&2
    exit 1
  fi
done

REGION="${TRAINING_WORKER_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
AMI_ID="${TRAINING_WORKER_AMI_ID}"
SECURITY_GROUP_IDS="${TRAINING_WORKER_SECURITY_GROUP_IDS}"
IAM_INSTANCE_PROFILE_ARN="${TRAINING_WORKER_IAM_INSTANCE_PROFILE_ARN}"
KEY_NAME="${TRAINING_WORKER_KEY_NAME:-}"
INSTANCE_NAME="${TRAINING_WORKER_INSTANCE_NAME_TAG:-lagrangian-remote-worker}"
INSTANCE_TYPE="c7i-flex.large"
VOLUME_SIZE=60

BOOTSTRAP_SCRIPT="${SCRIPT_DIR}/bootstrap_training_worker.sh"
if [[ ! -f "${BOOTSTRAP_SCRIPT}" ]]; then
  echo "error: bootstrap script not found at ${BOOTSTRAP_SCRIPT}" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH." >&2
  exit 1
fi

# ---- Launch instance ----
echo "Launching ${INSTANCE_TYPE} instance in ${REGION} ..."

LAUNCH_ARGS=(
  --region "${REGION}"
  --image-id "${AMI_ID}"
  --instance-type "${INSTANCE_TYPE}"
  --security-group-ids "${SECURITY_GROUP_IDS}"
  --iam-instance-profile "Arn=${IAM_INSTANCE_PROFILE_ARN}"
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOLUME_SIZE},\"VolumeType\":\"gp3\"}}]"
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

# ---- Update security group inbound rules for local machine's public IP ----
LOCAL_IP="$(curl -sf https://checkip.amazonaws.com)"
if [[ -z "${LOCAL_IP}" ]]; then
  echo "error: could not determine local public IP." >&2
  exit 1
fi
echo "Local public IP: ${LOCAL_IP}"

# Use the first security group ID (the one from .envrc)
SG_ID="${SECURITY_GROUP_IDS%%,*}"

echo "Updating security group ${SG_ID} inbound rules ..."
for PORT in 22 "${TRAINING_WORKER_PORT}"; do
  # Revoke any existing rule for this port (ignore errors if none exists)
  EXISTING_CIDRS="$(
    aws ec2 describe-security-groups \
      --region "${REGION}" \
      --group-ids "${SG_ID}" \
      --query "SecurityGroups[0].IpPermissions[?FromPort==\`${PORT}\` && ToPort==\`${PORT}\` && IpProtocol=='tcp'].IpRanges[].CidrIp" \
      --output text
  )"
  for CIDR in ${EXISTING_CIDRS}; do
    aws ec2 revoke-security-group-ingress \
      --region "${REGION}" \
      --group-id "${SG_ID}" \
      --protocol tcp \
      --port "${PORT}" \
      --cidr "${CIDR}" 2>/dev/null || true
  done

  aws ec2 authorize-security-group-ingress \
    --region "${REGION}" \
    --group-id "${SG_ID}" \
    --protocol tcp \
    --port "${PORT}" \
    --cidr "${LOCAL_IP}/32"
  echo "  port ${PORT}: allowed ${LOCAL_IP}/32"
done

# ---- Wait for SSH to become available ----
echo "Waiting for SSH to become available ..."
for i in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "ubuntu@${PUBLIC_IP}" true 2>/dev/null; then
    break
  fi
  if [[ "${i}" -eq 30 ]]; then
    echo "error: SSH did not become available after 30 attempts." >&2
    exit 1
  fi
  echo "  attempt ${i}/30 ..."
  sleep 10
done

# ---- Transfer bootstrap script ----
echo "Transferring bootstrap script ..."
scp -o StrictHostKeyChecking=no \
  "${BOOTSTRAP_SCRIPT}" \
  "ubuntu@${PUBLIC_IP}:/tmp/bootstrap_training_worker.sh"

# ---- Run bootstrap on instance ----
echo "Running bootstrap on instance ..."
ssh -o StrictHostKeyChecking=no "ubuntu@${PUBLIC_IP}" bash -s <<EOF
set -euo pipefail

AWS_REGION="${REGION}"
IMAGE_URI="${TRAINING_WORKER_IMAGE_URI}"
CONTAINER_NAME="${TRAINING_WORKER_CONTAINER_NAME}"
WORKER_PORT="${TRAINING_WORKER_PORT}"
WORKER_SHARED_SECRET="${TRAINING_WORKER_SHARED_SECRET}"

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
" /tmp/bootstrap_training_worker.sh > /tmp/bootstrap_training_worker.rendered.sh

chmod +x /tmp/bootstrap_training_worker.rendered.sh
sudo /tmp/bootstrap_training_worker.rendered.sh
EOF

echo "Done. Instance ${INSTANCE_ID} is up at ${PUBLIC_IP}"
