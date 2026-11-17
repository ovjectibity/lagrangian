#!/bin/bash
set -euxo pipefail

if command -v yum >/dev/null 2>&1; then
  # The GPU-ready AMI supplies the host NVIDIA driver. Updating the kernel or
  # driver here can leave the loaded kernel module out of sync with NVML until
  # the host is rebooted.
  # yum update -y
  yum install -y docker awscli
  systemctl enable docker
  systemctl start docker
  # NVIDIA driver + container toolkit (Amazon Linux 2 / AL2023)
  # yum install -y gcc kernel-devel-$(uname -r) || true
  # dnf install -y kernel-modules-extra || yum install -y kernel-modules-extra || true
  # yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo || true
  # yum install -y cuda-drivers || true
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "A working NVIDIA driver is required before bootstrap." >&2
    exit 1
  fi
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | tee /etc/yum.repos.d/nvidia-container-toolkit.repo
  yum install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
elif command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y docker.io awscli
  systemctl enable docker
  systemctl start docker
  # NVIDIA driver + container toolkit (Ubuntu/Debian)
  # The GPU-ready image supplies the host NVIDIA driver. Do not replace it
  # during bootstrap because the new module requires a host reboot.
  # apt-get install -y linux-headers-$(uname -r) || true
  # curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
  #   | gpg --dearmor -o /usr/share/keyrings/cuda-keyring.gpg
  # echo "deb [signed-by=/usr/share/keyrings/cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
  #   > /etc/apt/sources.list.d/cuda.list
  # apt-get update -y
  # apt-get install -y cuda-drivers
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "A working NVIDIA driver is required before bootstrap." >&2
    exit 1
  fi
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
else
  echo "Unsupported package manager" >&2
  exit 1
fi

__ECR_LOGIN__
docker pull __IMAGE_URI__
docker rm -f __CONTAINER_NAME__ || true

export HF_TOKEN=__HF_TOKEN__
bash /tmp/download_pi0_model.sh
cat >/etc/lagrangian-worker.env <<'ENV_EOF'
WS_HOST=0.0.0.0
WS_PORT=__WORKER_PORT__
WORKER_SHARED_SECRET=__WORKER_SHARED_SECRET__
WS_PING_TIMEOUT_SECONDS=60
HF_TOKEN=__HF_TOKEN__
CUDA_LAUNCH_BLOCKING=__CUDA_DEBUG_MODE__
CUDA_ENABLE_COREDUMP_ON_EXCEPTION=__CUDA_DEBUG_MODE__
CUDA_COREDUMP_SHOW_PROGRESS=__CUDA_DEBUG_MODE__
CUDA_DEBUG_MODE=__CUDA_DEBUG_MODE__
ENV_EOF

docker run -d --restart unless-stopped \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --name __CONTAINER_NAME__ \
  --env-file /etc/lagrangian-worker.env \
  -v /opt/lagrangian/pi0:/opt/lagrangian/pi0 \
  -p __WORKER_PORT__:__WORKER_PORT__ \
  __IMAGE_URI__

echo "### BOOTSTRAP_DONE ###"
