#!/usr/bin/env bash
set -euo pipefail

PI0_DIR="${PI0_CHECKPOINT_DIR:-/opt/lagrangian/pi0}"
HF_REPO="lerobot/pi0_libero_base"
BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"
FILES=(model.safetensors config.json)

mkdir -p "${PI0_DIR}"

for f in "${FILES[@]}"; do
  if [[ -f "${PI0_DIR}/${f}" ]]; then
    echo "Pi-0: ${f} already present, skipping."
  else
    echo "Pi-0: downloading ${f} ..."
    wget -q --show-progress -O "${PI0_DIR}/${f}.tmp" "${BASE_URL}/${f}"
    mv "${PI0_DIR}/${f}.tmp" "${PI0_DIR}/${f}"
  fi
done

echo "Pi-0 model ready at ${PI0_DIR}"
