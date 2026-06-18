#!/usr/bin/env bash
set -euo pipefail

PI0_DIR="${PI0_CHECKPOINT_DIR:-/opt/lagrangian/pi0}"
HF_REPO="lerobot/pi0_libero_base"
FILES=(model.safetensors config.json)

mkdir -p "${PI0_DIR}"

all_present=1
for f in "${FILES[@]}"; do
  if [[ ! -f "${PI0_DIR}/${f}" ]]; then
    all_present=0
    break
  fi
done

if [[ "${all_present}" -eq 1 ]]; then
  echo "Pi-0 model already present at ${PI0_DIR}, skipping download."
  exit 0
fi

echo "Pi-0: installing huggingface_hub ..."
python3 -m pip install -q huggingface_hub

echo "Pi-0: downloading ${HF_REPO} to ${PI0_DIR} ..."
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['model.safetensors', 'config.json']:
    print(f'Downloading {f} ...')
    hf_hub_download(repo_id='${HF_REPO}', filename=f, local_dir='${PI0_DIR}')
    print(f'Done: {f}')
"

echo "Pi-0 model ready at ${PI0_DIR}"
