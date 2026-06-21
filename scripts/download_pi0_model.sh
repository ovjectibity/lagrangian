#!/usr/bin/env bash
set -euo pipefail

PI0_DIR="${PI0_CHECKPOINT_DIR:-/opt/lagrangian/pi0}"
HF_REPO="lerobot/pi0_libero_base"
LEROBOT_FILES=(model.safetensors config.json policy_preprocessor.json policy_postprocessor.json)
GATED_FILES=(tokenizer.model)  # from google/paligemma-3b-pt-224, requires HF_TOKEN

mkdir -p "${PI0_DIR}"

all_present=1
for f in "${LEROBOT_FILES[@]}" "${GATED_FILES[@]}"; do
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
for f in ['model.safetensors', 'config.json', 'policy_preprocessor.json', 'policy_postprocessor.json']:
    print(f'Downloading {f} ...')
    hf_hub_download(repo_id='${HF_REPO}', filename=f, local_dir='${PI0_DIR}')
    print(f'Done: {f}')
"

echo "Pi-0: downloading tokenizer from google/paligemma-3b-pt-224 ..."
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN is required to download the tokenizer from the gated paligemma repo." >&2
  exit 1
fi
python3 -c "
from huggingface_hub import hf_hub_download
import os
print('Downloading tokenizer.model ...')
hf_hub_download(repo_id='google/paligemma-3b-pt-224', filename='tokenizer.model',
                local_dir='${PI0_DIR}', token=os.environ['HF_TOKEN'])
print('Done: tokenizer.model')
"

echo "Pi-0 model ready at ${PI0_DIR}"
