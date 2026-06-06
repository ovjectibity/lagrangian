#!/usr/bin/env bash
# Generate TypeScript and Python types from Rust source via typeshare.
# Run from the project root: ./scripts/gen_types.sh
#
# TypeScript targets use only client-facing commondef files; internal worker
# message types (remoteworkermessages.rs, workermessages.rs) are excluded
# to avoid duplicate interface names and unnecessary noise in the SDK.
#
# Python targets mirror the worker's commondef package:
#   - training.py  ← training.rs only
#   - remoteworkermessages.py ← remoteworkermessages.rs + training.rs
set -euo pipefail

TS_INCLUDE_FILES=(
  "app/src/commondef/clientmessages.rs"
  "app/src/commondef/model.rs"
  "app/src/commondef/experiments.rs"
  "app/src/commondef/training.rs"
  "app/src/commondef/testing.rs"
)

TS_TARGETS=(
  "client/src/common/types.ts"
  "sdk/src/types.ts"
)

# --- TypeScript targets ---
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for f in "${TS_INCLUDE_FILES[@]}"; do
  cp "$f" "$tmpdir/"
done

for target in "${TS_TARGETS[@]}"; do
  typeshare "$tmpdir" --lang=typescript --output-file="$target"
  echo "Generated: $target"
done

# --- Python: training.py (training types only) ---
py_training_dir=$(mktemp -d)
trap 'rm -rf "$py_training_dir"' EXIT
cp "app/src/commondef/training.rs" "$py_training_dir/"
typeshare "$py_training_dir" --lang=python \
  --output-file="worker/trainer/trainer-py/commondef/training.py"
echo "Generated: worker/trainer/trainer-py/commondef/training.py"

# --- Python: remoteworkermessages.py (remote worker protocol + training types) ---
py_remote_dir=$(mktemp -d)
trap 'rm -rf "$py_remote_dir"' EXIT
cp "app/src/commondef/remoteworkermessages.rs" "$py_remote_dir/"
cp "app/src/commondef/training.rs" "$py_remote_dir/"
typeshare "$py_remote_dir" --lang=python \
  --output-file="worker/trainer/trainer-py/commondef/remoteworkermessages.py"
echo "Generated: worker/trainer/trainer-py/commondef/remoteworkermessages.py"

echo "Done."
