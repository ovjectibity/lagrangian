#!/usr/bin/env bash
# Generate TypeScript and Python types from Rust source via typeshare.
# Run from the project root: ./scripts/gen_types.sh
#
# TypeScript targets use only client-facing commondef files; internal worker
# message types (remoteworkermessages.rs, workermessages.rs) are excluded
# to avoid duplicate interface names and unnecessary noise in the SDK.
#
# Python targets mirror the worker's commondef package:
#   - training.py  ← re-export shim (training types inlined into remoteworkermessages.py)
#   - remoteworkermessages.py ← remoteworkermessages.rs + training.rs
#   - testing.py ← testing.rs + training.rs (needed for TestGraph types)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# libtorch is required by gen_mongo_schema (via the lagrangian app crate / tch-rs).
# Source .envrc if present so LIBTORCH etc. are set, then fall back to a default.
if [[ -f "$REPO_ROOT/.envrc" ]]; then
  set -a
  source "$REPO_ROOT/.envrc"
  set +a
fi
export LIBTORCH="${LIBTORCH:-/Users/avi/Documents/projects/libtorch}"
export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

TS_INCLUDE_FILES=(
  "$REPO_ROOT/app/src/commondef/clientmessages.rs"
  "$REPO_ROOT/app/src/commondef/model.rs"
  "$REPO_ROOT/app/src/commondef/record.rs"
  "$REPO_ROOT/app/src/commondef/training.rs"
  "$REPO_ROOT/app/src/commondef/testing.rs"
)

TS_TARGETS=(
  "$REPO_ROOT/client/src/common/types.ts"
  "$REPO_ROOT/sdk/src/types.ts"
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

# --- Python: training.py (re-export shim) ---
echo "# Auto-generated re-export shim — do not edit. Run scripts/gen_types.sh to regenerate.
from commondef.remoteworkermessages import *" \
  > "$REPO_ROOT/worker/worker-py/commondef/training.py"
echo "Generated: worker/worker-py/commondef/training.py (re-export shim)"

# --- Python: remoteworkermessages.py (remote worker protocol + training types) ---
py_remote_dir=$(mktemp -d)
trap 'rm -rf "$py_remote_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/remoteworkermessages.rs" "$py_remote_dir/"
cp "$REPO_ROOT/app/src/commondef/training.rs" "$py_remote_dir/"
cp "$REPO_ROOT/app/src/commondef/testing.rs" "$py_remote_dir/"
typeshare "$py_remote_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/remoteworkermessages.py"
echo "Generated: worker/worker-py/commondef/remoteworkermessages.py"

# --- Python: testing.py (testing types for remote worker) ---
py_testing_dir=$(mktemp -d)
trap 'rm -rf "$py_testing_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/testing.rs" "$py_testing_dir/"
cp "$REPO_ROOT/app/src/commondef/training.rs" "$py_testing_dir/"
typeshare "$py_testing_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/testing.py"
echo "Generated: worker/worker-py/commondef/testing.py"

# --- MongoDB JSON Schemas ---
mkdir -p "$REPO_ROOT/schemas"

GEN_SCHEMA="cargo run --manifest-path $REPO_ROOT/scripts/gen_mongo_schema/Cargo.toml -q --"

$GEN_SCHEMA training_graph > "$REPO_ROOT/schemas/training_graph.json"
echo "Generated: schemas/training_graph.json"

$GEN_SCHEMA log_stream > "$REPO_ROOT/schemas/log_stream.json"
echo "Generated: schemas/log_stream.json"

$GEN_SCHEMA metrics_stream > "$REPO_ROOT/schemas/metrics_stream.json"
echo "Generated: schemas/metrics_stream.json"

$GEN_SCHEMA test_graph > "$REPO_ROOT/schemas/test_graph.json"
echo "Generated: schemas/test_graph.json"

echo "Done."
