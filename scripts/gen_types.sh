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

# --- Python: one generated module per Rust source file ---
# Each module only contains its own types; cross-file references are resolved via imports.
# typeshare uses cross-file type names as-is (no definition generated for them), so the
# prepended imports bring those names into scope at runtime.
#
# Dependency chain: record.py <- training.py <- testing.py <- remoteworkermessages.py

# record.py — standalone (no cross-file deps among #[typeshare] types)
py_record_dir=$(mktemp -d)
trap 'rm -rf "$py_record_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/record.rs" "$py_record_dir/"
typeshare "$py_record_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/record.py"
# typeshare serialized_as = "string" is correct for TypeScript but emits
# the literal name "string" in Python, which is not a built-in. Replace
# with "str" — the correct Python type — after generation.
python3 -c "
with open('$REPO_ROOT/worker/worker-py/commondef/record.py') as f:
    content = f.read()
content = content.replace(' = string\n', ' = str\n')
with open('$REPO_ROOT/worker/worker-py/commondef/record.py', 'w') as f:
    f.write(content)
"
echo "Generated: worker/worker-py/commondef/record.py"

# training.py — imports CheckpointReference from record.py
py_training_dir=$(mktemp -d)
trap 'rm -rf "$py_training_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/training.rs" "$py_training_dir/"
typeshare "$py_training_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/training.py"
python3 -c "
with open('$REPO_ROOT/worker/worker-py/commondef/training.py') as f:
    content = f.read()
marker = 'from __future__ import annotations'
content = content.replace(marker, marker + '\n\nfrom commondef.record import *', 1)
with open('$REPO_ROOT/worker/worker-py/commondef/training.py', 'w') as f:
    f.write(content)
"
echo "Generated: worker/worker-py/commondef/training.py"

# testing.py — imports training types (which transitively includes record types)
py_testing_dir=$(mktemp -d)
trap 'rm -rf "$py_testing_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/testing.rs" "$py_testing_dir/"
typeshare "$py_testing_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/testing.py"
python3 -c "
with open('$REPO_ROOT/worker/worker-py/commondef/testing.py') as f:
    content = f.read()
marker = 'from __future__ import annotations'
content = content.replace(marker, marker + '\n\nfrom commondef.training import *', 1)
with open('$REPO_ROOT/worker/worker-py/commondef/testing.py', 'w') as f:
    f.write(content)
"
echo "Generated: worker/worker-py/commondef/testing.py"

# remoteworkermessages.py — imports training + testing (which cover all transitive deps)
py_remote_dir=$(mktemp -d)
trap 'rm -rf "$py_remote_dir"' EXIT
cp "$REPO_ROOT/app/src/commondef/remoteworkermessages.rs" "$py_remote_dir/"
typeshare "$py_remote_dir" --lang=python \
  --output-file="$REPO_ROOT/worker/worker-py/commondef/remoteworkermessages.py"
python3 -c "
with open('$REPO_ROOT/worker/worker-py/commondef/remoteworkermessages.py') as f:
    content = f.read()
marker = 'from __future__ import annotations'
imports = '\n\nfrom commondef.training import *\nfrom commondef.testing import *'
content = content.replace(marker, marker + imports, 1)
with open('$REPO_ROOT/worker/worker-py/commondef/remoteworkermessages.py', 'w') as f:
    f.write(content)
"
echo "Generated: worker/worker-py/commondef/remoteworkermessages.py"

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
