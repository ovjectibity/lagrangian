#!/usr/bin/env bash
# Generate protobuf bindings for the testing-update wire format from the
# single source-of-truth .proto. Run from the project root: ./scripts/typegen/gen_proto.sh
#
# Target: worker/worker-py/commondef/testing_updates_pb2.py
#
# Requires grpcio-tools in the scripts venv for Python codegen.
# The worker only needs the protobuf runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROTO_DIR="$REPO_ROOT/docs/user/async_spec"
PROTO_FILE="$PROTO_DIR/testing_updates.proto"
GEN_PYTHON="$REPO_ROOT/scripts/.venv/bin/python"

# --- Python (worker) ---
PY_OUT="$REPO_ROOT/worker/worker-py/commondef"
echo "Generating Python protobuf -> $PY_OUT"
"$GEN_PYTHON" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PY_OUT" \
  "$PROTO_FILE"

echo "Done."
