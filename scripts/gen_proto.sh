#!/usr/bin/env bash
# Generate protobuf bindings for the testing-update wire format from the
# single source-of-truth .proto. Run from the project root: ./scripts/gen_proto.sh
#
# Targets:
#   - Python worker: worker/worker-py/commondef/testing_updates_pb2.py
#   - TypeScript SDK: sdk/src/proto/testing.{js,d.ts}
#
# Requires: grpcio-tools in the scripts venv (protoc for Python codegen — NOT a
# worker runtime dependency) and protobufjs-cli in the SDK node env (pbjs/pbts,
# pure JS — no protoc needed). The worker only needs the `protobuf` runtime.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$REPO_ROOT/docs/user/specs"
PROTO_FILE="$PROTO_DIR/testing_updates.proto"
GEN_PYTHON="$REPO_ROOT/scripts/.venv/bin/python"

# --- Python (worker) ---
PY_OUT="$REPO_ROOT/worker/worker-py/commondef"
echo "Generating Python protobuf -> $PY_OUT"
"$GEN_PYTHON" -m grpc_tools.protoc \
  -I "$PROTO_DIR" \
  --python_out="$PY_OUT" \
  "$PROTO_FILE"

# --- TypeScript (SDK) ---
SDK_DIR="$REPO_ROOT/sdk"
TS_OUT="$SDK_DIR/src/proto"
mkdir -p "$TS_OUT"
echo "Generating TypeScript protobuf -> $TS_OUT"
(
  cd "$SDK_DIR"
  npx pbjs -t static-module -w es6 -o "$TS_OUT/testing.js" "$PROTO_FILE"
  npx pbts -o "$TS_OUT/testing.d.ts" "$TS_OUT/testing.js"
)

echo "Done."
