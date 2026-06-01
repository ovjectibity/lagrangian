#!/usr/bin/env bash
# Generate TypeScript types from Rust source via typeshare.
# Run from the project root: ./scripts/gen_types.sh
#
# Only the client-facing commondef files are included; internal worker
# message types (remoteworkermessages.rs, workermessages.rs) are excluded
# to avoid duplicate interface names and unnecessary noise in the SDK.
set -euo pipefail

INCLUDE_FILES=(
  "app/src/commondef/clientmessages.rs"
  "app/src/commondef/model.rs"
  "app/src/commondef/experiments.rs"
  "app/src/commondef/training.rs"
  "app/src/commondef/testing.rs"
)

TARGETS=(
  "client/src/common/types.ts"
  "sdk/src/types.ts"
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for f in "${INCLUDE_FILES[@]}"; do
  cp "$f" "$tmpdir/"
done

for target in "${TARGETS[@]}"; do
  typeshare "$tmpdir" --lang=typescript --output-file="$target"
  echo "Generated: $target"
done

echo "Done."
