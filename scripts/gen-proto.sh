#!/usr/bin/env sh
# Regenerate the TypeScript client and Python exporter bindings from proto/.
set -eu
cd "$(dirname "$0")/.."

protoc \
  --proto_path=proto \
  --plugin=protoc-gen-es=scripts/protoc-gen-es \
  --es_out=src/gen \
  --es_opt=target=ts \
  --python_out=export \
  natlas/v1/natlas.proto \
  natlas/v2/natlas.proto \
  natlas/partition/v1/partition.proto
