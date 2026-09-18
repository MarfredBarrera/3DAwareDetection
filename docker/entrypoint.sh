#!/usr/bin/env bash
set -euo pipefail
mkdir -p /data/home /data/cache /data/checkpoints /data/objaverse /outputs
exec "$@"
