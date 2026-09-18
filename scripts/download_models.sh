#!/usr/bin/env bash
# Download SAM 3D Objects weights to the shared data mount; never bake them into the image.
set -euo pipefail
cd "$(dirname "$0")/.."

sam3_target=/data/checkpoints/sam3
sam3d_target=/data/checkpoints/sam3d/hf
docker compose run --rm dev sam3d-env sam3d bash -c "
  set -euo pipefail
  test ! -e '$sam3d_target' || { echo '$sam3d_target already exists; refusing to overwrite it.' >&2; exit 1; }
  mkdir -p /data/checkpoints/sam3d/_download
  hf download --repo-type model --local-dir /data/checkpoints/sam3d/_download --max-workers 1 facebook/sam-3d-objects
  test -d /data/checkpoints/sam3d/_download/checkpoints
  mv /data/checkpoints/sam3d/_download/checkpoints '$sam3d_target'
  rm -rf /data/checkpoints/sam3d/_download
  hf download --repo-type model --local-dir /data/checkpoints/sam3d/moge-vitl --max-workers 1 Ruicheng/moge-vitl
  test -f /data/checkpoints/sam3d/moge-vitl/model.pt
"
docker compose run --rm dev sam3d-env core bash -c "
  set -euo pipefail
  test ! -e '$sam3_target' || { echo '$sam3_target already exists; refusing to overwrite it.' >&2; exit 1; }
  hf download --repo-type model --local-dir '$sam3_target' --max-workers 1 facebook/sam3
  test -f '$sam3_target/sam3.pt'
"
echo "SAM3 and SAM 3D Objects checkpoints are available under /data/checkpoints"
