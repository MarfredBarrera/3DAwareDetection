#!/usr/bin/env bash
set -euo pipefail
repo=$1 ref=$2 dest=$3
git init "$dest"
git -C "$dest" remote add origin "https://github.com/$repo.git"
git -C "$dest" fetch --depth=1 origin "$ref"
git -C "$dest" checkout --detach FETCH_HEAD
test "$(git -C "$dest" rev-parse HEAD)" = "$ref"
git -C "$dest" submodule update --init --recursive --depth=1
