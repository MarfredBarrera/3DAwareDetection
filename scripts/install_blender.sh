#!/usr/bin/env bash
# Run inside the dev container. Keep Blender outside the image and repo checkout.
set -euo pipefail

version=4.5.14
archive="blender-${version}-linux-x64.tar.xz"
sha256=9ba871ff2ecd36526b77432745980b7e6664ecd0c7ca11c48849073dcfe06da3
root=/data/cache/blender
base="https://download.blender.org/release/Blender4.5"
mkdir -p "$root"
if [[ -x "$root/blender-${version}-linux-x64/blender" ]]; then
    "$root/blender-${version}-linux-x64/blender" --version | head -1
    exit 0
fi
curl -fL --retry 3 "$base/$archive" -o "$root/$archive"
printf '%s  %s\n' "$sha256" "$archive" | (cd "$root" && sha256sum -c -)
tar -xf "$root/$archive" -C "$root"
"$root/blender-${version}-linux-x64/blender" --version | head -1
