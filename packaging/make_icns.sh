#!/usr/bin/env bash
# Builds assets/icon.icns from crates/scaleup-app/assets/icon.png with the macOS `iconutil`.
# Usage: packaging/make_icns.sh [src.png] [out.icns]
set -euo pipefail
src="${1:-crates/scaleup-app/assets/icon.png}"
out="${2:-assets/icon.icns}"
tmp="$(mktemp -d)"
set="$tmp/icon.iconset"
mkdir -p "$set"
for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$src" --out "$set/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$src" --out "$set/icon_${size}x${size}@2x.png" >/dev/null
done
mkdir -p "$(dirname "$out")"
iconutil -c icns "$set" -o "$out"
rm -rf "$tmp"
echo "wrote $out"
