#!/usr/bin/env bash
# Packs the Linux release archive: scaleup-<version>-<target>.tar.gz with both binaries, the ONNX
# Runtime provider libraries that `ort` copied next to them, LICENSE, README.md and models/licenses/.
# Usage: packaging/tar_linux.sh <version> <target> [bin-dir] [out-dir]
set -euo pipefail
version="$1"
target="$2"
bin_dir="${3:-target/$target/release}"
out_dir="${4:-dist}"
name="scaleup-$version-$target"
stage="$(mktemp -d)/$name"
mkdir -p "$stage/models/licenses" "$out_dir"

cp -L "$bin_dir/scaleup" "$bin_dir/scaleup-app" "$stage/"
# `-L` dereferences the symlinks ort creates with the copy-dylibs feature.
shopt -s nullglob
for lib in "$bin_dir"/libonnxruntime*.so*; do
  cp -L "$lib" "$stage/"
done
shopt -u nullglob
cp LICENSE README.md "$stage/"
cp models/licenses/*.txt "$stage/models/licenses/"

tar -C "$(dirname "$stage")" -czf "$out_dir/$name.tar.gz" "$name"
rm -rf "$(dirname "$stage")"
echo "wrote $out_dir/$name.tar.gz"
tar -tzf "$out_dir/$name.tar.gz"
