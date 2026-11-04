#!/usr/bin/env bash
# Builds the macOS release artefacts from two release builds (x86_64 and aarch64):
#   dist/ScaleUP.app                               universal app bundle: Info.plist, icon.icns, both
#                                                  binaries lipo-merged, any libonnxruntime*.dylib in
#                                                  Contents/MacOS
#   dist/scaleup-<version>-macos.dmg               compressed disk image of the bundle
#   dist/scaleup-<version>-<target>.tar.gz         per-target archive (scaleup, scaleup-app, dylibs,
#                                                  LICENSE, README.md, models/licenses/) for both targets
# Usage: packaging/bundle_macos.sh <version> [x86_64-bin-dir] [aarch64-bin-dir] [out-dir]
set -euo pipefail
version="$1"
x86="${2:-target/x86_64-apple-darwin/release}"
arm="${3:-target/aarch64-apple-darwin/release}"
out="${4:-dist}"
app="$out/ScaleUP.app"
macos="$app/Contents/MacOS"
res="$app/Contents/Resources"

copy_docs() {
  mkdir -p "$1/models/licenses"
  cp LICENSE README.md "$1/"
  cp models/licenses/*.txt "$1/models/licenses/"
}

rm -rf "$app"
mkdir -p "$macos" "$res" "$out"

# --- universal app bundle -------------------------------------------------------------------------
lipo -create "$x86/scaleup" "$arm/scaleup" -output "$macos/scaleup"
lipo -create "$x86/scaleup-app" "$arm/scaleup-app" -output "$macos/scaleup-app"
chmod 755 "$macos/scaleup" "$macos/scaleup-app"

# ONNX Runtime is linked statically on macOS (pyke binaries); copy any dylib `ort` did produce so the
# bundle keeps working when moved, merging per-arch copies into a universal one.
shopt -s nullglob
for lib in "$arm"/libonnxruntime*.dylib; do
  base="$(basename "$lib")"
  if [ -f "$x86/$base" ]; then
    lipo -create "$x86/$base" "$lib" -output "$macos/$base"
  else
    cp -L "$lib" "$macos/$base"
  fi
done
shopt -u nullglob

sed "s/SCALEUP_VERSION/$version/g" packaging/Info.plist > "$app/Contents/Info.plist"
printf 'APPL????' > "$app/Contents/PkgInfo"
if [ ! -f assets/icon.icns ]; then
  packaging/make_icns.sh crates/scaleup-app/assets/icon.png assets/icon.icns
fi
cp assets/icon.icns "$res/icon.icns"
copy_docs "$res"

# Ad-hoc signature so Gatekeeper reports "unidentified developer" rather than "damaged".
codesign --force --deep --sign - "$app"

hdiutil create -volname ScaleUP -srcfolder "$app" -ov -format UDZO "$out/scaleup-$version-macos.dmg"

# --- per-target archives --------------------------------------------------------------------------
for pair in "x86_64-apple-darwin:$x86" "aarch64-apple-darwin:$arm"; do
  target="${pair%%:*}"
  bin_dir="${pair#*:}"
  name="scaleup-$version-$target"
  stage="$(mktemp -d)/$name"
  mkdir -p "$stage"
  cp -L "$bin_dir/scaleup" "$bin_dir/scaleup-app" "$stage/"
  shopt -s nullglob
  for lib in "$bin_dir"/libonnxruntime*.dylib; do
    cp -L "$lib" "$stage/"
  done
  shopt -u nullglob
  copy_docs "$stage"
  tar -C "$(dirname "$stage")" -czf "$out/$name.tar.gz" "$name"
  rm -rf "$(dirname "$stage")"
done

echo "bundle contents:"
find "$app" -type f | sort
lipo -info "$macos/scaleup"
ls -la "$out"
