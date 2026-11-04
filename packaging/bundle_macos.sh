#!/usr/bin/env bash
# Builds the macOS release artefacts from a release build:
#   dist/ScaleUP.app                               app bundle: Info.plist, icon.icns, both binaries, any
#                                                  libonnxruntime*.dylib in Contents/MacOS
#   dist/scaleup-<version>-macos.dmg               compressed disk image of the bundle
#   dist/scaleup-<version>-<target>.tar.gz         archive: scaleup, scaleup-app, dylibs, LICENSE,
#                                                  README.md, models/licenses/
# A second build directory makes the bundle universal (lipo) and adds that target's archive; `ort`
# 2.0.0-rc.15 ships no prebuilt ONNX Runtime for x86_64-apple-darwin, so releases are arm64-only.
# Usage: packaging/bundle_macos.sh <version> <aarch64-bin-dir> [x86_64-bin-dir] [out-dir]
set -euo pipefail
version="$1"
arm="${2:-target/aarch64-apple-darwin/release}"
x86="${3:-}"
out="${4:-dist}"
app="$out/ScaleUP.app"
macos="$app/Contents/MacOS"
res="$app/Contents/Resources"

copy_docs() {
  mkdir -p "$1/models/licenses"
  cp LICENSE README.md "$1/"
  cp models/licenses/*.txt "$1/models/licenses/"
}

# merge <name>: universal when an x86_64 build is present, else the arm64 file.
merge() {
  if [ -n "$x86" ] && [ -f "$x86/$1" ]; then
    lipo -create "$x86/$1" "$arm/$1" -output "$macos/$1"
  else
    cp -L "$arm/$1" "$macos/$1"
  fi
}

rm -rf "$app"
mkdir -p "$macos" "$res" "$out"

# --- app bundle -----------------------------------------------------------------------------------
merge scaleup
merge scaleup-app
chmod 755 "$macos/scaleup" "$macos/scaleup-app"

# ONNX Runtime is linked statically on macOS (pyke binaries); copy any dylib `ort` did produce so the
# bundle keeps working when moved.
shopt -s nullglob
for lib in "$arm"/libonnxruntime*.dylib; do
  merge "$(basename "$lib")"
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
archive() {
  local target="$1" bin_dir="$2" name stage
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
}
archive aarch64-apple-darwin "$arm"
if [ -n "$x86" ]; then
  archive x86_64-apple-darwin "$x86"
fi

echo "bundle contents:"
find "$app" -type f | sort
lipo -info "$macos/scaleup"
ls -la "$out"
