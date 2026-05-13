#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

APP_DIR="target/release/SwarmX.app"
BINARY="target/release/swarmx-desktop"
RESOURCES="apps/desktop/resources"

cargo build --release -p swarmx-desktop

rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$BINARY" "$APP_DIR/Contents/MacOS/"
cp "$RESOURCES/Info.plist" "$APP_DIR/Contents/"
cp "$RESOURCES/AppIcon.icns" "$APP_DIR/Contents/Resources/"

echo "Bundle: $APP_DIR"
echo "Run: open $APP_DIR"
