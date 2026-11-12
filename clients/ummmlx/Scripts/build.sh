#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
CONFIG=${1:-release}
cd "$ROOT"
swift build -c "$CONFIG"
BIN=$(swift build -c "$CONFIG" --show-bin-path)
APP="$ROOT/build/ummmlx.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN/ummmlx" "$APP/Contents/MacOS/ummmlx"
cp "$ROOT/Support/Info.plist" "$APP/Contents/Info.plist"
cp "$ROOT/../../LICENSE" "$APP/Contents/Resources/LICENSE"
cp "$ROOT/ThirdParty/Voxt/NOTICE" "$APP/Contents/Resources/Voxt-NOTICE"
printf '%s\n' "$APP"
