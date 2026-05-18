#!/bin/bash
# Fix framework MinimumOSVersion to match binary minos values.
# This resolves App Store error 90208 ("does not support the minimum OS Version
# specified in the Info.plist") caused by prebuilt frameworks from MediaPipe/TensorFlow.
#
# Usage: Run this after `flutter build ipa` but BEFORE uploading to App Store Connect.
#   ./ios/fix_framework_plists.sh
#
# It will:
#   1. Fix Info.plist MinimumOSVersion in each framework to match binary LC_BUILD_VERSION minos
#   2. Re-sign the frameworks and app
#   3. Re-export the IPA to build/ios/ipa-fixed/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ARCHIVE="$PROJECT_DIR/build/ios/archive/Runner.xcarchive"
APP="$ARCHIVE/Products/Applications/Runner.app"
FW_DIR="$APP/Frameworks"
# Team and signing identity are env-overridable so the same script works
# for both publishers on this repo (Yao on V9Q67SYWWQ, Yunxiang on
# DJD849Y8Q6). Default = the original collaborator (Yao). Override with:
#   TEAM_ID=DJD849Y8Q6 SIGNING_IDENTITY="Apple Distribution: Yunxiang Yan (DJD849Y8Q6)" ./ios/fix_framework_plists.sh
TEAM_ID="${TEAM_ID:-V9Q67SYWWQ}"
SIGNING_IDENTITY="${SIGNING_IDENTITY:-Apple Distribution: YAO XIAO ($TEAM_ID)}"

if [ ! -d "$ARCHIVE" ]; then
  echo "Error: Archive not found at $ARCHIVE"
  echo "Run 'flutter build ipa --release' first."
  exit 1
fi

# Fix each framework's MinimumOSVersion
FRAMEWORKS_TO_FIX=(
  "LiteRtTopKMetalSampler"
  "LiteRtMetalAccelerator"
  "StreamProxy"
  "GemmaModelConstraintProvider"
)

echo "=== Fixing framework MinimumOSVersion values ==="

for fw_name in "${FRAMEWORKS_TO_FIX[@]}"; do
  FW="$FW_DIR/$fw_name.framework"
  BINARY="$FW/$fw_name"
  PLIST="$FW/Info.plist"

  if [ ! -d "$FW" ]; then
    echo "  Skipping $fw_name (not found)"
    continue
  fi

  MINOS=$(otool -l "$BINARY" | grep -A5 LC_BUILD_VERSION | grep "minos" | awk '{print $2}')
  echo "  $fw_name: setting MinimumOSVersion = $MINOS"
  plutil -replace MinimumOSVersion -string "$MINOS" "$PLIST"
done

echo ""
echo "=== Re-signing frameworks ==="
for fw_name in "${FRAMEWORKS_TO_FIX[@]}"; do
  FW="$FW_DIR/$fw_name.framework"
  if [ ! -d "$FW" ]; then continue; fi
  echo "  Signing $fw_name..."
  codesign --force --sign "$SIGNING_IDENTITY" --timestamp=none "$FW"
done

echo ""
echo "=== Re-signing Runner.app ==="
codesign --force --sign "$SIGNING_IDENTITY" --timestamp=none --generate-entitlement-der "$APP"

echo ""
echo "=== Re-exporting IPA ==="
EXPORT_DIR="$PROJECT_DIR/build/ios/ipa-fixed"
mkdir -p "$EXPORT_DIR"

EXPORT_PLIST="$PROJECT_DIR/build/ios/ipa/ExportOptions.plist"
if [ ! -f "$EXPORT_PLIST" ]; then
  echo "Error: ExportOptions.plist not found. Run 'flutter build ipa' first."
  exit 1
fi

xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportOptionsPlist "$EXPORT_PLIST" \
  -exportPath "$EXPORT_DIR" \
  | tail -5

echo ""
echo "=== Done! ==="
echo "Fixed IPA: $EXPORT_DIR/ash.ipa"
echo "Upload it via Transporter."
