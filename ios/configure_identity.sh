#!/bin/bash
# Configure Xcode project signing identity for a known publisher.
#
# Usage:
#   ./ios/configure_identity.sh yunxiang
#   ./ios/configure_identity.sh yao
#   ./ios/configure_identity.sh --team-id ABCD1234EF --bundle-id com.example.ash
#
# Replaces {{TEAM_ID}} and {{BUNDLE_ID}} placeholders in
# ios/Runner.xcodeproj/project.pbxproj with real values.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PBXPROJ="$SCRIPT_DIR/Runner.xcodeproj/project.pbxproj"

# --- Parse arguments ---
TEAM_ID=""
BUNDLE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --team-id)
      TEAM_ID="$2"
      shift 2
      ;;
    --bundle-id)
      BUNDLE_ID="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      # Positional arg = publisher nickname
      case "$1" in
        yunxiang)
          TEAM_ID="DJD849Y8Q6"
          BUNDLE_ID="com.yunxiang.ash"
          ;;
        yao)
          TEAM_ID="V9Q67SYWWQ"
          BUNDLE_ID="com.yao.ash"
          ;;
        *)
          echo "Error: unknown publisher '$1'" >&2
          echo "Known publishers: yao, yunxiang" >&2
          exit 1
          ;;
      esac
      shift
      ;;
  esac
done

if [[ -z "$TEAM_ID" || -z "$BUNDLE_ID" ]]; then
  echo "Usage: $0 <publisher-nickname>" >&2
  echo "       $0 --team-id <ID> --bundle-id <bundle>" >&2
  echo "" >&2
  echo "Known publishers: yao, yunxiang" >&2
  exit 1
fi

if [[ ! -f "$PBXPROJ" ]]; then
  echo "Error: $PBXPROJ not found" >&2
  exit 1
fi

# --- Replace placeholders ---
sed -i '' \
  -e "s/{{TEAM_ID}}/$TEAM_ID/g" \
  -e "s/{{BUNDLE_ID}}/$BUNDLE_ID/g" \
  "$PBXPROJ"

REMAINING=$(grep -c '{{\(TEAM_ID\|BUNDLE_ID\)}}' "$PBXPROJ" 2>/dev/null || true)

echo "Done. Set TEAM_ID=$TEAM_ID BUNDLE_ID=$BUNDLE_ID in project.pbxproj"
if [[ -n "$REMAINING" && "$REMAINING" != "0" ]]; then
  echo "Warning: $REMAINING placeholder(s) still remain in the file." >&2
fi
