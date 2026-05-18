# TestFlight publishing — verified workflow

Step-by-step for shipping a new build of `com.yunxiang.ash` to TestFlight.
Captured 2026-05-18 after walking through it end-to-end on a fresh
Apple Distribution cert under team `DJD849Y8Q6` (Yunxiang Yan).

The collaborator (Yao Xiao, team `V9Q67SYWWQ`) has a parallel setup;
the framework fix script supports both via env-var overrides.

---

## One-time setup (per publisher)

Skip this section if you've already shipped a TestFlight build from
this Mac under your team.

### 1. Apple Developer Program

Active membership required (currently $99/yr). Verify at
<https://developer.apple.com/account>.

### 2. App ID registered

<https://developer.apple.com/account/resources/identifiers/list> →
look for `com.yunxiang.ash`. If absent, create it: **+ → App IDs →
App → Explicit**. Enable these capabilities (without them, vision
inference SIGKILLs silently mid-`engine_create` from iOS Jetsam):

- Extended Virtual Addressing
- Increased Memory Limit

`Increased Debugging Memory Limit` is **not** required for release
builds and App Store rejects it (collaborator's commit `8d79ad2`
stripped it from `ios/Runner/Runner.entitlements` for that reason).

### 3. Apple Distribution certificate

Xcode → Settings → Accounts → select your Apple ID → Manage Certificates
→ **+ → Apple Distribution**. Verify with:

```bash
security find-identity -p codesigning -v
```

You should see `Apple Distribution: <Your Name> (<TEAM_ID>)`.

### 4. App Store Connect record

<https://appstoreconnect.apple.com/apps> → **+ → New App**:

| Field | Value |
|-------|-------|
| Platforms | iOS only |
| Name | Ash: Survival AI |
| Primary Language | English (U.S.) |
| Bundle ID | `com.yunxiang.ash` (must appear in dropdown — depends on step 2) |
| SKU | `ash-001` (any unique string) |
| User Access | Full Access |

If the bundle ID doesn't appear: either the App ID isn't registered
under your team, or it's registered under a different team (bundle
IDs are globally unique — only one team can own each).

### 5. Xcode auto-managed signing

`ios/Runner.xcworkspace` → Runner target → Signing & Capabilities →
**Automatically manage signing** checked, Team set to your team.
Xcode generates the App Store distribution profile on first archive.
You may see "Signing Certificate: Apple Development" under the All tab
— Xcode swaps to Apple Distribution for Release archives automatically.

### 6. Keychain ACL (avoids prompt spam during build)

The fresh Apple Distribution key's ACL doesn't trust `codesign` by
default. A full build invokes `codesign` 20+ times — without
intervention you get a password prompt for each one. Two options:

- **Click "Always Allow"** on the very first prompt during a build.
  ACL persists.
- Or run interactively (one prompt total, then permanent):

  ```bash
  security set-key-partition-list \
    -S apple-tool:,apple:,codesign: \
    -s -k '' ~/Library/Keychains/login.keychain-db
  ```

---

## Per-build process

### F. Bump build number

`pubspec.yaml` `version: x.y.z+N` → bump `N`. Each TestFlight upload
needs a unique build number; Apple rejects duplicates.

### G. Build the IPA

```bash
flutter build ipa --release --export-method=app-store
```

Output: `build/ios/ipa/ash.ipa` (~140 MB), `build/ios/archive/Runner.xcarchive`,
`build/ios/ipa/ExportOptions.plist`. Takes ~10 minutes total
(archive ~50 s, IPA export ~9 min — most of that is dSYM packaging).

### H. Verify framework `MinimumOSVersion` mismatches

`flutter_gemma` ships prebuilt frameworks (from MediaPipe / TensorFlow)
whose `Info.plist` `MinimumOSVersion` is 13.0 but whose binaries
declare `LC_BUILD_VERSION minos` of 14.0 or 16.0. App Store Connect
rejects this with error 90208: *"does not support the minimum OS
Version specified in the Info.plist"*. Check before uploading:

```bash
APP=build/ios/archive/Runner.xcarchive/Products/Applications/Runner.app
for fw in LiteRtMetalAccelerator StreamProxy GemmaModelConstraintProvider; do
  PLIST="$APP/Frameworks/$fw.framework/Info.plist"
  BIN="$APP/Frameworks/$fw.framework/$fw"
  P=$(plutil -extract MinimumOSVersion raw "$PLIST" 2>/dev/null)
  B=$(otool -l "$BIN" 2>/dev/null | grep -A5 LC_BUILD_VERSION | grep minos | head -1 | awk '{print $2}')
  echo "$fw: plist=$P binary=$B $([ "$P" = "$B" ] && echo OK || echo MISMATCH)"
done
```

If everything reports OK (the upstream might fix this), skip step I.
Otherwise:

### I. Patch + re-sign + re-export

```bash
TEAM_ID=DJD849Y8Q6 \
SIGNING_IDENTITY="Apple Distribution: Yunxiang Yan (DJD849Y8Q6)" \
./ios/fix_framework_plists.sh
```

(For the collaborator's setup the defaults work — just run
`./ios/fix_framework_plists.sh`.)

The script:

1. Rewrites each framework's `MinimumOSVersion` to match its binary's
   `LC_BUILD_VERSION minos`.
2. Re-signs each patched framework with your Apple Distribution identity.
3. Re-signs `Runner.app` (with `--generate-entitlement-der` —
   required for any re-sign after the framework rewrite).
4. Re-exports the IPA via `xcodebuild -exportArchive` using the
   `ExportOptions.plist` flutter already wrote.

Output: `build/ios/ipa-fixed/ash.ipa`. **This is the file you upload**,
NOT the original `build/ios/ipa/ash.ipa`.

### J. Upload

Three methods, simplest first:

**Transporter** (recommended):

1. Open Transporter (Mac App Store, free).
2. Drag `build/ios/ipa-fixed/ash.ipa` into the window.
3. Click Deliver.

If you see *"Could not create a temporary .itmsp package … No
suitable application records were found"*: the App Store Connect
record (one-time-setup step 4) doesn't exist yet, or you're signed
into Transporter with an Apple ID that doesn't have access to that
team's apps. The Apple ID in Transporter's top-right must match a
user with access to the team that owns `com.yunxiang.ash`.

**Xcode Organizer**: Window → Organizer → Archives → select today's
build → Distribute App → App Store Connect → Upload. Walks through
Export Compliance ("Does your app use encryption?" → No, only standard
Apple APIs for HTTPS).

**`xcrun altool`**: needs an App Store Connect API key or
app-specific password. Not worth the setup unless you're scripting.

### K. Wait for processing

Apple processes the build in 5-30 min. Email arrives when ready.
While waiting:

- The build appears in App Store Connect → app → TestFlight tab,
  initially with status "Processing".
- Once processed, status becomes "Missing Compliance" until you answer
  the encryption question (one-tap in the TestFlight UI).

### L. Configure TestFlight (first build only)

App Store Connect → app → TestFlight tab:

1. **Test Information** (left sidebar) — required before any tester
   can install:
   - Beta App Description (one sentence is fine)
   - Feedback Email (yours)
   - Contact info (name + email, marketing/privacy URLs optional)
   - **Required Gemma attribution** (per Google's variant guidelines):
     include this line in the Beta App Description:
     > Uses a specialized Gemma model for AI-powered features.
     > Gemma is a trademark of Google LLC.

2. **Internal Testing** group:
   - "+" → name it (e.g. "Self test").
   - Add testers via Apple ID. Internal testers must be App Store
     Connect users in your team — invite them via Users and Access
     first if they're not yet in the team.
   - Internal testers do **not** need Beta App Review and can install
     immediately once the build finishes processing.

3. **External Testing** (only if you want external testers, up to 10k):
   - Requires Beta App Review (~1-2 days).
   - Needs a complete Test Information panel including beta review
     info (demo account if applicable, beta what's-new notes).

### M. Install on iPhone

1. Tester gets an email + push notification.
2. Open the invite on the iPhone → "View in TestFlight" → opens the
   TestFlight app.
3. Tap **Install**. The 140 MB IPA downloads.
4. First launch: the app downloads the Gemma `.litertlm` from
   HuggingFace (~2.5 GB for E2B, ~5 GB for E4B). Tester needs Wi-Fi.

---

## Gotchas hit during the first run

These are noted in case they recur on future builds.

1. **`Apple Distribution` cert was missing from this Mac initially.**
   Sideload-only setups have just an Apple Development cert. The fix
   is one-time-setup step 3. Symptom before fix: `flutter build ipa`
   succeeds but Transporter rejects with signing errors.

2. **App Store Connect record didn't exist.** Symptom: Transporter
   says "No suitable application records were found" at the upload
   step. Fix: one-time-setup step 4. The record creation requires
   the App ID to already exist under your team in the developer
   portal (step 2).

3. **Framework Info.plist / binary `minos` mismatch.** Symptom:
   `xcrun altool` or Transporter upload succeeds but Apple's
   post-upload validation rejects with error 90208 listing the
   four `LiteRt*` / `StreamProxy` / `GemmaModelConstraintProvider`
   frameworks. Fix: step I (framework fix script). Recheck with
   step H before each upload — upstream may patch this in a future
   `flutter_gemma` release.

4. **Codesign password-prompt spam.** Symptom: building or running
   the fix script triggers 20+ password prompts for the Apple
   Distribution key. Fix: one-time-setup step 6 (keychain ACL),
   or click "Always Allow" on the first prompt.

5. **"Increased Debugging Memory Limit" entitlement rejected.**
   Symptom: ASC rejects upload citing entitlement not allowed for
   App Store builds. Fix: collaborator's commit `8d79ad2` already
   stripped this from `ios/Runner/Runner.entitlements`. Don't add
   it back if you ever edit that file — it's debug-only.

6. **App name "Ash" was taken on the App Store.** Solved by using
   `Ash: Survival AI` as the App Store listing name; the on-device
   home-screen icon stays as just "Ash" (`CFBundleDisplayName` in
   `ios/Runner/Info.plist`). Apple allows them to differ.

---

## Quick reference: full clean-build command sequence

After all one-time setup is in place:

```bash
# 1. Bump build number in pubspec.yaml manually first

# 2. Build
flutter build ipa --release --export-method=app-store

# 3. Patch frameworks (use your team's env vars)
TEAM_ID=DJD849Y8Q6 \
SIGNING_IDENTITY="Apple Distribution: Yunxiang Yan (DJD849Y8Q6)" \
./ios/fix_framework_plists.sh

# 4. Upload — drag build/ios/ipa-fixed/ash.ipa into Transporter

# 5. Wait for "Build processed" email, then add to a TestFlight
#    internal test group via App Store Connect.
```

Total wall-clock from `flutter build ipa` to "Install" on phone:
roughly 25-50 minutes on a fast Mac and good network.
