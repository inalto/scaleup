# Code signing and notarisation (follow-up)

The release workflow ships unsigned binaries. Windows SmartScreen and macOS Gatekeeper therefore
warn on first launch (macOS: right-click → Open, or `xattr -d com.apple.quarantine`). This page
lists what signing would take; none of it is implemented.

## Windows (Authenticode)

1. Buy a code-signing certificate. An **EV** certificate removes the SmartScreen warning
   immediately; an OV certificate only builds reputation over time. EV keys live on a hardware
   token or in a cloud HSM (Azure Trusted Signing, DigiCert KeyLocker, SSL.com eSigner), so CI signs
   through the vendor's tool rather than with a `.pfx` file.
2. Sign `scaleup.exe` and `scaleup-app.exe` after `cargo build` and before `zip_windows.ps1` and
   `cargo wix`, then sign the `.msi` too:

   ```powershell
   signtool sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com `
     /a /n "ScaleUP contributors" target\x86_64-pc-windows-msvc\release\scaleup.exe
   signtool verify /pa /v dist\scaleup-<version>-x86_64-pc-windows-msvc.msi
   ```

   With Azure Trusted Signing use the `azure/trusted-signing-action` instead of `signtool /a`.
3. Secrets needed: the vendor's credentials (`AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
   `AZURE_CLIENT_SECRET` and the signing account/profile names for Trusted Signing; or the token
   vendor's user, password and TOTP secret for eSigner/KeyLocker). Never store a `.pfx` in the repo.

## macOS (Developer ID + notarisation)

1. Join the Apple Developer Program and create a **Developer ID Application** certificate. Export it
   as a `.p12` and store it base64-encoded in a secret.
2. In `bundle_macos.sh`, replace the ad-hoc `codesign --sign -` with a hardened-runtime signature,
   and sign the CLI inside the archive as well:

   ```bash
   codesign --force --deep --options runtime --timestamp \
     --sign "Developer ID Application: <name> (<TEAMID>)" dist/ScaleUP.app
   codesign --force --options runtime --timestamp --sign "Developer ID Application: ..." \
     dist/scaleup-<version>-aarch64-apple-darwin/scaleup
   ```

   The app needs an entitlements file only if it ever loads unsigned plug-ins; ONNX Runtime
   provider dylibs must be signed with the same identity (`--deep` covers the ones in
   `Contents/MacOS`).
3. Notarise the dmg and staple the ticket:

   ```bash
   xcrun notarytool submit dist/scaleup-<version>-macos.dmg \
     --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" --password "$APPLE_APP_PASSWORD" --wait
   xcrun stapler staple dist/scaleup-<version>-macos.dmg
   xcrun stapler validate dist/scaleup-<version>-macos.dmg
   ```

   Build the dmg from the signed app, notarise the dmg (it notarises the app inside), then staple.
4. Secrets needed: `MACOS_CERTIFICATE_P12` (base64), `MACOS_CERTIFICATE_PASSWORD`,
   `MACOS_KEYCHAIN_PASSWORD` (temporary keychain on the runner), `APPLE_ID`, `APPLE_TEAM_ID`,
   `APPLE_APP_PASSWORD` (an app-specific password, not the account password). Import the
   certificate into a temporary keychain with `security import` at the start of the
   `bundle-macos` job and delete the keychain at the end.

## Linux

Debian packages are not signed by the workflow. Options: sign the release assets with a GPG
detached signature (`gpg --armor --detach-sign`) and publish the public key, or host an apt
repository (reprepro, aptly) signed with a release key. Secrets: `GPG_PRIVATE_KEY`,
`GPG_PASSPHRASE`.

## Checklist when implementing

- Sign before packaging, verify after, and fail the job if verification fails.
- Timestamp every signature so it outlives the certificate.
- Keep the unsigned path working for forks without secrets (`if: secrets.X != ''` guards).
- Document in `CHANGELOG.md` the first release that ships signed binaries.
