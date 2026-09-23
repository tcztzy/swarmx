# macOS packaging and releases

`pnpm package:mac` builds SwarmX from source and exports a drag-to-Applications DMG for
the Mac running the command. It requires the development prerequisites in the README,
including Rust and the Xcode command-line tools. Run from the repository root:

```sh
pnpm install --frozen-lockfile
pnpm package:mac
```

The installer is `apps/desktop/release/SwarmX-<version>-mac-<arch>.dmg`, where `arch` is
`arm64` for Apple Silicon or `x64` for Intel. Open it and drag `SwarmX.app` to Applications.
The app includes Electron, the compiled renderer and Host, production dependencies,
native Agent resources and the Typst executable compiled for that architecture. These
files remain outside ASAR because native child processes need ordinary filesystem paths.
`pnpm deploy --prod --config.node-linker=hoisted` stages the complete production dependency
tree from the lockfile, including SDK peers. The deployed directory uses ordinary package
directories so macOS signing does not recursively follow pnpm's dependency symlinks.
Workspace packages are injected and synchronized after
their `bundle` scripts so the deployed tree contains the freshly compiled libraries.
The build hook uses `@electron/rebuild` for native modules and tells electron-builder to
preserve that tree. Recomputing it with electron-builder's dependency collector omits
runtime peers used by DSH. The staging directory is removed by `pnpm clean`.
Native Agent credentials and external tools such as Codex, Hermes and Docker retain the
requirements documented in [Agent platform](runtime-platform.md).

The app uses an ad-hoc signature, with no Apple Developer ID certificate or notarization.
Downloaded builds can require approval in macOS Privacy & Security before opening.
This workflow does not provide Apple-verified distribution or automatic updates.

## Local acceptance

Local export validates the packaging command; local build files are never release inputs.
After export, verify the DMG, mount it, and check the app from the mounted volume:

```sh
hdiutil verify apps/desktop/release/SwarmX-<version>-mac-<arch>.dmg
hdiutil attach -readonly -nobrowse apps/desktop/release/SwarmX-<version>-mac-<arch>.dmg
SWARMX_PACKAGED_APP='/Volumes/<volume>/SwarmX.app' pnpm vitest run apps/desktop/tests/macos-package.test.ts
```

The package test launches the bundled Electron executable from a temporary working directory
with isolated settings. It uses Node's debugger client to inspect the running app, loads every
native integration, checks the bundled resources, compiles a PDF through the Typst runtime,
and verifies the renderer and preload bridge. It does not send model requests or validate
external Agent authentication.
On first launch, macOS can scan an ad-hoc signed app before Electron starts its debugger.
If startup times out without debugger output, inspect the process exit code, termination signal
and `syspolicyd`/`amfid` logs before attributing the failure to the app. A later successful launch
does not establish why the first launch failed or validate a newly rebuilt installer.
Detach the volume after validation. The ordinary test suite skips this test unless
`SWARMX_PACKAGED_APP` names an exported app.

## GitHub Release

Release tags are `v<version>` and must match `apps/desktop/package.json`. Keep SwarmX
workspace packages, the Rust runtime and owned protocol identities at that same version.
Commit the release changes, then push the version tag. The existing Release workflow can
also be run manually with an existing tag.

GitHub checks out the tag and installs dependencies from the lockfile. Separate native
Apple Silicon and Intel macOS jobs run the same `pnpm package:mac` command and validate
their packaged apps. The source job runs the repository quality checks. Only after all
jobs succeed does the publish job create a GitHub Release containing both DMGs, the source
archive and `SHA256SUMS`. Versions with a prerelease suffix are marked as prereleases.
The workflow uses `GITHUB_TOKEN` with write access limited to the publish job; no release
token or Apple credentials are needed. Publishing an already released tag fails rather
than replacing its assets.
