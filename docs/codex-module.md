# Codex module

SwarmX ships Codex as the first repository-owned external Harness module. The
module is a direct Codex App Server JSON-RPC transport registered as a DSH
Cordis plugin; it is not an ACP adapter, a direct Provider implementation, a
second workflow format, or a SwarmX Project-tool host.

## Runtime contract

The built-in `codex` Harness keeps the trusted `swarmx-codex` command token but
declares `transport: "codex_server"`. Every CLI, Desktop, or ACP Server host
boots one `@deepseek-ai/cordis` Context, installs the Core execution, Harness
transport, Provider, Harness catalog, Harness permission, and Swarm strategy
Services, then loads the `@swarmx/codex` plugin. The plugin registers the
command with `acpLaunchers` and registers a client factory with
`harnessTransports` under `codex_server`. For Codex backends, `harnessTransports` therefore constructs a
`CodexServerClient` instead of `AcpClient`; other Custom Harness commands keep
the ACP path unless another plugin registers their transport.

`CodexServerClient` starts the resolved Codex command in `app-server` mode.
Command resolution order is deterministic: an explicit `codexCommand` first,
then an executable `codex` discovered on PATH, then the pinned
`@openai/codex` module through the repository-owned `swarmx-codex` shim. It
speaks the Codex
App Server JSON-RPC protocol directly: `initialize`, `thread/start` or
`thread/resume`, `turn/start`, `turn/completed`, item notifications,
`thread/list`, and `thread/read`. ACP is not initialized or translated on this
path. Codex continues to own its native tools, authentication, configuration,
permissions, and session history; SwarmX projects its terminal items into
normalized message chunks and does not inject duplicate Project tools.

Registration belongs to the plugin Fiber and is revoked when that Fiber
unloads. Resolution happens before process creation and does not invoke `npm`,
`npx`, an installer, or the network. Codex permission decisions are themselves
DSH plugin contributions: Core installs a fail-closed
`harnessPermissions` resolver, the Codex transport consumes the
highest-priority registered resolver. The `@swarmx/codex` plugin itself
inspects a composed DSH `permissionPresets`/`approval` stack at client creation:
policy `never` maps directly to deterministic rejection, while interactive
`ask` falls through to the SwarmX resolver and remains fail-closed until a
typed DSH Agent/Session adapter is added. Unknown or cancelled request classes
fail closed.

The module test suite additionally ships an opt-in real-binary acceptance
test: set `CODEX_E2E_BIN` to a Codex CLI path and it verifies both
`codexCommand` and PATH resolution before an actual `app-server` initialize and
`thread/list` exchange. The pinned-package path is the offline fallback when
neither selector resolves.

The module preserves the existing request-scoped `CODEX_CONFIG` selection and
Desktop's isolated `CODEX_HOME` behavior. A user may still set upstream runtime options explicitly through an authorized
host environment. Module startup never installs a Codex CLI; it only discovers
an existing executable `codex` on PATH before falling back to the pinned
package. Node is the only runtime prerequisite for the pinned fallback.

Packaged Electron builds unpack the module and compatible Codex binary
packages. The Codex plugin rewrites an entry beneath `app.asar` to the matching
`app.asar.unpacked` path before handing it to the child Node process. Protected
Desktop execution keeps the same direct protocol inside the sandbox: the
wrapper mounts the repository Codex module and the bundled Linux Codex runtime
into fixed `/opt/swarmx/*` paths, runs the dependency-free
`swarmx-codex-container.js` bootstrap with the container image's Node, and
points `SWARMX_CODEX_RUNTIME_DIR` at the mounted runtime. When those bundled
assets are unavailable, protected Codex execution fails closed instead of
falling back to native. Each execution
owns one child Fiber; process completion, cancellation, failure, or request
completion disposes that Fiber and its Harness/MCP effects, while the host
Context remains alive until explicit host shutdown.

Cordis is the Core composition and lifecycle authority, not an alternate
workflow or wire protocol. `SwarmConfig` remains the only persisted workflow
format and may name a registered DSH Swarm strategy. Cordis does not inject
SwarmX tools into external Harnesses.

## Upstream maintenance

The npm dependency is preferred while its public Codex App Server behavior
satisfies this contract. Upgrades change the exact dependency version,
lockfile, module acceptance tests, and this document together. If SwarmX needs
behavior upstream cannot provide, vendor the required upstream source at a
recorded commit with its license and a repeatable sync procedure; do not patch
generated or bundled files without provenance.

Core depends directly on the DSH-maintained `@deepseek-ai/cordis` release. It is
not vendored while the published lifecycle and service contracts satisfy this
design. If SwarmX needs changes that upstream cannot accept, vendor the pinned
Cordis source with the same provenance, license, and repeatable-sync rules.
