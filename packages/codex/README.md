# `@swarmx/codex`

Repository-owned executable module for the built-in SwarmX Codex Harness. It
pins the official `@openai/codex` runtime, exposes it as `swarmx-codex`, and
exports the Cordis plugin that registers the direct `codex_server` transport
with Core. Codex App Server JSON-RPC is used directly; ACP is not used on this
path.

SwarmX resolves this package locally; normal Harness execution never invokes
`npm`, `npx`, or an installer. A host may instead pass `codexCommand` pointing
to a user-managed Codex CLI, which is launched as `codex app-server`. Codex
retains ownership of native tools, authentication, configuration, and session
history. Protected container execution mounts the repository module and the
bundled Linux Codex runtime into `/opt/swarmx/*` and runs the dependency-free
`swarmx-codex-container.js` bootstrap. Permission decisions come from the DSH
`harnessPermissions` registry:
Core installs a fail-closed resolver and a DSH host may register a
higher-priority bridge to its own permission plugin. Plugin registration is
effect-scoped: unloading the Codex plugin removes the managed command and
transport from the Core Cordis container.

```shell
swarmx-codex --version
swarmx-codex
```

See [`docs/codex-module.md`](../../docs/codex-module.md) for the runtime and
upstream-maintenance contract.
