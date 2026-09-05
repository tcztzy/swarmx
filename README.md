# SwarmX

Local-first research desktop with recursive Swarms and native Agents.

```text
Electron / Browser → assistant-ui + AG-UI → Host → Swarm → native Agent
                                            ↑ ACP / A2A (external)
                                            └ MCP → ProductServices
```

Codex App Server is the default. Claude Agent SDK, Hermes TUI gateway and OpenClaw Gateway are
selected lazily; startup failure never selects another Agent. DSH, ZCode and Kimi are deferred.
Install and authenticate the native `codex` CLI before building or launching the desktop.

```sh
pnpm install
pnpm dev
```

Use `SWARMX_AGENT=claude|hermes|openclaw` or the Agent selector. Native setup and external
ACP/A2A access: [Agent platform](docs/runtime-platform.md).

[Memory](docs/memory.md) provides shared semantic memory in private OKF Markdown. Agents use the
`memory` product tool to retrieve and curate research knowledge across sessions; native runtimes
keep their own conversation histories. Existing vaults move to the current storage path on startup.

```sh
pnpm typecheck
pnpm test
pnpm build
pnpm lint
pnpm docs:check
```

[SPEC.md](SPEC.md): contract. [DESIGNS.md](DESIGNS.md): ownership and boundaries.
[CODEBASE.md](CODEBASE.md): source map.
