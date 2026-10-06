# SwarmX

SwarmX is a local-first research work system with native Agents, recursive delegation and
inspectable execution evidence. Persistent goals share budgets, deadlines and acceptance criteria.

Start with the [product direction](docs/product-direction.md) for the vision and ownership,
[product readiness](docs/product-readiness.md) for the usable and tested scope, and
[roadmap](ROADMAP.md) for the next implementation milestones.

```text
Electron → assistant-ui + AG-UI → IPC → Host → Swarm → Native Agent
                                        ↑ ACP / A2A (external)
                                        └ MCP → ProductServices → Swarm
```

SwarmX orchestrates external Agents and contains no builtin Agent runtime. Codex App Server
is the default; use its native login/configuration. To use an independent ACP agent, configure
`SWARMX_ACP_AGENT` and explicitly admit `acp` in Host policy; see [ACP setup](docs/acp.md).
Builds generate App Server declarations with the official CLI pinned in the development
dependencies; `CODEX_PATH` can explicitly select a different installed executable.
Claude and DSH use their official Agent SDK and SDK Client. Hermes uses its installed TUI Gateway;
OpenClaw uses the official Gateway Client with an explicit address and credentials. DSH provides
independent executions and Host log viewing; it does not resume tasks across processes.
Integrations load lazily; startup failure never selects another Agent. Native login is
needed for conversation, not for inspecting saved execution evidence.

Install Node.js and pnpm matching `package.json`. macOS packaging also requires the Xcode
command-line tools. Linux and macOS are the CI targets; Windows has not been validated.
Scientific runtimes, data models and domain tools belong to independent applications such as
GEEPilot; SwarmX has no bundled Python/Docker or Typst/Rust execution runtime.

```sh
pnpm install --frozen-lockfile
pnpm dev
```

`pnpm dev` starts Vite HMR for the renderer. React/CSS edits update the open window; compatible
React edits preserve component state. Main-process, Host and workspace-package source changes
trigger an incremental TypeScript build and restart Electron after a successful build. A failed
build stays visible in the terminal and the watcher waits for the next edit. Main-process restarts
end active runs; saved conversations and their permissions remain on disk. Ctrl+C stops development.
Use `pnpm start` for the normal static production build and launch.

The Host uses its starting directory for execution. Set `SWARMX_CWD=/absolute/research/path`
to select another directory at launch, and `SWARMX_HOME` to choose the private data directory
(default `~/.swarmx`).

Independent projects can keep their own programs, dependencies and native skills in
that directory. See [running an independent project](docs/domain-projects.md) for desktop/ACP launch
commands, project-local resources, output ownership and execution-log export. Domain applications own their scientific data and execution.

Start with a conversation. **Observe / 观测与溯源** shows execution evidence and react-o11y
traces. Work manages durable goals, budgets and independent acceptance. Settings controls language,
Host permissions and Memory. English and Simplified Chinese preferences survive restart.

Domain applications retain their scientific records, revisions and artifact bytes. Coordinating Agents use ordinary capability descriptions and authorized Agent/tool calls for
domain operations. SwarmX records opaque identities and observed execution evidence; it does not
centrally resolve scientific resources. See [domain references](docs/domain-projects.md). Existing
scientific data is not deleted or rewritten by this extraction.

Use `SWARMX_AGENT=codex|claude|dsh|hermes|openclaw|acp` or the Agent selector. Native setup and external
ACP/A2A access: [Agent platform](docs/runtime-platform.md).

[Memory](docs/memory.md) retains research knowledge across sessions in private OKF Markdown.
[Work management](docs/work-management.md) describes persistent goals, Agent choices, runtime
limits and acceptance. Native runtimes keep their own conversation histories.

See [verification commands](docs/product-readiness.md#verification) for engineering checks.
An advisory [AgentRC](https://github.com/microsoft/agentrc) report runs separately.
Its prerelease CLI and transitive dependencies are pinned in the isolated
`.github/agentrc/` development manifest and npm lockfile:

```sh
npm --prefix .github/agentrc ci --ignore-scripts --omit=optional --no-audit --no-fund
npm --prefix .github/agentrc test
npm --prefix .github/agentrc run --silent report > /tmp/swarmx-readiness.json
```

The report uses automatic pnpm workspace detection and a small local policy that
recognizes the packages' existing `build` or `bundle` commands. Presence checks do
not certify compilation or tests, and agent-tooling suggestions may not apply.
AgentRC's default app aggregation passes a criterion when at least 80% of
detected packages pass. A passing aggregate can contain missing package commands;
inspect the JSON `appSummary` and `appFailures` for the complete breakdown.
No maturity level or pass-rate threshold gates CI. JSON and the scanned commit
are CI artifacts, never committed reports. Official `agentrc init` was inspected,
not executed: its default instruction generation uses Copilot and its other
selections create editor/MCP settings. The official-schema config is an empty
stub; readiness needs no model login or calls and changes no runtime dependency.

Release tags use `v<version>` and must match `apps/desktop/package.json`. SwarmX workspace
packages and SwarmX-owned protocol identities use the same release version.
`pnpm package:mac` builds a macOS DMG for the current machine's architecture. The release
workflow builds Apple Silicon and Intel installers on GitHub's macOS runners, then publishes
them with a source archive and SHA256 checksums. See [macOS packaging and releases](docs/macos-release.md)
for local validation, tag publishing and signing limitations. [CITATION.cff](CITATION.cff) contains
software citation metadata.

Manuscript sources, Zotero references and publication evidence live in the separate `swarmx-paper`
project. Its paper revisions do not change this repository's software release archive or checksum.
Software builds, tests and releases work independently of the paper project.

[SPEC.md](SPEC.md): contract. [DESIGNS.md](DESIGNS.md): ownership and boundaries.
[CODEBASE.md](CODEBASE.md): source map.
