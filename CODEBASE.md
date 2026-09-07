# SwarmX codebase

## Desktop

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/main.ts` | Electron lifecycle |
| `apps/desktop/src/web-main.ts` | browser-only local Host lifecycle and one-use launch URL |
| `apps/desktop/src/settings.ts` | shared strict workspace policy and environment schemas |
| `apps/desktop/src/permissions.ts` | Host tool/delegation grants, harness/model admission and non-widening inheritance; see `docs/permissions.md` |
| `apps/desktop/src/platform.ts` | startup project restoration and Host lifecycle |
| `apps/desktop/src/agent.ts` | lazy upstream ACP harness selection and session ownership |
| `apps/desktop/src/agents/` | official ACP stdio clients, native mode discovery, review-only launch restrictions and Host event projection |
| `apps/desktop/src/acp-main.ts` | external ACP stdio entry point |
| `apps/desktop/src/window.ts` | BrowserWindow and navigation policy |
| `apps/desktop/src/host/` | secure HTTP, AG-UI, external ACP/A2A, MCP and ProductServices |
| `apps/desktop/src/host/acp-extension.ts` | negotiated ACP permission extension and acknowledgement validation; see `docs/acp.md` |
| `apps/desktop/src/host/acp-client.ts` | operation-scoped official ACP connections and Host gateway projections; native calls exist only at leaf boundaries |
| `apps/desktop/src/host/execution-journal.ts` | append-only execution records, persisted conversation permission grants, causal context and workspace-scoped reads; see `docs/execution-log.md` |
| `apps/desktop/src/host/recorded-agent.ts` | persistent native Agent observation below every gateway |
| `apps/desktop/src/host/memory.ts` | frozen session context, learning thresholds, proposals and journal-backed approval |
| `apps/desktop/src/host/memory-review.ts` | restricted, cancellable review execution through ACP, with tool rejection at its native leaf |
| `apps/desktop/src/memory.ts` | shared memory settings, graph and review UI schemas |
| `apps/desktop/src/host/research-environment.ts` | Docker setup, immutable Python image, isolation, limits and cancellation |
| `apps/desktop/src/host/workspace-settings.ts` | persistent project catalog, canonical directory identity and atomic project/private settings |
| `apps/desktop/src/execution-record.ts` | shared journal record, read response and run-control boundary schemas |
| `apps/desktop/src/renderer/` | React 18 assistant-ui, Codex-inspired task navigation, default Tailwind and optional react-o11y trace pane; see `docs/desktop-ui.md` |
| `apps/desktop/src/renderer/agent-controls.tsx` | Harness menu, assistant-ui-style model/Thinking selector and native mode selection |
| `apps/desktop/src/renderer/subagents.tsx` | persistent delegation list, per-run conversation/log inspection and native child controls |
| `apps/desktop/src/renderer/research.tsx` | conversation side view for assets, assistant edit drafts, react-o11y, scientific runs and RO-Crate |
| `apps/desktop/src/renderer/i18n.ts` and `locales/en.json` | English/Chinese UI, browser-language detection and shared translation catalog |
| `apps/desktop/src/renderer/research-graph.tsx` | bounded RO-Crate projection and shared React Flow view for research and memory |
| `apps/desktop/src/renderer/memory.tsx` | bilingual note editing, approval, review status and vault dependency graph |
| `apps/desktop/src/renderer/projects.tsx` | saved project navigation, directory registration and project settings entry |
| `apps/desktop/src/renderer/figure-studio.tsx` | cancellable Python figure generation and edits through Science tools |
| `apps/desktop/tests/` | native integrations, recursive gateways, AG-UI, renderer interactions, MCP and boundaries |
| `apps/desktop/vite.config.ts` | Renderer bundle and development HMR configuration |

## Public packages

| Path | Ownership |
| --- | --- |
| `packages/core/annotation/` | portable artifact annotations |
| `packages/core/dvc/` | Git/DVC inspection and explicit operations |
| `packages/core/memory/` | bounded Markdown notes, private OKF concepts, dependency validation/loading and lint |
| `packages/core/swarm/` | recursive ACP Agent/Client relay using official SDK connections |
| `packages/science/core/` | scientific journal, artifacts, tools, and previews |

Public packages do not depend on Electron, Renderer, AG-UI, A2A or provider SDKs.
The Swarm package uses the official ACP SDK for recursive communication.

## Repository tooling

`.pre-commit-config.yaml` uses official remote Biome and Ruff hooks for staged TS/TSX and Python.
Checks never rewrite files. Install with `prefligit install`;
verify with `prefligit run --all-files`.

Build, cleanup, documentation coverage, and manuscript-model utilities live under `scripts/`.
`scripts/dev.ts` compiles watched backend changes and owns the development Electron child process;
the Host serves Vite middleware on its authenticated origin in development.
Reproducible SoftwareX examples live under `examples/`; the manuscript is `swarmx.tex`.
`examples/softwarex/current-release/` contains the current native workflow driver and offline evidence verifier.
The pinned Jupyter Data Science container recipe lives under `apps/desktop/resources/python/`.
