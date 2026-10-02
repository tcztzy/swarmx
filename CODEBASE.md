# SwarmX codebase

## Desktop

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/main.ts` | Electron lifecycle |
| `apps/desktop/src/settings.ts` | shared strict execution policy and environment schemas |
| `apps/desktop/src/bridge-contract.ts` | validated Electron IPC payloads and renderer response schemas |
| `apps/desktop/src/tool-manifest.ts` | shared product-tool manifest contract between the Host and native integrations |
| `apps/desktop/src/private-json.ts` | atomic private JSON writes with restricted permissions |
| `apps/desktop/src/ipc.ts` | trusted-window IPC handlers and AG-UI event delivery |
| `apps/desktop/src/permissions.ts` | Host tool/delegation grants, harness/model admission and non-widening inheritance; see `docs/permissions.md` |
| `apps/desktop/src/platform.ts` | Host startup and lifecycle |
| `apps/desktop/src/agent.ts` | lazy integration selection and native session ownership |
| `apps/desktop/src/agents/` | external ACP, native Codex App Server, Claude/DSH SDKs, Hermes TUI Gateway and OpenClaw Gateway Client, settings and Host event projection; see `docs/native-agents.md` |
| `examples/legacy-pi/` | preserved builtin Pi adapter and test evidence; historical source only, excluded from production and default tests |
| `apps/desktop/src/agents/external-acp.ts` | opt-in external ACP client, owned stdio process, remote session references, permission forwarding and terminal lifecycle |
| `apps/desktop/resources/hermes-native.py` | Hermes native gateway bootstrap, process-local Host MCP registration and read-only native execution-thread wait |
| `apps/desktop/resources/openclaw-plugin/` | OpenClaw gateway plugin that forwards session-scoped Host tool calls over the private bridge |
| `apps/desktop/src/agents/openclaw-auth.ts` | private device identity, signing and origin-scoped token storage for the official Gateway Client |
| `apps/desktop/src/agents/dsh.ts` | independent DSH SDK executions, explicit provider/model and profile selection, native MCP configuration and terminal-outcome validation |
| `apps/desktop/src/acp-main.ts` | external ACP stdio entry point |
| `apps/desktop/src/window.ts` | BrowserWindow and navigation policy |
| `apps/desktop/src/host/` | renderer operations, AG-UI, external ACP/A2A, MCP and ProductServices |
| `apps/desktop/src/host/acp-extension.ts` | negotiated ACP permission extension and acknowledgement validation; see `docs/acp.md` |
| `apps/desktop/src/host/capabilities.ts` | projection of internal capabilities into the stable public gateway shape |
| `apps/desktop/src/host/execution-journal.ts` | append-only execution records, keyed Host event publication, same-run delegation preparation evidence, persisted conversation grants and directory-scoped reads; see `docs/execution-log.md` |
| `apps/desktop/src/host/evaluation-crate.ts` | selected evaluation and review evidence projected into an Attached RO-Crate; see `docs/ro-crate.md` |
| `apps/desktop/src/host/execution-evidence.ts` | deterministic summaries and coverage counts for cited executions, derived from original lifecycle records |
| `apps/desktop/src/host/delegation-skill.ts` | prepares the delegation skill body with authorized Memory evaluations and scoped harness/model/effort/provider evidence |
| `apps/desktop/src/host/work.ts` | persistent work cycles/items, Agent presets and temporary choices, runtime budgets, trusted acceptance and journal-derived costs; see `docs/work-management.md` |
| `apps/desktop/src/work.ts` | shared work domain and trusted desktop command/response schemas; renderer and Host use the same contracts |
| `apps/desktop/src/host/recorded-agent.ts` | persistent native Agent observation below every gateway |
| `apps/desktop/preload.cjs` | sandboxed CommonJS preload exposing the Electron IPC product-tool bridge |
| `apps/desktop/src/host/mcp-bridge.ts` | stdio MCP entry point that forwards product-tool calls to the Host socket |
| `apps/desktop/src/host/mcp-socket.ts` | Host socket that authorizes and dispatches bridged product-tool calls |
| `apps/desktop/src/host/agent-registry.ts` | Host ownership of one lazy native runtime per harness |
| `apps/desktop/src/host/operations.ts` | product operations exposed through Electron IPC |
| `apps/desktop/src/host/memory.ts` | Memory authoring, session context, delegation knowledge, durable execution review plans, replay and approval; see `docs/memory.md` |
| `apps/desktop/src/host/learning-resources.ts` | opted-in prompt/skill snapshots, fixed structural validation and optional baseline/candidate behavior evaluation before revision-checked atomic replacement; see `docs/learning-resources.md` |
| `apps/desktop/src/host/memory-review.ts` | restricted, cancellable direct Agent execution with tool rejection |
| `apps/desktop/src/memory.ts` | shared memory settings, graph and review UI schemas |
| `apps/desktop/src/host/research-environment.ts` | Docker setup, immutable Python image, isolation, limits and cancellation |
| `apps/desktop/src/host/settings-store.ts` | atomic private execution settings and user preferences |
| `apps/desktop/src/execution-record.ts` | shared journal record, read response and run-control boundary schemas |
| `apps/desktop/src/evaluation-crate.ts` | shared evaluation export request and Attached RO-Crate payload schemas |
| `apps/desktop/src/message-activity.ts` | validated Codex message phases, turn timing and native tool kind/status for history and streaming |
| `apps/desktop/src/renderer/` | assistant-ui task navigation, Tailwind and optional react-o11y trace pane; see `docs/desktop-ui.md` |
| `apps/desktop/src/renderer/agent-controls.tsx` | native catalog and run-setting ownership, Harness menu and assistant-ui model/Thinking selector |
| `apps/desktop/src/renderer/components/` | official assistant-ui elements and Radix/shadcn controls with at most ten changed lines per source; upstream attribution and MIT license included |
| `apps/desktop/src/renderer/hooks/` and `apps/desktop/src/renderer/lib/` | unchanged official clipboard hook and class-name utility used by the imported components |
| `apps/desktop/src/renderer/fonts/` | bundled homepage Demo Public Sans and JetBrains Mono Latin variable fonts with SIL Open Font Licenses |
| `apps/desktop/src/renderer/commentary.tsx` | per-turn collapsible work, native duration labels and live phase observation |
| `apps/desktop/src/renderer/tool-ui.tsx` | assistant-ui tool grouping, native activity summaries and shell terminal output |
| `apps/desktop/src/renderer/subagents.tsx` | persistent delegation list, per-run conversation/log inspection and native child controls |
| `apps/desktop/src/renderer/research.tsx` | conversation side view for assets, assistant edit drafts, react-o11y, scientific runs and RO-Crate |
| `apps/desktop/src/renderer/source-inspection.tsx` | pinned artifact and execution evidence inspection, original records, scoped statistics, input identities and recorded computations |
| `apps/desktop/src/renderer/saved-concept.tsx` | validated Memory-read result cards and source navigation from the current answer |
| `apps/desktop/src/renderer/evaluation-export.tsx` | local ZIP download of evaluation and review evidence |
| `apps/desktop/src/renderer/i18n.ts` and `locales/en.json` | English/Chinese UI, browser-language detection and shared translation catalog |
| `apps/desktop/src/renderer/research-graph.tsx` | bounded RO-Crate projection and shared React Flow view for research and memory |
| `apps/desktop/src/renderer/memory.tsx` | bilingual user-note editing, approval, review status and concept dependency graph |
| `apps/desktop/src/renderer/work.tsx` | long-term work goals, shared budgets, explicit execution controls, pinned evidence and user acceptance |
| `apps/desktop/src/renderer/interaction-form.tsx` | native confirmation fields and responses shared by conversation and managed work |
| `apps/desktop/src/renderer/bridge.ts` | typed access to the preload's product operations |
| `apps/desktop/src/renderer/agui.ts` | assistant-ui AG-UI adapter over Electron IPC |
| `apps/desktop/src/renderer/figure-studio.tsx` | cancellable Python figure generation and edits through Science tools |
| `apps/desktop/tests/mcp-bridge-support.ts` | spawns the stdio MCP bridge against a test Host socket |
| `apps/desktop/tests/` | native integrations, recursive gateways, AG-UI, renderer interactions, MCP and boundaries |
| `apps/desktop/vite.config.ts` | Renderer bundle and development HMR configuration |
| `electron-builder.yml` | macOS DMG contents, native resource layout and ad-hoc signing; see `docs/macos-release.md` |

## Public packages

| Path | Ownership |
| --- | --- |
| `packages/core/annotation/` | portable artifact annotations |
| `packages/core/dvc/` | Git/DVC inspection and explicit operations |
| `packages/core/memory/` | bounded Markdown notes, private OKF concepts, dependency validation/loading and lint |
| `packages/core/swarm/` | direct recursive Agent composition; the caller owns the borrowed lead |
| `packages/core/swarm/skills/delegate/SKILL.md` | shared delegation skill exported with the Swarm package: combination selection, task fit and evidence requirements |
| `packages/core/memory/skills/memory/SKILL.md` | shared Memory authoring guide exported with the Memory package, read on demand and used directly by background reviews |
| `packages/science/core/` | scientific journal, artifacts, tools, and previews |

Public packages do not depend on Electron, Renderer, AG-UI, A2A or provider SDKs.
The Swarm package has no transport or protocol dependency.

## Repository tooling

`.pre-commit-config.yaml` uses official remote Biome and Ruff hooks for staged TS/TSX and Python.
Checks never rewrite files. Install with `prefligit install`;
verify with `prefligit run --all-files`.

Build, cleanup, and documentation coverage utilities live under `scripts/`.
`scripts/prepare-macos-dependencies.mjs` rebuilds the deployed production dependencies for
Electron; `.github/workflows/release.yml` builds and checks native macOS DMGs before publishing.
`scripts/generate-codex-types.ts` generates ignored App Server declarations from the pinned official CLI development dependency before development and builds; `CODEX_PATH` is an explicit executable override.
`scripts/dev.ts` compiles watched backend changes, starts the local Vite renderer server and owns
the development Electron child process.
Manuscript sources, bibliography, publication evidence and their verification tools belong to the
separate `swarmx-paper` project. Software build, test and release tooling has no dependency on it.
The pinned Jupyter Data Science container recipe lives under `apps/desktop/resources/python/`.
Local research downloads, visual comparisons, logs and preview scripts belong in ignored
`runs/`. Promote software material that must ship into `docs/` or `scripts/` before committing;
preserve publication evidence in `swarmx-paper`.
