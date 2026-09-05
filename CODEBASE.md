# SwarmX codebase

## Desktop

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/main.ts` | Electron lifecycle |
| `apps/desktop/src/platform.ts` | workspace, Host and single ProductServices owner |
| `apps/desktop/src/agent.ts` | lazy native selection and session ownership |
| `apps/desktop/src/agents/` | Codex App Server, Claude SDK, Hermes and OpenClaw; transient native projections |
| `apps/desktop/src/acp-main.ts` | external ACP stdio entry point |
| `apps/desktop/src/window.ts` | BrowserWindow and navigation policy |
| `apps/desktop/src/host/` | secure HTTP, AG-UI, external ACP/A2A, MCP and ProductServices |
| `apps/desktop/src/renderer/` | React 18 assistant-ui, default Tailwind, Agent sidebar and react-o11y waterfall |
| `apps/desktop/tests/` | native integrations, recursive gateways, AG-UI, MCP and boundaries |
| `apps/desktop/vite.config.ts` | Renderer bundle configuration |

## Public packages

| Path | Ownership |
| --- | --- |
| `packages/core/annotation/` | portable artifact annotations |
| `packages/core/dvc/` | Git/DVC inspection and explicit operations |
| `packages/core/memory/` | shared semantic memory in private OKF Markdown |
| `packages/core/swarm/` | protocol-neutral recursive Agent composition |
| `packages/science/core/` | scientific journal, artifacts, tools, and previews |

Public packages do not depend on Electron, Renderer, AG-UI, ACP, A2A or provider SDKs.

## Repository tooling

`.pre-commit-config.yaml` uses official remote Biome and Ruff hooks for staged TS/TSX and Python.
Checks never rewrite files. Install with `prefligit install`;
verify with `prefligit run --all-files`.

Build, cleanup, documentation coverage, and manuscript-model utilities live under `scripts/`.
Reproducible SoftwareX examples live under `examples/`; the manuscript is `swarmx.tex`.
