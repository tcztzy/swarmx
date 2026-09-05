# Agent platform

## Native integrations

| Agent | Interface | Setup |
| --- | --- | --- |
| Codex (default) | App Server, official CLI-generated types | Installed `codex` in PATH, native login/config |
| Claude | Official Agent SDK | SDK authentication and native Claude settings |
| Hermes | Installed `tui_gateway.entry` over stdio | `SWARMX_HERMES_PYTHON`: absolute Python path in the Hermes environment |
| OpenClaw | Official Gateway client over WebSocket | `OPENCLAW_GATEWAY_URL`, `OPENCLAW_GATEWAY_TOKEN` |

`--agent` → `SWARMX_AGENT` → `codex`. The browser can select any configured Agent. Only a selected
integration loads; errors are reported without fallback. DSH, ZCode and Kimi are not registered.

SwarmX preserves native settings rather than reproducing every vendor control in the UI.
Codex/Claude receive the product MCP server. Hermes/OpenClaw use their own configured tools;
SwarmX does not rewrite their global configuration. Native history is read on demand. Empty
Claude sessions have no native transcript until the first prompt.

## External interfaces

After `pnpm build`, an ACP client launches:

```sh
node apps/desktop/dist/acp-main.js --agent codex
```

Or run `pnpm --silent acp`. ACP uses the official SDK and stdio; stdout contains only protocol
messages. The process owns one workspace (`SWARMX_WORKSPACE`, otherwise cwd) and prints its A2A
URL to stderr. ACP supports initialize, list/new/load/prompt/cancel and form elicitation.
Client-injected MCP servers and alternate workspaces are rejected; configure the native Agent.

A2A discovery: `/a2a/swarm/.well-known/agent-card.json`. JSON-RPC: `/a2a/swarm`.
Calls require `A2A-Version: 1.0` and a bearer token. Set `SWARMX_API_TOKEN` before startup for
external clients; otherwise the Host generates a private process token for product carriers.
Only text SendMessage, GetTask and CancelTask are provided. Native approvals/questions require
ACP with form elicitation or the browser, not this A2A endpoint.

The browser uses official AG-UI input/events and assistant-ui's adapter. Stream disconnect stops
native work. Interaction resume completes the original request without starting another run.

## Verification

`agents.test.ts` checks lazy loading and native API mappings; `gateways.test.ts` exercises the
official ACP/A2A clients and browser security/streaming. Codex types are generated from the installed
CLI by `pnpm agents:types` and are not authored or bundled runtime code.

Opt-in real checks:

```sh
SWARMX_REAL_CODEX=1 pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
SWARMX_HERMES_PYTHON=/path/to/hermes/venv/bin/python pnpm exec vitest run apps/desktop/tests/agents-real.test.ts
```

The Codex check sends one no-tool prompt and reads its native history. The Hermes check exercises
session discovery/create/history/interrupt without an LLM call. Claude and OpenClaw mappings
currently have simulated SDK tests, not authenticated live tests. DVC tests run when its CLI is
available. UI trace timing is observational; it is not a verified provenance claim.
