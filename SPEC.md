# SwarmX Contract

Local research Host; recursive Swarms; native Agents.

## Architecture

- Renderer → AG-UI → Host → Swarm → native Agent.
- Codex App Server (default), Claude Agent SDK, Hermes TUI gateway, OpenClaw Gateway.
- Codex uses the installed `codex` executable; generate types with that same CLI.
- ACP stdio and A2A 1.0 JSON-RPC/HTTP → external SwarmX entry points only.
- Internal composition uses a small in-process Agent interface. A Swarm implements the same
  interface as its members; parents never inspect provider identity. No nesting limit.
- Native integrations own provider configuration, sessions, events, approvals and cancellation.
  Load only the selected integration. Startup errors propagate; no fallback.
- DSH, ZCode and Kimi deferred. No placeholder integrations.

## Ownership

- Native transcripts remain the only conversation records. History hydration reads native data.
- New sessions without a first native turn hydrate as empty, without reading an absent transcript.
- Claude completes on its SDK idle notification after a result; queued native output is not truncated.
- Swarm owns membership and delegation, not another transcript or transport state machine.
- One Host-owned `ProductServices`: Science, Memory, Git and DVC; shared by REST and MCP.
- Memory is a shared semantic memory store of OKF research concepts, accessed through explicit
  search/read/create/update/deprecate/lint operations. Native Agents own conversation history.
- The storage upgrade moves the previous vault to `memory/vault` without changing stored bytes;
  conflicting vaults fail closed. Current package and tool names have no aliases.
- MCP carries product tool calls, never inter-Agent messages.
- ACP/A2A gateways translate external requests into the same Swarm operations. A2A Task storage
  holds communication lifecycle and SDK-owned ingress messages; not native transcripts or research tasks.
- Native events retain their provider identifiers; UI projections and react-o11y spans are
  transient views, not audit evidence. Science/Memory retain their own durable domain records.

## Interfaces and security

- Agent: list/create/read/start/steer/interrupt/dispose. Interaction replies use the pending
  native request's callback; no separate controller or polling state machine.
- Browser: official AG-UI input schema; native history, streaming text/reasoning, tool cards,
  interaction forms, Stop, Agent selector and conversation sidebar.
- React 18, assistant-ui, react-o11y, unconfigured Tailwind; monochrome; no product theme.
- ACP: official SDK over stdio, `pnpm acp`; stdout reserved for protocol messages.
- A2A: official SDK JSON-RPC endpoint and discoverable Agent Card; bearer required for calls.
- Host: random loopback port; one-use launch token → HttpOnly Strict cookie; Host, Origin,
  session ownership and canonical static-path checks remain.
- No Retry/Edit/Fork UI, Assistant Cloud, protocol shims, automatic approval, retries or fallback.

## Acceptance

- `swarm.test.ts`: recursive delegation and cancellation without provider/protocol imports.
- `agents.test.ts`: default Codex, lazy selection, native requests/events/interactions/Stop;
  failures remain failures. Native configuration is not narrowed to protocol capabilities.
- `gateways.test.ts`: official ACP/A2A clients reach the same native Agent; Card discovery and
  cancellation; official AG-UI schema, hydration, interaction resume, foreign-session rejection.
- `boundaries.test.ts`: no provider/UI/protocol dependencies in public packages; no old ACP
  adapters or deferred Agents in production dependencies.
- `pnpm typecheck`, `pnpm test`, `pnpm build`, `pnpm lint`, `pnpm docs:check` pass.
- Real Agent/DVC tests require configured environments; skipped coverage is reported explicitly.
