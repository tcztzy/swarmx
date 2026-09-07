# SwarmX Contract

Local research Host; recursive Swarms; native Agents.

## Architecture

- Renderer → AG-UI → Host → Swarm → native Agent.
- Upstream codex-acp (default), claude-agent-acp, `hermes acp`, `openclaw acp` over official ACP.
- Codex uses the installed `codex` executable; generate types with that same CLI.
- ACP provides shared capability, prompt-result, approval and cancellation semantics;
  official SDK connections work in process or over stdio. SwarmX extends ACP through negotiated
  metadata for persistent Host API grants and trusted delegation context. A2A is external ingress.
- Every internal Swarm/member edge uses an official ACP connection. A Swarm is an ACP Agent
  upstream and ACP Client downstream; parents never inspect provider identity. No nesting limit.
- Native integrations own provider configuration, sessions, events, approvals and cancellation.
  Load only the selected integration. Startup errors propagate; no fallback.
- DSH, ZCode and Kimi deferred. No placeholder integrations.

## Ownership

- Native transcripts own runtime resume and context. History hydration reads native data.
- A Host-owned, append-only execution journal preserves observed Agent events and product-tool
  requests/results independently of native retention. It records requested settings, reported
  generation metadata, interactions, failures, cancellation requests and delegation links.
- New sessions without a first native turn hydrate as empty, without reading an absent transcript.
- Claude completes on its SDK idle notification after a result; queued native output is not truncated.
- Swarm owns membership and delegation, not another transcript or transport state machine.
- Projects persist a name and canonical working directory. Each project owns its ProductServices,
  native tasks, research records, permissions, environment and workspace memory. Project-scoped
  browser/MCP/A2A routes never follow another window's selected project.
- The sidebar adds and opens projects. Global Settings contains language and user preferences;
  project settings show that project's directory, permissions, environment and memory.
- One `ProductServices` per loaded project: Science, Memory, Git and DVC; shared by REST and MCP.
- Memory combines bounded user/workspace notes, frozen per-session context, journal-backed
  conversation recall, and an OKF vault with revision-pinned acyclic prerequisites and ordered loading.
- Configurable post-turn reviews propose durable learning using a restricted native runtime.
  Host validation, revision checks and optional user-owned approval govern writes; Agents cannot
  approve their own changes. Native skills remain separate from structured vault knowledge.
- The storage upgrade moves the previous vault to `memory/vault` without changing stored bytes;
  conflicting vaults fail closed. Current package and tool names have no aliases.
- MCP carries product tool calls, never inter-Agent messages.
- ACP/A2A gateways translate external requests into the same Swarm operations. A2A Task storage
  holds communication lifecycle and SDK-owned ingress messages; not native transcripts or research tasks.
- Native events retain their provider identifiers and JSON payloads in the execution journal.
  UI projections and react-o11y spans remain transient. Science/Memory retain their authoritative
  domain records; logged tool results link execution to their revisions and provenance locators.

## Interfaces and security

- The Host authorizes SwarmX product tools, harness/model admission and delegation. Effective grants
  intersect project, caller, named Swarm and saved conversation grants; children cannot widen them.
  Tool grants distinguish Memory/Science reads and writes. Every native product MCP call binds to
  an active execution. These checks govern Host APIs, not arbitrary native processes or files.
- Ordinary tasks use harness-native permission modes, discovered and selected through ACP without
  a shared ranking or sandbox/approval overrides. Approvals retain native options and scope; late
  responses cannot approve ended runs. ACP carries requests and responses, not filesystem isolation.
  A Plan mode does not narrow a child's Host grant; YOLO does not expand one. SwarmX does not promise
  a filesystem permission ceiling across harnesses. Background memory reviews retain a separate
  restricted, tool-free execution boundary and never inherit an ordinary task's YOLO selection.
- Permissions follow each conversation across messages, aliases and Host restarts, including empty
  sessions. Creation and admitted-run grants are persisted in the execution journal and intersected
  on reuse. Grants can only narrow; omitted arguments, failed turns and broader project settings
  never reset them. Session model catalogs respect the same saved model ceiling.

- Research sandbox, settings, environments and artifact UI follow
  `docs/product-readiness.md`. Scientific execution fails closed without its isolated
  environment; native Agent permission boundaries remain independently visible.

- Agent: models/list/create/read/start/steer/interrupt/dispose. Model catalogs and per-run
  model/effort/mode selections delegate through nested Swarms to the native integration.
  Interaction replies use the pending
  native request's callback; no separate controller or polling state machine.
- Browser: official AG-UI input schema; native history, streaming text/reasoning, tool cards,
  interaction forms, Stop, Agent selector and conversation sidebar.
- Sub-Agent cards project recorded delegation links: per-run status, task, Harness and model,
  expandable observed conversation/logs, active-run steering and Stop. Child confirmations reach
  the parent; parallel completion order and restart do not fabricate progress or success.
- React 18, assistant-ui, react-o11y, unconfigured Tailwind; monochrome; no product theme.
- Codex-inspired desktop layout: collapsible task navigation, native session search, compact
  conversation header, centered composer, starter prompts and message copy. Assets open beside
  the conversation; Observe groups react-o11y, recorded runs and RO-Crate. The bottom-left
  workspace row opens full-page Settings. Navigation preserves the draft and active stream.
  English/Chinese UI selection is persistent and never translates scientific records or code.
  Narrow windows retain task and Agent navigation. Loading and failures remain visible.
- Composer: Harness picker and an assistant-ui-style model popover with Thinking segments below
  the model list and the active effort beside the model name, plus advertised native mode choices.
  Native catalogs remain authoritative. Changes apply to the next message in
  the same native session; switching Harness loads its own sessions. Do not rewrite global
  vendor settings or migrate transcripts. Configuration controls are disabled during a run.
- ACP: official SDK over stdio, `pnpm acp`; stdout reserved for protocol messages.
- A2A: official SDK JSON-RPC endpoint and discoverable Agent Card; bearer required for calls.
- Host: random loopback port; one-use launch token → HttpOnly Strict cookie; Host, Origin,
  session ownership and canonical static-path checks remain.
- No conversation Retry/Edit/Fork UI, Assistant Cloud, protocol shims, automatic approval,
  retries or fallback. Scientific artifact editing preserves its own revisions.

## Acceptance

- `swarm.test.ts`: recursive delegation and cancellation without provider/protocol imports.
- `agents.test.ts`: default Codex, lazy selection, native requests/events/interactions/Stop;
  failures remain failures. Native configuration is not narrowed to protocol capabilities.
- `permissions.test.ts`: recursive Host grants, native-mode independence, model admission on resumed
  sessions, project revocation, concurrent isolation, product writes and delegation rejection.
- `gateways.test.ts`: official ACP/A2A clients reach the same native Agent; Card discovery and
  cancellation; official AG-UI schema, hydration, interaction resume, foreign-session rejection.
- Execution-journal tests: append-only persistence, restart reads, workspace isolation, writes
  before dispatch/delivery, raw payload preservation, per-run settings and causal tool/delegation links.
- `boundaries.test.ts`: no provider/UI dependencies in public packages; ACP belongs to Swarm; no old ACP
  adapters or deferred Agents in production dependencies.
- Renderer interaction tests: session search/create/switch, stale request isolation, native
  history, suggested drafts, streaming/Stop, message copy and explicit interaction responses.
- `pnpm typecheck`, `pnpm test`, `pnpm build`, `pnpm lint`, `pnpm docs:check` pass.
- Real Agent/DVC tests require configured environments; skipped coverage is reported explicitly.
