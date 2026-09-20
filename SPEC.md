# SwarmX Contract

Local research Host; recursive Swarms; native Agents.

## Architecture

- Renderer → AG-UI over Electron IPC → Host → Swarm → native Agent.
- Pi SDK is the default Swarm runtime. Pi owns its Agent loop, sessions, compaction, model
  configuration, built-in tools and native resource discovery. Skills use Pi's on-demand loading.
  SwarmX adds its existing product tools and permission boundary through SDK custom tools.
- Other native integrations: Codex App Server, Claude Agent SDK, DSH SDK Client,
  Hermes TUI Gateway JSON-RPC and OpenClaw Gateway Client SDK.
- Codex uses the installed `codex` executable; generate types with that same CLI.
- Internal Swarm/member edges call the Agent interface directly. No ACP connections, envelopes,
  provider adapters or protocol types inside composition. Parents never inspect provider identity.
  No nesting limit. Each hop intersects trusted Host grants in the execution context.
- ACP and A2A are external gateways only. ACP retains its negotiated permissions extension.
- Native integrations own provider configuration, sessions, events, approvals and cancellation.
  Load only the selected integration. Startup errors propagate; no fallback.
- ZCode and Kimi deferred. No placeholder integrations.
- Titles and native history stay owned by the runtime. SwarmX never requests a second title.
- DSH exposes its SDK's actual capabilities: cancellation closes the owned runtime;
  unsupported interactive approval is not represented as an automatic grant.
  Each task is an independent execution with Host log viewing, without cross-process resume.
  Explicit provider/model routes, reasoning effort and SDK profiles are selected at launch;
  profiles remain separate from permission modes. The SDK validates routes without model discovery.
- Hermes execution uses native resume, including automatic continuation; no runtime patch suppresses it.
  OpenClaw accepts explicit Gateway address and credentials.

## Ownership

- Native transcripts own runtime resume and context. History hydration reads native data.
- A Host-owned, append-only execution journal preserves observed Agent events and product-tool
  requests/results independently of native retention. It records requested settings, reported
  generation metadata, interactions, failures, cancellation requests and delegation links.
- Manuscript sources, references and reproducible publication cases belong to the separate
  `swarmx-paper` project. Software builds, tests and releases do not depend on that project.
  Case archives preserve raw records and input/code/output identities independently of readers.
- New sessions without a first native turn hydrate as empty, without reading an absent transcript.
- Claude completes on its SDK idle notification after a result; queued native output is not truncated.
- Swarm owns membership and delegation, not another transcript or transport state machine.
- Before Agent-originated delegation, the lead receives admitted candidates, versioned bundled
  selection knowledge and authorized current private Memory. Dispatch requires a completed
  preparation for the same parent run/task and a recorded choice reason; evidence never grants authority.
- The Host owns `ProductServices`: Science and Memory, shared by the renderer bridge
  and product tools. Native Agent runtimes load lazily, once per harness, and are disposed at shutdown.
- One canonical execution directory is fixed at Host startup, using `SWARMX_CWD` or the process's
  starting directory. Native sessions, scientific records and execution logs retain their directory
  ownership. ACP and A2A cannot retarget the Host to another directory.
- Settings contains language, permissions, environment and memory preferences.
- Memory combines a bounded user note, frozen per-session context, journal-backed
  conversation recall, and a flat OKF concept pool with revision-pinned acyclic prerequisites
  and ordered loading.
- Memory authors use American English metadata, retaining original names for concepts/entities
  specific to a language community, culture or institution when needed for their identity.
  Bodies may use any language or mix languages; Host instructions guide all Agent writing paths.
- New Memory concept filenames use concise canonical names without random suffixes. An occupied
  filename requires reading and updating the existing concept.
- Configurable post-turn reviews propose durable learning using a restricted native runtime.
  Host validation, revision checks and optional user-owned approval govern writes; Agents cannot
  approve their own changes. Native skills remain separate from structured vault knowledge.
- Eligible success, failure and cancellation outcomes feed a persistent execution-directory review
  backlog. Reviews retain original grants, survive restart and save their operation plans before
  writes. Completed batches acknowledge only their own evidence; no-change outcomes state a reason.
  One review covers H/M/P selection, agent prompts and skills. Explicitly registered project
  resources evolve only at the supplied revision after their fixed project validator succeeds.
- Selection evaluations and prompt/skill improvement proposals cite original execution evidence
  with explicit task, criteria and limitations. The Host validates references and computes scoped
  execution statistics; subjective judgments, user preferences and unknown data remain distinct.
  Users and delegating leads can inspect the same evidence. A valid citation is not proof of a claim.
  Selected evaluations and their generation process export as self-contained RO-Crate evidence
  packages, retaining original bytes, pinned revisions and explicit limits on what was observed.
- Concepts live directly under `memory/` with reserved `index.md`, `README.md` and `USER.md`;
  there are no scope folders and no private revision store.
- MCP carries product tool calls, never inter-Agent messages.
- ACP/A2A gateways translate external requests into the same Swarm operations. A2A Task storage
  holds communication lifecycle and SDK-owned ingress messages; not native transcripts or research tasks.
- Native events retain their provider identifiers and JSON payloads in the execution journal.
  UI projections and react-o11y spans remain transient. Science/Memory retain their authoritative
  domain records; logged tool results link execution to their revisions and provenance locators.

## Interfaces and security

- The Host authorizes SwarmX product tools, harness/model admission and delegation. Effective grants
  intersect Host policy, caller, named Swarm and saved conversation grants; children cannot widen them.
  Tool grants distinguish Memory/Science reads and writes. Every native product MCP call binds to
  an active execution. These checks govern Host APIs, not arbitrary native processes or files.
- Ordinary tasks use harness-native permission modes, discovered and selected through native APIs without
  a shared ranking or sandbox/approval overrides. Approvals retain native options and scope; late
  responses cannot approve ended runs. External ACP carries requests and responses, not filesystem isolation.
  A Plan mode does not narrow a child's Host grant; YOLO does not expand one. SwarmX does not promise
  a filesystem permission ceiling across harnesses. Background memory reviews retain a separate
  Host boundary rejecting observed tools and approvals and never inherit an ordinary task's YOLO
  selection. Requested native review restrictions follow unmodified upstream behavior; no
  cross-harness tool-free, title-suppression or ephemeral-session guarantee is claimed.
- Permissions follow each conversation across messages, aliases and Host restarts, including empty
  sessions. Creation and admitted-run grants are persisted in the execution journal and intersected
  on reuse. Grants can only narrow; omitted arguments, failed turns and broader Host settings
  never reset them. Session model catalogs respect the same saved model ceiling.

- Research sandbox, settings, environments and artifact UI follow
  `docs/product-readiness.md`. Scientific execution fails closed without its isolated
  environment; native Agent permission boundaries remain independently visible.

- Agent: models/list/create/read/start/steer/interrupt/dispose. Model catalogs and per-run
  model/effort/mode selections delegate through nested Swarms to the native integration.
  Interaction replies use the pending
  native request's callback; no separate controller or polling state machine.
- Renderer: official AG-UI input schema over Electron IPC; native history, streaming text, tool cards,
  interaction forms, Stop, Agent selector and conversation sidebar.
- Sub-Agent cards project recorded delegation links: per-run status, task, Harness and model,
  expandable observed conversation/logs, active-run steering and Stop. Child confirmations reach
  the parent; parallel completion order and restart do not fabricate progress or success.
- React ref-as-prop, assistant-ui, react-o11y and Tailwind with the official homepage Demo's
  light tokens, typography and controls; see `docs/desktop-ui.md`.
- Codex-inspired desktop layout: collapsible task navigation, native session search, compact
  conversation header, centered composer, starter prompts and message copy. Assets open beside
  the conversation; Observe groups react-o11y, recorded runs and RO-Crate. The bottom-left
  settings control opens full-page Settings. Navigation preserves the draft and active stream.
  English/Chinese UI selection is persistent and never translates scientific records or code.
  Narrow windows retain task and Agent navigation. Loading and failures remain visible.
  Explicit history reload reads the same native session without sending a prompt or replacing
  the draft. Pinned Science sources retain their producing computation beyond the recent-run list.
- Composer: Harness picker and an assistant-ui-style model popover with Thinking segments below
  the model list and the active effort beside the model name, plus advertised native mode choices.
  Native catalogs remain authoritative. Changes apply to the next message in
  the same native session; switching Harness loads its own sessions. Do not rewrite global
  vendor settings or migrate transcripts. Configuration controls are disabled during a run.
- ACP: official SDK over stdio, `pnpm acp`; stdout reserved for protocol messages.
- A2A: official SDK JSON-RPC endpoint and discoverable Agent Card; bearer required for calls.
- Electron IPC accepts validated operations from the application's top-level window. The preload
  exposes named operations. A2A uses a random loopback port and bearer-authenticated calls;
  native product tools use an execution-bound credential over the Host MCP socket.
- No message regeneration/edit/fork UI, Assistant Cloud, protocol shims, automatic approval,
  automatic retries or fallback. Scientific artifact editing preserves its own revisions.

## Acceptance

- `swarm.test.ts`: recursive delegation and cancellation without provider/protocol imports.
- `agents.test.ts`: default Pi, lazy selection, native requests/events/interactions/Stop;
  failures remain failures. Native configuration is not narrowed to protocol capabilities.
- `permissions.test.ts`: recursive Host grants, native-mode independence, model admission on resumed
  sessions, policy revocation, concurrent isolation, product writes and delegation rejection.
- `gateways.test.ts`: official ACP/A2A clients reach the same native Agent; Card discovery and
  cancellation; official AG-UI schema, hydration, interaction resume, foreign-session rejection.
- Execution-journal tests: append-only persistence, restart reads, directory isolation, writes
  before dispatch/delivery, raw payload preservation, per-run settings and causal tool/delegation links.
- `boundaries.test.ts`: no provider/protocol/UI dependencies in public packages; ACP is confined
  to external ingress; no upstream ACP harness adapters or patches in production dependencies.
- Renderer interaction tests: session search/create/switch, stale request isolation, native
  history, suggested drafts, streaming/Stop, message copy and explicit interaction responses.
- `pnpm typecheck`, `pnpm test`, `pnpm build`, `pnpm lint`, `pnpm docs:check` pass.
- Real Agent/DVC tests require configured environments; skipped coverage is reported explicitly.
