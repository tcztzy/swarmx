# SwarmX Contract

Durable product requirements. See [product direction](docs/product-direction.md) for the objective,
[DESIGNS](DESIGNS.md) for ownership and feature documentation for implementation details.

## Architecture

- Renderer → AG-UI over Electron IPC → Host → Swarm → native Agent.
- SwarmX implements orchestration, not a built-in Agent. External runtimes own Agent loops,
  model/context management, sessions, compaction, tools and skill discovery.
- Codex is the default external runtime; a configured ACP endpoint becomes the default only
  with an explicit Host harness grant. The old `pi` selection fails with migration guidance;
  it never falls back to a different Agent or rewrites native session files.
- Native integrations: Codex App Server, Claude Agent SDK, DSH SDK Client,
  Hermes TUI Gateway JSON-RPC and OpenClaw Gateway Client SDK.
- Internal Swarm/member edges call the Agent interface directly. No ACP connections, envelopes,
  provider adapters or protocol types inside composition. Parents never inspect provider identity.
  No nesting limit. Each hop intersects trusted Host grants in the execution context.
- ACP and A2A remain external boundaries. ACP also supports explicitly configured external
  agent processes through the optional `acp` integration; composition still calls the Agent
  interface directly. The inbound ACP gateway retains its negotiated permissions extension.
- Native integrations own provider configuration, sessions, events, approvals and cancellation.
  Load only the selected integration. Startup errors propagate without implicit adapter fallback.
  An explicit Host policy may retry or choose another admitted configuration while retaining the
  original failure, accounting for every attempt and protecting non-repeatable side effects.
- ZCode and Kimi deferred. No placeholder integrations.
- Titles and native history stay owned by the runtime. SwarmX never requests a second title.
- Integrations preserve native lifecycle, approval, configuration and cancellation behavior;
  see [native Agents](docs/native-agents.md) for each runtime's supported contract.

## Ownership

- Native transcripts own runtime resume and context. History hydration reads native data.
  External ACP agents retain their own runtime, tools and native history; SwarmX does not load
  their underlying harness or reconstruct their context from its observation journal.
  Failed external resume never creates a replacement session or resubmits an old prompt.
- A Host-owned, append-only execution journal preserves observed Agent events and product-tool
  requests/results independently of native retention. The requested Agent configuration is the
  shared identity; an explicit native mismatch fails. Original native payloads retain observed
  metadata without a second reported-configuration projection. See [execution logs](docs/execution-log.md).
- Manuscript sources, references and reproducible publication cases belong to the separate
  `swarmx-paper` project. Software builds, tests and releases do not depend on that project.
  Case archives preserve raw records and input/code/output identities independently of readers.
- New sessions without a first native turn hydrate as empty, without reading an absent transcript.
- Swarm owns membership and delegation, not another transcript or transport state machine.
- Before Agent-originated delegation, the lead receives admitted candidates, the delegation skill
  and authorized evaluation evidence. Dispatch requires preparation for the same parent run/task
  and a recorded choice reason; evidence never grants authority. See [Swarm](docs/swarm.md).
- The Host owns `ProductServices`: Memory, Work and execution evidence, shared by the renderer bridge
  and product tools. Native Agent runtimes load lazily, once per harness, and are disposed at shutdown.
- The Host owns persistent work cycles and items, runtime budgets/timeouts, journal-derived costs
  and independent acceptance. Agent presets and temporary selections combine harness and model;
  reasoning effort belongs to the model, tools/skills to the native harness. Configuration selection
  precedes managed execution; full management uses an explicitly selected supervisor to decide
  continuations. Sessions, execution IDs and domain artifact revisions are references;
  delegation or retry does not create new business value, budget or responsibility.
  Runtime completion and task acceptance are independent. Only an authorized user or trusted
  validator records acceptance; an Agent's self-report cannot establish verified correctness.
  Every Agent's total cost includes all descendants and is used in configuration comparisons.
  Each independent charge contributes once to cycle spending; missing LLM costs remain incomplete.
  Resource governance belongs above the lightweight Swarm layer; see [work management](docs/work-management.md).
- One canonical execution directory is fixed at Host startup, using `SWARMX_CWD` or the process's
  starting directory. Native sessions, domain references and execution logs retain their directory
  ownership. ACP and A2A cannot retarget the Host to another directory.
- Settings contains language, permissions and Memory preferences. Retired environment metadata remains readable without enabling a runtime.
- [Memory](docs/memory.md) combines a bounded user note, frozen session context, journal-backed
  recall and flat OKF concepts with revision-pinned acyclic prerequisites. Native skills remain
  separate from structured vault knowledge. New sessions receive a short tool entry point and
  user note; the complete authoring guide and relevant concepts load on demand.
- Configurable post-turn reviews propose durable learning using a restricted native runtime.
  Host validation, revision checks and optional user-owned approval govern writes; Agents cannot
  approve their own changes.
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
- MCP carries product tool calls, never inter-Agent messages.
- ACP/A2A gateways translate external requests into the same Swarm operations. A2A Task storage
  holds communication lifecycle and SDK-owned ingress messages; not native transcripts or research tasks.
- Native events retain their provider identifiers and JSON payloads in the execution journal.
  UI projections and react-o11y spans remain transient. Domain applications and Memory retain their authoritative
  records; logged tool results link execution to their revisions and provenance locators.

## Interfaces and security

- The Host authorizes SwarmX product tools, harness/model admission and delegation. Effective grants
  intersect Host policy, caller, named Swarm and saved conversation grants; children cannot widen them.
  Active tool grants distinguish Memory reads and writes; old domain grant strings are inert saved data. Every native product MCP call binds to
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

- Scientific models, notebooks, artifact stores and execution environments belong to domain
  applications. SwarmX does not expose Science tools or scientific IPC routes.
- Domain checks use ordinary authorized Agent/tool calls discovered from their capability
  descriptions. Work records opaque artifact IDs/revisions with required producing/verification
  execution references. The Host validates directory, Work and runtime provenance, not domain
  syntax or scientific validity. Self-reports remain claims; independent acceptance pins the
  identities, evidence, criteria and evaluator report. There is no domain resolver registration.
  Legacy records without evidence stay readable but do not qualify as new artifact submissions.
  Memory validates owned execution URNs and labels other URI references unverified.

- Agent: models/list/create/read/start/steer/interrupt/dispose. Model catalogs and per-run
  model/effort/mode selections delegate through nested Swarms to the native integration.
  Interaction replies use the pending
  native request's callback; no separate controller or polling state machine.
- Renderer: official AG-UI input schema over Electron IPC; native history, streaming text, tool cards,
  interaction forms, Stop, Agent selector and conversation sidebar.
- Sub-Agent cards project recorded delegation links: per-run status, task, Harness and model,
  expandable observed conversation/logs, active-run steering and Stop. Child confirmations reach
  the parent; parallel completion order and restart do not fabricate progress or success.
- The [desktop UI](docs/desktop-ui.md) uses assistant-ui, react-o11y and Tailwind. Navigation
  preserves the draft and active stream; English/Chinese preferences persist without translating
  scientific records or code. Explicit history reload never sends a prompt.
- Native model/mode catalogs remain authoritative. Configuration changes apply to the next
  message; switching harnesses loads its own sessions without migrating transcripts or rewriting
  vendor settings. Configuration controls are disabled during a run.
- ACP: official SDK over stdio, `pnpm acp`; stdout reserved for protocol messages.
  Outbound `acp` is opt-in through an explicit endpoint descriptor and Host harness grant.
  It forwards the remote agent's actual capabilities, choices and permission requests; it
  does not grant filesystem/terminal access or inject Host credentials into the child.
  Cancel waits for the original prompt terminal outcome; disposal owns and awaits child exit.
- A2A: official SDK JSON-RPC endpoint and discoverable Agent Card; bearer required for calls.
- Electron IPC accepts validated operations from the application's top-level window. The preload
  exposes named operations. A2A uses a random loopback port and bearer-authenticated calls;
  native product tools use an execution-bound credential over the Host MCP socket.
- Message regeneration/edit/fork UI and Assistant Cloud are not currently implemented.
  Domain artifact editing and revisions belong to the domain application. Protocol integrations use official SDKs;
  native approval requests require their authorized response. Explicit retry/escalation and learning
  policies cannot expand grants or silently rewrite failures.

## Acceptance

- `swarm.test.ts`: recursive delegation and cancellation without provider/protocol imports.
- `agents.test.ts`: external default, retired builtin rejection, lazy selection and native requests;
  failures remain failures. Native configuration is not narrowed to protocol capabilities.
- `permissions.test.ts`: recursive Host grants, native-mode independence, model admission on resumed
  sessions, policy revocation, concurrent isolation, product writes and delegation rejection.
- `gateways.test.ts`: official ACP/A2A clients reach the same native Agent; Card discovery and
  cancellation; official AG-UI schema, hydration, interaction resume, foreign-session rejection.
- Execution-journal tests: append-only persistence, restart reads, directory isolation, writes
  before dispatch/delivery, raw payload preservation, per-run settings and causal tool/delegation links.
- `boundaries.test.ts`: no provider/protocol/UI dependencies in public packages; ACP is confined
  to external ingress/egress adapters; no ACP imports in recursive composition or upstream
  harness patches in production dependencies.
- `external-acp.test.ts`: real stdio child lifecycle, explicit endpoint admission, remote-owned
  sessions, permission forwarding/rejection, cancellation and disconnect cleanup.
- Renderer interaction tests: session search/create/switch, stale request isolation, native
  history, suggested drafts, streaming/Stop, message copy and explicit interaction responses.
- Run the [engineering checks](docs/product-readiness.md#verification) against the current candidate.
  Real Agent/DVC tests require configured environments; skipped coverage is reported explicitly.
