# SwarmX Product Specification

Status: current

SwarmX is a local-first desktop workspace and TypeScript platform for running
direct model agents, ACP-compatible coding agents, and durable background work.
It composes a runtime Harness with an independent Model, gives that Agent bounded
access to a Project, preserves conversations as resumable Sessions, and persists
durable execution independently as WorkItems.

This document defines the durable product contract. It is intentionally not an
implementation map, test plan, backlog, changelog, or incident log.

## Product model

| Concept | Contract |
| --- | --- |
| Project | A local folder bookmark and the containment root for task tools. It is not a remote workspace or authorization domain. |
| Provider | A connection and credential source that may supply Models. It does not own Model identity. |
| Model | A primary entity with a stable id, API compatibility, and capabilities. |
| Harness | A reproducible runtime recipe: Software, selected Skills and MCP servers, Project context, delivery capabilities, and permission policy. |
| Agent | Exactly one Harness paired with one Model. Its identity is `harnessId:modelId`; Provider routing and effort do not change it. |
| Session | The canonical, resumable conversation record, persisted as an append-only event log. It may observe a WorkItem but is not task authority. |
| Memory | User-owned subjective durable knowledge. `USER.md` and `MEMORY.md` are its bounded global snapshot; the current linked-page organization provides explicit entity CRUD and revisions. An organization method is replaceable implementation, not a separate product concept. Memory is not Activity Profile data, Session history, Project context, task authority, or an executable workflow. |
| Reference Library | Explicitly configured, read-only objective sources: a local ZIM archive and the local Zotero library. It is not editable Memory, Session history, Project context, Web Search, or a download service. |
| WorkItem | A durable unit of language-independent work whose runs, leases, checkpoints, artifacts, and events survive Session changes. |
| Workflow | A `SwarmConfig` graph of agents, tools, nested swarms, and explicit edges. |

## Product invariants

1. **One workflow format.** `SwarmConfig` is the only persisted execution graph.
   Imports, editors, previews, CLI calls, and Desktop execution converge on it.
2. **Explicit composition.** Model, Provider, Harness, Agent, Extension, and
   Project remain separate concepts. The ordinary composer asks for Harness,
   Model, and Effort; trusted runtime code resolves supply details.
3. **Host-owned tools.** Direct SwarmX tasks may receive bounded Project tools.
   External ACP Harnesses retain their native tools and permission systems and
   never receive a duplicate SwarmX tool surface.
4. **Local-first authority.** Sessions, Projects, settings, credentials, and
   managed media are local by default. Optional server and telemetry features
   are explicit and disabled unless configured.
5. **Isolation by construction.** The Electron renderer has no direct
   filesystem, process, network-credential, or secret access. Privileged work
   crosses a typed Preload/Main boundary and is authorized for the main frame.
6. **Schema and secret boundaries.** External and persisted data is validated at
   its boundary. Metadata stores secret references or status, never plaintext
   credentials, raw environment snapshots, or secret-bearing logs. The dedicated
   local Provider auth file is an explicit user-editable plaintext credential
   store and remains outside metadata, Renderer data, and logs.
7. **Explicit side effects.** Discovery and planning are read-only. Installation,
   repair, trust changes, permission grants, destructive actions, and external
   writes require an explicit action and the appropriate confirmation.
8. **Honest capabilities.** SwarmX advertises only behavior it implements.
   Unsupported host tools, media transports, provider features, and ACP
   capabilities fail clearly instead of silently degrading to different
   semantics.
9. **Portable core.** Generic schemas and decision helpers stay host-neutral and
   side-effect free. Filesystem, process, keychain, installer, and UI effects
   belong to host adapters.
10. **Independent task authority.** Durable WorkItems use their own append-only
    event authority. Sessions may create or observe the same WorkItem, while
    Session switching, unlinking, or archiving does not cancel execution.
11. **Concise auditability.** Privileged decisions and side effects produce
    correlated, structured, secret-free audit events. Intent is durable before
    authority expands or an effect starts, and an unavailable audit authority
    fails closed at those boundaries.
12. **Transparent bounded global memory.** `USER.md` and `MEMORY.md` are
    explicitly viewable, editable, and forgettable, have strict independent
    capacities, and are injected only as one read-only per-run snapshot.
    `USER.md` contains stable user identity, preferences, and working habits;
    `MEMORY.md` contains cross-Project environment facts, conventions,
    decisions, and reusable experience. Direct Agents, external ACP Harness
    prompts, and Agent-bearing workflows report the actual snapshot use;
    plaintext Memory is never audit, trace, telemetry, or unrelated transport
    data. Agent-initiated save or forget requests require an in-run human
    confirmation before effect.
13. **Independent durable execution.** Eligible WorkItems run under one
    authenticated local supervisor that owns leases and cancellation independently
    from Electron, so closing Desktop does not terminate active durable work.
14. **Revision-safe linked-page Memory.** Memory pages are created, read, listed,
    searched, updated, deleted, versioned, diffed, and restored through strict
    bounded schemas. Updates, deletes, and restores require the current page
    revision; success is reported only after the Markdown write, Git commit,
    and search-index refresh complete. Markdown plus Git is the current Memory
    authority, while search indexes and knowledge edges remain rebuildable
    projections. SwarmX-owned Agents receive bounded on-demand reads
    and may mutate only after one-call user confirmation; external ACP Harnesses
    do not receive the local tool.
15. **Verified module runtimes.** Language-native feature modules run
    as SwarmX-managed, version-pinned MCP servers over private stdio. Runtime
    inspection is read-only, launch revalidates code or executable digests,
    protocol version, server identity, and an exact allowlisted tool surface;
    ordinary module calls never download, compile, install, or repair
    dependencies. The Rust `swarmx-mem` server is a root-workspace crate under
    `crates/`. Python ships as one standard `swarmx` distribution with regular
    `swarmx.rsi`, `swarmx.ref`, and `swarmx.worker` subpackages; its private MCP
    server identities remain `swarmx-rsi` and `swarmx-ref`, but they are not
    separately discovered distributions or dependency groups. They receive
    sanitized environments and bounded,
    explicitly granted inputs, are never registered as Agent-facing MCP
    servers, and do not acquire persistence, scheduling, permission, Provider
    credential, or audit authority from MCP itself.
16. **Subjective Memory versus objective Reference.** Memory is curated,
    user/Agent-authored subjective knowledge with CRUD and Git versions.
    Reference Library is read-only access to explicitly configured sources: a
    local ZIM archive and Zotero Desktop's local
    API. It can report source metadata, search, and return bounded plaintext
    records, but cannot download, create, update, delete, read Zotero attachment
    content, access Web Search, fetch arbitrary URLs, or silently promote
    reference text into Memory. Active HTML is stripped before model use. A
    direct Agent configured with official DeepSeek, OpenAI API, or Codex
    Responses credentials exposes that Provider's server-side Web Search.
    Hosted search calls remain visible as correlated tool lifecycle events, and
    continuation preserves Provider-owned search state. Anthropic `pause_turn`
    responses are continued by replaying the complete assistant content without
    fabricating a client tool result. DeepSeek's official Anthropic endpoint
    keeps the same provider-native behavior for models that do not support
    Responses. Gateways, unsupported protocols, and lookalike endpoints do not
    gain hosted search. When no Reference source is configured, the Agent tool
    is not injected and SwarmX never claims that Reference was used.
17. **Session-scoped reflective Memory.** An explicit user request to remember
    may prompt an immediate Memory proposal. Otherwise each persisted Session
    maintains an independent review cursor and receives a reflection reminder
    after each ten completed foreground user-Agent turns; switching Sessions or
    restarting Desktop preserves that Session's cursor but never combines raw
    content from different Sessions into one reflection window. An archived or
    sufficiently idle Session may review its remaining tail independently.
    The reminder is attached only to that Session's normal bounded execution
    context and does not add dialogue from any other Session. It asks the Agent
    to emit typed, source-bearing candidates for `USER.md`, `MEMORY.md`, or exact entity-page upsert;
    candidates do not become Memory until the configured human admission gate
    approves the proposed write. Repeated review and retry are idempotent.

## Required capabilities

### Runtime and workflows

- Run a direct single-Agent task without requiring a workflow.
- Parse, validate, preview, and execute `SwarmConfig` workflows.
- Persist WorkItems, runs, fenced leases, cancellation, retry decisions,
  execution checkpoints, artifact references, approvals, and side-effect
  receipts independently of Sessions, then rebuild state by replaying events.
- Run replaceable language workers through a versioned, strictly validated
  protocol. Workers execute granted operations but never own scheduling,
  authoritative state, or unrestricted Provider credentials.
- Recover expired leases conservatively and resume only from an execution
  checkpoint produced by the same verified environment. Context packets and
  summary checkpoints remain model-context aids, not execution checkpoints.
- Compile bounded model context from immutable event snapshots through
  replaceable, provenance-preserving projections. Tool calls remain atomic with
  their results, derived historical claims cite source events, live repository
  observations outrank historical projections, and overflow fails explicitly
  before a Provider request instead of being silently truncated.
- Budget the complete Provider request against the selected Model/Supply window,
  including instructions, current input, attachments, tool schemas, projected
  history, and output reserve. Below the configured pressure threshold, preserve
  prior history losslessly; above it, use a source-linked checkpoint, verbatim
  recent atomic tail, and verified evidence without replacing canonical history.
- Select context behavior through a serializable, named evaluation profile. Ship
  reproducible profiles for OpenCode, Codex CLI local compaction, Claude Code,
  Hermes, Reasonix, Lossless Context Management, Parallel Context Compaction,
  and ReSum alongside SwarmX and full-history baselines. Every manifest records
  the selected profile, its public-source/behavior/paper fidelity class, summary
  path, and subcall count; a closed or provider-hosted implementation is never
  advertised as exact parity.
- Keep Recursive Language Model execution outside the compaction-profile
  boundary until the host supplies an explicitly authorized sandboxed program
  environment and bounded recursive model-call protocol. Retrieval or ordinary
  summarization alone must not be labelled RLM.
- Evaluate context profiles through versioned agentic-coding suites. Every arm
  receives the same immutable history and a fresh clone of the same simulated
  environment; seeded execution order changes only arm order. Score observable
  final state, exact retained constraints, recovery, repeated/blocked actions,
  safety, strategy failures, Provider failures, tokens, cost, and latency.
- Keep context-evaluation artifacts content-free: store source/config/output
  hashes, action ids and statuses, manifests, metrics, and score evidence, never
  raw prompts, histories, model responses, tool output, credentials, or state
  values. Matrix size and adaptive search rounds are bounded before any model
  call; summary and continuation Agents cannot receive MCP, hooks, hosted Web
  Search, or real Project mutation authority.
- Describe external effects as at-least-once, require stable idempotency keys
  and durable outcome receipts, and preserve an `unknown` outcome when a crash
  prevents proof of completion. Never advertise exactly-once delivery.
- Submit, inspect, and cancel eligible WorkItems through an authenticated local
  supervisor that reuses the canonical event store, fencing, worker protocol,
  capability grants, and recovery logic. After accepting a verified run recipe,
  the same supervisor automatically redispatches retryable failures and approved
  human pauses within the WorkItem's attempt budget; Electron is a client, not
  task authority.
- Structurally import n8n workflow JSON into `SwarmConfig`, preserving topology
  and inert metadata without importing secrets or executing n8n node runtimes.
- Run native Provider APIs and external ACP Harnesses with streaming,
  cancellation, MCP integration, and resumable sessions where the host supports
  them.
- Keep ACP server capabilities, prompt history, working directory, resources,
  MCP state, and cancellation consistent with the persisted Session.

### Composition and extensions

- Discover Models from explicit Provider connections, Extension metadata, and
  manual declarations, then resolve a compatible `Harness x Model` route.
- Export source-dated task guidance for Models, Harnesses, and exact Agents as
  passive product metadata. Keep benchmark configuration, measured target,
  review date, and limitations visible; never let guidance override runtime
  compatibility or silently turn upstream Harness results into SwarmX parity.
  Missing guidance means unrated, not weak or unsupported, and product guidance
  remains separate from user-owned Memory and local evaluation evidence.
- Create reusable Custom Agents from Harness recipes and Models, including
  deterministic Agent/Model-specific Skill variants.
- Run a governed Skill self-improvement loop with immutable optimizer
  candidates, hidden-holdout paired evaluation, static and evaluation gates,
  human compare-and-swap promotion of a per-Skill active revision, and
  rollback; learning never mutates an active request, a running Session,
  Skill files, or the persisted workflow, and optimizers cannot self-promote.
- Load Extension, marketplace, Agent profile, Skill, MCP, connector, LSP, hook,
  command, asset, permission, and UI-contribution metadata as passive inventory.
- Resolve and display composition readiness before execution. Inventory loading
  alone never executes bundled code, starts services, changes trust, or mutates
  host configuration.
- Execute configured Agent and Swarm lifecycle hooks only through an explicit
  host-owned capability executor. Matching handlers run concurrently with
  bounded timeouts and structured input/output; missing executors, malformed
  output, timeouts, denials, and handler failures fail closed. Hook target
  strings are capability names and never become implicit shell commands.
- Read native Claude Code Markdown and Codex TOML Agent definitions without
  activating or rewriting their source files.

### Project tools and permissions

- Let direct SwarmX tasks inspect, edit, search, and validate the active Project
  through bounded host tools with containment, stale-read protection,
  cancellation, output limits, and a fail-closed platform sandbox where
  supported.
- Project direct tools as `auto`, Claude Code, Codex, or Kimi Code contracts
  while dispatching through the same safety boundary. The resolved style is
  fixed for a persisted Session.
- Support bounded foreground, background, PTY, LSP, planning, child-Agent, and
  scheduled work only when the selected contract has real backing behavior.
- Resolve direct-tool authority from managed, Project, personal, Agent, and
  conversation layers. Denials and lower-authority ceilings win; approval never
  escapes the operating-system sandbox.
- In Auto mode, keep bounded Project reads and edits on the deterministic fast
  path and send remaining one-call permission requests to a separate tool-free
  LLM review. The reviewer may approve only the offered one-call option; it
  cannot override explicit asks or denials, grant durable authority, or widen
  Project, network, environment, secret, or sandbox scope. Missing, malformed,
  timed-out, or unsupported review falls back to human approval.
- Use Auto immediately as the fallback whenever no user, Agent, Project,
  managed, or conversation layer declares a mode, including persisted Sessions
  that still inherit the default. Preserve every explicit mode and every
  lower-authority ceiling; vendor rollout dates never gate this product default.
- Preserve permission decisions, privileged host requests, process lifecycle,
  and externally reachable request outcomes as a tamper-evident local audit
  chain with bounded query, verification, and export.

### Desktop experience

- Let the user view, edit, and explicitly forget bounded `USER.md` and
  `MEMORY.md` global Memory in Settings. Direct SwarmX runs, external ACP
  Harness prompts, and Agent-bearing workflows receive one frozen read-only
  snapshot and persist a concise Session-visible usage receipt. Direct Agents
  may request save or forget, but Main applies it only after explicit user
  confirmation. Existing Settings-backed Personal Memory remains readable as a
  migration source until the user first saves `USER.md`; successful migration
  removes the obsolete Settings value.
- Let SwarmX-owned Agents list, get, search, and inspect linked-page Memory graph data
  on demand. Agent-proposed page creation, update, or deletion must show the
  proposed change and require explicit one-call confirmation; mutation audit
  records must exclude titles, aliases, and Markdown bodies.
- Let a SwarmX-owned Agent propose a structured research capture containing
  entities, aliases, typed observations, source references, confidence, and why
  each observation is costly to reconstruct from ordinary public documentation.
  After one confirmation per proposed entity mutation, exact normalized
  title/alias matching creates or updates the corresponding versioned Wiki page
  without overwriting unrelated authored content or duplicating an already
  captured observation.
- When no compatible Model is available, replace starter task suggestions with
  an actionable readiness state. Its primary action opens Provider setup, and
  the UI must not imply that a blocked task can run.
- Keep Project and Session history in the sidebar's bounded scroll region while
  the persistent `Local workspace` control remains anchored to its bottom.
- Allow every persisted or discovered Project group to expand and collapse
  independently. Label sessions without a Project as `Recents`, keep that group
  after all Project groups, indent Session titles grouped under a Project, and
  keep `Recents` Session titles flush with the sidebar content edge. Sessions
  whose persisted Project reference no longer resolves remain reachable in a
  discovered Project group or `Recents` instead of disappearing.
- Group local task history by Project and support resume, title editing, pinning,
  archiving, timing, and cancellation. Persist each Project's Session history
  in its own local directory, keep sessions without a Project in `Recents`, and
  use only append-only JSONL plus a rebuildable per-directory index. Older JSON
  Session files are unsupported.
- Stream reasoning, commentary, tool progress, and results while a task runs;
  collapse transient work after completion and retain the final answer and
  canonical trace.
- Float the ordinary conversation composer over the bottom of the conversation
  surface instead of allocating it a separate application-layout row. Reserve
  matching scroll space beneath the transcript so the composer never obscures
  the final message, while bottom and side panels remain outside that overlay.
- Keep each user or assistant message's action row visually hidden until that
  message is hovered or contains keyboard focus. Preserve author-aligned
  placement and render every action as the same neutral, unselected control.
  Keep the actions directly available on input devices that cannot hover.
- Render conversational Markdown, math, code, sanitized trace cards, and typed
  attachments without executing content or fetching arbitrary remote media.
- Attach, persist, transport, preview, and capability-gate images, PDFs, audio,
  video, text, and general files without storing inline binary data in Sessions.
- Support transient read-only side chats anchored to a parent snapshot, with
  explicit promotion as the only route into normal persisted task history.
- Provide dedicated Settings surfaces for Providers, permissions, Extensions,
  Custom Agents, runtime health, and updates; Doctor inspection is read-only and
  repair is separately confirmed.

### Reusable platform

- Export validated TypeScript contracts for orchestration, Sessions, Providers,
  Models, Harnesses, source-dated Agent guidance, Agent profiles, Extensions,
  Skills, actions, context, normalized rendering, telemetry, managed
  dependencies, and the generic durable-task runtime.
- Persist bounded local Memory as actual `USER.md` and `MEMORY.md` global files
  plus Markdown entity pages with CRUD, optimistic revision checks,
  title/alias/content search, and derived double-bracket links. All authored
  Memory shares one local Git authority. Unknown, malformed, and self references
  remain explicit graph diagnostics; the linked Markdown organization is not a
  `SwarmConfig` workflow or a second Memory concept.
- Keep browser-safe public subpaths free of Node-only imports.
- Provide a CLI, ACP server adapter, runtime Doctor, and Desktop-first npm
  launcher without launching GUI code during package installation.
- Discover the product Python worker, `uv`, a compatible uv-managed Python, and
  the one locked `swarmx` environment without mutation. Runtime dependencies
  for the worker, RSI, and Reference subpackages are normal project
  dependencies, never module-specific dependency groups. Installation and
  synchronization are explicit setup/repair actions and never occur implicitly
  during a run.
- Discover and verify the packaged Memory runtime without mutation, then start it
  only from a digest-checked launch description. A missing, incompatible, or
  modified runtime produces an explicit setup/repair state rather than falling
  back to an unverified binary or installing Cargo on the user's machine.
- Package releases from one version-aligned, quality-gated commit; release
  artifacts must not contain generated residue, credentials, or known avoidable
  production vulnerabilities.

## Explicit limits

- SwarmX core does not own downstream domain workflows such as biosecurity,
  HPC/LSF execution, scientific interpretation, benchmark policy, paper
  generation, or memory-claim admission.
- It does not provide cloud team identity, email activation, hosted knowledge
  storage, or product-specific analytics and collaboration policy.
- n8n import does not make SwarmX an n8n node runtime.
- A Python task operation is an executor capability, not an Agent, Harness, ACP
  Session, or new `SwarmConfig` node kind. The worker lease/checkpoint protocol
  is not carried over ACP.
- The local task supervisor continues active eligible WorkItems after Desktop
  closes. Automatic login/startup installation and remote/cloud execution remain
  outside this local service contract.
- Extension manifests are declarative metadata, not executable UI or script
  delivery. Executable UI components must be registered by the embedding host.
- Raw conversations, prompts, responses, source files, terminal output, and
  credentials are neither telemetry nor audit payloads. Canonical Session and
  task histories retain their own product data; the audit chain records only
  compact decision/effect metadata.
- Global Memory is admitted durable context, not inferred Activity Profile data,
  Session search, Project context, a Skill, or an unreviewed conversation copy.
  Agents may explicitly or periodically propose bounded saves, forgets, and
  entity observations, but cannot combine unrelated Session windows, bypass the
  configured admission gate, or mutate the active run's frozen snapshot.
- The linked-page organization does not ingest sources, call an LLM, invent missing entities,
  mutate itself without confirmation, run vector retrieval, or render an
  interactive graph. BM25 search and linked-Markdown graph analysis operate only over
  explicit current pages and remain rebuildable from Markdown plus Git.
- The Memory runtime stops with Desktop and is not hosted by the detached task
  supervisor. Persisted Memory survives normally, but this slice does not
  claim Desktop-closed background Memory execution.
- Local audit verification is tamper-evidence, not remote attestation or
  non-repudiation against a user who controls both the log and its checkpoint.
- Claude-compatible `PowerShell`, `SendMessage`, and `Workflow` remain absent
  until SwarmX has the corresponding Windows sandbox, concurrent team runtime,
  and persisted workflow VM. Similar-looking existing features are not aliases
  for those contracts.

## Acceptance and change policy

The zod schemas, public TypeScript types, and focused tests are the executable
authority for field-level and protocol-level behavior. `DESIGNS.md` explains
architecture; `docs/` explains feature details and operator behavior.

Changes must preserve the invariants above and pass validation proportional to
their scope. The canonical Node quality gate is:

```shell
pnpm run ci:node
```

Tagged releases additionally run the production dependency audit and the
repository's release workflow gates.

Keep this file short:

- Add a requirement only for a durable user-visible capability or cross-cutting
  invariant.
- Put schemas and implementation details in code, architecture rationale in
  `DESIGNS.md`, feature guidance in `docs/`, and acceptance details in tests.
- Track unfinished work in `ROADMAP.md` or the issue tracker.
- Track completed work and failures in Git history and release notes.
- Do not add task ids, test ids, file-by-file implementation indexes, or bug
  ledgers here.

The former `G/C/I/V/T/B` ledger remains available in Git history:

```shell
git show 780fb8e:SPEC.md
```
