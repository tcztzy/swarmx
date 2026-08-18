# `@swarmx/core`

Portable domain layer. It owns the schemas and behavior shared by Desktop, CLI,
and the ACP server. Root exports are assembled in `index.ts`; public browser
consumers use the package subpaths declared in `packages/core/package.json`.

## Execution spine

1. `types.ts` validates a `SwarmConfig` (including its named DSH strategy) and
   message/config primitives.
2. `core-runtime.ts` is the only public execution constructor. Its Cordis
   `swarmRuntime` Service creates internal graphs in one child Fiber per request;
   `dsh-plugin.ts` supplies the effect-scoped `providerConnectors`,
   `harnessConnectors`, `swarmStrategies`, and `harnessTransports` registries.
3. `swarm.ts` and `agent.ts` are internal execution entities. Direct model
   requests use `native-model.ts`; external harnesses use `acp.ts` or a
   plugin-registered transport through the Fiber-scoped `harnessTransports`
   Service. Codex uses the direct `codex_server` transport.
4. `local-tool-contracts.ts` defines Provider-independent local tools;
   `mcp.ts` and `tool.ts` adapt them to MCP-backed execution, while
   `request-scope.ts` supplies protocol-neutral cancellation. `edge.ts` and
   `hook.ts` control graph transitions and lifecycle callbacks.
5. `rendering.ts`, `conversation.ts`, `activity.ts`, and `telemetry.ts` derive
   bounded output/metrics without replacing canonical Session history;
   Activity persists one aggregate `run_summary` per run and is not audit.

Privileged adapters share `audit.ts`: a strict, secret-free hash-chained event
authority with stable request/Session/task correlations. It is operational
evidence, not a copy of conversation or terminal content.

Durable work follows a separate execution spine: `task-runtime.ts` is the
language-independent event reducer, `task-runtime-store.ts` is its append-only
authority, `task-control-service.ts` owns leases and recovery, and
`task-worker-process.ts` hosts a replaceable executor through the schemas in
`task-worker-protocol.ts`. Sessions link to WorkItems only as observers.
The `./task-runtime` and `./task-worker-protocol` public subpaths are browser-safe;
filesystem/process control remains in Node-only host modules.

## Identity and supply spine

`harness.ts` describes runtime recipes; `model-capabilities.ts` describes
independent models; `providers.ts` describes explicit supply/credential routes;
`extensions.ts` discovers passive metadata; `harness-management.ts`,
`extension-management.ts`, and `dependencies.ts` resolve readiness and plans.
An Agent remains `harnessId:modelId`.

## Source map

### Contracts, policy, and graph primitives

| Source | Contract |
| --- | --- |
| `packages/core/src/types.ts` | Zod schemas/types for MCP servers, hooks, backends (including the `codex_server` custom transport), agents, tools, edges, `SwarmConfig` (including optional named DSH strategy), messages, media, and Sessions; central persistence boundary. `pure` |
| `packages/core/src/project-bootstrap.ts` | Browser-safe strict contract for one bounded immutable per-execution-attempt Project service snapshot, exact Project matching, deterministic instruction projection, and content-free revision/digest receipts. `pure` |
| `packages/core/src/project-contracts.ts` | Browser-safe canonical Project record schema/type shared by persistence and Desktop transport; it has no filesystem or registry authority. `pure` |
| `packages/core/src/model-api.ts` | Allowed provider API and request-mode literals (`anthropic`, OpenAI chat/responses, Ollama; standard/Codex responses). `pure` |
| `packages/core/src/version.ts` | Single `SWARMX_VERSION` constant consumed by manifests, ACP, diagnostics, and telemetry. `pure` |
| `packages/core/src/canonical-json.ts` | Internal deterministic JSON serialization and stable non-cryptographic hashing shared by ids, digests, and canonical records. `pure` |
| `packages/core/src/secret-scanner.ts` | Internal recursive sensitive-field classifier shared by secret-free metadata boundaries; allows explicit references/redaction and path-scoped vault exceptions. `pure` |
| `packages/core/src/local-tool-contracts.ts` | Browser-safe, Provider-independent local function/text tool contracts, progress/result envelopes, and branded model-facing result helper. It has no adapter or runtime dependency. `pure` |
| `packages/core/src/edge.ts` | `Edge` graph object and CEL condition evaluation used by `Swarm`. `pure` |
| `packages/core/src/memory-links.ts` | Browser-safe zod contracts and bounded double-bracket-link scanner/resolver that projects caller-owned entity Markdown into directed, non-executable knowledge edges with explicit diagnostics. `pure` |
| `packages/core/src/memory.ts` | Memory zod schemas, async host-backend contract, graph projection, and strict `Memory` local Agent tool: bounded global-file and entity CRUD/search/version reads, optimistic revisions, confirmed mutations, and typed source-bearing research capture with Session provenance. Persistence belongs only to the Rust Memory MCP server. `pure` |
| `packages/core/src/memory-runtime-protocol.ts` | Strict versioned request/operation-matched response schemas for the private `swarmx-mem` MCP server, including global `USER.md` / `MEMORY.md` operations, exact `swarmx_memory` tool identity, and bounded structured results. `pure` |
| `packages/core/src/reference-library.ts` | Browser-safe zod contracts and the read-only `ReferenceLibrary` Agent tool for bounded, source-qualified ZIM/Zotero status, search, and plaintext reads; reports unavailable paths as unsupported and has no Web or mutation operation. `pure` |
| `packages/core/src/hook.ts` | Hook config plus fail-closed lifecycle dispatcher: explicit host executor, structured input/output, concurrent same-event handlers, bounded timeouts, denial, and additional-context limits. `pure` |
| `packages/core/src/builtin-tools.ts` | Built-in tool style/revision schemas and style resolution for Claude Code, Codex, and Kimi Code contracts. `pure` |
| `packages/core/src/actions.ts` | Action intent/confirmation/risk schemas, secret-safe payload sanitization, and explicit-confirmation checks for side effects. `pure` |
| `packages/core/src/context.ts` | Context strategy, packet, summary checkpoint, and invocation metadata schemas/builders; controls isolated vs thread context. `pure` |
| `packages/core/src/context-engine.ts` | Coding-agent Context Engine contracts and projections: immutable event snapshots, atomic tool units, complete-request accounting, two-phase finalization, named OpenCode/Codex/Claude Code/Hermes/Reasonix/LCM/Parallel/ReSum evaluation profiles, injected bounded summary providers plus a request-scoped evaluation prompt override bound by checkpoint digest, LCM read-only source tools, deterministic fallback, verified BM25 evidence, priority assembly, explicit overflow, shared host-receipt replay exclusion, and fidelity-bearing replay manifests. `pure` except injected model calls |
| `packages/core/src/context-evaluation.ts` | Strict versioned coding-agent context benchmark: bounded profile/model/seed/parameter matrices, cloned-state seeded paired replay, declared task families, one mutable simulator tool, isolated model-backed summary calls, digest-bound prompt-only candidate arms, capability/contained-attempt/uncontained-safety scoring, strategy-vs-infrastructure failure separation, family-clustered confirmation gates, usage/cost/request-to-completion accounting, content-free JSONL records, per-Agent leaderboards, and bounded non-promoting neighbor search. `pure` except injected Agent model calls |
| `packages/core/src/context-engine-store.ts` | Standalone append-only SQLite WAL `EventStore`, JSONL replay adapter, and digest-verified local content-addressed `ArtifactStore`; production Session/WorkItem authority remains separate. `fs` |
| `packages/core/src/skill-variants.ts` | Skill bindings, delivery modes, lineage, evaluation, promotion, optimization request, candidate/evaluation manifests, promotion receipts, active pointer, and policy schemas. `pure` |
| `packages/core/src/skill-evolution.ts` | Pure skill evolution state machine: strict per-kind secret-free ledger records with typed payloads, immutable candidates/evaluations, candidate status transitions, gate verdicts (quality up; safety/failure/context not down; sample count and improvement-ratio minima), canonical optimizer config digests, and replay that enforces request-anchored compare-and-swap, staged-candidate/eligible-evaluation prerequisites, and idempotency. `pure` |
| `packages/core/src/skill-evolution-store.ts` | Append-only evolution ledger under `~/.swarmx/skill-evolution/`: strict replay, torn-tail recovery, idempotency, CAS on promotion receipts, and content-addressed blobs for candidate/evaluation content. `fs` |
| `packages/core/src/skill-evolution-service.ts` | Skill evolution orchestration: optimization WorkItems through the durable task runtime, immutable candidate ingestion with static checks (missing worker lineage fails closed), paired evaluation and strictly validated external-evidence recording, human-gate promotion with CAS, rollback, audit intent-before-effect with honest terminal-outcome failure reporting, the grant-checked `skill_evolution` capability gateway, and eval-safe swarm configuration validation. `fs` + `proc` + audit |
| `packages/core/src/skill-delivery.ts` | Request-scoped `prompt_fragment` Skill delivery: content-addressed artifact loading with digest/size verification, explicit rejection of native-plugin/rules-file/unsupported/external-Harness delivery, and instruction assembly that reaches the model-visible system message. `fs` + `pure` |
| `packages/core/src/skill-evaluation.ts` | Paired baseline/candidate evaluation through the same real SwarmX path, executing in the seeded-randomized per-case order it records, with deterministic scoring, metric aggregation, and gate verdicts. `pure` + `net` via Swarm |

### Durable task runtime

| Source | Contract |
| --- | --- |
| `packages/core/src/task-runtime.ts` | Language-independent Zod authority for WorkItems, Runs, fenced leases, budgets, progress, execution checkpoints, artifact refs, approvals, schedules, Session links, at-least-once side-effect receipts, strict events, replay/idempotency, lease expiry, and retry/schedule decisions. Contains no project-iteration or analysis-execution enum. `pure` + deterministic hashing |
| `packages/core/src/task-runtime-store.ts` | Append-only, fsynced task event store under `~/.swarmx/task-runtime/`, strict replay, narrow writer lock/stale-lock recovery, explicit torn-tail truncation, and content-addressed secret-safe JSON/binary blobs. Session files never participate in task replay. `fs` |
| `packages/core/src/task-worker-protocol.ts` | Core-owned version 1 strict JSONL/stdio schemas and codecs for hello/capability negotiation, start, heartbeat/progress, checkpoint/artifact, human-needed, complete/fail/cancel, and grant-checked capability calls; rejects oversized, malformed, direction-invalid, secret-bearing, and unsafe-path messages. `pure` |
| `packages/core/src/task-worker-process.ts` | One-run subprocess host: strict secret-key-free explicit launch schema, hello/backend/digest/operation negotiation, lease-envelope and sequence validation, heartbeat/wall-time/terminal-exit watchdogs, post-terminal rejection, cancellation grace/process-group termination, bounded redacted stderr, and selected-grant capability dispatch. Does not inherit ambient Provider credentials. `proc` |
| `packages/core/src/task-control-service.ts` | Restartable process-local control primitive: creates Session-independent WorkItems, links observing Sessions, acquires fenced leases, records worker events/checkpoints/artifacts/receipts, persists cancellation and decisions, schedules retries, and recovers torn tails/expired leases. Resume binds checkpoint identity/checksum/environment to the new Run; detached lifecycle is supplied only by `task-supervisor.ts`. `fs` + `proc` + explicit gateway effects |
| `packages/core/src/task-supervisor.ts` | Authenticated local JSONL socket server/client for detached WorkItem execution: mode-restricted random token, strict create/run/list/cancel/decision schemas, canonical store/control-service reuse, in-memory verified run recipes for automatic retry/approved-resume dispatch, per-run ownership independent from clients, and no Renderer token/process authority. `fs` + `net` + `proc` |

### Agents and execution adapters

| Source | Contract |
| --- | --- |
| `packages/core/src/agent.ts` | `Agent` runtime: direct native model or external ACP backend; DSH transport-aware Harness client factory for custom backends; explicit built-in service variant resolution before prompt/tool assembly; hooks; optional and fail-closed required MCP/tools (including Project bootstrap, Context Engine, and Memory-owned tools); cancellation, streaming, Provider limits, two-phase context compilation, request-scoped runtime environment, hosted-search endpoint detection, and Memory/reflection instruction assembly. `net`/`proc` through adapters |
| `packages/core/src/service-registry.ts` | Browser-safe, evaluation-only built-in Agent service ablation contracts and registry: strict complete `AblationProfile`, deterministic Swarm/Agent topology, typed Context Engine/Memory/Skill-delivery variants, shipped `production`/`baseline` providers, duplicate/missing activation rejection, and content-free activation/eval receipts. Echo and external ACP Harnesses never resolve these seams. `pure` |
| `packages/core/src/native-model.ts` | Native Anthropic/OpenAI/Ollama request construction, enforced Provider output and continuation-step limits, streaming, fail-closed tool/pause continuation settlement, token usage, request environment handling, and opt-in Responses/DeepSeek-Anthropic hosted Web Search with opaque-state replay plus visible tool lifecycle chunks. `net` + `secret` |
| `packages/core/src/request-scope.ts` | Node-only, protocol-neutral AsyncLocalStorage request scope with exclusive ids, cooperative `AbortSignal`, idempotent external cancellation, and registered adapter cleanup participants. It imports no ACP/MCP implementation. `node` |
| `packages/core/src/dsh-plugin.ts` | Effect-scoped DSH Cordis registry Services: `providerConnectors` selects typed Provider supply builders, `harnessConnectors` owns the Harness catalog plus runtime-model routes, `harnessPermissions` selects the highest-priority Harness permission resolver, `swarmStrategies` owns named `SwarmConfig` executors, `taskGuidance` merges effect-scoped task-guidance contributions over the static baseline, and `harnessTransports` selects wire clients by backend transport or command before ACP fallback. Every registration is revoked with its owning Fiber and duplicate ownership fails closed. `proc` composition |
| `packages/core/src/harness-client.ts` | Transport-neutral Harness launch and client contracts shared by Core and external DSH plugins: launch request/spec/resolver, prompt/session client interfaces, ACP permission aliases, DSH Harness approval request/outcome/resolver types, and cancellation signal. `pure` types |
| `packages/core/src/core-runtime.ts` | `coreRuntimePlugin` installs every Core Service and built-in Provider/Harness/permission/dag plugin into an existing DSH Cordis Context; `createCoreRuntime` is the convenience host wrapper around it. Owns request-Fiber `swarmRuntime`, `acpRuntime`, `mcpRuntime`, `providerConnectors`, `harnessConnectors`, `harnessPermissions`, `swarmStrategies`, `taskGuidance`, and `harnessTransports` Services; exposes plugin-backed `harnessCatalog` and Task-guidance queries; loads first-party Codex and host-supplied DSH plugins, resolves per-backend transport clients, and tears the host down idempotently. `proc` composition |
| `packages/core/src/acp.ts` | Internal ACP client lifecycle with injected Cordis launch resolution, optional backend transport selector, subprocess/session negotiation, prompt/update decoding through the terminal prompt response, tracked permission callbacks, and protocol/process cancellation registered into the shared request scope. `net` + `proc` |
| `packages/core/src/mcp.ts` | MCP client/server lifecycle, identity/version/exact-tool-surface verification when required, object-root native function-schema projection, model-facing and direct host-side tool calls, raw content-block preservation for strict host verification, resource discovery, content normalization, timeouts, and cancellation. `net`/`proc` |
| `packages/core/src/tool.ts` | Internal validated MCP tool wrapper; receives its manager factory from the request Fiber, normalizes structured content, and closes servers. `net` |
| `packages/core/src/swarm.ts` | Internal workflow graph: materializes config, derives nested Agent topology, detects cycles, evaluates edges, waits for predecessors, enforces the step bound, and collects ordered chunks/metrics. Construction is owned by `core-runtime.ts`. `net`/`proc` via nodes |
| `packages/core/src/conversation.ts` | Message construction, model-message conversion, the shared browser-safe host-receipt exclusion used by Session and Context Engine model replay, tool/progress filtering, and bounded conversation normalization. `pure` |
| `packages/core/src/media.ts` | Attachment validation and provider/ACP prompt content conversion; preserves metadata and bounds payloads. `pure` |
| `packages/core/src/server.ts` | Loopback-first HTTP adapter over the public Core execution interface with explicit auth/origin behavior; SSE streams `onChunk` message chunks as they are emitted instead of waiting for completion. `net` |
| `packages/core/src/n8n.ts` | Read-only n8n topology/import adapter into canonical `SwarmConfig`; never imports credentials or runs n8n nodes. `pure` |

### Catalog, composition, and extensions

| Source | Contract |
| --- | --- |
| `packages/core/src/harness.ts` | Canonical built-in Harness recipes, including the repository-owned Codex module token with its direct `codex_server` transport, software/version/command metadata, backend declarations, supported APIs, model controls, environment allowlists, runtime/model compatibility, and the injectable `HarnessCatalog` seam with its static default. `pure` |
| `packages/core/src/model-capabilities.ts` | Independent Model registry, capability metadata, reasoning normalization, validated Model/ModelSupply context-window and output limits, and Harness × Model inventory resolution through an injectable DSH `HarnessCatalog` (static default). `pure` |
| `packages/core/src/task-guidance.ts` | Browser-safe, source-dated passive task guidance for Models, Harnesses, and exact Agents; strict benchmark provenance/attribution schemas, DSH `HarnessCatalog`-aware target validation, and deterministic layered queries. The DSH `taskGuidance` Service merges effect-scoped plugin contributions over the static baseline. It never changes compatibility, selection, Memory, or evaluation state. `pure` |
| `packages/core/src/providers.ts` | Provider profile/supply schemas, compatibility modes, secret-reference validation/redaction, runtime environment construction, and route selection. `pure` + `secret` at call boundary |
| `packages/core/src/extensions.ts` | Passive extension manifest discovery/validation with host-observed source/content digests, installed-extension-aware component inventory, deterministic composition preflight before runtime effects, agent composition, `off`/`auto`/`required` MCP policy, Project-root runtime binding, extension-provided execution metadata, optional host hook-executor forwarding, and bounded Project/global Memory/reflection forwarding to the selected Agent execution path. `fs` + `pure` |
| `packages/core/src/extension-composition.ts` | Strict declarative Extension composition contract and read-only deterministic capability closure/topological ordering; identifies custom Harness, stdio MCP, LSP, Hook, Command, Software-command, and connector-entrypoint execution, then rejects uninstalled/disabled/untrusted/integrity-mismatched executable bundles, missing/cyclic requirements, duplicate capability/tool/Provider ownership, protected-kernel replacement, phase/order ambiguity, conflicts, and ungranted permissions while returning a content-free load/permission preview. `pure` |
| `packages/core/src/extension-management.ts` | Explicit extension install/update/rollback/trust/enable/repair/grant/revoke plans and state transitions; separates requested from granted permissions, prevents extension-owned authority changes, applies reductions safely, and requires fail-closed semantic audit around trust or permission expansion. Discovery remains passive. `fs` through host callbacks |
| `packages/core/src/dependencies.ts` | Managed dependency manifest schemas, runtime readiness, install/repair planning, and safe process/environment descriptors. `pure` + `fs`/`proc` through host |
| `packages/core/src/harness-management.ts` | Harness inventory, setup/repair requirements, and composition readiness aggregation. `pure` |
| `packages/core/src/agent-profiles.ts` | Parse reusable Markdown frontmatter with the standard YAML parser, serialize canonical agent profiles, and project Claude Markdown/Codex TOML definitions; rejects inline secrets. `pure` |

### State, persistence, security, and telemetry

| Source | Contract |
| --- | --- |
| `packages/core/src/session.ts` | Claude Code-style per-Project JSONL-only Session authority under `~/.swarmx/projects/`, durable `(sessionId, requestId)` start/settle receipts with digest-bound replay and conflict/unknown outcomes, additive foreground request ids on atomic message batches, projectless `__recents__`, rebuildable per-directory indexes, cross-Project lookup, locking, summaries, edits/forks/promotion, receipt-free transient model projection, write-time relocation of the prior flat JSONL layout, and a no-write causal-source reader. `fs` |
| `packages/core/src/session-timeline.ts` | Browser-safe strict deterministic causal projection over ordered Session records plus verified compact audit evidence: Turn/Step/tool/approval/task/external-operation correlations, conservative legacy inference, late/duplicate diagnostics, derived unsettled hints, and fixed content-free summaries. It owns no execution, completion-barrier, audit, or persistence authority. `pure` |
| `packages/core/src/session-discovery.ts` | Discover/group/load external Harness Sessions and convert them to Core Session data without claiming ownership. Production hosts enter through Core Runtime methods with the plugin-backed `harnessCatalog`; a DSH transport-aware client factory selects ACP or a plugin transport per backend, and clients remain request-Fiber scoped. `fs`/`proc` through harness adapters |
| `packages/core/src/project.ts` | Node-only canonical Project registry under `~/.swarmx/projects.json`: realpath identity, default registration, pin/rename/dismiss/remove, sorting, and restrictive atomic persistence. It consumes the browser-safe Project record contract. `fs` |
| `packages/core/src/desktop-settings.ts` | Shared Desktop settings schemas/defaults, including per-Session Memory reflection cursors, legacy Personal Memory migration input, and secret-free metadata sections. `pure` |
| `packages/core/src/personal-memory.ts` | Strict bounded, credential-rejecting global `USER.md` / `MEMORY.md` file, snapshot, migration, receipt, and per-Session reflection schemas; native/ACP instruction assembly; explicit/ten-turn/idle-tail decisions; nested workflow Agent counting; and legacy Personal Memory compatibility. `pure` |
| `packages/core/src/secrets.ts` | Secret-reference and local vault document schemas, file-mode checks, redaction, and safe parsing; no renderer exposure. `fs` + `secret` |
| `packages/core/src/rendering.ts` | Sanitized render event/artifact/provenance schemas and conversion from message chunks; rendering state is not canonical history. `pure` |
| `packages/core/src/activity.ts` | Strict one-event-per-run `run_summary`, token estimates, aggregate tool/Skill counts, daily/profile summaries, and non-authoritative activity store. `fs` |
| `packages/core/src/audit.ts` | Strict concise audit event/query schemas plus locked, fsynced, hash-chained JSONL persistence, head-checkpoint verification, explicit interrupted-tail recovery, recursive secret/raw-content omission, verified export, and fail-closed no-write diagnostic queries under `~/.swarmx/audit/`. Local integrity evidence is not external attestation. `fs` |
| `packages/core/src/quota.ts` | Shared quota/rate-limit value types and normalization helpers. `pure` |
| `packages/core/src/telemetry.ts` | Opt-in telemetry envelope/config/status/ingest schemas and redaction-safe envelope builder. `pure` |

### Public barrel

| Source | Contract |
| --- | --- |
| `packages/core/src/index.ts` | Root public barrel. Version 4 exports `createCoreRuntime` and execution interfaces, not the internal `Swarm`, `Agent`, `AcpClient`, or `McpManager` constructors. Browser consumers should use declared safe subpaths where available. `pure` |

## Core subpaths

The manifest is authoritative and exposes root plus focused public subpaths for
rendering, telemetry, activity, dependencies, conversation, Context, Provider,
Harness, Agent profile/guidance, Desktop settings, Memory/Reference, security,
actions, Skills, extension management, Projects, and durable task contracts.
`local-tool-contracts`, `memory-links`, `memory`, `memory-runtime-protocol`,
`reference-library`, `project-contracts`, `project-bootstrap`, `task-runtime`, and
`task-worker-protocol` are browser-safe. `request-scope` is explicitly Node-only
because it uses AsyncLocalStorage. Memory persistence remains exclusively in the
Rust Memory process; `memory.ts` owns only schemas, backend contracts, and the
projected Agent tool.
The task store, controller, supervisor, and process host remain available only
from the Node-capable root barrel, not browser-safe Renderer subpaths.
The Context Engine contracts and standalone store are Node-only because they use
cryptographic hashing and local filesystem persistence.
The context-evaluation runner is exported from the Node-capable root barrel; it
uses cryptographic hashes and injected native model execution, not a
browser-safe subpath.
The audit store is likewise Node-only and exported from the root barrel.
Skill evolution and delivery modules are Node-capable root-barrel exports.
When adding a public module, update the manifest and this map together.
