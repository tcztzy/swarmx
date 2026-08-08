# `@swarmx/core`

Portable domain layer. It owns the schemas and behavior shared by Desktop, CLI,
and the ACP server. Root exports are assembled in `index.ts`; public browser
consumers use the package subpaths declared in `packages/core/package.json`.

## Execution spine

1. `types.ts` validates a `SwarmConfig` and message/config primitives.
2. `swarm.ts` materializes nodes and edges, rejects unconditional cycles, and
   schedules each node within a bounded execution.
3. `agent.ts` runs one configured agent; direct model requests use
   `native-model.ts`, while external harnesses use `acp.ts`.
4. `mcp.ts` and `tool.ts` provide MCP-backed tools; `edge.ts` and `hook.ts`
   control graph transitions and lifecycle callbacks.
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
| `packages/core/src/types.ts` | Zod schemas/types for MCP servers, hooks, backends, agents, tools, edges, `SwarmConfig`, messages, media, Sessions, and Projects; central persistence boundary. `pure` |
| `packages/core/src/model-api.ts` | Allowed provider API and request-mode literals (`anthropic`, OpenAI chat/responses, Ollama; standard/Codex responses). `pure` |
| `packages/core/src/version.ts` | Single `SWARMX_VERSION` constant consumed by manifests, ACP, diagnostics, and telemetry. `pure` |
| `packages/core/src/canonical-json.ts` | Internal deterministic JSON serialization and stable non-cryptographic hashing shared by ids, digests, and canonical records. `pure` |
| `packages/core/src/secret-scanner.ts` | Internal recursive sensitive-field classifier shared by secret-free metadata boundaries; allows explicit references/redaction and path-scoped vault exceptions. `pure` |
| `packages/core/src/edge.ts` | `Edge` graph object and CEL condition evaluation used by `Swarm`. `pure` |
| `packages/core/src/hook.ts` | Hook config/runtime callback contract used by agent and swarm lifecycle events. `pure` |
| `packages/core/src/builtin-tools.ts` | Built-in tool style/revision schemas and style resolution for Claude Code, Codex, and Kimi Code contracts. `pure` |
| `packages/core/src/actions.ts` | Action intent/confirmation/risk schemas, secret-safe payload sanitization, and explicit-confirmation checks for side effects. `pure` |
| `packages/core/src/context.ts` | Context strategy, packet, summary checkpoint, and invocation metadata schemas/builders; controls isolated vs thread context. `pure` |
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
| `packages/core/src/task-control-service.ts` | Restartable app-attached control-plane slice: creates Session-independent WorkItems, links multiple observing Sessions, acquires fenced leases, records worker events/checkpoints/immutable artifacts/receipts, persists cancellation and human decisions, schedules eligible retries, and recovers torn tails/expired leases. Resume reparses the blob and binds checkpoint identity/checksum/environment to the new Run and launch; cross-Desktop-close execution remains a future daemon boundary. `fs` + `proc` + explicit gateway effects |

### Agents and execution adapters

| Source | Contract |
| --- | --- |
| `packages/core/src/agent.ts` | `Agent` runtime: direct native model or external backend, hooks, MCP/tools, cancellation, streaming chunks, and request-scoped runtime environment. `net`/`proc` through adapters |
| `packages/core/src/native-model.ts` | Native Anthropic/OpenAI/Ollama request construction, streaming, tool continuation, token usage, and request environment handling. `net` + `secret` |
| `packages/core/src/acp.ts` | ACP client lifecycle, subprocess/session negotiation, prompt/update decoding, permission callbacks, request cancellation, and request-local abort scope. `net` + `proc` |
| `packages/core/src/mcp.ts` | MCP client/server lifecycle, tool/resource discovery and calls, local tool contracts, content normalization, and cancellation. `net`/`proc` |
| `packages/core/src/tool.ts` | Validated named MCP tool wrapper; creates a manager, calls the tool, normalizes structured content, and closes servers. `net` |
| `packages/core/src/swarm.ts` | `Swarm` workflow runtime: parse/materialize config, detect cycles, evaluate edges, wait for predecessors, bound steps, and collect ordered chunks/metrics. `net`/`proc` via nodes |
| `packages/core/src/conversation.ts` | Message construction, model-message conversion, tool/progress filtering, and bounded conversation normalization. `pure` |
| `packages/core/src/media.ts` | Attachment validation and provider/ACP prompt content conversion; preserves metadata and bounds payloads. `pure` |
| `packages/core/src/server.ts` | Loopback-first HTTP server adapter exposing a `Swarm` execution endpoint with explicit auth/origin behavior. `net` |
| `packages/core/src/n8n.ts` | Read-only n8n topology/import adapter into canonical `SwarmConfig`; never imports credentials or runs n8n nodes. `pure` |

### Catalog, composition, and extensions

| Source | Contract |
| --- | --- |
| `packages/core/src/harness.ts` | Canonical built-in Harness recipes, software/version/command metadata, backend declarations, supported APIs, model controls, environment allowlists, and runtime/model compatibility. `pure` |
| `packages/core/src/model-capabilities.ts` | Independent Model registry, capability metadata, reasoning normalization, model/supply schemas, and Harness × Model inventory resolution. `pure` |
| `packages/core/src/providers.ts` | Provider profile/supply schemas, compatibility modes, secret-reference validation/redaction, runtime environment construction, and route selection. `pure` + `secret` at call boundary |
| `packages/core/src/extensions.ts` | Passive extension manifest discovery/validation, component inventory, trust-safe normalization, agent composition/preflight, and extension-provided execution metadata. `fs` + `pure` |
| `packages/core/src/extension-management.ts` | Explicit extension install/update/rollback/trust/enable/repair plans and state transitions; discovery remains passive. `fs` |
| `packages/core/src/dependencies.ts` | Managed dependency manifest schemas, runtime readiness, install/repair planning, and safe process/environment descriptors. `pure` + `fs`/`proc` through host |
| `packages/core/src/harness-management.ts` | Harness inventory, setup/repair requirements, and composition readiness aggregation. `pure` |
| `packages/core/src/agent-profiles.ts` | Parse reusable Markdown frontmatter with the standard YAML parser, serialize canonical agent profiles, and project Claude Markdown/Codex TOML definitions; rejects inline secrets. `pure` |

### State, persistence, security, and telemetry

| Source | Contract |
| --- | --- |
| `packages/core/src/session.ts` | Append-only JSONL Session authority, rebuildable index, locking, summaries, edits/forks/promotion, legacy JSON replay, and migration with verified backups. `fs` |
| `packages/core/src/session-discovery.ts` | Discover/group/load external ACP Sessions and convert them to Core Session data without claiming ownership. `fs`/`proc` through ACP |
| `packages/core/src/project.ts` | Project bookmark registry and normalization under `~/.swarmx/projects.json`; list/rename/pin/dismiss/remove. `fs` |
| `packages/core/src/desktop-settings.ts` | Shared Desktop settings schemas/defaults and secret-free metadata sections. `pure` |
| `packages/core/src/secrets.ts` | Secret-reference and local vault document schemas, file-mode checks, redaction, and safe parsing; no renderer exposure. `fs` + `secret` |
| `packages/core/src/rendering.ts` | Sanitized render event/artifact/provenance schemas and conversion from message chunks; rendering state is not canonical history. `pure` |
| `packages/core/src/activity.ts` | Strict one-event-per-run `run_summary`, token estimates, aggregate tool/Skill counts, daily/profile summaries, and non-authoritative activity store. `fs` |
| `packages/core/src/audit.ts` | Strict concise audit event/query schemas plus locked, fsynced, hash-chained JSONL persistence, head-checkpoint verification, explicit interrupted-tail recovery, recursive secret/raw-content omission, and verified export under `~/.swarmx/audit/`. Local integrity evidence is not external attestation. `fs` |
| `packages/core/src/quota.ts` | Shared quota/rate-limit value types and normalization helpers. `pure` |
| `packages/core/src/telemetry.ts` | Opt-in telemetry envelope/config/status/ingest schemas and redaction-safe envelope builder. `pure` |

### Public barrel

| Source | Contract |
| --- | --- |
| `packages/core/src/index.ts` | Root public barrel. Re-exports the Core API and types; changes here affect every package. Browser consumers should use declared safe subpaths where available. `pure` |

## Core subpaths

The manifest currently exposes root plus `rendering`, `telemetry`, `activity`,
`dependencies`, `conversation`, `providers`, `model-capabilities`,
`builtin-tools`, `harness-management`, `agent-profiles`, `desktop-settings`,
`secrets`, `actions`, `skill-variants`, `extension-management`, `harness`, and `project`.
The durable task modules are exposed from the Node-capable root barrel; the
store, controller, and process host are not browser-safe Renderer subpaths.
The audit store is likewise Node-only and exported from the root barrel.
Skill evolution and delivery modules are Node-capable root-barrel exports.
When adding a public module, update the manifest and this map together.
