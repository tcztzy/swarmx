# CLI, runtime, ACP server, and launcher

These packages are thin boundary adapters over Core. They parse user/host
inputs, call reusable services, and format results; they do not introduce a
second workflow or persistence model.

## Codex module (`@swarmx/codex`)

| Source | Contract |
| --- | --- |
| `packages/codex/src/index.ts` | DSH Cordis Harness plugin: registers the managed `swarmx-codex` launcher and the direct `codex_server` transport, resolves Codex in `codexCommand` > PATH `codex` > pinned module priority, resolves Electron-as-Node execution for the pinned path, resolves bundled Linux Codex runtimes for protected containers, consumes the DSH `harnessPermissions` resolver, and rewrites packaged asar paths. Registrations are revoked with the owning Fiber. `proc` composition |
| `packages/codex/src/codex-server-client.ts` | Direct Codex App Server JSON-RPC client used by the `codex_server` transport: newline-delimited request/notification framing, initialize/thread/turn/thread-list/thread-read operations, fail-closed approval request handling, cancellation/termination, stderr capture, and turn-item message projection. No ACP import. `proc` + `net` through Codex |
| `packages/codex/bin/swarmx-codex.js` | Stdout-clean executable shim that starts the pinned `@openai/codex` binary in `app-server` mode with inherited stdio; handles the repository module version probe without starting the server. `proc` + `net` through Codex |
| `packages/codex/bin/swarmx-codex-container.js` | Dependency-free bootstrap run inside the protected `node:22-slim` container: resolves the mounted Linux Codex runtime, starts `app-server`, and mirrors child signals/exit. `proc` through Codex |
| `packages/codex/tsconfig.json` | TypeScript build boundary for the published Cordis plugin. |
| `packages/codex/tests/codex-module.test.ts` | Package/version/bin/DSH-plugin, bundled Linux runtime, and Electron unpacking contracts plus fake app-server JSON-RPC acceptance tests for the direct Codex transport, notification coalescing, process exit, and cancellation. |

The package is intentionally a narrow owned DSH plugin and launch module rather
than a fork. Its compatible Codex runtime is an exact package and lockfile
input; the vendoring threshold and packaged-Desktop path behavior are
documented in [`docs/codex-module.md`](../codex-module.md).

## Runtime (`@swarmx/runtime`)

| Source | Contract |
| --- | --- |
| `packages/runtime/src/sandbox-policy.ts` | Host-owned zod-validated `native_allowed` / `protected_required` strategy and immutable protected-profile registry; protected resolution never falls back to native. `pure` |
| `packages/runtime/src/harness-environment.ts` | Detects Harness executables/versions, container runtimes, protection modes, and setup requirements from an injectable DSH `HarnessCatalog` (static default); native Codex readiness depends on Node plus the repository module or an explicit `codexCommand` and never installs a separate Codex CLI, protected Codex mounts the repository module and bundled Linux Codex runtime into the container, and legacy persisted Codex ACP commands remain identifiable for protection; explicit host callbacks perform other installation/setup. `fs` + `proc` |
| `packages/runtime/src/doctor.ts` | `HarnessDoctor` converts runtime plus host-supplied Provider/Project/offline readiness into deterministic `ok`/warning/blocking/repairable/decision findings with symptom/cause/impact/next-action fields, curated first-run guidance, idempotent change previews, and explicit fix results. Inspection and planning are read-only; repair is opt-in. `proc` through host |
| `packages/runtime/src/python-environment.ts` | Read-only product-worker asset, `uv`, uv-managed Python, and digest-addressed locked-environment inspection; computes the environment digest (including opt-in module sources) and returns an asynchronously reverified direct-Python launch with a hash-checked source snapshot, or an explicit install/repair plan. Status checks are offline/no-download and never mutate during task execution. `fs` + `proc` |
| `packages/runtime/src/memory-runtime-environment.ts` | Zod-validated packaged Memory runtime manifest, read-only executable/digest/version inspection, explicit repair planning, launch-time revalidation, and secret-minimal launch spec for the private Rust MCP server. Never installs or compiles on an operation path. `fs` + `proc` |
| `src/swarmx/__init__.py` | Regular package marker and installed `swarmx` distribution version. Python does not use a PEP 420 namespace or feature-distribution discovery. `pure` |
| `src/swarmx/worker.py` | Python 3.11+ reference backend for protocol v1. Executes the minimal operations plus `swarmx.evolve_skill`; the deterministic fake stays local while DSPy/GEPA is delegated to the locked RSI MCP server. Reports durable-task events and issues grant-checked capability calls, but owns no task authority. `proc` worker |
| `src/swarmx/rsi/__init__.py`, `src/swarmx/rsi/contract.py`, `src/swarmx/rsi/errors.py` | RSI subpackage plus strict process-boundary request validation and module-local error contract. |
| `src/swarmx/rsi/server.py`, `src/swarmx/rsi/client.py` | Private MCP Python SDK v2 server and worker-side client: exact `swarmx_rsi_optimize` surface, sanitized module launch, identity/tool checks, cancellation/progress, and MCP sampling mapped to the grant-checked gateway. The private client deliberately initializes a pre-2026 stdio session because one optimizer call may issue multiple server-initiated sampling requests. `proc` |
| `src/swarmx/rsi/skill_program.py`, `src/swarmx/rsi/capability_lm.py`, `src/swarmx/rsi/optimize.py` | DSPy Skill component mapping, credential-free gateway/deterministic LM adapters, and GEPA runner with metric, proposer, config digest, and bounded report. `proc` worker |
| `src/swarmx/ref/__init__.py`, `src/swarmx/ref/service.py`, `src/swarmx/ref/server.py` | Reference subpackage and private read-only multi-source MCP: official libzim adapter, fixed-loopback Zotero reads, strict source-qualified status/search/get requests, response bounds, and active-HTML-to-text filtering. It has no Web Search backend; Provider credentials and native server tools remain in Core. `fs` + `net` + `proc` |
| `tests/python/package_layout_test.py` | Single-distribution package/dependency/script boundary and absence of namespace entry points or split distributions. |
| `tests/python/roundtrip_test.py`, `tests/python/worker_capability_test.py`, `tests/python/rsi_mcp_server_test.py` | Python round-trip, worker budget/cancel/timeout, and real stdio RSI MCP acceptance tests. |
| `tests/python/reference_service_test.py`, `tests/python/reference_mcp_server_test.py` | Strict read-only/boundary tests plus a real generated-ZIM stdio MCP acceptance test. |
| `packages/runtime/src/index.ts` | Runtime public barrel for Doctor, Harness, sandbox policy, and Python worker-environment APIs. `pure` |
| `packages/runtime/src/sandbox-policy.test.ts` | Protected-profile validation, immutable host registration, native/protected resolution, and fail-closed fallback tests. `pure` |
| `packages/runtime/src/doctor.test.ts` | Contract tests for inspection, plan risk, and explicit repair behavior. |
| `packages/runtime/src/python-environment.test.ts` | Contract tests for read-only/no-download discovery, digest stability, environment readiness, forged/stale-status rejection, hash-snapshotted sanitized launch specs, and explicit setup/repair plans. |
| `packages/runtime/src/python-worker-smoke.test.ts` | End-to-end smoke contract between the Core app-attached controller and reference Python worker, including progress/checkpoint completion, partial and final-checkpoint resume, cancellation, and persisted human-decision resume. |
| `packages/runtime/src/memory-runtime-environment.test.ts` | Memory runtime inspection tests for read-only status, digest tampering, protocol identity, sanitized launch, explicit repair, and launch-time revalidation. |

Exact RSI module paths are
`src/swarmx/rsi/__init__.py`, `contract.py`, `errors.py`, `server.py`,
`client.py`, `skill_program.py`, `capability_lm.py`, and `optimize.py`. Its
acceptance tests are `tests/python/roundtrip_test.py`,
`worker_capability_test.py`, and `rsi_mcp_server_test.py`.

Exact Reference module paths are `src/swarmx/ref/__init__.py`, `service.py`, and
`server.py`. Its acceptance tests are `tests/python/reference_service_test.py`
and `reference_mcp_server_test.py`.

The root Cargo workspace discovers `crates/*`; `crates/swarmx-mem/` is the
private Rust Memory MCP module. Its manifest and root `Cargo.lock` pin the build graph;
`crates/swarmx-mem/src/main.rs` exposes only `swarmx_memory`;
`crates/swarmx-mem/src/lib.rs` owns the human-readable recursive Markdown vault,
stable frontmatter identity/aliases/kinds/sources, portable natural filenames,
index/disambiguation/backlink views, atomic inbound-link rename, external-edit
reconciliation, idempotent legacy filename migration, crash-recoverable
single-writer WAL transactions, durable Git publication, secret rejection,
Git/BM25 CRUD, and version semantics;
`crates/swarmx-mem/tests/memory_service.rs` covers persistence, Unicode and
portable names, same-title disambiguation, aliases/backlinks/views, human edits
and moves, migration, failed-transaction rollback, conflicts,
history/diff/restore, bounds, and restart recovery.

The root `pyproject.toml` owns the one installable, regular `swarmx` package.
DSPy, MCP, and libzim are direct project dependencies in the same lock and
environment; there are no Python workspace members, module entry points, or
`rsi`/`ref` dependency groups. Inspect evaluation tooling alone remains in the
opt-in `inspect` group.

## CLI (`@swarmx/cli`)

| Source | Contract |
| --- | --- |
| `packages/cli/src/cli.ts` | Commander entrypoint and owner of one lazy process-level Core Cordis Runtime for `send`, eval, serve, evolution, and REPL execution. Request resources remain Fiber-scoped and the root is disposed at process shutdown. `proc`/`net` via Core/Runtime |
| `packages/cli/src/audit-command.ts` | Verifies and filters the canonical audit chain, formats compact human/JSON output, and writes 0600 verified JSONL exports with intent/outcome events. `fs` via Core |
| `packages/cli/src/doctor.ts` | Interactive/noninteractive Doctor runner plus stable human/JSON formatting for classification and symptom/cause/impact/next-action/change previews, with explicit confirmation handling. `proc` |
| `packages/cli/src/session-timeline-command.ts` | Read-only `sessions timeline` adapter: loads canonical Session records and verified audit evidence, calls the Core causal projector, and formats concise safe human or strict JSON output. `fs` through Core readers |
| `packages/cli/src/eval-run.ts` | Loads/validates workflow and eval arguments, then executes single-sample and Context evaluation through the injected process-level Core Runtime. It retains strict ablation, bounded JSONL reservation, and request-scoped Skill delivery contracts. `fs` + injected model calls |
| `packages/cli/src/evolution-command.ts` | Thin CLI adapters for the skill evolution loop: digest over worker/project/lockfile/RSI sources/Python and exact DSPy/MCP versions; optimization WorkItem + RSI MCP/gateway wiring; evaluation, status, human decisions, rollback, and active-revision delivery resolution. Business rules stay in Core. `fs` + `proc` via Core |
| `packages/cli/src/send-config.ts` | Builds a canonical one-agent `SwarmConfig` from CLI model/harness options. `pure` |

## ACP server (`@swarmx/acp-server`)

| Source | Contract |
| --- | --- |
| `packages/acp-server/src/server.ts` | Implements ACP Agent around persisted Core Sessions and one lazily owned Core Cordis Runtime. Each prompt executes through a request Fiber; connection/signal shutdown disposes the root. ACP lifecycle and cancellation audit contracts remain unchanged. `net` + `fs` |
| `packages/acp-server/src/index.ts` | Public re-export of `run` and `SwarmXAgent`. `pure` |
| `packages/acp-server/src/cli.ts` | Executable shim that invokes the ACP server `run` entrypoint. `net` |
| `packages/acp-server/src/server.test.ts` | ACP lifecycle/capability/session contract tests. |

## npm launcher (`swarmx`)

| Source | Contract |
| --- | --- |
| `packages/swarmx/bin/swarmx.js` | Package-installed bootstrap: selects Desktop for no/`desktop` args, delegates CLI args to `@swarmx/cli`, repairs Electron path metadata, forwards signals, and routes Electron stderr through the Desktop launch filter. `fs` + `proc` |

## Package/launcher tests

The exact test inventory and what each contract protects are in
[`tests.md`](tests.md). CLI command modules intentionally remain adapters, so
new behavior should normally be implemented/tested in Core or Runtime first.
