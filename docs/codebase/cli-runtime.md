# CLI, runtime, ACP server, and launcher

These packages are thin boundary adapters over Core. They parse user/host
inputs, call reusable services, and format results; they do not introduce a
second workflow or persistence model.

## Runtime (`@swarmx/runtime`)

| Source | Contract |
| --- | --- |
| `packages/runtime/src/harness-environment.ts` | Detects Harness executables/versions, container runtimes, protection modes, and setup requirements; explicit host callbacks perform installation/setup. `fs` + `proc` |
| `packages/runtime/src/doctor.ts` | `HarnessDoctor` converts environment status into inspect reports, risk-labelled repair plans, and explicit fix results. Discovery is read-only; repair is opt-in. `proc` through host |
| `packages/runtime/src/python-environment.ts` | Read-only product-worker asset, `uv`, uv-managed Python, and digest-addressed locked-environment inspection; computes the environment digest (including opt-in module sources) and returns an asynchronously reverified direct-Python launch with a hash-checked source snapshot, or an explicit install/repair plan. Status checks are offline/no-download and never mutate during task execution. `fs` + `proc` |
| `src/swarmx/__init__.py` | Regular package marker and installed `swarmx` distribution version. Python does not use a PEP 420 namespace or feature-distribution discovery. `pure` |
| `src/swarmx/worker.py` | Python 3.11+ reference backend for protocol v1. Executes the minimal operations plus `swarmx.evolve_skill`; the deterministic fake stays local while DSPy/GEPA is delegated to the locked RSI MCP server. Reports durable-task events and issues grant-checked capability calls, but owns no task authority. `proc` worker |
| `src/swarmx/rsi/__init__.py`, `src/swarmx/rsi/contract.py`, `src/swarmx/rsi/errors.py` | RSI subpackage plus strict process-boundary request validation and module-local error contract. |
| `src/swarmx/rsi/server.py`, `src/swarmx/rsi/client.py` | Private RSI FastMCP server and worker-side client: exact `swarmx_rsi_optimize` surface, sanitized module launch, identity/tool checks, cancellation/progress, and MCP sampling mapped to the grant-checked gateway. `proc` |
| `src/swarmx/rsi/skill_program.py`, `src/swarmx/rsi/capability_lm.py`, `src/swarmx/rsi/optimize.py` | DSPy Skill component mapping, credential-free gateway/deterministic LM adapters, and GEPA runner with metric, proposer, config digest, and bounded report. `proc` worker |
| `src/swarmx/ref/__init__.py`, `src/swarmx/ref/service.py`, `src/swarmx/ref/server.py` | Reference subpackage and private read-only ZIM MCP: official libzim adapter, strict status/search/get requests, serialized search, item/result bounds, and active-HTML-to-text filtering. `fs` + `proc` |
| `tests/python/package_layout_test.py` | Single-distribution package/dependency/script boundary and absence of namespace entry points or split distributions. |
| `tests/python/roundtrip_test.py`, `tests/python/worker_capability_test.py`, `tests/python/rsi_mcp_server_test.py` | Python round-trip, worker budget/cancel/timeout, and real stdio RSI MCP acceptance tests. |
| `tests/python/reference_service_test.py`, `tests/python/reference_mcp_server_test.py` | Strict read-only/boundary tests plus a real generated-ZIM stdio MCP acceptance test. |
| `packages/runtime/src/index.ts` | Runtime public barrel for Doctor, Harness, and Python worker-environment APIs. `pure` |
| `packages/runtime/src/doctor.test.ts` | Contract tests for inspection, plan risk, and explicit repair behavior. |
| `packages/runtime/src/python-environment.test.ts` | Contract tests for read-only/no-download discovery, digest stability, environment readiness, forged/stale-status rejection, hash-snapshotted sanitized launch specs, and explicit setup/repair plans. |
| `packages/runtime/src/python-worker-smoke.test.ts` | End-to-end smoke contract between the Core app-attached controller and reference Python worker, including progress/checkpoint completion, partial and final-checkpoint resume, cancellation, and persisted human-decision resume. |

Exact RSI module paths are
`src/swarmx/rsi/__init__.py`, `contract.py`, `errors.py`, `server.py`,
`client.py`, `skill_program.py`, `capability_lm.py`, and `optimize.py`. Its
acceptance tests are `tests/python/roundtrip_test.py`,
`worker_capability_test.py`, and `rsi_mcp_server_test.py`.

Exact Reference module paths are `src/swarmx/ref/__init__.py`, `service.py`, and
`server.py`. Its acceptance tests are `tests/python/reference_service_test.py`
and `reference_mcp_server_test.py`.

The root `pyproject.toml` owns the one installable, regular `swarmx` package.
DSPy, MCP, and libzim are direct project dependencies in the same lock and
environment; there are no Python workspace members, module entry points, or
`rsi`/`ref` dependency groups. Inspect evaluation tooling alone remains in the
opt-in `inspect` group.

## CLI (`@swarmx/cli`)

| Source | Contract |
| --- | --- |
| `packages/cli/src/cli.ts` | Commander entrypoint for `doctor`, `send`, `eval-run`, `serve`, `audit`, `sessions`, `harnesses`, `evolution`, and REPL input; `send`/eval/REPL share content-free `agent.run` audit events distinguished by `surface`, while other commands retain semantic lifecycle actions. `proc`/`net` via Core/Runtime |
| `packages/cli/src/audit-command.ts` | Verifies and filters the canonical audit chain, formats compact human/JSON output, and writes 0600 verified JSONL exports with intent/outcome events. `fs` via Core |
| `packages/cli/src/doctor.ts` | Interactive/noninteractive doctor runner plus stable human/JSON formatting and confirmation handling. `proc` |
| `packages/cli/src/eval-run.ts` | Loads/validates workflow and eval arguments, executes `Swarm`, and formats deterministic eval result/error records with context-token usage; supports request-scoped `prompt_fragment` Skill delivery via `--skill-delivery` + `--skill-content-path`, and `--resolve-skill` binding to the evolved active revision. `fs` |
| `packages/cli/src/evolution-command.ts` | Thin CLI adapters for the skill evolution loop: digest over worker/project/lockfile/RSI sources/Python and exact DSPy/MCP versions; optimization WorkItem + RSI MCP/gateway wiring; evaluation, status, human decisions, rollback, and active-revision delivery resolution. Business rules stay in Core. `fs` + `proc` via Core |
| `packages/cli/src/send-config.ts` | Builds a canonical one-agent `SwarmConfig` from CLI model/harness options. `pure` |

## ACP server (`@swarmx/acp-server`)

| Source | Contract |
| --- | --- |
| `packages/acp-server/src/server.ts` | Implements ACP Agent around a persisted Core Session and `Swarm` executor; the executable injects Core audit authority and records content-free Session/prompt lifecycles before effects. Prompt cancellation reuses `acp.prompt` with correlated cancellation outcomes; the no-op mode setter emits no audit noise. `net` + `fs` |
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
