# CLI, runtime, ACP server, and launcher

These packages are thin boundary adapters over Core. They parse user/host
inputs, call reusable services, and format results; they do not introduce a
second workflow or persistence model.

## Runtime (`@swarmx/runtime`)

| Source | Contract |
| --- | --- |
| `packages/runtime/src/harness-environment.ts` | Detects Harness executables/versions, container runtimes, protection modes, and setup requirements; explicit host callbacks perform installation/setup. `fs` + `proc` |
| `packages/runtime/src/doctor.ts` | `HarnessDoctor` converts environment status into inspect reports, risk-labelled repair plans, and explicit fix results. Discovery is read-only; repair is opt-in. `proc` through host |
| `packages/runtime/src/python-environment.ts` | Read-only product-worker asset, `uv`, uv-managed Python, and digest-addressed locked-environment inspection; computes the environment digest (including evolution sidecar sources) and returns an asynchronously reverified direct-Python launch with a hash-checked source snapshot, or an explicit install/repair plan. Status checks are offline/no-download and never mutate during task execution. `fs` + `proc` |
| `packages/runtime/python/swarmx_worker.py` | Dependency-free Python 3.11+ reference backend for protocol v1. Executes the minimal `swarmx.echo`, `swarmx.count`, `swarmx.fail`, and `swarmx.needs_human` operations plus `swarmx.evolve_skill` (deterministic fake optimizer, or DSPy/GEPA through the locked evolution sidecar); reports heartbeats/progress/checkpoints/artifacts/terminal events, acknowledges cancellation, and issues grant-checked `capability_call` requests with correlation, timeout, cancel, and budget handling. It is replaceable execution, not task authority, an Agent/Harness, ACP, or a workflow node. `proc` worker |
| `packages/runtime/python/evolution/__init__.py` | Opt-in DSPy/GEPA skill optimization sidecar package marker; imported only by the `dspy.gepa.v1` optimizer under the locked `evolution` dependency group and never touches active pointers, promotions, Sessions, credentials, or Skill installs. |
| `packages/runtime/python/evolution/skill_program.py` | Maps Skill Markdown to a DSPy `Predict` component whose instructions are exactly the Skill bytes, and exports optimized component instructions back to candidate Markdown. `proc` worker |
| `packages/runtime/python/evolution/capability_lm.py` | DSPy `BaseLM` adapters: `CapabilityLm` relays model calls through the grant-checked capability gateway (credentials stay host-side); `DeterministicLm` is the offline test adapter. `proc` worker |
| `packages/runtime/python/evolution/optimize.py` | GEPA runner with a score-plus-feedback-plus-deterministic-validator metric, gateway or deterministic instruction proposer (no silent fallback), canonical config digest, and optimizer report. `proc` worker |
| `packages/runtime/python/evolution/tests/__init__.py` | Python test package marker for the evolution sidecar. |
| `packages/runtime/python/evolution/tests/roundtrip_test.py` | Python round-trip tests: baseline -> DSPy program -> GEPA mutates the real component -> exported candidate digest differs, plus metric/proposer/config-digest contracts. |
| `packages/runtime/python/evolution/tests/worker_capability_test.py` | Python worker capability-client tests: per-operation budget keys, token budget charging, timeout, and cancel. |
| `packages/runtime/src/index.ts` | Runtime public barrel for Doctor, Harness, and Python worker-environment APIs. `pure` |
| `packages/runtime/src/doctor.test.ts` | Contract tests for inspection, plan risk, and explicit repair behavior. |
| `packages/runtime/src/python-environment.test.ts` | Contract tests for read-only/no-download discovery, digest stability, environment readiness, forged/stale-status rejection, hash-snapshotted sanitized launch specs, and explicit setup/repair plans. |
| `packages/runtime/src/python-worker-smoke.test.ts` | End-to-end smoke contract between the Core app-attached controller and reference Python worker, including progress/checkpoint completion, partial and final-checkpoint resume, cancellation, and persisted human-decision resume. |

The root `pyproject.toml` names the product project `swarmx` and has no required
Python dependencies. Inspect evaluation tooling lives only in the opt-in
`inspect` dependency group, the DSPy/GEPA skill optimizer lives only in the
opt-in locked `evolution` dependency group, and product setup selects no default
groups; eval and evolution dependencies therefore never enter the worker
environment accidentally.

## CLI (`@swarmx/cli`)

| Source | Contract |
| --- | --- |
| `packages/cli/src/cli.ts` | Commander entrypoint for `doctor`, `send`, `eval-run`, `serve`, `audit`, `sessions`, `harnesses`, `evolution`, and REPL input; `send`/eval/REPL share content-free `agent.run` audit events distinguished by `surface`, while other commands retain semantic lifecycle actions. `proc`/`net` via Core/Runtime |
| `packages/cli/src/audit-command.ts` | Verifies and filters the canonical audit chain, formats compact human/JSON output, and writes 0600 verified JSONL exports with intent/outcome events. `fs` via Core |
| `packages/cli/src/doctor.ts` | Interactive/noninteractive doctor runner plus stable human/JSON formatting and confirmation handling. `proc` |
| `packages/cli/src/eval-run.ts` | Loads/validates workflow and eval arguments, executes `Swarm`, and formats deterministic eval result/error records with context-token usage; supports request-scoped `prompt_fragment` Skill delivery via `--skill-delivery` + `--skill-content-path`, and `--resolve-skill` binding to the evolved active revision. `fs` |
| `packages/cli/src/evolution-command.ts` | Thin CLI adapters for the skill evolution loop: `digest` (launch environment digest over worker/project/lockfile/evolution sources/Python), `evolve` (optimization WorkItem + candidate ingest with credential-free `--model-command` gateway wiring), `evaluate` (paired holdout or strictly validated Inspect evidence), `status` (ledger-only), `promote`, `reject`, `quarantine`, `rollback`, and `resolveActiveSkillDeliveriesForAgent` (the production entry that binds promoted revisions to named Agent nodes for new executions); business rules stay in Core. `fs` + `proc` via Core |
| `packages/cli/src/send-config.ts` | Builds a canonical one-agent `SwarmConfig` from CLI model/harness options. `pure` |
| `packages/cli/src/session-migration.ts` | CLI adapter for dry-run/migrate Session commands and concise result formatting. `fs` via Core |

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
| `packages/swarmx/bin/migrate-sessions.js` | Minimal executable alias for the Core/CLI Session migration command. `proc` |

## Package/launcher tests

The exact test inventory and what each contract protects are in
[`tests.md`](tests.md). CLI command modules intentionally remain adapters, so
new behavior should normally be implemented/tested in Core or Runtime first.
