# SwarmX codebase map

This is the agent entry point for document-driven development. Read this file
and the package map for the area being changed before opening implementation
files. The maps describe ownership, contracts, data flow, side effects, and
every authored runtime/test source path. Generated output, dependencies, and
release bundles are intentionally excluded.

## Fast read order

1. [`AGENTS.md`](AGENTS.md): repository rules and non-negotiable invariants.
2. [`SPEC.md`](SPEC.md): durable product behavior.
3. [`DESIGNS.md`](DESIGNS.md): architecture and security rationale.
4. The relevant source map:
   - [`core`](docs/codebase/core.md)
   - [`desktop`](docs/codebase/desktop.md)
   - [`CLI, runtime, ACP server, and launcher`](docs/codebase/cli-runtime.md)
   - [`tests and auxiliary source`](docs/codebase/tests.md)
5. The exact source file and its adjacent test only when implementation detail
   is needed.

The source maps are deliberately denser than prose: one row is one file, and
the row states the file's job, public contract, and important effects. This
keeps agent context small while preserving navigation fidelity.

## System shape

```text
Desktop Renderer
      │ typed contextBridge API
      ▼
Desktop Preload ───────► Desktop Main ───────► Core / Runtime
                              │                  │
                              ├─ Sessions         ├─ Agent / Swarm
                              ├─ task controller  ├─ task kernel/store
                              ├─ Projects         ├─ Providers / Models
                              ├─ settings/auth    ├─ ACP / MCP
                              ├─ workspace tools  └─ schemas/contracts
                              └─ external harnesses
                                      │ strict JSONL stdio
                                      ▼
                                replaceable worker
                                (Python first)

CLI ───────────────► Core + Runtime
ACP server ────────► Core Session + Swarm
swarmx launcher ───► Desktop or CLI
```

## Ownership boundaries

| Area | Owns | Must not own |
| --- | --- | --- |
| `packages/core` | portable domain contracts, direct execution, durable task state/control, ACP/MCP adapters, Sessions, Projects, catalog and policy logic | Electron UI or renderer-only imports |
| `packages/runtime` | executable-harness and Python/uv discovery, environment checks, explicit setup/repair planning | product persistence, scheduling, or silent installation |
| `packages/desktop/src/main` | filesystem, subprocesses, credentials, IPC handlers, host integrations | renderer presentation or generic unvalidated IPC |
| `packages/desktop/src/preload` | narrow typed bridge and bootstrap validation | Node authority exposed to the browser |
| `packages/desktop/src/renderer` | React presentation and transient UI state | filesystem, subprocess, credentials, arbitrary network |
| `packages/cli` | Commander commands and terminal formatting | duplicate domain schemas or persistence rules |
| `packages/acp-server` | ACP transport around a Core `Swarm` and Session | duplicate agent runtime semantics |
| `packages/swarmx` | npm launcher selection/bootstrap | application behavior |

## Stable data paths

- Persisted workflow: `SwarmConfigSchema` only; graph nodes are `agent`, `tool`,
  or nested `swarm`.
- Agent identity: `harnessId:modelId`; Provider routes and effort do not create
  new identities.
- Session authority: append-only JSONL under `~/.swarmx/sessions/`; its index is
  rebuildable.
- Durable task authority: append-only runtime events under
  `~/.swarmx/task-runtime/`; WorkItems are independent of Sessions and workers.
- Skill evolution authority: append-only ledger records and content-addressed
  blobs under `~/.swarmx/skill-evolution/`; optimization WorkItems and
  candidate artifacts reuse the task-runtime store.
- Audit authority: concise hash-chained events and a head checkpoint under
  `~/.swarmx/audit/`; raw prompts, responses, source, terminal streams,
  credentials, and environment snapshots are excluded.
- Activity statistics: one aggregate `run_summary` per run in
  `~/.swarmx/activity.jsonl`; this profile data is not audit evidence.
- Worker transport: versioned strict JSONL over stdio with Core-owned schemas;
  it is not ACP and does not create a Python `SwarmConfig` node.
- Project authority: local folder bookmark/containment root in
  `~/.swarmx/projects.json`.
- Provider auth exception: `~/.swarmx/provider-auth.json`, schema version 2,
  plaintext credentials with restrictive permissions; Main only.
- Renderer transport: types in `packages/desktop/src/shared/desktop-api.ts`,
  bridge implementation in `packages/desktop/src/preload/api.ts`, handlers in
  `packages/desktop/src/main/ipc.ts`.

## Common change routes

| Change | Start here | Then inspect |
| --- | --- | --- |
| New domain field or persisted format | `packages/core/src/types.ts` or owning schema | `DESIGNS.md`, all boundary parsers, focused tests, desktop shared type |
| New direct model behavior | `packages/core/src/agent.ts` and `native-model.ts` | `providers.ts`, `model-capabilities.ts`, rendering, activity, tests |
| New external harness behavior | `packages/core/src/acp.ts`, `harness.ts` | desktop harness/session runtime, runtime environment, ACP tests |
| New durable task state or event | `packages/core/src/task-runtime.ts` | store replay, control service, worker protocol, focused runtime tests, `DESIGNS.md` |
| New task worker/backend | `packages/core/src/task-worker-protocol.ts` | process host, capability gateway, `packages/runtime` detection/plan, backend smoke test |
| New desktop capability | shared API → preload API → Main IPC → service → renderer | `window-security.ts`, request registry, permissions, focused UI tests |
| New workspace operation | `workspace-tools.ts` / `workspace-shell.ts` | containment, patch, cancellation, permission tests |
| New Provider or credential flow | Core provider/model modules, then Main auth/catalog/usage | secret redaction and Renderer types; never log plaintext |
| New privileged decision or side effect | `packages/core/src/audit.ts` | semantic action or transport policy, intent-before-effect where authority expands, safe correlations, focused redaction/failure tests |
| New skill evolution behavior | `packages/core/src/skill-evolution.ts` and `skill-evolution-service.ts` | ledger records, delivery module, paired evaluation, CLI adapter, Inspect adapter, `DESIGNS.md` skill loop section |
| New CLI command | `packages/cli/src/cli.ts` | command adapter, output formatter, CLI test |
| Session change or migration | `packages/core/src/session.ts` | discovery, desktop session messages, migration tests |

## Documentation rule

For every new or moved authored source file:

- add or update its row in the relevant map;
- record new public exports, persistence/protocol effects, and security
  boundaries in the row or the package overview;
- update the flow/contract section if the dependency direction changes;
- add/update the adjacent focused test map;
- run `pnpm docs:check` before handoff.

The check intentionally scans source and test paths, not `dist`, `node_modules`,
release artifacts, or generated reports. It is a navigation guard, not a
replacement for type checking or tests.

## Verification commands

```shell
pnpm docs:check
pnpm lint
pnpm test
pnpm -r build
```

Run the focused package test first when changing one boundary; run the broader
gate in proportion to risk. The final response must say what actually ran.
