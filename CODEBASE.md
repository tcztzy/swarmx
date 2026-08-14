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
                              ├─ task client      ├─ task kernel/store
                              ├─ Projects         ├─ Providers / Models
                              ├─ settings/auth    ├─ ACP / MCP
                              ├─ workspace tools  └─ schemas/contracts
                              └─ external harnesses
                              │ authenticated local socket
                              ▼
                       detached task supervisor
                                      │ strict JSONL stdio
                                      ▼
                                replaceable worker
                                (Python first)
                                      │ private MCP stdio
                                      ▼
                               RSI module (Python)

Desktop Main ───── private MCP stdio ─────► Memory module (Rust)
Desktop Main ───── private MCP stdio ─────► Reference module (Python)

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
- Memory authority: global `USER.md` / `MEMORY.md`, linked entity Markdown pages,
  and one local Git history under `~/.swarmx/memory/`; the search index and knowledge edges are rebuildable and
  never execute as workflow edges. There is no JSON fallback or second
  persistence authority.
  SwarmX-owned Agents receive an on-demand CRUD/search/graph/version tool whose
  mutations and restores require one-call user confirmation. Research capture
  stores typed, sourced Session provenance on entity pages.
- Agent identity: `harnessId:modelId`; Provider routes and effort do not create
  new identities.
- Session authority: append-only JSONL grouped by Project under
  `~/.swarmx/projects/`, with projectless history in `__recents__`; each index
  is rebuildable.
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
- Global Memory: bounded user-editable `USER.md` and `MEMORY.md` files under the
  Memory authority; direct runs receive a read-only snapshot and Sessions retain
  only a concise usage receipt. Reflection cursors persist in settings per
  Session, trigger on explicit requests or ten unreviewed user turns, and never
  aggregate raw dialogue across Sessions. The former settings-backed Personal
  Memory record is migration input for `USER.md` only.
- Reference Library: no SwarmX persistence authority; configured ZIM and Zotero
  sources remain authoritative and are accessed read-only by `swarmx.ref`.
  Direct Agents on exact official DeepSeek, OpenAI API, or Codex Responses
  endpoints expose provider-hosted search, with the official DeepSeek Anthropic
  route retained where Responses is unsupported. Hosted calls produce visible,
  correlated tool lifecycle events. `swarmx.ref` has no Web Search backend.
  Zotero access is confined to its fixed loopback read API without
  attachment/full-text reads.
- Worker transport: versioned strict JSONL over stdio with Core-owned schemas;
  it is not ACP and does not create a Python `SwarmConfig` node.
- Managed module transport: private MCP over stdio with verified runtime/source
  digests, sanitized environments, exact server/tool allowlists, bounded
  structured results, and host-owned authorization. Current modules are the
  Rust `swarmx-mem` server plus Python `swarmx.rsi` and `swarmx.ref` servers;
  neither raw tool surface is registered with Agents. Python is one regular
  `swarmx` package and locked environment with explicitly owned subpackages;
  it has no namespace-distribution discovery or module dependency groups. Rust
  crates are root Cargo-workspace members.
- Project authority: local folder bookmark/containment root in
  `~/.swarmx/projects.json`.
- Provider auth exception: `~/.swarmx/provider-auth.json`, schema version 2,
  plaintext credentials with restrictive permissions; Main only.
- Renderer transport: feature contracts in
  `packages/desktop/src/shared/ipc-contracts/` compose behind the compatibility
  facade in `shared/desktop-api.ts`; bridge implementation is in
  `preload/api.ts`, and audited Main routing composes in `main/ipc.ts`.

## Common change routes

| Change | Start here | Then inspect |
| --- | --- | --- |
| New domain field or persisted format | `packages/core/src/types.ts` or owning schema | `DESIGNS.md`, all boundary parsers, focused tests, desktop shared type |
| New linked-Memory graph behavior | `packages/core/src/memory-links.ts` | `docs/memory.md`, focused tests, and the workflow/knowledge-edge separation in `DESIGNS.md` |
| New Memory CRUD behavior | `packages/core/src/memory.ts` and `memory-runtime-protocol.ts` | `crates/swarmx-mem`, Git/revision/conflict tests, Desktop projection, graph projection, `docs/memory.md` |
| New Reference source behavior | `src/swarmx/ref/service.py` and `server.py` | source selection, ZIM/Web/Zotero bounds, real stdio MCP test, standard package boundary, `docs/reference-library.md` |
| New Python package capability | `src/swarmx/` and root `pyproject.toml` | one locked environment, direct project dependencies, private MCP identity tests, `tests/python/package_layout_test.py` |
| New managed feature module | owning Core zod contract | verified Runtime environment, private MCP host/client, exact tool allowlist, packaging, focused cross-process test, `DESIGNS.md` |
| New direct model behavior | `packages/core/src/agent.ts` and `native-model.ts` | `providers.ts`, `model-capabilities.ts`, rendering, activity, tests |
| New external harness behavior | `packages/core/src/acp.ts`, `harness.ts` | desktop harness/session runtime, runtime environment, ACP tests |
| New durable task state or event | `packages/core/src/task-runtime.ts` | store replay, control service, worker protocol, focused runtime tests, `DESIGNS.md` |
| New context projection, retrieval, or replay behavior | `packages/core/src/context-engine.ts` | context store, `context.ts`, focused context-engine tests, `docs/context-engine.md`, `DESIGNS.md` |
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
