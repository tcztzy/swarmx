# Tests and auxiliary source

Tests are executable contracts, not a second implementation. A test file with
the same stem as a production module exercises that module's public behavior;
the list below makes the coverage route searchable without loading every test.
The primary coverage gate spans media/window boundaries plus audit, context
assembly, durable task state, detached supervision, Session message projection,
and CLI send composition. Per-file thresholds live only in `vitest.config.ts`.

## Core tests

| Test paths |
| --- |
| `packages/core/tests/acp.test.ts`, `actions.test.ts`, `activity.test.ts`, `audit.test.ts`, `agent-profiles.test.ts`, `agent.test.ts`, `builtin-tools.test.ts`, `context.test.ts` |
| `packages/core/tests/conversation.test.ts`, `dependencies.test.ts`, `desktop-settings.test.ts`, `edge.test.ts`, `memory-links.test.ts`, `memory.test.ts`, `packages/core/tests/memory-runtime-protocol.test.ts`, `extension-management.test.ts`, `extensions.test.ts`, `harness-management.test.ts`, `harness.test.ts` |
| `packages/core/tests/local-tool-contracts.test.ts`, `mcp.test.ts`, `media.test.ts`, `model-capabilities.test.ts`, `agent-guidance.test.ts`, `n8n.test.ts`, `package-boundaries.test.ts`, `personal-memory.test.ts`, `project-bootstrap.test.ts`, `project-contracts.test.ts`, `project.test.ts`, `providers.test.ts`, `rendering.test.ts`, `request-scope.test.ts`, `secrets.test.ts` |
| `packages/core/tests/server.test.ts`, `session-discovery.test.ts`, `session.test.ts`, `skill-variants.test.ts`, `swarm-eval.test.ts`, `swarm.test.ts`, `telemetry.test.ts`, `version.test.ts` |
| `packages/core/tests/skill-evolution.test.ts`, `skill-evolution-store.test.ts`, `skill-evolution-service.test.ts`, `skill-evaluation.test.ts`, `skill-delivery.test.ts` |
| `packages/core/tests/task-runtime.test.ts` |
| `packages/core/tests/task-runtime-store.test.ts` |
| `packages/core/tests/task-worker-protocol.test.ts` |
| `packages/core/tests/task-worker-process.test.ts` |
| `packages/core/tests/task-control-service.test.ts` |
| `packages/core/tests/task-supervisor.test.ts` |
| `packages/core/tests/context-engine.test.ts` |
| `packages/core/tests/context-evaluation.test.ts` |
| `packages/core/tests/context-engine-store.test.ts` |

Coverage focus: schema acceptance/rejection, graph scheduling/cycles, provider
redaction and routing, ACP/MCP cancellation, Provider/workflow continuation
settlement, post-terminal ACP update rejection, per-Project JSONL-only append-only
persistence and recovery, capability composition, rendering sanitization, and
deterministic workflow evaluation. Activity tests enforce one aggregate `run_summary` per
run; audit tests enforce secret-free hash-chain replay and recovery. Durable
runtime tests additionally cover event replay and
idempotency collisions, fenced lease expiry, cancellation, retry and checkpoint
lineage/resume (including corrupted identity/environment rejection), torn-tail
recovery, Session-observer links, rejected human approval, protocol rejection,
app-attached control-plane behavior, authenticated supervisor rejection, and
execution continuing after a requesting client disconnects. Supervisor tests
also require automatic redispatch after a retryable failure and after an
approved persisted human pause, without a second client-supplied run request.

Skill evolution tests cover the end-to-end closed loop with the real Python
worker (deterministic optimizer), candidate immutability and lineage, static
check rejection (secrets, size), paired evaluation through the same Swarm path
with model-visible instruction digests, stale-parent compare-and-swap rejection,
rollback to retained revisions, promotion affecting only new execution
snapshots, audit intent-before-effect with fail-closed audit failure, the
fail-closed policy gate, and grant-scoped capability denial. Delivery tests
prove digest verification and that candidate instructions change what a stub
model observes.

Exact Core test paths: `packages/core/tests/acp.test.ts`,
`packages/core/tests/actions.test.ts`, `packages/core/tests/activity.test.ts`,
`packages/core/tests/audit.test.ts`,
`packages/core/tests/agent-profiles.test.ts`, `packages/core/tests/agent.test.ts`,
`packages/core/tests/builtin-tools.test.ts`,
`packages/core/tests/task-runtime.test.ts`,
`packages/core/tests/task-runtime-store.test.ts`,
`packages/core/tests/task-worker-protocol.test.ts`,
`packages/core/tests/task-worker-process.test.ts`,
`packages/core/tests/task-control-service.test.ts`,
`packages/core/tests/security-utils.test.ts`,
`packages/core/tests/skill-evolution.test.ts`,
`packages/core/tests/skill-evolution-store.test.ts`,
`packages/core/tests/skill-evolution-service.test.ts`,
`packages/core/tests/skill-evaluation.test.ts`,
`packages/core/tests/skill-delivery.test.ts`,
`packages/core/tests/context.test.ts`, `packages/core/tests/conversation.test.ts`,
`packages/core/tests/context-engine.test.ts`, `packages/core/tests/context-evaluation.test.ts`,
`packages/core/tests/context-engine-store.test.ts`,
`packages/core/tests/dependencies.test.ts`, `packages/core/tests/desktop-settings.test.ts`,
`packages/core/tests/edge.test.ts`, `packages/core/tests/memory-links.test.ts`, `packages/core/tests/memory.test.ts`, `packages/core/tests/extension-management.test.ts`,
`packages/core/tests/extensions.test.ts`, `packages/core/tests/harness-management.test.ts`,
`packages/core/tests/harness.test.ts`, `packages/core/tests/mcp.test.ts`,
`packages/core/tests/local-tool-contracts.test.ts`,
`packages/core/tests/media.test.ts`, `packages/core/tests/model-capabilities.test.ts`,
`packages/core/tests/agent-guidance.test.ts`,
`packages/core/tests/n8n.test.ts`, `packages/core/tests/personal-memory.test.ts`,
`packages/core/tests/package-boundaries.test.ts`,
`packages/core/tests/project-bootstrap.test.ts`,
`packages/core/tests/project-contracts.test.ts`,
`packages/core/tests/project.test.ts`,
`packages/core/tests/providers.test.ts`, `packages/core/tests/rendering.test.ts`,
`packages/core/tests/request-scope.test.ts`,
`packages/core/tests/secrets.test.ts`, `packages/core/tests/server.test.ts`,
`packages/core/tests/session-discovery.test.ts`, `packages/core/tests/session.test.ts`,
`packages/core/tests/skill-variants.test.ts`, `packages/core/tests/swarm-eval.test.ts`,
`packages/core/tests/swarm.test.ts`, `packages/core/tests/telemetry.test.ts`,
`packages/core/tests/version.test.ts`.

## Desktop tests

### Main/Preload

| Test paths |
| --- |
| `packages/desktop/src/main/acp-session-runtime.test.ts`, `agent-interactions.test.ts`, `browser-host.test.ts`, `builtin-tool-settings.test.ts`, `child-agent-host.test.ts`, `claude-scheduled-tasks.test.ts`, `claude-session-runtime.test.ts`, `codex-auth.test.ts`, `direct-harness-release.e2e.test.ts` |
| `packages/desktop/src/main/composer-preferences.test.ts`, `custom-agents.test.ts`, `extension-manager.test.ts`, `harness-environment.test.ts`, `ipc-router.test.ts`, `library.test.ts`, `lsp-host.test.ts`, `media-faults.test.ts`, `media-preview-hash.test.ts` |
| `packages/desktop/src/main/media.test.ts`, `model-catalog.test.ts`, `permission-review.test.ts`, `permission-service.test.ts`, `global-memory-service.test.ts`, `personal-memory.test.ts`, `preload.test.ts`, `private-json-file.test.ts`, `provider-auth.test.ts`, `provider-error.test.ts`, `provider-key-pool.test.ts`, `provider-usage.test.ts` |
| `packages/desktop/src/main/memory-runtime-host.test.ts`, `packages/desktop/src/main/memory-runtime-integration.test.ts`, `packages/desktop/src/main/memory-runtime-backend.test.ts` |
| `packages/desktop/src/main/browser-ipc.test.ts`, `global-memory-ipc.test.ts`, `project-ipc.test.ts`, `project-service.test.ts`, `request-registry.test.ts`, `session-title.test.ts`, `settings-store.test.ts`, `side-chat-service.test.ts`, `task-runtime-ipc.test.ts`, `task-supervisor.test.ts`, `terminal-host.test.ts`, `terminal-ipc.test.ts`, `updater.test.ts`, `window-security.test.ts`, `workspace-inspection-ipc.test.ts`, `workspace-shell.test.ts`, `workspace-tool-permissions.test.ts`, `workspace-tools.test.ts` |
| `packages/desktop/src/shared/ipc-contracts/app-update.test.ts`, `browser.test.ts`, `global-memory.test.ts`, `project.test.ts`, `task-runtime.test.ts`, `terminal.test.ts`, `workspace-inspection.test.ts` |

### Renderer

| Test paths |
| --- |
| `packages/desktop/src/renderer/src/App.test.tsx`, `agent-interaction-dialog.test.tsx`, `app-brand.test.tsx`, `app-icon-data.test.ts`, `composer.test.tsx`, `conversation-messages.test.ts`, `doctor-controller.test.tsx` |
| `packages/desktop/src/renderer/src/media-preview.test.tsx`, `message-attachments.test.tsx`, `message-content.test.tsx`, `model-display.test.ts`, `session-navigation.test.ts`, `settings-workspace.test.ts` |
| `packages/desktop/src/renderer/src/styling-architecture.test.ts`, `terminal-controller.test.tsx`, `ui-primitives.test.tsx`, `text-utils.test.ts`, `workflow-workspace.test.ts`, `workspace-panel.test.tsx`, `harness-icon-data.test.ts` |

Coverage focus: IPC boundary validation, renderer-safe data, floating-composer
layout and transcript clearance, permission and containment rules, provider
credential isolation, the isolated explicit-Provider direct-Harness release path
through native streaming and restart recovery, terminal cancellation,
transport-policy audit compaction, semantic terminal close reasons, workspace
patching, media access, detached-supervisor startup/reconnection and environment
isolation, live read-only WorkItem refresh while Runtime Settings is visible,
session navigation, UI state transitions, and a built-Electron smoke
probe that measures the actual split-panel and Agent-picker geometry. Layout
behavior lives in rendered interaction/smoke tests rather than source-string
class assertions.

Exact Desktop test paths: `packages/desktop/src/main/acp-session-runtime.test.ts`,
`packages/desktop/src/main/agent-interactions.test.ts`,
`packages/desktop/src/main/browser-host.test.ts`,
`packages/desktop/src/main/browser-ipc.test.ts`,
`packages/desktop/src/main/global-memory-ipc.test.ts`,
`packages/desktop/src/main/builtin-tool-settings.test.ts`,
`packages/desktop/src/main/child-agent-host.test.ts`,
`packages/desktop/src/main/claude-scheduled-tasks.test.ts`,
`packages/desktop/src/main/claude-session-runtime.test.ts`,
`packages/desktop/src/main/codex-auth.test.ts`,
`packages/desktop/src/main/direct-harness-release.e2e.test.ts`,
`packages/desktop/src/main/composer-preferences.test.ts`,
`packages/desktop/src/main/custom-agents.test.ts`,
`packages/desktop/src/main/extension-manager.test.ts`,
`packages/desktop/src/main/harness-environment.test.ts`,
`packages/desktop/src/main/ipc-router.test.ts`,
`packages/desktop/src/main/library.test.ts`,
`packages/desktop/src/main/lsp-host.test.ts`,
`packages/desktop/src/main/media-faults.test.ts`,
`packages/desktop/src/main/media-preview-hash.test.ts`,
`packages/desktop/src/main/media.test.ts`,
`packages/desktop/src/main/model-catalog.test.ts`,
`packages/desktop/src/main/permission-review.test.ts`,
`packages/desktop/src/main/permission-service.test.ts`,
`packages/desktop/src/main/global-memory-service.test.ts`,
`packages/desktop/src/main/personal-memory.test.ts`,
`packages/desktop/src/main/preload.test.ts`,
`packages/desktop/src/main/private-json-file.test.ts`,
`packages/desktop/src/main/project-ipc.test.ts`,
`packages/desktop/src/main/project-service.test.ts`,
`packages/desktop/src/main/provider-auth.test.ts`,
`packages/desktop/src/main/provider-error.test.ts`,
`packages/desktop/src/main/provider-key-pool.test.ts`,
`packages/desktop/src/main/provider-usage.test.ts`,
`packages/desktop/src/main/request-registry.test.ts`,
`packages/desktop/src/main/session-title.test.ts`,
`packages/desktop/src/main/settings-store.test.ts`,
`packages/desktop/src/main/side-chat-service.test.ts`,
`packages/desktop/src/main/task-supervisor.test.ts`,
`packages/desktop/src/main/task-runtime-ipc.test.ts`,
`packages/desktop/src/main/terminal-host.test.ts`,
`packages/desktop/src/main/terminal-ipc.test.ts`,
`packages/desktop/src/main/updater.test.ts`,
`packages/desktop/src/main/window-security.test.ts`,
`packages/desktop/src/main/workspace-inspection-ipc.test.ts`,
`packages/desktop/src/main/workspace-shell.test.ts`,
`packages/desktop/src/main/workspace-tool-permissions.test.ts`,
`packages/desktop/src/main/workspace-tools.test.ts`,
`packages/desktop/src/shared/ipc-contracts/app-update.test.ts`,
`packages/desktop/src/shared/ipc-contracts/browser.test.ts`,
`packages/desktop/src/shared/ipc-contracts/global-memory.test.ts`,
`packages/desktop/src/shared/ipc-contracts/project.test.ts`,
`packages/desktop/src/shared/ipc-contracts/task-runtime.test.ts`,
`packages/desktop/src/shared/ipc-contracts/terminal.test.ts`,
`packages/desktop/src/shared/ipc-contracts/workspace-inspection.test.ts`,
`packages/desktop/src/renderer/src/App.test.tsx`,
`packages/desktop/src/renderer/src/agent-interaction-dialog.test.tsx`,
`packages/desktop/src/renderer/src/app-brand.test.tsx`,
`packages/desktop/src/renderer/src/app-icon-data.test.ts`,
`packages/desktop/src/renderer/src/composer.test.tsx`,
`packages/desktop/src/renderer/src/conversation-messages.test.ts`,
`packages/desktop/src/renderer/src/doctor-controller.test.tsx`,
`packages/desktop/src/renderer/src/harness-icon-data.test.ts`,
`packages/desktop/src/renderer/src/media-preview.test.tsx`,
`packages/desktop/src/renderer/src/message-attachments.test.tsx`,
`packages/desktop/src/renderer/src/message-content.test.tsx`,
`packages/desktop/src/renderer/src/model-display.test.ts`,
`packages/desktop/src/renderer/src/session-navigation.test.ts`,
`packages/desktop/src/renderer/src/settings-workspace.test.ts`,
`packages/desktop/src/renderer/src/styling-architecture.test.ts`,
`packages/desktop/src/renderer/src/terminal-controller.test.tsx`,
`packages/desktop/src/renderer/src/text-utils.test.ts`,
`packages/desktop/src/renderer/src/ui-primitives.test.tsx`,
`packages/desktop/src/renderer/src/workflow-workspace.test.ts`,
`packages/desktop/src/renderer/src/workspace-panel.test.tsx`.

## CLI, ACP, launcher, and eval tests

| Test paths |
| --- |
| `packages/cli/tests/audit-command.test.ts`, `cli-entry.test.ts`, `doctor.test.ts`, `eval-run.test.ts`, `send-config.test.ts`, `evolution-command.test.ts` (incl. real-chain promoted-revision resolution) |
| `packages/acp-server/src/server.test.ts`, `packages/runtime/src/doctor.test.ts`, `packages/runtime/src/python-environment.test.ts`, `packages/runtime/src/python-worker-smoke.test.ts`, `packages/runtime/src/memory-runtime-environment.test.ts`, `packages/swarmx/tests/launcher.test.ts` (including exact macOS InputMethodKit diagnostic filtering and preservation of all other Electron stderr) |
| `evals/inspect/__init__.py`, `evals/inspect/tasks.py`, `evals/inspect/tasks_test.py`, `evals/inspect/skill_eval.py`, `evals/inspect/skill_eval_test.py` |

Exact CLI test paths: `packages/cli/tests/audit-command.test.ts`,
`packages/cli/tests/cli-entry.test.ts`,
`packages/cli/tests/doctor.test.ts`,
`packages/cli/tests/eval-run.test.ts`, `packages/cli/tests/send-config.test.ts`,
`packages/cli/tests/evolution-command.test.ts`.

CLI coverage includes the real Session/Harness list entrypoints, multi-turn REPL
history, strict context-suite loading and Commander routing, exclusive 0600
content-free JSONL output, checked-in matrix bounds, the shared `agent.run`
action and its `cli_send`/`eval`/`repl` surface metadata, plus verified audit
export. ACP
coverage includes correlated `acp.prompt` cancellation without a duplicate
cancel action or no-op mode events.

Exact Runtime test paths: `packages/runtime/src/doctor.test.ts`,
`packages/runtime/src/python-environment.test.ts`, and
`packages/runtime/src/python-worker-smoke.test.ts`.

Reference boundary coverage lives in
`packages/core/tests/reference-library.test.ts` and
`packages/desktop/src/main/reference-library-host.test.ts`; Python service,
real-ZIM MCP, and standard-package paths are indexed in
[`cli-runtime.md`](cli-runtime.md).

## Auxiliary authored source

| Source | Contract |
| --- | --- |
| `packages/desktop/scripts/electron-stderr.mjs` | Shared launcher stream filter that removes only the known macOS InputMethodKit `IMKCFRunLoopWakeUpReliable` diagnostic and forwards all other Electron stderr. `proc` |
| `packages/desktop/scripts/start-electron.mjs` | Development launcher that starts Electron against the Vite renderer and applies the shared stderr filter. `proc` |
| `packages/desktop/scripts/build-macos-artifacts.mjs` | Packages the built Desktop application into macOS artifacts. `fs` + `proc` |
| `packages/desktop/scripts/build-mem-runtime.mjs` | Builds the locked `swarmx-mem` crate for one target, copies the executable into Desktop resources, and writes its digest/version manifest. No install-at-runtime path. `fs` + `proc` |
| `packages/desktop/scripts/test-mem-runtime.mjs` | Cross-platform managed-Memory acceptance runner: builds the locked runtime and executes the real private-MCP integration test with its explicit manifest. `proc` |
| `packages/desktop/scripts/test-desktop-smoke.mjs` | macOS CI acceptance runner: launches the built Electron app in an isolated home and requires real rendered CSS geometry for the main split panel and Agent picker. `proc` |
| `scripts/publish-npm.mjs` | Release helper for npm package publication: packs in dependency order, rejects unresolved internal ranges and test artifacts, verifies integrity, and publishes through the registry. `proc` |
| `scripts/rebuild-icon.py` | Rebuilds packaged icon assets from the canonical icon input. `fs` |
| `scripts/check-codebase-docs.mjs` | CI/navigation guard: scans authoritative authored source/test roots and fails if a path is absent from `docs/codebase`. `fs` |
| `evals/inspect/tasks.py` | Inspect evaluation task definitions and adapter entrypoints. `proc` through evaluator |
| `evals/inspect/tasks_test.py` | Python evaluation adapter tests. |
| `evals/inspect/skill_eval.py` | Inspect `skill_paired_eval` adapter: runs baseline/candidate through the real `swarmx eval-run` path on a hidden holdout, deterministic scoring, and Core evidence JSON emission. Never writes the active revision or decides promotion. `proc` through evaluator |
| `evals/inspect/skill_eval_test.py` | Python tests for the paired adapter: seeded order, sample scoring, evidence serialization. |
| `evals/inspect/__init__.py` | Python package marker for inspect evaluation discovery. |
| `evals/context/README.md` | Operator guidance and validity limits for the editable context-strategy benchmark fixture. |
| `evals/context/smoke-suite.json` | Strict development-split matrix with 10 harness/paper profiles, two replaceable Model slots, and five tool-rich coding continuation cases; its declared first-pass bound is 100 runs. |

## Non-code assets

Desktop public SVG/PNG icons, `index.html`, and CSS are visual/build inputs;
their authored code paths are indexed by the guard and summarized in
[`desktop.md`](desktop.md). Generated `dist`, `out`, `release`, coverage, paper,
and PDF output are intentionally not documented as implementation.
