# Tests and auxiliary source

Tests are executable contracts, not a second implementation. A test file with
the same stem as a production module exercises that module's public behavior;
the list below makes the coverage route searchable without loading every test.

## Core tests

| Test paths |
| --- |
| `packages/core/tests/acp.test.ts`, `actions.test.ts`, `activity.test.ts`, `audit.test.ts`, `agent-profiles.test.ts`, `agent.test.ts`, `builtin-tools.test.ts`, `context.test.ts` |
| `packages/core/tests/conversation.test.ts`, `dependencies.test.ts`, `desktop-settings.test.ts`, `edge.test.ts`, `extension-management.test.ts`, `extensions.test.ts`, `harness-management.test.ts`, `harness.test.ts` |
| `packages/core/tests/mcp.test.ts`, `media.test.ts`, `model-capabilities.test.ts`, `n8n.test.ts`, `project.test.ts`, `providers.test.ts`, `rendering.test.ts`, `secrets.test.ts` |
| `packages/core/tests/server.test.ts`, `session-discovery.test.ts`, `session.test.ts`, `skill-variants.test.ts`, `swarm-eval.test.ts`, `swarm.test.ts`, `telemetry.test.ts`, `version.test.ts` |
| `packages/core/tests/skill-evolution.test.ts`, `skill-evolution-store.test.ts`, `skill-evolution-service.test.ts`, `skill-evaluation.test.ts`, `skill-delivery.test.ts` |
| `packages/core/tests/task-runtime.test.ts` |
| `packages/core/tests/task-runtime-store.test.ts` |
| `packages/core/tests/task-worker-protocol.test.ts` |
| `packages/core/tests/task-worker-process.test.ts` |
| `packages/core/tests/task-control-service.test.ts` |

Coverage focus: schema acceptance/rejection, graph scheduling/cycles, provider
redaction and routing, ACP/MCP cancellation, append-only persistence and
migration, capability composition, rendering sanitization, and deterministic
workflow evaluation. Activity tests enforce one aggregate `run_summary` per
run; audit tests enforce secret-free hash-chain replay and recovery. Durable
runtime tests additionally cover event replay and
idempotency collisions, fenced lease expiry, cancellation, retry and checkpoint
lineage/resume (including corrupted identity/environment rejection), torn-tail
recovery, Session-observer links, rejected human approval, protocol rejection,
and app-attached control-plane behavior.

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
`packages/core/tests/dependencies.test.ts`, `packages/core/tests/desktop-settings.test.ts`,
`packages/core/tests/edge.test.ts`, `packages/core/tests/extension-management.test.ts`,
`packages/core/tests/extensions.test.ts`, `packages/core/tests/harness-management.test.ts`,
`packages/core/tests/harness.test.ts`, `packages/core/tests/mcp.test.ts`,
`packages/core/tests/media.test.ts`, `packages/core/tests/model-capabilities.test.ts`,
`packages/core/tests/n8n.test.ts`, `packages/core/tests/project.test.ts`,
`packages/core/tests/providers.test.ts`, `packages/core/tests/rendering.test.ts`,
`packages/core/tests/secrets.test.ts`, `packages/core/tests/server.test.ts`,
`packages/core/tests/session-discovery.test.ts`, `packages/core/tests/session.test.ts`,
`packages/core/tests/skill-variants.test.ts`, `packages/core/tests/swarm-eval.test.ts`,
`packages/core/tests/swarm.test.ts`, `packages/core/tests/telemetry.test.ts`,
`packages/core/tests/version.test.ts`.

## Desktop tests

### Main/Preload

| Test paths |
| --- |
| `packages/desktop/src/main/acp-session-runtime.test.ts`, `agent-interactions.test.ts`, `browser-host.test.ts`, `builtin-tool-settings.test.ts`, `child-agent-host.test.ts`, `claude-scheduled-tasks.test.ts`, `claude-session-runtime.test.ts`, `codex-auth.test.ts` |
| `packages/desktop/src/main/composer-preferences.test.ts`, `custom-agents.test.ts`, `extension-manager.test.ts`, `harness-environment.test.ts`, `library.test.ts`, `lsp-host.test.ts`, `media-faults.test.ts`, `media-preview-hash.test.ts` |
| `packages/desktop/src/main/media.test.ts`, `model-catalog.test.ts`, `permission-review.test.ts`, `permission-service.test.ts`, `preload.test.ts`, `provider-auth.test.ts`, `provider-error.test.ts`, `provider-key-pool.test.ts`, `provider-usage.test.ts` |
| `packages/desktop/src/main/request-registry.test.ts`, `session-title.test.ts`, `settings-store.test.ts`, `side-chat-service.test.ts`, `terminal-host.test.ts`, `updater.test.ts`, `window-security.test.ts`, `workspace-shell.test.ts`, `workspace-tools.test.ts` |

### Renderer

| Test paths |
| --- |
| `packages/desktop/src/renderer/src/App.test.tsx`, `agent-interaction-dialog.test.tsx`, `agent-picker-layout.test.ts`, `app-brand.test.tsx`, `app-icon-data.test.ts`, `composer.test.tsx`, `conversation-messages.test.ts` |
| `packages/desktop/src/renderer/src/media-preview.test.tsx`, `message-attachments.test.tsx`, `message-content.test.tsx`, `model-display.test.ts`, `session-navigation.test.ts`, `settings-workspace.test.ts` |
| `packages/desktop/src/renderer/src/text-utils.test.ts`, `workflow-workspace.test.ts`, `workspace-panel-layout.test.ts`, `workspace-panel.test.tsx`, `harness-icon-data.test.ts` |

Coverage focus: IPC boundary validation, renderer-safe data, permission and
containment rules, provider credential isolation, terminal cancellation,
transport-policy audit compaction, semantic terminal close reasons, workspace
patching, media access, session navigation, and UI state transitions.

Exact Desktop test paths: `packages/desktop/src/main/acp-session-runtime.test.ts`,
`packages/desktop/src/main/agent-interactions.test.ts`,
`packages/desktop/src/main/browser-host.test.ts`,
`packages/desktop/src/main/builtin-tool-settings.test.ts`,
`packages/desktop/src/main/child-agent-host.test.ts`,
`packages/desktop/src/main/claude-scheduled-tasks.test.ts`,
`packages/desktop/src/main/claude-session-runtime.test.ts`,
`packages/desktop/src/main/codex-auth.test.ts`,
`packages/desktop/src/main/composer-preferences.test.ts`,
`packages/desktop/src/main/custom-agents.test.ts`,
`packages/desktop/src/main/extension-manager.test.ts`,
`packages/desktop/src/main/harness-environment.test.ts`,
`packages/desktop/src/main/library.test.ts`,
`packages/desktop/src/main/lsp-host.test.ts`,
`packages/desktop/src/main/media-faults.test.ts`,
`packages/desktop/src/main/media-preview-hash.test.ts`,
`packages/desktop/src/main/media.test.ts`,
`packages/desktop/src/main/model-catalog.test.ts`,
`packages/desktop/src/main/permission-review.test.ts`,
`packages/desktop/src/main/permission-service.test.ts`,
`packages/desktop/src/main/preload.test.ts`,
`packages/desktop/src/main/provider-auth.test.ts`,
`packages/desktop/src/main/provider-error.test.ts`,
`packages/desktop/src/main/provider-key-pool.test.ts`,
`packages/desktop/src/main/provider-usage.test.ts`,
`packages/desktop/src/main/request-registry.test.ts`,
`packages/desktop/src/main/session-title.test.ts`,
`packages/desktop/src/main/settings-store.test.ts`,
`packages/desktop/src/main/side-chat-service.test.ts`,
`packages/desktop/src/main/terminal-host.test.ts`,
`packages/desktop/src/main/updater.test.ts`,
`packages/desktop/src/main/window-security.test.ts`,
`packages/desktop/src/main/workspace-shell.test.ts`,
`packages/desktop/src/main/workspace-tools.test.ts`,
`packages/desktop/src/renderer/src/App.test.tsx`,
`packages/desktop/src/renderer/src/agent-interaction-dialog.test.tsx`,
`packages/desktop/src/renderer/src/agent-picker-layout.test.ts`,
`packages/desktop/src/renderer/src/app-brand.test.tsx`,
`packages/desktop/src/renderer/src/app-icon-data.test.ts`,
`packages/desktop/src/renderer/src/composer.test.tsx`,
`packages/desktop/src/renderer/src/conversation-messages.test.ts`,
`packages/desktop/src/renderer/src/harness-icon-data.test.ts`,
`packages/desktop/src/renderer/src/media-preview.test.tsx`,
`packages/desktop/src/renderer/src/message-attachments.test.tsx`,
`packages/desktop/src/renderer/src/message-content.test.tsx`,
`packages/desktop/src/renderer/src/model-display.test.ts`,
`packages/desktop/src/renderer/src/session-navigation.test.ts`,
`packages/desktop/src/renderer/src/settings-workspace.test.ts`,
`packages/desktop/src/renderer/src/text-utils.test.ts`,
`packages/desktop/src/renderer/src/workflow-workspace.test.ts`,
`packages/desktop/src/renderer/src/workspace-panel-layout.test.ts`,
`packages/desktop/src/renderer/src/workspace-panel.test.tsx`.

## CLI, ACP, launcher, and eval tests

| Test paths |
| --- |
| `packages/cli/tests/audit-command.test.ts`, `doctor.test.ts`, `eval-run.test.ts`, `send-config.test.ts`, `session-migration.test.ts`, `evolution-command.test.ts` (incl. real-chain promoted-revision resolution) |
| `packages/acp-server/src/server.test.ts`, `packages/runtime/src/doctor.test.ts`, `packages/runtime/src/python-environment.test.ts`, `packages/runtime/src/python-worker-smoke.test.ts`, `packages/swarmx/tests/launcher.test.ts` (including exact macOS InputMethodKit diagnostic filtering and preservation of all other Electron stderr) |
| `evals/inspect/__init__.py`, `evals/inspect/tasks.py`, `evals/inspect/tasks_test.py`, `evals/inspect/skill_eval.py`, `evals/inspect/skill_eval_test.py` |

Exact CLI test paths: `packages/cli/tests/audit-command.test.ts`,
`packages/cli/tests/doctor.test.ts`,
`packages/cli/tests/eval-run.test.ts`, `packages/cli/tests/send-config.test.ts`,
`packages/cli/tests/session-migration.test.ts`,
`packages/cli/tests/evolution-command.test.ts`.

CLI coverage includes the shared `agent.run` action and its
`cli_send`/`eval`/`repl` surface metadata, plus verified audit export. ACP
coverage includes correlated `acp.prompt` cancellation without a duplicate
cancel action or no-op mode events.

Exact Runtime test paths: `packages/runtime/src/doctor.test.ts`,
`packages/runtime/src/python-environment.test.ts`, and
`packages/runtime/src/python-worker-smoke.test.ts`.

## Auxiliary authored source

| Source | Contract |
| --- | --- |
| `packages/desktop/scripts/electron-stderr.mjs` | Shared launcher stream filter that removes only the known macOS InputMethodKit `IMKCFRunLoopWakeUpReliable` diagnostic and forwards all other Electron stderr. `proc` |
| `packages/desktop/scripts/start-electron.mjs` | Development launcher that starts Electron against the Vite renderer and applies the shared stderr filter. `proc` |
| `packages/desktop/scripts/build-macos-artifacts.mjs` | Packages the built Desktop application into macOS artifacts. `fs` + `proc` |
| `scripts/publish-npm.mjs` | Release helper for npm package publication. `proc` |
| `scripts/rebuild-icon.py` | Rebuilds packaged icon assets from the canonical icon input. `fs` |
| `scripts/check-codebase-docs.mjs` | CI/navigation guard: scans authoritative authored source/test roots and fails if a path is absent from `docs/codebase`. `fs` |
| `evals/inspect/tasks.py` | Inspect evaluation task definitions and adapter entrypoints. `proc` through evaluator |
| `evals/inspect/tasks_test.py` | Python evaluation adapter tests. |
| `evals/inspect/skill_eval.py` | Inspect `skill_paired_eval` adapter: runs baseline/candidate through the real `swarmx eval-run` path on a hidden holdout, deterministic scoring, and Core evidence JSON emission. Never writes the active revision or decides promotion. `proc` through evaluator |
| `evals/inspect/skill_eval_test.py` | Python tests for the paired adapter: seeded order, sample scoring, evidence serialization. |
| `evals/inspect/__init__.py` | Python package marker for inspect evaluation discovery. |

## Non-code assets

Desktop public SVG/PNG icons, `index.html`, and CSS are visual/build inputs;
their authored code paths are indexed by the guard and summarized in
[`desktop.md`](desktop.md). Generated `dist`, `out`, `release`, coverage, paper,
and PDF output are intentionally not documented as implementation.
