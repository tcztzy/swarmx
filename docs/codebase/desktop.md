# `@swarmx/desktop`

Electron application split into Main, Preload, shared transport types, and
Renderer. Main is the only privileged process. Preload exposes a finite typed
API. Renderer receives normalized data and renders React UI.

## Runtime flow

1. `main/index.ts` creates the secure `BrowserWindow`, registers the media
   protocol, installs IPC handlers, and starts update checks.
2. `preload/index.ts` validates bootstrap data, wraps `ipcRenderer`, and exposes
   `window.swarmxAPI` through `contextBridge`.
3. `main/ipc.ts` validates channel inputs, resolves request ownership and
   permissions, applies the channel's audit policy, and delegates to Main
   services/Core.
4. `renderer/src/App.tsx` owns screen composition; feature workspaces call the
   shared API through `renderer-api.ts` and render sanitized Core/runtime data.
5. Direct execution uses `workspace-tools.ts` and `workspace-shell.ts`; ACP
   execution uses Core ACP and external Harness runtimes without duplicate
   Project tools.

## Main process

| Source | Contract |
| --- | --- |
| `packages/desktop/src/main/index.ts` | Electron entrypoint: app lifecycle, secure window, renderer URL, media protocol, IPC registration, update timer, terminal disposal, and an isolated content-free CI smoke probe for the built Renderer. `ipc` + `proc` |
| `packages/desktop/src/main/ipc.ts` | Main IPC router/authorization boundary; maps transport evidence to `ipc.request` plus normalized audit policy, records semantic effects, validates sender, injects Personal Memory snapshots, exposes local Memory/Reference tools only to SwarmX-owned execution, brokers explicit Agent memory mutation confirmation, exposes list/cancel/decision-only WorkItem controls, and excludes raw arguments/results from audit. `ipc` + `fs` + all privileged effects |
| `packages/desktop/src/main/library.ts` | Reusable Main-process public barrel; exports host services/types without starting Electron. `pure` |
| `packages/desktop/src/main/window-security.ts` | Trusted renderer URL/IPC checks, safe external URL policy, secure WebPreferences, navigation guards. `ipc` |
| `packages/desktop/src/main/request-registry.ts` | Request owner/context registry for cancellation, session association, and cleanup across IPC and agent runs. `pure` |
| `packages/desktop/src/main/agent-chunk-publisher.ts` | Batches/throttles bounded live agent/terminal chunks and sends only while the owner window exists. `ipc` |
| `packages/desktop/src/main/agent-interactions.ts` | Broker and typed events for questions, plan approval, and tool approval; request-scoped interaction ownership. `ipc` |
| `packages/desktop/src/main/acp-session-runtime.ts` | External ACP identity/binding and ephemeral Codex home management; attachment-aware session routing. `fs` |
| `packages/desktop/src/main/child-agent-host.ts` | Bounded Claude child-agent execution and inherited model/session history; rejects unsupported overrides. `proc` |
| `packages/desktop/src/main/browser-host.ts` | Embedded browser view lifecycle, bounds/navigation normalization, permission handling, and owner-scoped state. `net` + `ipc` |
| `packages/desktop/src/main/builtin-tool-settings.ts` | Reads stored tool-style preferences and resolves effective direct-run tool style/source. `fs` through settings |
| `packages/desktop/src/main/claude-scheduled-tasks.ts` | Locked scheduled-task store, cron file watching, owner/process liveness, and atomic task updates. `fs` + `proc` |
| `packages/desktop/src/main/claude-session-runtime.ts` | Claude session activation, monitor event buffering/rate limits, cron timers, and scheduled-task dispatch through WorkspaceShell. `proc` |
| `packages/desktop/src/main/codex-auth.ts` | Reads/validates Codex local access-token sources for Main-only use; returns no renderer secret. `fs` + `secret` |
| `packages/desktop/src/main/composer-preferences.ts` | Settings adapter for composer model/harness/effort preferences. `fs` through settings |
| `packages/desktop/src/main/custom-agents.ts` | Discovers native Claude/Codex agent definitions, converts to canonical profiles, validates workspace/home sources, and persists user agents. `fs` |
| `packages/desktop/src/main/extension-manager.ts` | Loads marketplace catalogs and coordinates explicit extension lifecycle actions with settings-backed state. `fs` + `net` |
| `packages/desktop/src/main/lsp-host.ts` | Spawns JSON-RPC language servers, serves bounded file/skill completions, and handles cancellation/shutdown. `proc` + `fs` |
| `packages/desktop/src/main/media.ts` | Imports attachments into managed content-addressed storage, validates MIME/size/identity, creates preview URLs, and supports safe text previews. `fs` |
| `packages/desktop/src/main/model-catalog.ts` | Provider/model discovery, credential-backed catalog refresh, official multi-protocol endpoint normalization for DeepSeek/OpenCode Go/OpenRouter, model-scoped DeepSeek Responses availability, model supply inventory, readiness summaries, and persisted manual models. `fs` + `net` + `secret` |
| `packages/desktop/src/main/permission-service.ts` | Loads managed/project/personal/profile policy layers, immediately applies the undeclared-mode Auto fallback without overwriting explicit settings, resolves effective permissions, records bounded human/model approval receipts, and fails closed on malformed policy. `fs` |
| `packages/desktop/src/main/permission-review.ts` | Tool-free LLM auto-review boundary: sends only bounded user messages plus the pending executable payload, strictly parses one-call allow/defer verdicts, and falls back to human review on any failure. `net` through an injected model call |
| `packages/desktop/src/main/personal-memory.ts` | Settings-backed Personal Memory read/save/explicit-forget service, immutable execution snapshot boundary, and direct Agent mutation tool with mandatory confirmation plus body-free audit events. `fs` through settings |
| `packages/desktop/src/main/memory-runtime-host.ts` | Private MCP-over-stdio host for `swarmx-mem`: verifies server version and exact single-tool surface, enforces structured/text consistency and response bounds, validates operation-matched Core schemas, and owns close semantics. The connection is never exposed to Renderer or Agent MCP registration. `proc` |
| `packages/desktop/src/main/reference-library-host.ts` | Main-only private MCP host for explicitly configured `swarmx-ref` ZIM and local Zotero sources; validates launch inputs, uses a credential-free environment, verifies the exact server/tool surface, enforces structured/text consistency and bounds, and closes with the app. `proc` + `fs` + loopback `net` through module |
| `packages/desktop/src/main/memory-runtime-backend.ts` | Implements the async Core `MemoryBackend` by mapping strict CRUD/search/version requests to private Memory MCP calls and rebuilding graph projection from the bounded snapshot. `proc` through host |
| `packages/desktop/src/main/memory-runtime-service.ts` | Lazy app-lifecycle owner for verified Memory runtime launch. It exposes the generic Core backend and owns no fallback format or migration authority. `fs` + `proc` |
| `packages/desktop/src/main/task-supervisor.ts` | Main-only connector/launcher for the detached Core supervisor; creates the mode-restricted token, reconnects safely, starts with a credential-free ambient environment, and never exposes launch/token authority to Renderer. `fs` + `net` + `proc` |
| `packages/desktop/src/main/task-supervisor-entry.ts` | Bundled Node-mode supervisor process entry; owns the Core socket server independently from Electron lifecycle. `net` + `proc` |
| `packages/desktop/src/main/provider-auth.ts` | Schema-version-2 user-editable Provider auth file store with restrictive permissions and credential lookup; plaintext never leaves Main. `fs` + `secret` |
| `packages/desktop/src/main/provider-error.ts` | Classifies Provider failures into stable user-facing error codes/notices without leaking response credentials. `pure` |
| `packages/desktop/src/main/provider-key-pool.ts` | Persists per-key usage/cooldown state and runs bounded Provider key selection/retry callbacks. `fs` + `secret` |
| `packages/desktop/src/main/provider-usage.ts` | Provider balance/window/credit/token usage queries and normalized snapshots; includes Codex app-server/New API adapters. `proc` + `net` + `secret` |
| `packages/desktop/src/main/pty-runtime.ts` | Ensures the packaged node-pty spawn helper is executable and resolves its runtime path. `fs` |
| `packages/desktop/src/main/session-messages.ts` | Converts Core Session events into desktop message streams, publishes chunks, handles timing/interruption, asserts final assistant output, and excludes persisted Memory receipts from subsequent model history. `ipc` |
| `packages/desktop/src/main/session-title.ts` | Generates/normalizes bounded Session titles and detects placeholders. `pure` |
| `packages/desktop/src/main/settings-store.ts` | Queued atomic JSON settings store with section-level merge and schema-safe persistence. `fs` |
| `packages/desktop/src/main/side-chat-service.ts` | In-memory transient Session forks, edits, unread/title state, and explicit promotion to canonical Sessions. `fs` only on promotion |
| `packages/desktop/src/main/terminal-host.ts` | Owner-scoped PTY lifecycle and fail-closed semantic audit actions `create`/`write`/`resize`/`close`/`exit`; close reasons distinguish user kill, owner cleanup, and app disposal. Records byte counts/lifecycle only, never terminal data/cwd/env. `proc` + `ipc` |
| `packages/desktop/src/main/updater.ts` | Npm release check/download/install/restart with host allowlist, integrity checks, progress state, and disabled mode. `net` + `fs` + `proc` |
| `packages/desktop/src/main/workspace-patch.ts` | Parses Codex patch hunks and applies complete-update operations with deterministic errors. `pure` |
| `packages/desktop/src/main/workspace-shell.ts` | Bounded shell/PTY sessions: sanitized environment, Project-root execution, output limits, polling, cancellation, process-group termination, and agent-tool adapter. `proc` + `fs` |
| `packages/desktop/src/main/workspace-tools.ts` | Direct Project tools for read/list/search/write/edit/delete/patch, review, browser, LSP, child agents, monitors, and cron; derives model-facing JSON Schema and call-time validation from shared Zod definitions, then enforces containment, digests, limits, permissions, and atomic writes. `fs` + `proc` |

## Preload and shared transport

| Source | Contract |
| --- | --- |
| `packages/desktop/src/preload/api.ts` | Typed IPC invoke/subscribe functions, bootstrap parser, Personal Memory controls, list/cancel/decision WorkItem controls, verified audit calls, and the finite `SwarmxDesktopApi` surface. `ipc` |
| `packages/desktop/src/preload/index.ts` | Electron-only bridge bootstrap; calls `createSwarmxDesktopApi` and exposes it with `contextBridge`. `ipc` |
| `packages/desktop/src/shared/desktop-api.ts` | Renderer-safe DTOs/events for Sessions, chunks, Personal Memory, WorkItem observation/control, media, browser, workspace, permissions, Providers, extensions, updates, audit, and all API methods. `pure` |

## Renderer application and feature UI

| Source | Contract |
| --- | --- |
| `packages/desktop/src/renderer/App.ts` | Package-level re-export of the Renderer App entry. `pure` |
| `packages/desktop/src/renderer/index.html` | Vite HTML shell, CSP, icon link, and `src/main.tsx` bootstrap. `ui` |
| `packages/desktop/src/renderer/src/main.tsx` | React DOM mount and global stylesheet import. `ui` |
| `packages/desktop/src/renderer/src/App.tsx` | Top-level application composition: navigation, conversation, composer, Personal Memory settings, workflow, extensions, runtime, profile, terminal, and GUI contribution registry; dead transitional panels do not remain as hidden alternatives. `ui` |
| `packages/desktop/src/renderer/src/renderer-api.ts` | Single renderer access point for `window.swarmxAPI`. `ipc` |
| `packages/desktop/src/renderer/src/agent-interaction-dialog.tsx` | Renders question/plan/tool approval dialogs and returns typed responses. `ui` |
| `packages/desktop/src/renderer/src/agent-picker.tsx` | Harness/model selection options, grouping, preferred/default resolution, and composition display. `ui` |
| `packages/desktop/src/renderer/src/app-brand.tsx` | Branded app icon component. `ui` |
| `packages/desktop/src/renderer/src/app-icon-data.ts` | Stable URL for the canonical SVG copied from Renderer `public/` by Vite. `pure` |
| `packages/desktop/src/renderer/src/composer.tsx` | Prompt editor, send/stop controls, attachment import, and LSP/server mention completion. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/conversation-history.tsx` | Renders normalized message/render events, always-visible Personal Memory use/not-used receipts, streaming progress, tool calls, edits, forks, attachments, and copy controls. `ui` |
| `packages/desktop/src/renderer/src/conversation-messages.ts` | Pure merge/key/timing helpers for streaming desktop message DTOs. `pure` |
| `packages/desktop/src/renderer/src/doctor-panel.tsx` | Displays Harness requirements, versions, doctor issues, and setup/fix state. `ui` |
| `packages/desktop/src/renderer/src/extension-presentation.ts` | Pure extension/agent composition labels, counts, chips, and plan status. `pure` |
| `packages/desktop/src/renderer/src/extension-workspace.tsx` | Marketplace source/catalog/inventory UI and explicit extension lifecycle actions. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/harness-icon-data.ts` | Packaged Harness icon URL registry. `pure` |
| `packages/desktop/src/renderer/src/harness-presentation.tsx` | Built-in Harness labels, icons, and selection presentation. `ui` |
| `packages/desktop/src/renderer/src/internal-terminal.tsx` | Bottom terminal panel using xterm and the typed terminal event API. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/media-preview.tsx` | Attachment preview panel for image/audio/video/document/text metadata and errors. `ui` |
| `packages/desktop/src/renderer/src/message-attachments.tsx` | Attachment list, icons, and bounded byte formatting. `ui` |
| `packages/desktop/src/renderer/src/message-content.tsx` | Sanitized Markdown/GFM/math/code rendering, local media loading, syntax highlighting, and copy controls; remote media policy remains blocked. `ui` |
| `packages/desktop/src/renderer/src/model-display.ts` | Pure model brand/order/reasoning presentation descriptors. `pure` |
| `packages/desktop/src/renderer/src/profile-workspace.tsx` | Activity profile summary, daily heatmap, rankings, and usage presentation. `ui` |
| `packages/desktop/src/renderer/src/provider-presentation.tsx` | Provider branding, protocol labels, and URL/provider classification. `ui` |
| `packages/desktop/src/renderer/src/runtime-settings.tsx` | Runtime environment/doctor setup plus detached WorkItem status, cancel, and pending-decision UI. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/session-navigation.ts` | Pure session discovery/grouping/cache/ordering/project navigation helpers. `pure` |
| `packages/desktop/src/renderer/src/settings-workspace.tsx` | General, Personal Memory edit/forget, permission, Provider usage, tool-style, and custom-agent settings screens. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/stylesheet-test-utils.ts` | Test-only stylesheet loader helper. `fs` |
| `packages/desktop/src/renderer/src/text-utils.ts` | Pure labels, errors, timestamps, path/project names, and slug helpers. `pure` |
| `packages/desktop/src/renderer/src/ui-primitives.tsx` | Tailwind-backed shared `Button` and `Badge` primitives with CVA-derived semantic variants, plus plain structural class composition. `ui` |
| `packages/desktop/src/renderer/src/workflow-workspace.tsx` | Workflow JSON parsing, node/edge graph UI, import validation, and execution controls for canonical `SwarmConfig`. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/workspace-panel.tsx` | Files/review/terminal/browser panel, diff parsing, file previews, and workspace navigation. `ui` + `ipc` |
| `packages/desktop/src/renderer/src/code-highlighter.ts` | Shiki-backed bounded syntax highlighting DTOs. `pure` |
| `packages/desktop/src/renderer/src/env.d.ts` | Vite/renderer environment type declarations. `pure` |

## Styles and visual inputs

| Source | Contract |
| --- | --- |
| `packages/desktop/src/renderer/src/assets/styles.css` | Sole Tailwind v4 stylesheet entry: semantic theme tokens, ordered inclusive max-width variants, explicit reset-free cascade layers, and bounded base/component imports; Vite emits the public compiled CSS artifact. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/base.css` | Owned reset-free Electron base styles and light/dark semantic token values; imported into Tailwind's `base` layer. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/app-shell.css` | Bounded Tailwind `utilities`-layer overrides for app-shell, navigation, and relational layout selectors. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/conversation.css` | Bounded Tailwind `utilities`-layer overrides for rich conversation, Markdown, tool, and attachment relationships not expressed locally. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/responsive.css` | Bounded Tailwind `utilities`-layer compound narrow-window and host-preference overrides. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/settings.css` | Bounded Tailwind `utilities`-layer overrides for compound settings, provider, permission, doctor, and profile relationships. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/workflow.css` | Bounded Tailwind `utilities`-layer overrides for workflow graph and editor relationships. `ui` |
| `packages/desktop/src/renderer/src/assets/styles/workspace.css` | Bounded Tailwind `utilities`-layer overrides for workspace files, terminal, browser, and review relationships. `ui` |
| `packages/desktop/src/renderer/public/ICON_VARIANTS.md` | Design notes for packaged app icon variants. |
| `packages/desktop/src/renderer/public/harness-icons/README.md` | Harness icon asset conventions. |
| `packages/desktop/src/renderer/public/provider-icons/README.md` | Provider icon asset conventions. |

The SVG/PNG files in `renderer/public` are static build assets rather than
runtime source contracts. Their filenames are intentionally discoverable from
the directory and are not duplicated in the code map.
