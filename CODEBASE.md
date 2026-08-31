# SwarmX codebase map

SwarmX is one Electron product whose only application UI/plugin host is the published DeepSeek Harness Web profile. DSH and Codex App Server are peer conversation runtimes below a narrow Host Cordis service; local-first Science, durable Swarm, and private PKB policy are runtime-neutral, with DSH and Codex code confined to bottom carriers.

## Runtime flow

`apps/desktop/src/main.ts` resolves the default runtime and canonical workspace through `runtime/platform.ts`. The platform always boots DSH Web, registers the available peer adapters on its existing `webServer`, then loads only the DSH Web URL through `apps/desktop/src/window.ts`.

Each native Session/Thread is its only transcript truth. `ConversationRuntimeRegistry` multiplexes disposable projections through `/api/swarmx/conversation-runtimes` on the DSH listener; it serves no HTML and creates no synthetic DSH Session. DSH product features use Cordis carriers, while Codex receives the same Science/PKB/Swarm authorities through one owned required MCP stdio server.

## Authored source and tests

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/app-lifecycle.ts` | Electron single-primary ownership/focus plus once-only quit coordination, pending-exit fencing, and explicit cleanup-failure reporting |
| `apps/desktop/src/harness.ts` | Profile creation, profile dependency-closure module resolution, DSH + Science + Swarm system-preset roots, patch composition, in-process Harness boot, authenticated loopback launch URL |
| `apps/desktop/src/main.ts` | Explicit runtime/workspace selection, shared-platform startup, window recreation, and ordered shutdown |
| `apps/desktop/src/window.ts` | BrowserWindow construction, navigation and permission boundary |
| `apps/desktop/src/runtime/approval.ts` | Runtime-qualified single-use pending approval registry and fail-closed disposal |
| `apps/desktop/src/runtime/bridge.ts` | Token-authenticated loopback bridge granting bounded native operations and root-claiming newly created Codex member Threads before response publication |
| `apps/desktop/src/runtime/contracts.ts` | Platform-neutral conversation, item, event, approval, workspace, and revision/lifecycle contracts |
| `apps/desktop/src/runtime/controller.ts` | Runtime-neutral Retry/Edit revision and explicit Fork orchestration |
| `apps/desktop/src/runtime/index.ts` | Shared runtime public exports |
| `apps/desktop/src/runtime/platform.ts` | Always-on DSH Web host, selected peer adapters, workspace authority, product migration, bridge, protocol plugin, and teardown composition |
| `apps/desktop/src/runtime/product-home.ts` | Owner-only `$SWARMX_HOME` resolution plus bounded verified one-time legacy Science/PKB/Swarm import |
| `apps/desktop/src/runtime/selection.ts` | Strict CLI/environment `dsh|codex` selection without fallback |
| `apps/desktop/src/runtime/registry.ts` | Exact peer-adapter registration, lookup, once-sequenced global event fanout, and once-only disposal |
| `apps/desktop/src/runtime/science-config.ts` | Strict bounded projection of composed Science settings across the Host-to-Codex-MCP process boundary |
| `apps/desktop/src/runtime/swarm-recovery-owner.ts` | Non-configurable product-lifetime cold/final recovery owner and exact provisional Codex-member claim authority outside the patchable Harness tree |
| `apps/desktop/src/runtime/web-plugin.ts` | Host Cordis service registering the bounded runtime HTTP/SSE protocol on DSH Web's existing listener |
| `apps/desktop/src/runtime/workspace.ts` | Canonical Host-minted workspace identity/token and descendant authorization |
| `apps/desktop/src/runtime/dsh/runtime.ts` | Minimal Session/Agent/query adapter from DSH native state to `ConversationRuntime`; DSH Web retains its sole native approval path |
| `apps/desktop/src/runtime/codex/connection.ts` | Bounded JSONL JSON-RPC App Server connection with structured protocol errors, allowlisted experimental initialization, pending-call, stderr, and child lifecycle |
| `apps/desktop/src/runtime/codex/index.ts` | Codex process launch plus required SwarmX MCP configuration and secret-safe environment handoff |
| `apps/desktop/src/runtime/codex/member-bindings.ts` | Exact transactional mapping from workspace member authority to native Codex Thread handles over owner-initialized storage |
| `apps/desktop/src/runtime/codex/runtime.ts` | Thread/turn/item/approval adapter with cursor-bounded list, transient blank summaries, stored-thread resume, typed MCP elicitation validation, exact native fork boundaries, cascade archive cleanup, and same-thread last-turn revert |
| `apps/desktop/src/runtime/codex/swarm-recovery.ts` | Root-runtime-only reconciliation of interrupted Codex provisioning and stale native member bindings |
| `apps/desktop/src/runtime/codex/mcp-server.ts` | Required stdio Science/PKB/Swarm MCP carrier with carrier-exact schema projection plus shared strict validation, native-Thread/session/call-scoped identity, workspace bridge, native Codex member threads, and bounded image output |
| `apps/desktop/tests/app-lifecycle.test.ts` | Single-primary focus/secondary exit plus once-only quit, successful continuation, and fail-loud cleanup rejection contract |
| `apps/desktop/tests/harness.test.ts` | Real profile boot, profile-installed bundle resolution, Science preset discovery/scoping, browser-handoff and HTTP integration |
| `apps/desktop/tests/codex-connection.test.ts` | App Server handshake, framing bounds, malformed payload, pending-call, and process disposal contract |
| `apps/desktop/tests/codex-member-bindings.test.ts` | Concurrent per-member binding claim/release, workspace isolation, validation, and owner-permission contract |
| `apps/desktop/tests/codex-real.test.ts` | Installed real Codex App Server schema, stored-thread resume, required product MCP startup, and `SWARMX_CODEX_FULL_ACCEPTANCE=1` schema/approval/PKB scope/native-member restart/archive/lifecycle gate |
| `apps/desktop/tests/codex-runtime.test.ts` | Codex history-mode selection, paged/transient list, resume, projection, exact native fork/revert revision, cascade cleanup, ordered events, approvals, questions, and elicitation contract |
| `apps/desktop/tests/codex-swarm-recovery.test.ts` | Root-only Codex provisioning resume, fail-closed missing/transient/cancel classification, stale binding, and native orphan cleanup contract |
| `apps/desktop/tests/dsh-runtime.test.ts` | DSH projection, native operations/forked revision, ordered events, and no-second-approval-handler contract |
| `apps/desktop/tests/product-mcp.test.ts` | Shared product tool list, Science project/image path, exact identity, workspace recovery, concurrent native-member/mailbox ownership, live child lifecycle, and Swarm carrier contract |
| `apps/desktop/tests/runtime-bridge.test.ts` | Bearer authorization, Host-owned workspace/model routing, and response-loss-safe root member-claim contract |
| `apps/desktop/tests/runtime-contract.test.ts` | Shared Retry/Edit revision/Fork behavior contract |
| `apps/desktop/tests/runtime-selection.test.ts` | Strict runtime CLI/environment selection and invalid/duplicate rejection |
| `apps/desktop/tests/swarm-recovery-owner.test.ts` | Product-level cold/final recovery ownership independent of configurable Harness plugins |
| `apps/desktop/tests/runtime-platform.test.ts` | Real DSH Web startup proving the runtime registry shares its origin and no second UI URL |
| `apps/desktop/tests/runtime-registry.test.ts` | Peer registration, no-fallback lookup, qualified event multiplexing, and disposal contract |
| `apps/desktop/tests/runtime-web-plugin.test.ts` | Existing-listener Cordis mount, strict runtime routes, origin rejection, and no-root ownership contract |
| `apps/desktop/tests/workspace-platform.test.ts` | Canonical workspace containment plus bounded idempotent product-home migration contract |
| `apps/desktop/tests/window.test.ts` | Window security regression tests |
| `native/writing-preview-runtime/Cargo.toml` | Native semantic writing preview runtime dependency and release boundary; Typst is the initial engine |
| `native/writing-preview-runtime/Cargo.lock` | Reproducible native writing runtime dependency graph |
| `native/writing-preview-runtime/src/main.rs` | Typst engine implementation for same-snapshot compile/watch, PDF export, and IDE click-to-source resolution |
| `packages/core/annotation/tsdown.config.ts` | Provider-neutral annotation package artifact configuration |
| `packages/core/annotation/vitest.config.ts` | Focused annotation contract test configuration |
| `packages/core/annotation/src/index.ts` | OpenAI Responses annotation superset plus bounded SwarmX document, image, and source-addressed message targets |
| `packages/core/annotation/tests/annotation.test.ts` | Official branch round-trip, provider-field preservation, message quote, and extension rejection contract |
| `packages/core/dvc/tsdown.config.ts` | Host-only DVC service artifact configuration |
| `packages/core/dvc/src/git-worktree.ts` | Package-private committed Git identity and owner-only detached-worktree lifecycle used only by DVC replay |
| `packages/core/dvc/src/index.ts` | `ctx.dvc` path-free status, explicit pull, isolated exact-HEAD reproduction, request validation, redaction, and cleanup |
| `packages/core/dvc/tests/dvc.test.ts` | DVC status digest, explicit mutation, isolated replay, dirty-workspace rejection, failure redaction, lazy CLI, and disposal contract |
| `packages/core/dvc/tests/dvc-real.test.ts` | Opt-in-required real DVC CLI initialization, relative-root inspection, exact-HEAD replay, source isolation, and cleanup gate |
| `packages/core/dvc/tests/fixtures/fake-dvc.mjs` | Deterministic local DVC CLI fixture for status, pull, cache, repro, and failed-stage tests |
| `packages/core/pkb/tsdown.config.ts` | Host-only PKB service artifact configuration |
| `packages/core/pkb/src/conversation.ts` | Workspace-authorized DSH Session search, exact event expansion, and bounded conversation evidence capture |
| `packages/core/pkb/src/errors.ts` | Stable PKB error taxonomy |
| `packages/core/pkb/src/index.ts` | Public PKB package and default Cordis plugin entry |
| `packages/core/pkb/src/markdown.ts` | Bounded OKF frontmatter, portable Markdown, provenance, and revision contract |
| `packages/core/pkb/src/plugin.ts` | Runtime-neutral PKB operation executor plus thin DSH Tool/approval service and per-Agent frozen index carrier |
| `packages/core/pkb/src/vault.ts` | Owner-only Markdown Vault, workspace authorization, idempotent admission creates, revisions, indexes, logs, and conversation references |
| `packages/core/pkb/src/workspace.ts` | Salted canonical workspace identity without host-path disclosure |
| `packages/core/pkb/tests/conversation.test.ts` | Cross-Session workspace isolation, all-scope authorization, CJK fallback, and exact evidence contract |
| `packages/core/pkb/tests/okf-fixture.test.ts` | Executable OKF, Obsidian, MyST, and conversation-footnote fixture contract |
| `packages/core/pkb/tests/plugin.test.ts` | Aggregate tool approval and frozen prompt-index contract |
| `packages/core/pkb/tests/vault.test.ts` | Vault permissions, isolation, admission idempotency, revision, history, malformed-page, and portability contract |
| `packages/core/swarm/src/capabilities.ts` | Exact-participant role restrictions, member delegation/PKB denial, mutating-Tool classification, and active-write-attempt guard |
| `packages/core/swarm/src/contracts.ts` | Strict bounded role/model/budget, attempt, submission/verdict, monitor, R/W/K task, mailbox, snapshot, and UI schemas |
| `packages/core/swarm/src/coordinator.ts` | Platform-neutral exact-actor authority, heterogeneous routing, attempt economics/budgets, submission/verdict fencing, mailbox delivery, evidence admission, and single-writer scheduling |
| `packages/core/swarm/src/errors.ts` | Stable Swarm error taxonomy |
| `packages/core/swarm/src/index.ts` | Thin `ctx.swarm` DSH carrier, Session-usage reducer, event-driven budget monitor, native continuable adapter, Tool-effect wrapper, revision wait, and disposal |
| `packages/core/swarm/src/journal.ts` | Owner/client SQLite WAL v5 event/attempt/message ledger with atomic mailbox delivery, exact runtime-member claims, replayable archive fences, deterministic owner projection rebuild, serialized migration, salted workspace identity, and unsafe-state recovery |
| `packages/core/swarm/src/knowledge.ts` | Platform-neutral lead-only owner-preserving Science evidence and approved PKB committer |
| `packages/core/swarm/src/monitor.ts` | Pure deterministic budget, stall, mailbox, lifecycle, submission, verification, and usage finding evaluator |
| `packages/core/swarm/src/privacy.ts` | Bounded absolute-path and credential-shaped text redaction for semantic-monitor and browser projections |
| `packages/core/swarm/src/remote-contract.ts` | Strict read-only Swarm UI snapshot/wait invocation descriptors |
| `packages/core/swarm/src/remote.ts` | Client Swarm Remote contribution and namespace typing |
| `packages/core/swarm/src/routing.ts` | Platform-neutral durable per-member provider/model/max-token policy application |
| `packages/core/swarm/src/tools.ts` | Runtime-neutral aggregate operation invocation plus the preset-scoped DSH `swarm` Tool and Team-mode prompt carrier |
| `packages/core/swarm/src/team-policy.ts` | Deterministic read/write DAG width, write-conflict, and Tool-density Team preflight |
| `packages/core/swarm/src/typert.ts` | Host Swarm Typert contribution |
| `packages/core/swarm/src/verification-model.ts` | Bounded executable state exploration and enforced-versus-prompt-only fault benchmark |
| `packages/core/swarm/tsdown.config.ts` | Deterministic Host package artifact configuration |
| `packages/core/swarm/tests/capabilities.test.ts` | Member delegation, PKB, exact authority, and write-attempt guard contract |
| `packages/core/swarm/tests/build.test.ts` | Published entry and shared runtime chunk allowlist contract |
| `packages/core/swarm/tests/coordinator.test.ts` | Exact identity, R/W/K attempts, Tool effects, evidence admission, mailbox idempotency, reassignment, and archival contract |
| `packages/core/swarm/tests/contracts.test.ts` | Strict role/model/budget, acceptance/submission/verdict, legacy-default, and UI privacy schema contract |
| `packages/core/swarm/tests/journal.test.ts` | Atomic replay/mailbox claims, v1–v5 migration, owner/client permissions, workspace identity, uncertain intent recovery, and fenced archive contract |
| `packages/core/swarm/tests/knowledge.test.ts` | Science/PKB owner boundary, typed source, idempotency-key, and PKB approval contract |
| `packages/core/swarm/tests/monitor.test.ts` | Deterministic bounded budget, stall, mailbox, lifecycle, and submission finding/dedupe contract |
| `packages/core/swarm/tests/preset.test.ts` | Exact Science composition plus Swarm-only preset row and metadata contract |
| `packages/core/swarm/tests/remote-contract.test.ts` | Strict read-only Remote method and cancellation surface contract |
| `packages/core/swarm/tests/routing.test.ts` | Distinct member routes, durable cold-resume reapplication, and legacy deployment-default contract |
| `packages/core/swarm/tests/service.test.ts` | Strict path-free Swarm service UI projection regression |
| `packages/core/swarm/tests/tools.test.ts` | Bounded aggregate Tool actions and exact Agent carrier contract |
| `packages/core/swarm/tests/team-policy.test.ts` | Deterministic Team sizing, serialization-pressure, and DAG rejection contract |
| `packages/core/swarm/tests/verification-model.test.ts` | Exhaustive bounded properties, fault metrics, and prompt-only control regression |
| `packages/client/ui-swarm/src/css.d.ts` | Swarm activity CSS Module type boundary |
| `packages/client/ui-swarm/src/index.ts` | Stable Host marker for the browser-only Swarm activity contribution |
| `packages/client/ui-swarm/src/client/activity-store.ts` | One shared cancellable revision wait loop per rendered Session |
| `packages/client/ui-swarm/src/client/index.ts` | Strict Remote mount, per-Session header action, and keyed Side View registration |
| `packages/client/ui-swarm/src/client/swarm-locales.ts` | Complete English/Chinese role, budget, usage, task, verdict, and monitor presentation contract |
| `packages/client/ui-swarm/src/client/swarm-view.tsx` | Safe Team/member/task activity projection and Side View rendering |
| `packages/client/ui-swarm/tsdown.config.ts` | DSH-compatible Host marker and single-file browser bundle configuration |
| `packages/client/ui-swarm/tests/activity-store.test.ts` | Shared single-poll and last-subscriber cancellation contract |
| `packages/client/ui-swarm/tests/swarm-locales.test.ts` | English/Chinese Swarm locale key parity and non-empty value contract |
| `packages/client/ui-swarm/tests/swarm-view.test.tsx` | Serializable Side View entry and de-identified activity rendering contract |
| `packages/client/tsdown.client.ts` | DSH-compatible client-plugin build preset and module-table external policy |
| `packages/client/ui-dvc/tsdown.config.ts` | DVC UI host/client artifacts and bundled strict Remote entries |
| `packages/client/ui-dvc/src/contracts.ts` | Bounded path-free DVC inspection and typed unavailable-state schemas |
| `packages/client/ui-dvc/src/index.ts` | Session-authorized, read-only `ctx.dvcUi` adapter over Host-only `ctx.dvc.inspect` |
| `packages/client/ui-dvc/src/remote-contract.ts` | Shared strict DVC snapshot invocation descriptor |
| `packages/client/ui-dvc/src/remote.ts` | Client DVC Remote contribution and namespace typing |
| `packages/client/ui-dvc/src/typert.ts` | Host DVC UI Typert contribution |
| `packages/client/ui-dvc/src/client/index.ts` | DVC Remote mount consumed by the shared Version Control presentation |
| `packages/client/ui-dvc/tests/client-build.test.ts` | Shared single-file DVC client-bundle contract |
| `packages/client/ui-dvc/tests/client.test.tsx` | DVC client Remote-only registration contract |
| `packages/client/ui-dvc/tests/dvc-snapshot.test.ts` | Session delegation, path privacy, typed optional-state, and cancellation contract |
| `packages/client/ui-dvc/tests/typert.test.ts` | Host/Client DVC descriptor parity, strict path boundary, and read-only method surface |
| `packages/client/ui-git/tsdown.config.ts` | Git UI host/client artifacts and bundled strict Remote entries |
| `packages/client/ui-git/src/contracts.ts` | Bounded repository-relative Git status and typed unavailable-state schemas |
| `packages/client/ui-git/src/index.ts` | Session-authorized, read-only `ctx.gitUi` Host service over porcelain-v2 Git status |
| `packages/client/ui-git/src/remote-contract.ts` | Shared strict Git snapshot invocation descriptor |
| `packages/client/ui-git/src/remote.ts` | Client Git Remote contribution and namespace typing |
| `packages/client/ui-git/src/typert.ts` | Host Git Typert contribution |
| `packages/client/ui-git/src/css.d.ts` | Git UI CSS Module type boundary |
| `packages/client/ui-git/src/client/index.ts` | Git Remote mount plus one per-Session Version Control action/view combining independent Git and optional DVC loads |
| `packages/client/ui-git/src/client/version-control-view.tsx` | Read-only default-open Git Changes and detected DVC accordion presentation with shared refresh/error states |
| `packages/client/ui-git/src/client/version-control-view.module.css` | Compact Source Control-inspired disclosure, status-row, and header-action styling |
| `packages/client/ui-git/tests/client-build.test.ts` | Shared single-file client-bundle contract |
| `packages/client/ui-git/tests/client.test.tsx` | Single Version Control registration, default-open Git/DVC disclosures, DVC hiding, and read-only rendering contract |
| `packages/client/ui-git/tests/git-snapshot.test.ts` | Real Git repository status, path privacy, entry cap, lazy CLI, Session, and cancellation contract |
| `packages/client/ui-git/tests/typert.test.ts` | Host/Client Git descriptor parity and read-only method surface |
| `packages/client/ui-conversation/tsdown.config.ts` | Conversation extensions host/client artifact configuration |
| `packages/client/ui-conversation/src/annotation-reference.ts` | Stable reusable annotation-reference module entry for client plugins |
| `packages/client/ui-conversation/src/index.ts` | Conversation extensions plugin host half |
| `packages/client/ui-conversation/src/css.d.ts` | CSS Module type boundary |
| `packages/client/ui-conversation/src/fork-boundary.ts` | Safe rerun preparation strategy resolution |
| `packages/client/ui-conversation/src/rerun.ts` | Fork, open, and prompt orchestration |
| `packages/client/ui-conversation/src/turn-origin.ts` | User text lookup within a conversation turn |
| `packages/client/ui-conversation/src/error-turn.ts` | Terminal failure selection for the turn-tail row |
| `packages/client/ui-conversation/src/user-edit-node.ts` | Derived Edit node for user-authored messages |
| `packages/client/ui-conversation/src/client/actions.tsx` | User Edit and failed-turn Retry icon controls |
| `packages/client/ui-conversation/src/client/annotation-composer.module.css` | Composer-aligned annotation tray, selection action, optional-note editor, and hover/focus actions |
| `packages/client/ui-conversation/src/client/annotation-composer.tsx` | Message-selection annotation controller plus ordered tray edit/remove interaction |
| `packages/client/ui-conversation/src/client/annotation-locales.ts` | Complete English and Chinese product annotation namespace |
| `packages/client/ui-conversation/src/client/annotation-projection.ts` | Hidden `dsh-annotation` transport, legacy persisted projection, localized card metadata, and readable Copy text |
| `packages/client/ui-conversation/src/client/annotation-reference.ts` | Generic annotation codec and detached per-session composer occurrence operations |
| `packages/client/ui-conversation/src/client/annotation-user-message.tsx` | Safe-Markdown user/steering renderer with bounded annotation cards |
| `packages/client/ui-conversation/src/client/controller.ts` | Per-session Retry/Edit orchestration plus alpha.2 addressable first-turn sibling creation with source Workspace/cwd and agent-preset inheritance |
| `packages/client/ui-conversation/src/client/index.ts` | Retry/Edit, generic Side View, and selected peer-runtime Conversation registration |
| `packages/client/ui-conversation/src/client/runtime-client.ts` | Same-origin list/read/revision/mutation/SSE client with completed-authoritative projection and typed approval form submission |
| `packages/client/ui-conversation/src/client/runtime-conversation.tsx` | Runtime-neutral Conversation slot occupant with typed elicitation controls, submit-time Edit state, and an explicit metadata-failure surface, used only when the selected default is not native DSH |
| `packages/client/ui-conversation/src/client/runtime-conversation.module.css` | DSH-frame-native layout for peer conversation list, transcript, edit composer, controls, and approvals |
| `packages/client/ui-conversation/src/client/icons.ts` | Single semantic Edit/Retry icon mapping |
| `packages/client/ui-conversation/src/client/message-selection.ts` | Same-message selection validation, durable source addressing, popover positioning, and note keyboard policy |
| `packages/client/ui-conversation/src/client/slots.ts` | Injected action contract |
| `packages/client/ui-conversation/src/client/side-view.ts` | Serializable per-Session Side View service contract and deterministic tab state |
| `packages/client/ui-conversation/src/client/side-view-panel.tsx` | Generic right-column tab shell and keyed content dispatch |
| `packages/client/ui-conversation/src/client/side-view-registration.ts` | Side View service, details geometry, and additive turn-tail item seat |
| `packages/client/ui-conversation/src/client/turn-tail-items.tsx` | Pass-through renderer for explicitly registered completed-turn contributions |
| `packages/client/ui-conversation/tests/client-build.test.ts` | Client build face, entry, external, and loader wrapper contract |
| `packages/client/ui-conversation/tests/annotation-composer.test.tsx` | Ordered tray content plus hover/focus edit/remove affordance contract |
| `packages/client/ui-conversation/tests/annotation-locales.test.ts` | English/Chinese annotation namespace parity contract |
| `packages/client/ui-conversation/tests/annotation-projection.test.ts` | `dsh-annotation` suppression, Markdown preservation, readable Copy, and legacy recovery contract |
| `packages/client/ui-conversation/tests/annotation-reference.test.ts` | Detached codec, zero-draft-geometry insertion, source/destination isolation, edit, and removal contract |
| `packages/client/ui-conversation/tests/controller.test.ts` | Retry/Edit controller plus first-turn creation during an in-flight list refresh and ungrouped cwd adoption contract |
| `packages/client/ui-conversation/tests/error-turn.test.ts` | Terminal failure selection behavior |
| `packages/client/ui-conversation/tests/detached-reference-patch.test.ts` | Exact-baseline detached-reference runtime seam contract |
| `packages/client/ui-conversation/tests/fork-boundary.test.ts` | Boundary behavior |
| `packages/client/ui-conversation/tests/history-projection.test.ts` | Append-only branch and model-projection isolation |
| `packages/client/ui-conversation/tests/message-selection.test.ts` | Selection eligibility, cross-session-ready source identity, positioning, and keyboard behavior |
| `packages/client/ui-conversation/tests/rerun.test.ts` | Re-run orchestration behavior |
| `packages/client/ui-conversation/tests/runtime-client.test.ts` | Runtime-qualified HTTP revision/operations, error handling, and SSE subscription contract |
| `packages/client/ui-conversation/tests/runtime-registration.test.ts` | Native DSH preservation and Codex-only Conversation slot shadowing contract |
| `packages/client/ui-conversation/tests/turn-origin.test.ts` | Turn lookup behavior |
| `packages/client/ui-conversation/tests/user-edit-node.test.ts` | User-message Edit node behavior |
| `packages/client/ui-conversation/tests/user-edit-layout.test.ts` | User-message clock, Copy, and Edit layout contract |
| `packages/client/ui-conversation/tests/conversation-icons.test.ts` | Conversation icon mapping contract |
| `packages/client/ui-conversation/tests/side-view.test.ts` | Side View tabs, Session isolation, serialization, dismissal, and HMR disposal contract |
| `packages/client/ui-conversation/tests/side-view-registration.test.ts` | Published details/slot ownership and service lifecycle contract |
| `packages/client/ui-conversation/tests/side-view-panel.test.ts` | Wrapping Arrow/Home/End tab keyboard navigation contract |
| `packages/client/ui-conversation/tests/turn-tail-items.test.tsx` | Completed-turn pass-through and no-derived-Tool-action contract |
| `packages/client/ui-science/tsdown.config.ts` | Science host/client artifact configuration and bundled Remote entry |
| `packages/client/ui-science/src/index.ts` | Science plugin Host half and preset-independent Markdown deliverable-link prompt |
| `packages/client/ui-science/src/css.d.ts` | Science CSS Module type boundary |
| `packages/client/ui-science/src/client/index.ts` | Strict Remote mount, Chat turn-tail artifact cards, shared annotation insertion, and keyed artifact DetailsPanel registration |
| `packages/client/ui-science/src/client/annotation-reference.ts` | Science image/paper target conversion into the shared annotation contract |
| `packages/client/ui-science/src/client/science-artifact-side-view.tsx` | Claude-style artifact DetailsPanel, on-demand RO-Crate provenance projection, bounded previews, image point annotations, and native fullscreen |
| `packages/client/ui-science/src/client/science-conversation-artifacts.tsx` | Ordered same-Turn Science artifact recovery plus generated thumbnail/card group in Chat |
| `packages/client/ui-science/src/client/science-deliverables.tsx` | Completed-turn file evidence recovery, safe Typst reference parsing, shared Markdown/tail opener, and Typst workbench routing |
| `packages/client/ui-science/src/client/science-pdf-geometry.ts` | PDF page, annotation, and inverse-search click geometry normalization |
| `packages/client/ui-science/src/client/science-pdf-viewer.tsx` | Bounded PDF rendering, page navigation, selection/figure annotations, and text-click inverse search |
| `packages/client/ui-science/src/client/science-table-grid.tsx` | AG Grid Community typed tabular artifact renderer with sorting, filtering, resizing, and bounded pagination |
| `packages/client/ui-science/src/client/science-tool-artifact.tsx` | Strict same-Session Science Tool artifact locator parser for generated cards |
| `packages/client/ui-science/src/client/science-typst-side-view.tsx` | Typst source/live PDF split view, imported-source switching, exact caret reveal, and PDF figure workbench tabs |
| `packages/client/ui-science/tests/client-registration.test.ts` | Remote and slot registration lifecycle contract |
| `packages/client/ui-science/tests/client-build.test.ts` | Shared client-bundle configuration and Jupyter adapter exclusion contract |
| `packages/client/ui-science/tests/annotation-reference.test.ts` | Unified image/paper comment conversion, hidden model serialization, and draft-preserving insertion contract |
| `packages/client/ui-science/tests/science-artifact-side-view.test.tsx` | Bounded artifact locator, on-demand path-free RO-Crate provenance projection, and image point normalization contract |
| `packages/client/ui-science/tests/science-conversation-artifacts.test.tsx` | Same-Turn artifact ordering/deduplication and complete clickable Chat card contract |
| `packages/client/ui-science/tests/science-deliverables.test.ts` | Paper deliverables state and action contract |
| `packages/client/ui-science/tests/science-pdf-viewer.test.ts` | PDF viewer geometry, navigation, and annotation contract |
| `packages/client/ui-science/tests/science-typst-side-view.test.tsx` | Per-paper workbench identity and cross-tab state-isolation contract |
| `packages/client/ui-science/tests/science-table-grid.test.ts` | Science scalar column to AG Grid data-type mapping regression contract |
| `packages/client/ui-science/tests/science-tool-artifact.test.ts` | Strict Science Tool artifact locator acceptance and rejection contract |
| `packages/science/core/demo/README.md` | Runnable local-only end-to-end Science IDE demo guide |
| `packages/science/core/config/agent-presets/dsh-science/agent.cordis.yml` | Complete locked DSH standard composition plus preset-scoped Science tool and contract rows |
| `packages/science/core/config/agent-presets/dsh-science/preset.yml` | Read-only Science mode display metadata and roster order |
| `packages/science/core/tsdown.config.ts` | Science service, contracts, Typert, tools, and preset-contract artifact build configuration |
| `packages/science/core/src/artifact-store.ts` | Owner-only streamed/uploaded/generated SHA256 capture, stable source fingerprinting, verified bounded readback, disposable Notebook input materialization, and deduplication |
| `packages/science/core/src/bibliography.ts` | Strict bounded BibTeX parser/serializer and path-bearing private-field removal for the provider-neutral literature exchange boundary |
| `packages/science/core/src/contracts.ts` | Client-safe operational requests/results plus RO-Crate 1.3 entity, metadata-document, export, and boundary schemas |
| `packages/science/core/src/demo.ts` | Executable public-service tour through the complete first Science product loop |
| `packages/science/core/src/errors.ts` | Stable Science service error codes |
| `packages/science/core/src/figure.ts` | Figure code hashing, semantic object inference, and accepted-patch range remapping |
| `packages/science/core/src/core.ts` | Platform-neutral workspace-scoped Science authority over journal, artifacts, notebooks, literature, writing, and resources |
| `packages/science/core/src/index.ts` | Thin DSH Cordis/Typert service carrier over `ScienceCore` and default live-Session workspace resolution |
| `packages/science/core/src/journal.ts` | WAL SQLite journal, v1–v5 migrations, replay, materialized operational views, and export receipts |
| `packages/science/core/src/jupymcp-runtime.ts` | Official-SDK MCP stdio controller pool with one persistent Jupyter kernel and bounded canonical Notebook per workspace Notebook |
| `packages/science/core/src/literature.ts` | Loopback-only Zotero v3 candidate retrieval, Zotero-to-BibTeX normalization, owner-only snapshot, deterministic local ranking, and bounded citation results |
| `packages/science/core/src/artifact-metadata.ts` | Format routing plus deterministic PNG `iTXt`, SVG `metadata`, and PDF XMP reproducibility encoding/extraction |
| `packages/science/core/src/python-runtime.ts` | Managed stateless Python execution over the structural Science process seam |
| `packages/science/core/src/preset.ts` | Preset-scoped Science annotation/literature/Typst prompt sections and direct Typst Bash guard |
| `packages/science/core/src/tabular-preview.ts` | Papa Parse CSV/TSV and bounded scalar-record JSON adaptation for typed table previews |
| `packages/science/core/src/writing-preview-runtime.ts` | Engine-identified strict NDJSON controller for the bundled semantic writing process and revision-bound point lookup |
| `packages/science/core/src/typst-preview.ts` | Workspace-authorized semantic Typst watcher with bounded PDF, diagnostics, source writes, and inverse search |
| `packages/science/core/src/remote-contract.ts` | Shared strict invocation descriptors |
| `packages/science/core/src/remote.ts` | Client Remote contribution and namespace typing |
| `packages/science/core/src/research-object.ts` | Deterministic project-scoped RO-Crate 1.3 projection with Schema.org entities and Action provenance |
| `packages/science/core/src/resource-id.ts` | Canonical typed local `sx:` logical/exact ID formatter and strict parser |
| `packages/science/core/src/resource-resolver.ts` | Pure workspace-snapshot typed resource index, revision assertion, digest mapping, and Host-only entity resolution |
| `packages/science/core/src/resource-view.ts` | Strict bounded resource heads, metadata projections, verified Artifact preview selection, and explicit relation neighbors |
| `packages/science/core/src/subprocess.ts` | Runtime-neutral managed-process structural contract and credential-scrubbed child environment |
| `packages/science/core/src/typert.ts` | Host Typert contribution |
| `packages/science/core/src/tools.ts` | Seven aggregate operation schemas plus direct literature search, with runtime-neutral invocation and DSH Tool rendering/registration carriers |
| `packages/science/core/src/writing.ts` | Deterministic source hashing, format routing, structural checks, and scientific diagnostics |
| `packages/science/core/tests/artifact-registry.test.ts` | Artifact security, deduplication, cancellation, migration, and replay contract |
| `packages/science/core/tests/build.test.ts` | Deterministic shared-chunk and publish allowlist regression contract |
| `packages/science/core/tests/demo.test.ts` | Runnable Notebook-to-export Science IDE demo contract |
| `packages/science/core/tests/experiment-export.test.ts` | Operational research facts, Experiment/Run lifecycle, v5 replay, comparison, and RO-Crate export contract |
| `packages/science/core/tests/fixture.ts` | Shared two-workspace Science service lifecycle fixture |
| `packages/science/core/tests/figure-studio.test.ts` | Figure libraries, semantic selection, artifact linkage, revisions, migration, RO-Crate Action projection, and replay contract |
| `packages/science/core/tests/jupymcp-runtime.test.ts` | JupyMCP controller isolation, MIME normalization, resource bounds, cancellation, input cleanup, and disposal contract |
| `packages/science/core/tests/notebook-execution.test.ts` | Python execution, materialized artifact input, output bounds, idempotency, cancellation, disposal, RO-Crate Action projection, and replay contract |
| `packages/science/core/tests/preset-contract.test.ts` | Science-only prompt sections and managed Typst guard contract |
| `packages/science/core/tests/preset.test.ts` | Standard composition parity, preset metadata, Host boundary, and published asset contract |
| `packages/science/core/tests/artifact-metadata.test.ts` | PNG/SVG/PDF replacement, round-trip, malformed input, source normalization, and opt-out contract |
| `packages/science/core/tests/bibliography.test.ts` | BibTeX nested-value round-trip, private-path removal, malformed input, and duplicate-key contract |
| `packages/science/core/tests/literature.test.ts` | Local Zotero boundary, Bib snapshot, deterministic result projection, cancellation, and unavailable-source contract |
| `packages/science/core/tests/provenance.test.ts` | Workspace-local RO-Crate lineage, Action, privacy, and isolation contract |
| `packages/science/core/tests/research-object.test.ts` | RO-Crate 1.3 structure, Schema.org mapping, Action provenance, privacy, and boundary rejection contract |
| `packages/science/core/tests/resource-id.test.ts` | All-kind canonical format/parse, encoding, revision, and malformed-address contract |
| `packages/science/core/tests/resource-resolver.test.ts` | Workspace isolation, typed lookup, digest/revision, collision, and privacy contract |
| `packages/science/core/tests/resource-view.test.ts` | Bounded head/metadata/select/batch/neighbors projection contract |
| `packages/science/core/tests/science-service.test.ts` | Journal, replay, isolation, cancellation, and disposal contract |
| `packages/science/core/tests/science-tools.test.ts` | Science aggregate/direct tools, strict input, cancellation, locator, literature separation, and disposal contract |
| `packages/science/core/tests/typert.test.ts` | Strict Host/Client Typert descriptor parity |
| `packages/science/core/tests/typst-integration.test.ts` | Real Typst CLI plus bundled semantic compile/inverse-search integration contract |
| `packages/science/core/tests/typst-preview.test.ts` | Typst preview lifecycle, bounds, revisions, updates, and cleanup contract |
| `packages/science/core/tests/writing-studio.test.ts` | Document validation, diagnostics, patch revision, cancellation, migration, and replay contract |
| `patches/@deepseek-ai__dsh-client-ui-agent-preset@0.1.2-alpha.2.patch` | Exact-baseline English/Chinese locale mapping for the system-owned `dsh-science` preset while retaining metadata fallback for unknown and user presets |
| `patches/@deepseek-ai__dsh-api-session-controller@0.1.2-alpha.2.patch` | Exact-baseline agent-preset support through alpha.2's synchronously addressable client session creation object layer |
| `patches/@deepseek-ai__dsh-app-boot@0.1.2-alpha.2.patch` | Exact-baseline shared module-fallback reconciliation that removes obsolete DSH-owned packages from previously populated homes |
| `patches/@deepseek-ai__dsh-client-ui-conversation@0.1.2-alpha.2.patch` | Exact-baseline detached reference occurrences outside Lexical draft geometry, annotation-only submission, and occurrence edit/remove operations |
| `patches/@deepseek-ai__dsh-client-ui-layout@0.1.2-alpha.2.patch` | Exact-baseline layout patch removing the fixed 520px details ceiling while retaining the 300px floor, preferred open width, and center-width concession |
| `scripts/build-writing-preview-runtime.ts` | Builds and stages the current-platform semantic writing preview executable for the Science package |
| `scripts/clean.ts` | Removes only known desktop, client-plugin, and staged native build outputs |
| `scripts/check-codebase-docs.mjs` | Checks every authored source/test path is mapped here |
| `scripts/workspace-layout.test.ts` | Grouped workspace, package-manager, development entry, root build-tool ownership, exact alpha.2 patch set, preset locale, and viewport-bound details solver contract |

Generated `dist/`, `lib/`, native `target/`, and staged `packages/science/core/bin/` artifacts are not source. DSH does not publish its client tsdown preset; the local minimal preset is release-coupled to the DSH module table and guarded by `client-build.test.ts`.

`docs/ro-crate.md` defines the RO-Crate 1.3 Research Object boundary, Science projection mapping,
extension policy, and legacy-export rule.

`docs/resource-addressing.md` defines canonical local Science IDs, revision assertions, bounded
resource views, workspace authorization, and the external-identifier extension boundary.

`docs/runtime-platform.md` records the audited DSH Web seam, shared conversation contract,
DSH/Codex adapter boundary, native transcript ownership, protocol/approval rules, explicit capability
differences, MCP carrier, workspace scope, and one-time product-state migration.

`docs/reproducibility-metadata.md` defines the portable Figure PNG/SVG/PDF metadata schema,
Python/R generation flow, source locators, security boundary, and opt-out semantics.

`docs/pkb.md` defines the owner-only OKF v0.2 Vault, Obsidian/MyST compatibility profile,
conversation evidence boundary, approval policy, and progressive-disclosure prompt behavior.

`docs/version-control.md` defines the read-only human Git/DVC UIs, Host-only DVC command capability,
and package-private Git isolation boundary used for clean-HEAD DVC reproduction.

`docs/swarm.md` defines exact Agent/role authority, heterogeneous continuable routing, attempt
economics and budgets, independent submission/verdict flow, event-driven monitoring,
single-writer coordination, durable recovery, archive behavior, and the read-only UI boundary.
