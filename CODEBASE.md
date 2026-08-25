# SwarmX codebase map

SwarmX is a thin Electron host around the published DeepSeek Harness Web profile, plus product-owned conversation, local-first science, and private PKB extensions composed after the published bundles and before user patches.

## Runtime flow

`apps/desktop/src/main.ts` boots `apps/desktop/src/harness.ts`, then loads its loopback URL through the security boundary in `apps/desktop/src/window.ts`.

The Harness supplies the complete browser UI and `/api` transport. SwarmX has no renderer, preload, alternate session store, or model client.

## Authored source and tests

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/harness.ts` | Profile creation, profile-installed bundle resolution, DSH + Science system-preset roots, patch composition, in-process Harness boot, loopback URL |
| `apps/desktop/src/markdown-file-links.ts` | Exact rc.2 frontend asset route that adds trusted Markdown file-link resolution and rejects upstream seam drift |
| `apps/desktop/src/main.ts` | Electron startup, window recreation, Harness shutdown |
| `apps/desktop/src/window.ts` | BrowserWindow construction, navigation and permission boundary |
| `apps/desktop/tests/harness.test.ts` | Real profile boot, profile-installed bundle resolution, Science preset discovery/scoping, browser-handoff and HTTP integration |
| `apps/desktop/tests/markdown-file-links.test.ts` | Markdown seam transformation, native-link fallback, and upstream drift rejection contract |
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
| `packages/core/pkb/src/plugin.ts` | `ctx.pkb`, aggregate model tool, approval gates, and per-Agent frozen index context |
| `packages/core/pkb/src/vault.ts` | Owner-only Markdown Vault, workspace authorization, revisions, indexes, logs, and conversation references |
| `packages/core/pkb/src/workspace.ts` | Salted canonical workspace identity without host-path disclosure |
| `packages/core/pkb/tests/conversation.test.ts` | Cross-Session workspace isolation, all-scope authorization, CJK fallback, and exact evidence contract |
| `packages/core/pkb/tests/okf-fixture.test.ts` | Executable OKF, Obsidian, MyST, and conversation-footnote fixture contract |
| `packages/core/pkb/tests/plugin.test.ts` | Aggregate tool approval and frozen prompt-index contract |
| `packages/core/pkb/tests/vault.test.ts` | Vault permissions, isolation, revision, history, malformed-page, and portability contract |
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
| `packages/client/ui-conversation/src/client/controller.ts` | Per-session Retry/Edit orchestration |
| `packages/client/ui-conversation/src/client/index.ts` | Retry/Edit and generic Side View client registration |
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
| `packages/client/ui-conversation/tests/controller.test.ts` | Controller behavior |
| `packages/client/ui-conversation/tests/error-turn.test.ts` | Terminal failure selection behavior |
| `packages/client/ui-conversation/tests/detached-reference-patch.test.ts` | Exact-baseline detached-reference runtime seam contract |
| `packages/client/ui-conversation/tests/fork-boundary.test.ts` | Boundary behavior |
| `packages/client/ui-conversation/tests/history-projection.test.ts` | Append-only branch and model-projection isolation |
| `packages/client/ui-conversation/tests/message-selection.test.ts` | Selection eligibility, cross-session-ready source identity, positioning, and keyboard behavior |
| `packages/client/ui-conversation/tests/rerun.test.ts` | Re-run orchestration behavior |
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
| `packages/science/core/src/index.ts` | Workspace-scoped `ctx.science` service and Remote methods |
| `packages/science/core/src/journal.ts` | WAL SQLite journal, v1–v5 migrations, replay, materialized operational views, and export receipts |
| `packages/science/core/src/jupymcp-runtime.ts` | Official-SDK MCP stdio controller pool with one persistent Jupyter kernel and bounded canonical Notebook per workspace Notebook |
| `packages/science/core/src/literature.ts` | Loopback-only Zotero v3 candidate retrieval, Zotero-to-BibTeX normalization, owner-only snapshot, deterministic local ranking, and bounded citation results |
| `packages/science/core/src/artifact-metadata.ts` | Format routing plus deterministic PNG `iTXt`, SVG `metadata`, and PDF XMP reproducibility encoding/extraction |
| `packages/science/core/src/python-runtime.ts` | Managed stateless Python execution over the DSH subprocess seam |
| `packages/science/core/src/preset.ts` | Preset-scoped Science annotation/literature/Typst prompt sections and direct Typst Bash guard |
| `packages/science/core/src/tabular-preview.ts` | Papa Parse CSV/TSV and bounded scalar-record JSON adaptation for typed table previews |
| `packages/science/core/src/writing-preview-runtime.ts` | Engine-identified strict NDJSON controller for the bundled semantic writing process and revision-bound point lookup |
| `packages/science/core/src/typst-preview.ts` | Workspace-authorized semantic Typst watcher with bounded PDF, diagnostics, source writes, and inverse search |
| `packages/science/core/src/remote-contract.ts` | Shared strict invocation descriptors |
| `packages/science/core/src/remote.ts` | Client Remote contribution and namespace typing |
| `packages/science/core/src/research-object.ts` | Deterministic project-scoped RO-Crate 1.3 projection with Schema.org entities and Action provenance |
| `packages/science/core/src/typert.ts` | Host Typert contribution |
| `packages/science/core/src/tools.ts` | Seven strict aggregate Science tools plus direct local literature search, preset-scoped registration, durable locators, and verified annotation image projection |
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
| `packages/science/core/tests/science-service.test.ts` | Journal, replay, isolation, cancellation, and disposal contract |
| `packages/science/core/tests/science-tools.test.ts` | Science aggregate/direct tools, strict input, cancellation, locator, literature separation, and disposal contract |
| `packages/science/core/tests/typert.test.ts` | Strict Host/Client Typert descriptor parity |
| `packages/science/core/tests/typst-integration.test.ts` | Real Typst CLI plus bundled semantic compile/inverse-search integration contract |
| `packages/science/core/tests/typst-preview.test.ts` | Typst preview lifecycle, bounds, revisions, updates, and cleanup contract |
| `packages/science/core/tests/writing-studio.test.ts` | Document validation, diagnostics, patch revision, cancellation, migration, and replay contract |
| `patches/@deepseek-ai__dsh-client-ui-agent-preset@0.1.1-rc.2.patch` | Exact-baseline English/Chinese locale mapping for the system-owned `dsh-science` preset while retaining metadata fallback for unknown and user presets |
| `patches/@deepseek-ai__dsh-client-ui-conversation@0.1.1-rc.2.patch` | Exact-baseline detached reference occurrences, annotation-only submission, and occurrence edit/remove operations |
| `patches/@deepseek-ai__dsh-client-ui-layout@0.1.1-rc.2.patch` | Exact-baseline layout patch removing the fixed 520px details ceiling while retaining the 300px floor, preferred open width, and center-width concession |
| `patches/@deepseek-ai__dsh-client-ui-primitives@0.1.1-rc.2.patch` | Exact-baseline optional Markdown destination resolver for verified produced-file buttons; unresolved relative and unsafe links remain inert |
| `scripts/build-writing-preview-runtime.ts` | Builds and stages the current-platform semantic writing preview executable for the Science package |
| `scripts/clean.ts` | Removes only known desktop, client-plugin, and staged native build outputs |
| `scripts/check-codebase-docs.mjs` | Checks every authored source/test path is mapped here |
| `scripts/workspace-layout.test.ts` | Grouped workspace, package-manager, development entry, root build-tool ownership, patched preset locale and safe Markdown seams, and viewport-bound details solver contract |

Generated `dist/`, `lib/`, native `target/`, and staged `packages/science/core/bin/` artifacts are not source. DSH does not publish its client tsdown preset; the local minimal preset is release-coupled to the DSH module table and guarded by `client-build.test.ts`.

`docs/ro-crate.md` defines the RO-Crate 1.3 Research Object boundary, Science projection mapping,
extension policy, and legacy-export rule.

`docs/reproducibility-metadata.md` defines the portable Figure PNG/SVG/PDF metadata schema,
Python/R generation flow, source locators, security boundary, and opt-out semantics.

`docs/pkb.md` defines the owner-only OKF v0.2 Vault, Obsidian/MyST compatibility profile,
conversation evidence boundary, approval policy, and progressive-disclosure prompt behavior.

`docs/version-control.md` defines the read-only human Git/DVC UIs, Host-only DVC command capability,
and package-private Git isolation boundary used for clean-HEAD DVC reproduction.
