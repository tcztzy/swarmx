# SwarmX codebase map

SwarmX is a thin Electron host around the published DeepSeek Harness Web profile, plus product-owned conversation and local-first science extensions composed after the published bundles and before user patches.

## Runtime flow

`apps/desktop/src/main.ts` boots `apps/desktop/src/harness.ts`, then loads its loopback URL through the security boundary in `apps/desktop/src/window.ts`.

The Harness supplies the complete browser UI and `/api` transport. SwarmX has no renderer, preload, alternate session store, or model client.

## Authored source and tests

| Path | Ownership |
| --- | --- |
| `apps/desktop/src/harness.ts` | Profile creation, profile-installed bundle resolution, patch composition, in-process Harness boot, loopback URL |
| `apps/desktop/src/main.ts` | Electron startup, window recreation, Harness shutdown |
| `apps/desktop/src/window.ts` | BrowserWindow construction, navigation and permission boundary |
| `apps/desktop/tests/harness.test.ts` | Real profile boot, profile-installed bundle resolution, browser-handoff and HTTP integration |
| `apps/desktop/tests/window.test.ts` | Window security regression tests |
| `packages/client/tsdown.client.ts` | DSH-compatible client-plugin build preset and module-table external policy |
| `packages/client/ui-conversation/tsdown.config.ts` | Conversation extensions host/client artifact configuration |
| `packages/client/ui-conversation/src/index.ts` | Conversation extensions plugin host half |
| `packages/client/ui-conversation/src/css.d.ts` | CSS Module type boundary |
| `packages/client/ui-conversation/src/fork-boundary.ts` | Safe rerun preparation strategy resolution |
| `packages/client/ui-conversation/src/rerun.ts` | Fork, open, and prompt orchestration |
| `packages/client/ui-conversation/src/turn-origin.ts` | User text lookup within a conversation turn |
| `packages/client/ui-conversation/src/error-turn.ts` | Terminal failure selection for the turn-tail row |
| `packages/client/ui-conversation/src/user-edit-node.ts` | Derived Edit node for user-authored messages |
| `packages/client/ui-conversation/src/client/actions.tsx` | User Edit and failed-turn Retry icon controls |
| `packages/client/ui-conversation/src/client/controller.ts` | Per-session Retry/Edit orchestration |
| `packages/client/ui-conversation/src/client/index.ts` | Retry/Edit and generic Side View client registration |
| `packages/client/ui-conversation/src/client/icons.ts` | Single semantic Edit/Retry icon mapping |
| `packages/client/ui-conversation/src/client/slots.ts` | Injected action contract |
| `packages/client/ui-conversation/src/client/side-view.ts` | Serializable per-Session Side View service contract and deterministic tab state |
| `packages/client/ui-conversation/src/client/side-view-panel.tsx` | Generic right-column tab shell and keyed content dispatch |
| `packages/client/ui-conversation/src/client/side-view-registration.ts` | Side View service, details geometry, Tool route, and turn entry lifecycle |
| `packages/client/ui-conversation/src/client/tool-side-view.tsx` | Public Conversation snapshot Tool locator, existing Tool output bridge, and additive Tool actions seat |
| `packages/client/ui-conversation/tests/client-build.test.ts` | Client build face, entry, external, and loader wrapper contract |
| `packages/client/ui-conversation/tests/controller.test.ts` | Controller behavior |
| `packages/client/ui-conversation/tests/error-turn.test.ts` | Terminal failure selection behavior |
| `packages/client/ui-conversation/tests/fork-boundary.test.ts` | Boundary behavior |
| `packages/client/ui-conversation/tests/history-projection.test.ts` | Append-only branch and model-projection isolation |
| `packages/client/ui-conversation/tests/rerun.test.ts` | Re-run orchestration behavior |
| `packages/client/ui-conversation/tests/turn-origin.test.ts` | Turn lookup behavior |
| `packages/client/ui-conversation/tests/user-edit-node.test.ts` | User-message Edit node behavior |
| `packages/client/ui-conversation/tests/user-edit-layout.test.ts` | User-message clock, Copy, and Edit layout contract |
| `packages/client/ui-conversation/tests/conversation-icons.test.ts` | Conversation icon mapping contract |
| `packages/client/ui-conversation/tests/side-view.test.ts` | Side View tabs, Session isolation, serialization, dismissal, and HMR disposal contract |
| `packages/client/ui-conversation/tests/side-view-registration.test.ts` | Published details/slot ownership and service lifecycle contract |
| `packages/client/ui-conversation/tests/side-view-panel.test.ts` | Wrapping Arrow/Home/End tab keyboard navigation contract |
| `packages/client/ui-conversation/tests/tool-side-view.test.ts` | Root and nested public Tool locator contract |
| `packages/client/ui-science/tsdown.config.ts` | Science host/client artifact configuration and bundled Remote entry |
| `packages/client/ui-science/src/index.ts` | Science Workspace plugin host half |
| `packages/client/ui-science/src/css.d.ts` | Science CSS Module type boundary |
| `packages/client/ui-science/src/client/index.ts` | Strict Remote mount, additive conversation view, and keyed artifact Side View registration |
| `packages/client/ui-science/src/client/notebook-output.tsx` | Untrusted JupyterLab OutputArea/RenderMime adapter with ordered nbformat conversion and plain-text fallback |
| `packages/client/ui-science/src/client/science-artifact-side-view.tsx` | Science artifact locator, bounded preview states, and fullscreen workbench handoff |
| `packages/client/ui-science/src/client/science-navigation.ts` | Deduplicated per-Session Science fullscreen deep-link retention and mount tracking |
| `packages/client/ui-science/src/client/science-table-grid.tsx` | AG Grid Community typed tabular artifact renderer with sorting, filtering, resizing, and bounded pagination |
| `packages/client/ui-science/src/client/science-tool-artifact.tsx` | Strict same-Session Science Tool artifact locator parser and Side View action |
| `packages/client/ui-science/src/client/science-workspace.tsx` | Science Workspace Files import/analysis, Notebook, Writing, Figure, Research Map search, Experiment/Run, and project export UI |
| `packages/client/ui-science/tests/client-registration.test.ts` | Remote and slot registration lifecycle contract |
| `packages/client/ui-science/tests/client-build.test.ts` | Client bundle dependency-SVG support required by the JupyterLab renderer contract |
| `packages/client/ui-science/tests/notebook-output-lifecycle.test.tsx` | Lumino widget layout-cleanup ordering regression contract |
| `packages/client/ui-science/tests/notebook-output.test.tsx` | Ordered Notebook MIME-to-nbformat adaptation and accessible fallback contract |
| `packages/client/ui-science/tests/science-artifact-side-view.test.tsx` | Bounded artifact locator and path-free provenance renderer contract |
| `packages/client/ui-science/tests/science-navigation.test.ts` | Fullscreen target deduplication, Session isolation, mounted fallback, and HMR disposal contract |
| `packages/client/ui-science/tests/science-table-grid.test.ts` | Science scalar column to AG Grid data-type mapping regression contract |
| `packages/client/ui-science/tests/science-tool-artifact.test.ts` | Strict Science Tool artifact locator acceptance and rejection contract |
| `packages/client/ui-science/tests/science-view.test.tsx` | Science Workspace states, all five destinations, selection/search helpers, mutation failure, and accessibility contract |
| `packages/science/core/demo/README.md` | Runnable local-only end-to-end Science IDE demo guide |
| `packages/science/core/tsdown.config.ts` | Science service, contracts, and Typert artifact build configuration |
| `packages/science/core/src/artifact-store.ts` | Owner-only streamed/uploaded/generated SHA256 capture, verified bounded readback, disposable Notebook input materialization, and deduplication |
| `packages/science/core/src/contracts.ts` | Client-safe project, notebook input, browser artifact import/preview, studio, Research Map, Experiment/Run, export, provenance, and request schemas |
| `packages/science/core/src/demo.ts` | Executable public-service tour through the complete first Science product loop |
| `packages/science/core/src/errors.ts` | Stable Science service error codes |
| `packages/science/core/src/figure.ts` | Figure code hashing, semantic object inference, and accepted-patch range remapping |
| `packages/science/core/src/index.ts` | Workspace-scoped `ctx.science` service and Remote methods |
| `packages/science/core/src/journal.ts` | WAL SQLite journal, v1–v5 migrations, replay, materialized views, and bounded provenance trace |
| `packages/science/core/src/jupymcp-runtime.ts` | Official-SDK MCP stdio controller pool with one persistent Jupyter kernel and bounded canonical Notebook per workspace Notebook |
| `packages/science/core/src/python-runtime.ts` | Managed stateless Python execution over the DSH subprocess seam |
| `packages/science/core/src/tabular-preview.ts` | Papa Parse CSV/TSV and bounded scalar-record JSON adaptation for typed table previews |
| `packages/science/core/src/remote-contract.ts` | Shared strict invocation descriptors |
| `packages/science/core/src/remote.ts` | Client Remote contribution and namespace typing |
| `packages/science/core/src/typert.ts` | Host Typert contribution |
| `packages/science/core/src/tools.ts` | Seven strict aggregate Science tools, durable locators, and HMR-safe registration |
| `packages/science/core/src/writing.ts` | Deterministic source hashing, format routing, structural checks, and scientific diagnostics |
| `packages/science/core/tests/artifact-registry.test.ts` | Artifact security, deduplication, cancellation, migration, and replay contract |
| `packages/science/core/tests/build.test.ts` | Deterministic shared-chunk and publish allowlist regression contract |
| `packages/science/core/tests/demo.test.ts` | Runnable Notebook-to-export Science IDE demo contract |
| `packages/science/core/tests/experiment-export.test.ts` | Research Map, Experiment/Run lifecycle, v5 replay, comparison, and project export contract |
| `packages/science/core/tests/fixture.ts` | Shared two-workspace Science service lifecycle fixture |
| `packages/science/core/tests/figure-studio.test.ts` | Figure libraries, semantic selection, artifact linkage, revisions, migration, provenance, and replay contract |
| `packages/science/core/tests/jupymcp-runtime.test.ts` | JupyMCP controller isolation, MIME normalization, resource bounds, cancellation, input cleanup, and disposal contract |
| `packages/science/core/tests/notebook-execution.test.ts` | Python execution, materialized artifact input, output bounds, idempotency, cancellation, disposal, provenance, and replay contract |
| `packages/science/core/tests/provenance.test.ts` | Workspace-local bounded lineage contract |
| `packages/science/core/tests/science-service.test.ts` | Journal, replay, isolation, cancellation, and disposal contract |
| `packages/science/core/tests/science-tools.test.ts` | Seven aggregate tools, strict input, cancellation, locator, and disposal contract |
| `packages/science/core/tests/typert.test.ts` | Strict Host/Client Typert descriptor parity |
| `packages/science/core/tests/writing-studio.test.ts` | Document validation, diagnostics, patch revision, cancellation, migration, and replay contract |
| `patches/@deepseek-ai__dsh-client-ui-layout@0.1.0-rc.8.patch` | Exact rc.8 layout patch removing the fixed 520px details ceiling while retaining the 300px floor and center-width concession |
| `scripts/clean.ts` | Removes only known desktop and client-plugin build outputs |
| `scripts/check-codebase-docs.mjs` | Checks every authored source/test path is mapped here |
| `scripts/workspace-layout.test.ts` | Grouped workspace, package-manager, root build-tool ownership, and patched viewport-bound details solver contract |

Generated `dist/` and `lib/` artifacts are not source. DSH does not publish its client tsdown preset; the local minimal preset is release-coupled to the DSH module table and guarded by `client-build.test.ts`.
