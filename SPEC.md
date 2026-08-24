# SPEC

## §G GOAL

SwarmX presents the published DeepSeek Harness Web profile as one safe, local Electron desktop surface with an additive local-first scientific IDE.

## §C CONSTRAINTS

- Harness owns sessions, tools, model access, persistence, permissions, and browser UI.
- SwarmX owns in-process boot, Electron lifecycle, navigation, desktop security policy, and product extensions mounted only through published Harness seams.
- Renderer remains sandboxed remote content with no preload or Node integration.
- Server binds `127.0.0.1` on an OS-assigned port.
- Workspace uses `pnpm@11.7.0`; build tooling follows DSH TypeScript 6 + tsdown client conventions.
- DSH session log remains agent-interaction truth; Science Journal owns scientific domain facts and never replaces Chat/Trajectory projections.
- Science defaults to local-only execution/storage with no implicit network capability.

## §I INTERFACES

- cmd: `pnpm start` → build, boot Harness, open one Electron window.
- cmd: `pnpm build` → emit desktop host + DSH-compatible client plugin artifacts.
- file: `$DSH_HOME/profiles/web/cordis.patch.yml` → user-owned profile overrides.
- package: profile dependencies installed by `dsh plugin --profile web add` → out-of-tree DSH bundles resolved from the profile before the app-owned fallback graph.
- package: `@swarmx/dsh-ui-conversation` → Retry/Edit client plugin composed after DSH bundles and before user patches.
- api: `ctx.sideView.open/activate/close/dismiss/getSnapshot/subscribe` → generic per-Session serializable right-column tabs over published layout geometry.
- slot: `side-view.content` → keyed Session content renderer; Side View service stores descriptors only, never React nodes.
- package: `@swarmx/dsh-science` → `ctx.science`, client-safe contracts, SQLite Science Journal, projections, and strict Remote descriptors.
- package: `@swarmx/dsh-ui-science` → additive `conversation.view` Science Workspace.
- api: `ctx.science.createProject/createNotebook/getWorkspace` → idempotent, abortable project/notebook vertical slice.
- api: `ctx.science.registerArtifact(sessionId, request, signal?)` → workspace-relative generic file capture; returns immutable metadata only.
- api: `ctx.science.importArtifact(sessionId, request, signal?)` → bounded browser-file admission into the immutable Artifact Registry; infers trusted type from one logical filename and returns metadata only.
- api: `ctx.science.traceProvenance(sessionId, request, signal?)` → bounded workspace-local entity/event/relation lineage derived from Science Journal.
- api: `ctx.science.executeNotebookCell(sessionId, request, signal?)` → run one non-interactive Python cell in the live workspace and return bounded output, runtime metadata, notebook revision, and optional captured artifact.
- api: `ctx.science.previewArtifact(sessionId, request, signal?)` → return one authorized, digest-verified, bounded text/image preview or an explicit unavailable reason without revealing an artifact path.
- api: `ctx.science.createDocument(sessionId, request, signal?)` → create one bounded logical `.typ`/`.tex`/`.md`/`.bib` source document with revision metadata and diagnostics.
- api: `ctx.science.modifyDocument(sessionId, request, signal?)` → propose or accept/reject one source-selection patch against an exact document revision.
- api: `ctx.science.createFigure(sessionId, request, signal?)` → create one bounded matplotlib/seaborn/ggplot2/plotly code figure with a semantic object map and optional registered figure artifact.
- api: `ctx.science.modifyFigureCode(sessionId, request, signal?)` → propose or accept/reject one semantic-object-linked code patch against an exact figure revision.
- api: `ctx.science.createQuestion/createHypothesis/recordClaim/linkEvidence` → append bounded Research Map facts and typed evidence relations without exposing Journal or SQL internals.
- api: `ctx.science.defineExperiment/startRun/finishRun/compareRuns` → own an exact-revision experiment/run ledger with redacted reproducibility metadata and bounded metric comparison.
- api: `ctx.science.exportProject(sessionId, request, signal?)` → create one content-addressed, replayable JSON project export whose Journal event contains metadata only.
- plugin: `@swarmx/dsh-science/tools` → seven aggregate `science_*` tools registered through `ctx.tools`, each returning a fact/inference/proposal classification and Science entity locator.
- api: `JupyMcpRuntime.execute/readNotebook/close` → Host-only persistent Notebook Controller keyed by workspace + Notebook, backed by one local JupyMCP MCP stdio session through the official TypeScript SDK; returns bounded standard MIME outputs and owns MCP/kernel teardown.

## §R RESEARCH

id|topic|finding|src
R1|Notebook separation|VS Code separates `NotebookSerializer`, `NotebookController`, and `NotebookRenderer`; core owns cells/simple outputs, extensions own execution and application MIME rendering|https://code.visualstudio.com/api/extension-guides/notebook
R2|Notebook viewport|VS Code Notebook editor is a virtualized list; each cell resolves to a Monaco text model, rich outputs render asynchronously in isolated webview/iframe context|https://github.com/microsoft/vscode/wiki/Notebook-documentation
R3|Jupyter execution|VS Code Jupyter maps kernelspecs to Notebook controllers and uses Jupyter Messaging Protocol through raw ZeroMQ kernels or Jupyter Server API|https://github.com/microsoft/vscode-jupyter/wiki/Kernels-%28Architecture%29
R4|Jupyter MIME|Microsoft ships Notebook renderer extensions for image, GeoJSON, Plotly, Vega, VDOM, and other Jupyter MIME outputs instead of embedding those renderers in the controller|https://github.com/microsoft/vscode-notebook-renderers
R5|JupyMCP substrate|JupyMCP 0.3.1 is a Python ≥3.11 MCP server with local kernel management, stdio by default, `execute`/kernel lifecycle tools, and `notebook://{path}` resources; it declares MCP 2.x and ipykernel directly, while installation and remote Jupyter remain explicit operator choices|https://pypi.org/project/jupymcp/
R6|Notebook output rendering|JupyterLab publishes `RenderMimeRegistry` standard MIME factories and `OutputArea` as separable renderer primitives; VS Code likewise separates Notebook serialization, execution controllers, and output renderers|https://jupyterlab.readthedocs.io/en/stable/api/modules/rendermime.html https://jupyterlab.readthedocs.io/en/stable/api/modules/outputarea.html https://code.visualstudio.com/api/extension-guides/notebook

## §V INVARIANTS

V1: ∀ desktop boot → Harness browser handoff disabled; exactly one SwarmX-owned surface opens.
V2: ∀ renderer navigation → same Harness origin stays in-window; foreign `http:`/`https:` may open externally; every other protocol is denied.
V3: ∀ renderer browser permission → denied unless a later documented capability adds a focused allow rule and regression test.
V4: ∀ generated build output → excluded from source lint; authored source remains formatted and checked.
V5: ∀ initial renderer load failure → error is reported and the window is shown instead of remaining invisibly blocked.
V6: ∀ re-run lookup → use real DSH conversation nodes and turn boundaries; missing or out-of-window evidence declines safely.
V7: ∀ Edit action → prepare a separate session before the selected turn, open it, seed its draft; source session remains unchanged.
V8: ∀ first-turn Retry/Edit → create a fresh session in the source Workspace (adopting its cwd when ungrouped) because DSH `session.fork` cannot represent an empty completed-turn prefix.
V9: ∀ closed user-originated message with a safe preparation strategy → Edit renders as an icon beside that user message; assistant action rows own no SwarmX Edit control.
V10: ∀ turn ending in error → visible failure message and Retry icon render even when no finalized assistant message exists.
V11: ∀ Edit/Retry → source log remains append-only and unchanged; active child starts before the superseded turn and appends only the revised/retried user input; superseded messages, `llm/retry`, and `turn/end(error)` remain in source history but stay outside child Chat/model projections and LLM prompt payloads.
V12: ∀ eligible user-message action row → the clock, Copy, and Edit render in that order; Copy and Edit each reserve their own 28px lane with the native 10px action gap, so neither icon overlaps the clock.
V13: ∀ SwarmX conversation action icon → its semantic component resolves through one local `conversationIcons` map; changing icon artwork requires no action/controller/layout or client-bundle configuration change.
V14: ∀ reusable package → `packages/<group>/<package>`; browser plugin at `packages/client/ui-conversation`; executable assembly at `apps/desktop`.
V15: ∀ clean root build → host/client TypeScript references + tsdown emit desktop `dist/` and plugin `lib/{index.js,client.js,types/**}`.
V16: ∀ shared build tool → root ownership; client bundle keeps DSH module-table identities external.
V17: ∀ Science UI mount → add one `conversation.view` entry; Chat snapshot, Trajectory snapshot, and agent loop remain unchanged.
V18: ∀ Science mutation → validate client-safe input & honor pre-aborted `AbortSignal`; one idempotency key commits ≤1 journal event and projection update in one transaction.
V19: ∀ committed Science fact → append-only SQLite WAL journal is truth; versioned migrations and deterministic replay rebuild materialized views.
V20: ∀ Science artifact bytes → immutable content-addressed object outside journal; journal stores bounded metadata/reference only.
V21: ∀ Science default storage/execution → local-only, owner-only paths, no implicit network, no shell-concatenated command.
V22: ∀ Science plugin disposal/reload → close owned database/process resources and withdraw slot/Remote contributions; replacement mount reopens durable state without duplicate effects.
V23: ∀ initial Science view → expose Notebook, Writing, Figures, Research Map, Experiments; project/notebook create flow renders keyboard-accessible loading, empty, success, and error states.
V24: ∀ Science Remote access → host derives an opaque workspace key from a live DSH session `cwd`, rejects cross-workspace entity ids, and returns no host path to UI.
V25: ∀ artifact registration → stream bytes through owner-only staging, compute SHA256, fsync, atomically publish one immutable object per digest, then append metadata; abort/failure before Journal commit may leave only an unreachable immutable object.
V26: ∀ artifact source → non-empty relative path resolves through realpath to a regular file inside live session `cwd` and within configured byte limit; reject absolute/traversal/symlink escape; Journal/Remote contain neither source path nor bytes.
V27: ∀ provenance trace → derive only workspace-local entity/event/relation facts from Science Journal, cap traversal depth and output, exclude host paths and model-private reasoning.
V28: ∀ client-supplied Science environment metadata → redact secret-key values and absolute paths before Journal commit or Remote return.
V29: ∀ explicit Isolated Run Python execution → resolve the configured executable through `ctx.subprocess`, pass source through stdin with explicit argv/cwd/stdio/env (no shell), bound both output streams, terminate on abort/disposal, and append no execution fact after cancellation.
V30: ∀ settled Notebook cell execution → one idempotency key atomically appends ≤1 `notebook/cell-executed` fact and updates notebook cells/revision plus at most one output artifact; replay restores both, including failed-execution evidence, ordered bounded outputs, and redacted path-free runtime metadata.
V31: ∀ provenance trace crossing an aggregate Science event → return each Journal sequence at most once and attribute the event to its owning mutation entity.
V32: ∀ root `pnpm test` → rebuild current library/plugin artifacts before the real Harness packaging test boots package `main` entries.
V33: ∀ Science document create → accept one traversal-free logical name ending `.typ`/`.tex`/`.md`/`.bib`, bounded UTF-16 source, workspace-local project ownership, and one replayable `document/created` fact.
V34: ∀ document patch mutation → require exact `expectedRevision`; proposal selection uses textarea-compatible UTF-16 `[start,end)` offsets and snapshots original/proposed text plus reasoning; propose/reject preserve source, accept replaces only the snapshotted range, and each action appends ≤1 replayable fact.
V35: ∀ Writing diagnostics → deterministic bounded ranges/messages derived from stored source; distinguish structural checks from full Typst/LaTeX compilation and never report structural success as compile success.
V36: ∀ Figure create → accept one supported plotting library and bounded source; derive bounded axis/legend/annotation/line/point/image-layer/data-series objects with valid UTF-16 code ranges, verify any artifact is a workspace-local `figure`, and append one replayable `figure/created` fact.
V37: ∀ Figure patch mutation → require exact `expectedRevision` and ≥1 current semantic object; proposal snapshots selected object ids, their enclosing code range, original/proposed code, instruction, and proposal reasoning; propose/reject preserve code and accept replaces only that range in one replayable fact.
V38: ∀ accepted Figure code patch → preserve selected semantic object identity, shift unaffected later ranges by the UTF-16 delta, reject ambiguous overlap, and keep every object range inside the new code.
V39: ∀ Research Map fact → belong to one workspace-local project, use a bounded question/hypothesis/claim/evidence/decision/review/open-question kind, and append exactly one replayable Journal fact; relations use the published scientific relation vocabulary and reject cross-project or cross-workspace endpoints.
V40: ∀ Experiment/Run mutation → require workspace-local ownership and exact revisions; a Run starts once, finishes once into succeeded/failed/cancelled, records redacted bounded environment/metrics/artifact references, and replay restores the complete lifecycle.
V41: ∀ Run comparison → compare 2–10 completed Runs of one Experiment without mutation, return bounded finite numeric deltas against the first Run, and classify the result as inference rather than fact.
V42: ∀ project export → serialize only one workspace-local project's client-safe scientific facts in deterministic order, store bytes as an immutable SHA256 object outside Journal, append only bounded digest/size/count metadata, and return an exact idempotent payload capped by configured export size.
V43: ∀ Science agent surface → expose exactly `science_notebook`, `science_write`, `science_figure`, `science_experiment`, `science_record`, `science_query`, and `science_export`; validate strict JSON, forward `AbortSignal`, expose no SQL/path/private reasoning, and rely on the profile's session-scoped spill policy for oversized rendered results.
V44: ∀ successful Science tool result → include a durable locator containing session id, tool call id, entity kind/id, and Journal sequence plus a fact/inference/proposal classification; DSH tool events remain the trajectory association and no agent-loop or Trajectory snapshot mutation is permitted.
V45: ∀ Science UI Remote mutation rejection → enter the visible retryable error state exactly once and never run the mutation's success reload path.
V46: ∀ Science view registration → occur only inside a Cordis child context injected with `remote.science`, so every rendered Remote call has an active namespace service.
V47: ∀ `ctx.sideView.open(sessionId, entry)` → accept one JSON-serializable `inspect|workbench` descriptor, upsert by entry id inside that Session, activate it, and open published `ctx.layout` details geometry; service stores no React node or host path.
V48: ∀ Side View tab mutation → open/switch/close order and active fallback remain deterministic per Session; one Session cannot observe another's tabs; dismiss preserves tabs while HMR disposal clears owned state/listeners, closes details, and withdraws registrations/service.
V49: ∀ generic Side View mount → shadow only published root `details` at an explicit lower render priority, declare keyed `side-view.content`, redeclare no upstream-owned child slot, expose labelled tabs with wrapping Arrow/Home/End keyboard navigation, render Tool input/output only from public `ToolCallBlock`, and never replace `conversation`, `conversation.session`, Chat snapshot, composer state, or chat scroll memory.
V50: ∀ Science artifact Side View entry → carry bounded client-safe artifact metadata/provenance only, route by `science-artifact`, deduplicate repeated opens into one active tab, and expose no artifact bytes or host path.
V51: ∀ Side View mode → details opens at 360px and remains draggable above its 300px floor without a fixed maximum; the concession solver derives the rendered maximum from the viewport after preserving the sidebar and a 640px center, while narrow concession may visually close details without mutating tabs; `workbench` remains an explicit descriptor with the fullscreen fallback.
V52: ∀ artifact preview → resolve artifact id only inside the requesting Session workspace, verify immutable bytes against the registered digest, return either bounded text (≤64 KiB), a bounded PNG/JPEG/GIF/WebP data URL (source ≤2 MiB), or an explicit unsupported/too-large result; support AbortSignal and expose no host path or network URL.
V53: ∀ Side View “Open in Science” → retain the same artifact locator/tab, publish one deduplicated per-Session workbench target, focus it immediately when the Science view is mounted, or retain it until the user selects Science when rc.8 exposes no public programmatic view switch; never mutate Chat view selection, composer draft, or scroll memory.
V54: ∀ completed aggregate Science Tool result carrying an artifact locator → a bounded strict parse may open that artifact in the same Session Side View through `side-view.tool.actions`; malformed, cross-Session, non-artifact, and oversized results contribute no action.
V55: ∀ future DSH Side View capability → extend `ctx.layout.openDetails` with an inspect/workbench preferred-width policy and expose a public Tool selection/renderer delegation, still subject to center minimum and narrow concession; until published, an exact-version package-manager patch may remove only rc.8's fixed details ceiling, but implementation must not directly edit installed dependencies, copy private DetailsPanel, redeclare its private child slot, or bypass the concession solver with CSS.
V56: ∀ ready Science workspace with ≥1 Notebook → exactly one active Notebook id belongs to the refreshed workspace; initial load selects the first, create selects its returned Notebook, explicit selection survives reload while present, and execute targets only the active Notebook; selector exposes current state to keyboard and assistive technology.
V57: ∀ Science destination navigation → Notebook is initial workbench; exactly one of Notebook/Writing/Figures/Research Map/Experiments exposes `aria-current` and renders its studio body; switching uses native buttons, keeps parent-owned drafts/selections, and never mutates Session, Journal, composer, or Side View state.
V58: ∀ Notebook optional output capture → empty path executes without artifact; non-empty path must be traversal-free workspace-relative with a supported deterministic extension→kind/MIME mapping, is sent in the same `executeNotebookCell` request, and on success appears once in Artifact Registry linked to that execution/Notebook; invalid capture input cannot run or reach Host.
V59: ∀ populated Artifact Registry card → render title and bounded kind/size/Journal metadata as one distinct readable identity block while its Side View action remains independently operable.
V60: ∀ ready Science view → use one Claude-Science-style project shell: a persistent compact project rail for New/Notebook/Writing/Figures/Files/Research Map/Experiments, one center activity or workbench surface, and the existing generic Side View for contextual artifacts; New reveals the create-Notebook form on demand; remove the hero and five-card dashboard without replacing DSH AppFrame, Chat, composer, or layout geometry.
V61: ∀ active Notebook in that shell → expose Notebook tabs, render only its ordered code/output activity, reveal one direct Python run composer on demand (or for an empty Notebook) without competing with the persistent agent composer, and surface generated artifacts as file actions that open the unchanged per-Session Side View locator.
V62: ∀ Science artifact Side View → prioritize the authorized file preview before compact provenance/details, keep filename and workbench action in a file header, and retain loading/error/unsupported states without exposing bytes, host paths, or network URLs.
V63: ∀ Writing/Figure/Research/Experiment workbench with an open inspect Side View → adapt to the remaining center track without horizontal overflow; collapse multi-column review layouts below 720px of actual stage width while retaining source, review, and actions.
V64: ∀ bundle installed into the active profile through the published DSH plugin command → load its patch and module from that profile's `node_modules` before the app-owned `$DSH_HOME/profiles/node_modules` fallback; keep in-box and SwarmX product bundles resolvable without copying third-party source or editing dependency internals.
V65: ∀ details drag preference >520px → retain that preference and render it when the viewport can preserve `CENTER_MIN`; when it cannot, shrink or close details through the existing concession chain and restore the preference on re-widening.
V66: ∀ browser artifact import → accept one basename ending `.csv|.tsv|.json|.txt|.md|.xlsx|.pdf|.png|.jpg|.jpeg|.gif|.webp`, infer kind/MIME on Host, require canonical base64 decoding to 1–8 MiB, bind project to the requesting Session workspace, fingerprint idempotency with decoded SHA256, and persist/return neither encoded bytes nor a host path.
V67: ∀ Notebook artifact input → accept 1–4 workspace-local artifact ids totaling ≤32 MiB, verify immutable bytes, materialize owner-only disposable copies, expose their paths only as `DSH_SCIENCE_INPUT_<n>` during controller-owned silent kernel setup, serialize only that setup→visible-cell critical section across a shared workspace peer, remove those variables and copies after success/failure/abort/disposal, and record the exact input ids on replayable cells without returning paths.
V68: ∀ Files import surface → expose keyboard-operable file selection plus drag/drop, announce importing/error state, import accepted files sequentially, refresh each artifact exactly once, and reuse the unchanged Side View locator/preview; rejected files never invoke Remote.
V69: ∀ “Analyze in Notebook” on an imported file → select or create one Notebook, switch only the Science destination to Notebook, execute one deterministic extension-aware inspection cell with that artifact as input, capture one bounded JSON analysis artifact, refresh Files/cells, and preserve Chat, composer, scroll, Side View tabs, and agent loop.
V70: ∀ CSV/TSV or scalar-record JSON artifact preview → parse verified workspace-local bytes on Host into at most 500 rows × 256 typed columns, preserve duplicate/blank headings without object-key loss, report truncation, return scalar client-safe cells only, and render through AG Grid Community in Side View/workbench rather than raw text; cancellation, digest verification, path secrecy, and no-network defaults remain unchanged.
V71: ∀ editable Notebook surface → follow Serializer/Controller/Renderer separation: canonical Notebook cells + MIME outputs are document truth, one persistent Jupyter controller owns kernel execution, mature editors/virtualization own visible cells, MIME renderers own outputs, and Science Journal adds provenance without replacing any layer.
V72: ∀ Jupyter kernel session → Host owns one controller per workspace + Notebook while one configured JupyMCP stdio server is shared only within that verified workspace through the official MCP SDK with DSH-scrubbed environment and workspace cwd; expose no MCP transport, kernel connection file/port/token/path to Client, perform no implicit `uvx` download, package install, remote-server connection, or outbound dependency fetch, and report explicit unavailable state when prerequisites fail.
V73: ∀ Jupyter cell execution → serialize requests per controller, call JupyMCP `execute` with a Host-derived workspace-relative `.ipynb` path, preserve kernel state across cells, normalize ordered MCP text/image/audio/embedded-resource blocks into bounded client-safe `stream|display_data|execute_result|error` outputs, classify cancellation before Journal mutation, and discard an aborted controller before later reuse.
V74: ∀ Jupyter controller abort/disposal → settle its queued/in-flight calls and shut down only its owned JupyMCP kernel before reuse; ∀ runtime disposal/reload → stop admitting calls, settle every controller, request `shutdown_all` once per workspace, close each official MCP Client/stdio transport exactly once, and leave no owned JupyMCP/kernel/input staging process behind; teardown failure is explicit and never reported as success.
V75: ∀ legacy stateless Python execution → label and invoke only as explicit Isolated Run; never silently substitute it when persistent Jupyter is unavailable and never present its fresh namespace as Notebook kernel semantics.
V76: ∀ JupyMCP 0.3.1 controller → start one kernel per Notebook, retain its returned `kernel_id`, pass that id to visible execution and controller-owned silent setup, install one-shot post-cell cleanup for temporary input environment, and target that id for shutdown; multiple Notebook controllers may share one workspace MCP server but never rely on running-kernel list order or share kernel state.
V77: ∀ JupyMCP Notebook document → store only as the Host-derived owner-only `.dsh-science-notebook-<notebook-id>.ipynb` hidden file inside the verified workspace, so JupyMCP 0.3.1's authority-only `notebook://{path}` resource template can address it without unsupported slash encoding; bound resource reads independently from one-cell outputs and return neither its name/path nor raw unbounded document to Client.
V78: ∀ Notebook MIME output → adapt canonical outputs to the official JupyterLab OutputArea/RenderMime stack with untrusted rendering by default, preserve stream/error/display ordering and plain-text fallback, detach widgets on cell change/unmount/HMR, and do not hand-code parallel HTML/image/LaTeX renderer dispatch.
V79: ∀ bundled JupyterLab renderer dependency → resolve its published browser entry even when the DSH closure factory provides a loader-scoped `require`; never execute a Node builtin fallback in Web, and retain the visible plain-text fallback if a later renderer dependency still fails.
V120: ∀ renderer clipboard write → permit only `clipboard-sanitized-write` from the SwarmX window's main frame at the exact Harness origin; clipboard read, subframes, foreign origins/WebContents, and every unrelated permission remain denied.

## §T TASKS

id|status|task|cites
T1|x|suppress duplicate system-browser handoff|V1
T2|x|restrict external navigation and permissions|V2,V3
T3|x|restore reliable source lint and code map check|V4
T4|x|compose and integration-test Retry/Edit plugin|I.package
T5|.|add packaging, signing, updates, and release smoke tests|I.cmd
T6|x|surface initial renderer load failures|V5
T7|x|render and execute Retry/Edit against real DSH conversation nodes|V6,V7,V8
T8|x|place Edit on user messages and Retry on failed turns|V9,V10
T9|x|pin append-only branch and model-projection isolation|V11
T10|x|keep user clock, Copy, and Edit in one non-overlapping visual sequence|V12
T11|x|centralize conversation icon selection and consolidate retained legacy icon assets|V13
T12|x|align workspace layout and build system with DSH|V14,V15,V16,I.cmd,I.package
T13|x|ship Science project/notebook journal + Remote + visible Workspace vertical slice|V17,V18,V19,V21,V22,V23,V24,I.api
T14|x|add content-addressed Artifact Registry and provenance trace/replay|V18,V19,V20,V21,V25,V26,V27,V28,I.api
T15|x|execute Python notebook cells through subprocess with environment/artifact capture|V18,V20,V21,V22,V29,V30,V31,V32,I.api
T16|x|add Typst/LaTeX/Markdown Writing Studio patch revisions and claim checks|V17,V18,V19,V23,V33,V34,V35,I.api
T17|x|add Figure Studio registration, semantic selection, and code patches|V17,V18,V19,V20,V23,V36,V37,V38,I.api
T18|x|add Research Map facts, bounded science tools, experiment lifecycle, export, demo, and trajectory locators|V17,V18,V19,V20,V21,V39,V40,V41,V42,V43,V44,I.api
T19|x|ship generic Side View, public-data Tool inspector, and Science artifact tab deep link|V47,V48,V49,V50,V51,I.api,I.slot
T20|x|add bounded artifact preview, Chat locator entry, and fullscreen Science deep-link fallback|V52,V53,V54,V55,I.api,I.slot
T21|x|add explicit active Notebook selection and exact execution ownership|V23,V56,I.api
T22|x|turn Science destination cards into state-preserving first-level workbench navigation|V17,V23,V57,I.package
T23|x|connect Notebook output capture to Artifact Registry and Side View preview|V20,V25,V26,V30,V50,V52,V58,V59,I.api,I.slot
T24|x|replace the Science dashboard with a Claude-Science-style project/activity/artifact shell|V17,V23,V47,V51,V57,V60,V61,V62,V63,I.package,I.slot
T25|x|honor profile-installed out-of-tree plugins and run dsh-cowork in the desktop harness|V64,I.package
T26|x|remove the fixed DetailsPanel width ceiling without weakening layout concessions|V51,V55,V65,I.package
T27|x|ship Files import and one-click artifact-to-Notebook analysis loop|V17,V18,V20,V21,V24,V25,V27,V30,V45,V50,V52,V56,V58,V61,V62,V66,V67,V68,V69,I.api,I.slot
T28|x|replace raw tabular previews with mature typed AG Grid data views|V17,V20,V24,V51,V52,V62,V70,I.api,I.package,I.slot
T29|x|ship JupyMCP-backed persistent Notebook Controller and standard MIME execution contract|V17,V18,V20,V21,V22,V24,V30,V31,V56,V58,V61,V67,V71,V72,V73,V74,V75,V76,V77,I.api,I.package
T30|x|render Notebook MIME outputs through official JupyterLab renderer primitives|V24,V30,V33,V35,V49,V58,V60,V61,V71,V73,V78,V79,I.package,I.slot

## §B BUGS

id|date|cause|fix
B1|2026-08-20|Electron launcher passed empty Web args, preserving upstream auto-open default|V1
B2|2026-08-20|navigation fence delegated unvalidated protocols to `shell.openExternal`|V2
B3|2026-08-20|renderer session installed no explicit permission policy|V3
B4|2026-08-20|Biome scanned generated `lib/` output and used a stale schema URL|V4
B5|2026-08-20|the launcher ignored `loadURL` rejection while waiting indefinitely for `ready-to-show`|V5
B6|2026-08-20|turn lookup read nonexistent `location.turn.turn`; real assistant nodes expose top-level `turn`, so both actions always rendered null|V6
B7|2026-08-20|Edit staged private controller state but no composer submission path consumed it|V7
B8|2026-08-20|the extension treated omitted `atSeq` as session start, but DSH forks through the last completed turn when the anchor is omitted|V8
B9|2026-08-20|Edit registered in the finalized-assistant-only action slot, so it appeared on the wrong author row|V9
B10|2026-08-20|Retry registered in the finalized-assistant-only action slot, so error turns without an assistant message had no action or visible terminal failure|V10
B11|2026-08-20|source-log preservation and model-projection isolation were implicit, leaving future Edit/Retry changes free to overwrite history or leak failure metadata into prompts|V11
B12|2026-08-20|the derived Edit row was pulled left of Copy without reserving space in the native clock-and-Copy row, so longer clock labels overlapped Edit and the icon order was reversed|V12
B13|2026-08-20|the new icon map used a multiline import that differed from Biome's canonical formatting|V4
B14|2026-08-20|README required Corepack although supported Node 26 omits it|§C package-manager requirement only
B15|2026-08-20|pnpm script policy allowed Electron only, so DSH native/build dependencies stayed unbuilt|§C DSH-aligned `allowBuilds`
B16|2026-08-21|workspace test equated an asset directory with a package boundary, conflicting with preserved `packages/desktop/build` artwork|§C package ownership is identified by `package.json`
B17|2026-08-21|artifact environment metadata accepted secret values and absolute paths without applying the client-visible data boundary|V28
B18|2026-08-21|aggregate notebook execution attached the same Journal event to both notebook and artifact trace nodes, duplicating one scientific fact|V31
B19|2026-08-21|root tests booted package `main` entries without rebuilding, so Harness integration assertions observed stale plugin artifacts|V32
B20|2026-08-21|non-clean multi-entry Science builds emitted hashed shared chunks, so pack allowlists retained stale runtime files|V15
B21|2026-08-21|Figure creation passed a third callback to `Promise.then`, so rejection used the success reload callback and the intended error handler was unreachable|V45
B22|2026-08-21|migration startup trusted only `MAX(version)` and used non-idempotent v5 DDL, so a missing middle migration or retained projection table could not self-repair|V19
B23|2026-08-21|the Science slot closed over the plugin root context, which mounted `remote.science` but never injected that namespace into the view registration fiber|V46
B24|2026-08-21|generic Side View registered the occupied single `details` slot at upstream priority 0, so browser plugin boot rejected the extension|V49
B25|2026-08-21|generic Tool content redeclared upstream DetailsPanel's `conversation.details.tool` child; rc.8 enforces one declaring parent and rejected browser plugin boot|V49,V55
B26|2026-08-21|Tool inspector signature drifted from Biome's canonical single-line formatting during the rc.8 fallback edit|V4
B27|2026-08-21|Science Workspace discarded `createNotebook` result and executed against `workspace.notebooks[0]`, so a newly created Notebook was visible but could not become the execution owner|V56
B28|2026-08-21|active Notebook resolver call drifted from Biome's canonical single-line formatting during T21|V4
B29|2026-08-21|T22 conditional Studio wrappers changed JSX nesting without applying Biome's canonical indentation before lint|V4
B30|2026-08-21|T23 capture metadata and Remote forwarding were added without applying Biome's canonical import and line wrapping before the focused check|V4
B31|2026-08-21|Artifact Registry rendered its title and metadata as adjacent inline text, visually concatenating `live-demo.csvdataset` in the real workbench|V59
B32|2026-08-21|T24 imported the aggregate DSH primitives entry for two rail icons, which pulled its CSS-bearing component graph into the CSS-free server-rendered view test|V60
B33|2026-08-21|T24 project-heading JSX exceeded Biome's canonical wrapping after the accessible id correction|V4
B34|2026-08-21|the embedded launcher forced all bare plugin imports to resolve from the installed DSH package, bypassing the active profile's own `node_modules` and breaking plugins installed through the published `dsh plugin` command|V64
B35|2026-08-21|the profile-plugin harness fixture import was expanded across lines instead of Biome's canonical single-line form|V4
B49|2026-08-24|the desktop's blanket renderer permission denial rejected the async Clipboard API used by the native assistant Copy action, so the button produced neither clipboard content nor success feedback|V120
