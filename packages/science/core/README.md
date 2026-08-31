# `@swarmx/dsh-science`

Local-first scientific domain authority for SwarmX. `ScienceCore` owns all workspace, journal,
artifact, notebook, literature, and writing behavior behind structural workspace/process seams.
The default export is the thin DSH Cordis/Typert carrier; Codex calls `ScienceCore` through the
owned SwarmX MCP carrier. T13 establishes the first end-to-end slice:

- `ctx.science.createProject(sessionId, request, signal?)`
- `ctx.science.createNotebook(sessionId, request, signal?)`
- `ctx.science.getWorkspace(sessionId, signal?)`
- append-only SQLite Science Journal in WAL mode
- deterministic project/notebook projection replay
- strict Typert Host and Client Remote descriptors

T14 extends that same ownership boundary with:

- `await ctx.science.registerArtifact(sessionId, request, signal?)`
- `ctx.science.getResearchObject(sessionId, { projectId }, signal?)`
- generic immutable objects addressed by streamed SHA256 content digests
- metadata-only Journal events and a replayable artifact projection
- RO-Crate 1.3 project metadata with Schema.org entities and provenance Actions

T15 adds one stateless, non-interactive Python execution path:

- `ctx.science.executeNotebookCell(sessionId, request, signal?)`
- configured Python resolution and managed process-tree lifecycle through `ctx.subprocess`
- source over stdin, explicit argv/cwd/stdio/env, bounded stdout/stderr, and no shell
- Python implementation/version plus a deterministic installed-package-set SHA256
- atomic notebook cell/revision update with an optional single output artifact
- nonzero exits retained as reproducible output; cancellation appends no execution fact

T29 replaces that path as the default Notebook controller while retaining it as an explicit
`isolated` run mode:

- one configured local JupyMCP MCP stdio process per verified workspace with one explicitly routed
  persistent Jupyter kernel per Notebook, connected through the official MCP TypeScript SDK
- serialized `execute` calls with ordered bounded text/image/audio/resource MIME outputs
- owner-only hidden `.ipynb` persistence and separately bounded Notebook resource reads
- no implicit package download, install, remote Jupyter connection, or fallback to a fresh process
- cancellation discards and shuts down only that Notebook's owned kernel; runtime disposal requests
  `shutdown_all` once and closes the workspace MCP transport

T27 closes the Files-to-Notebook agent loop:

- `ctx.science.importArtifact(sessionId, request, signal?)` accepts one canonical-base64 browser
  file up to 8 MiB, infers kind/MIME from a path-free supported filename, and returns metadata only
- imported bytes enter the same owner-only content-addressed object store; encoded bytes and host
  paths enter neither the idempotency record, Journal event, projection, nor Remote response
- `executeNotebookCell` may address up to four workspace-local input artifacts totaling 32 MiB;
  verified owner-only temporary copies are exposed only to controller-owned kernel setup as
  `DSH_SCIENCE_INPUT_<n>` and removed after success, failure, cancellation, or disposal
- replayable Notebook cells retain exact input artifact ids, and the Research Object reports them
  through `isBasedOn` and Action `object` references without exposing a materialized path

There is no browser Science Workspace with direct Notebook create/execute controls. Users ask the
model in Chat; the aggregate `science_notebook` tool remains the user-facing
entry to these Host APIs and JupyMCP, while generated artifacts remain visible through Files and
Side View.

T16 adds the first Writing Studio revision path:

- `ctx.science.createDocument(sessionId, request, signal?)`
- `ctx.science.modifyDocument(sessionId, request, signal?)`
- logical `.typ`, `.tex`, `.md`, and `.bib` documents with bounded client-safe source
- textarea-compatible UTF-16 source selections and exact-revision conflict rejection
- append-only propose/accept/reject facts preserving original text, proposed text, reasoning summary,
  resolution, source hashes, and provenance receipts
- deterministic structural, missing-figure-reference, and unsupported-claim diagnostics

Writing diagnostics are local heuristics, not compiler results. Every document explicitly reports
`structural: checked` and `compilation: not-run`; that result alone never supports a compilability or
rendered-validity claim. Workspace Typst papers are compiled only by the managed preview watcher
below, never by a model tool call. Other formats require a separately owned compiler boundary.

Workspace papers produced by the agent use a separate, real Typst preview boundary:

- `ctx.science.previewTypstDocument(sessionId, { relativePath }, signal?)` authorizes one
  workspace-relative `.typ`/`.typst` file and starts/reuses the bundled semantic Typst watcher.
- the watcher retains one Typst document/world snapshot and exports the PDF from that exact snapshot;
  imported source and asset changes are observed automatically without a model tool call.
- the paper entrypoint comes from a verified alpha.2 inline-code file mention or explicit DetailsPanel
  open; after registration, the watcher observes source/import changes without another model turn.
- the response contains bounded source plus its SHA256 revision, the latest bounded compiled PDF when
  available, the exact source revision that produced that PDF, and current bounded compiler
  diagnostics; no canonical path leaves the Host.
- `ctx.science.updateTypstSource(...)` atomically writes only when `expectedSourceRevision` still
  matches, so an agent edit and a source-mode save cannot silently overwrite each other.
- `ctx.science.resolveTypstSourceAtPoint(...)` accepts a page, normalized point, and exact PDF
  revision. It uses Typst IDE's semantic frame mapping, then re-authorizes the returned project file,
  SHA256, and UTF-16 caret offset before returning it to the browser. It never searches extracted PDF
  text or asks the model to infer a source location.
- watcher output lives in an owner-only disposable directory and every process/directory is closed on
  replacement or plugin disposal. The writing preview runtime is built and packaged for the current
  desktop platform with an explicit `typst` engine identity; an unavailable runtime is explicit,
  never a structural-check success or an automatic network install. Future writing engines such as
  LaTeX extend this boundary instead of renaming it.

T17 adds the first Figure Studio revision path:

- `ctx.science.createFigure(sessionId, request, signal?)`
- `ctx.science.modifyFigureCode(sessionId, request, signal?)`
- bounded matplotlib, seaborn, ggplot2, and plotly source with inferred semantic objects
- axis, legend, annotation, line, point, image-layer, and data-series objects linked to UTF-16 code
  ranges
- exact-revision propose/accept/reject facts with selected object ids and proposal reasoning
- accepted-patch range remapping that preserves object identity and rejects ambiguous overlap
- optional linkage to a workspace-local immutable `figure` artifact

The first Figure slice edits plotting source and its semantic map. It does not execute plotting code,
serve artifact bytes to the browser, or claim that a proposed patch renders successfully.

T18 completes the first product loop inside the same release boundary:

The model-facing surface is composed only by the system-owned `dsh-science` preset. Its checked-in
agent composition matches the complete locked DSH `standard` preset and then adds the Science tool
and contract rows. The Host `ctx.science` service remains global so Remote methods and artifact UI
continue to work independently of the selected agent preset.

- `ctx.science.createQuestion/createHypothesis/recordClaim/linkEvidence` for bounded operational
  research facts and typed internal relations that project to RO-Crate
- `ctx.science.defineExperiment/startRun/finishRun/compareRuns` for exact-revision Experiment and
  Run lifecycle records, redacted environments, metrics, and artifact association
- `ctx.science.exportProject` for deterministic RO-Crate 1.3 JSON stored as a content-addressed
  object while the Journal retains only digest, byte count, entity counts, and provenance
- exactly seven aggregate tools from `@swarmx/dsh-science/tools`: `science_notebook`,
  `science_write`, `science_figure`, `science_experiment`, `science_record`, `science_query`, and
  `science_export`, plus the direct read-only `literature_search` tool
- `science_notebook.create_project` gives every runtime the same model-facing project-root action
- `science_query.inspect_annotation` re-authorizes a structured image point and projects the
  verified artifact as a durable image block for model discussion
- `runScienceDemo(ctx.science, sessionId, signal?)`, an executable local-only tour covering Notebook,
  Experiment, Figure, Writing, RO-Crate Research Object, and export

Local literature discovery is a separate read-only boundary:

- `ctx.science.searchLiterature(sessionId, { query, limit, filters }, signal?)` searches the running
  Zotero Desktop library through its loopback Web API v3 implementation. It never contacts
  zotero.org or another scholarly index and never reads attachment file URLs.
- `literature_search` is the sole model-facing literature tool. `science_query` remains the local
  RO-Crate/annotation reader and does not search publications.
- every Zotero candidate is converted to a sanitized, owner-only BibTeX snapshot before ranking or
  Tool output; provider-native JSON is not a public result contract.
- results distinguish the Zotero item key from the BibTeX citation key, carry bounded portable
  BibTeX entries, and label ranking as `zotero-local-v1`. Search is ephemeral and appends no Science
  Journal event; only a later explicit citation/evidence action persists a selected work.

Every successful tool result distinguishes `fact`, `inference`, or `proposal` and carries a locator
with the DSH session, tool call, Science entity, and Journal sequence. The ordinary DSH tool events
therefore associate agent operations with scientific facts in Trajectory without reading private
reasoning or modifying the agent loop. The current Harness profile does not publish a Trajectory
anchor-routing extension, so the locator remains in the ordinary Tool result but is not rendered as
a custom clickable Trajectory anchor. Oversized textual results use the Web profile's existing
session-scoped spill policy.

The Host defaults to the configured local `jupymcp` executable, 256 KiB per cell output, a 5 MiB
Notebook resource bound, and a 60-second MCP request timeout. It uses the same official SDK and
scrubbed child environment precedent as the Harness MCP client, additionally removes Python
injection and proxy variables, and never returns an executable, Notebook, or kernel path.
The controller uses JupyMCP's native MCP and kernel dependencies; Science adds no downgrade or
compatibility pin.

The `isolated` mode defaults to `python3` and a two-second process termination grace. It is
available only when explicitly configured and is never an automatic fallback for a missing
JupyMCP/kernel prerequisite. Callers may name one workspace-relative output file for capture and
may attach up to four immutable Science inputs.

Artifact registration accepts a workspace-relative source path. The Host rejects absolute paths,
`..` traversal, symlink escape, non-regular files, and files above `maxArtifactBytes`. Source paths
and file bytes never enter the Journal or a Remote response. Identical content shares one immutable
object while each scientific registration retains its own project metadata and provenance receipt.
The default capture limit is 1 GiB and can be reduced in the plugin configuration.
Environment values under secret-like keys and absolute-path values are replaced with `[redacted]`
before idempotency hashing, Journal commit, projection materialization, or Remote return.

Figure PNG/SVG/PDF capture can add one format-standard `dsh-science.provenance` record containing
exact plotting code, code hash, supported library, redacted runtime, generation request id, and
normalized workspace-relative/Artifact/S3 source references. The same capture postprocessor handles
matplotlib and ggplot2 output without modifying the workspace file. Injection is enabled by default
for declared Notebook Figures, may be supplied explicitly for standalone registration, and can be
disabled globally with `embedArtifactMetadata: false` or per artifact with
`reproducibilityMetadata: false`. Absolute source paths are never accepted. See
`../../../docs/reproducibility-metadata.md`.

Storage migration v1 owns project/notebook facts and projections. Migration v2 adds the artifact
projection and digest index. Migration v3 adds the document projection, migration v4 adds the
figure projection, and migration v5 adds operational research facts, Experiment/Run, relation, and export
projections; every mount rebuilds all projections from the append-only Journal.
The project Research Object is generated on read from those bounded projections as one validated,
flat RO-Crate 1.3 graph. It follows explicit source identities, remains restricted to the live
Session workspace, and rejects an oversized, duplicate, or dangling graph instead of truncating it.

The Host derives an opaque workspace key from the addressed live DSH session. Requests and
responses never carry a filesystem path. A caller-provided UUID is the idempotency key; reusing it
with different input is rejected.

The selected runtime's native Session or Thread remains the interaction truth. Science Journal events contain scientific
facts only and never enter Chat, Trajectory, or model history.

ID-addressed reads use canonical typed `sx:` logical IDs and revision-guarded exact IDs. The
`science_query` actions `head`, `batch_head`, `get`, `select`, and `neighbors` resolve only within the
live authorized workspace and return bounded refs/projections rather than a workspace snapshot or full
entity. Artifact selection is limited to the existing verified 64 KiB/500-row preview, with at most
100 returned table rows or 16 KiB text per call. See `../../../docs/resource-addressing.md`.

No `dsh-memory` package or stable memory service is present in the current profile. Science therefore
does not invent a memory adapter: its stable integration surfaces are the client-safe contracts,
the platform-neutral `ScienceCore`, strict DSH Typert descriptors, the Codex MCP carrier, and the
selected runtime's bounded Tool result policy.
