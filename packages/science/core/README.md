# `@swarmx/dsh-science`

Local-first scientific domain service for SwarmX. T13 establishes the first end-to-end slice:

- `ctx.science.createProject(sessionId, request, signal?)`
- `ctx.science.createNotebook(sessionId, request, signal?)`
- `ctx.science.getWorkspace(sessionId, signal?)`
- append-only SQLite Science Journal in WAL mode
- deterministic project/notebook projection replay
- strict Typert Host and Client Remote descriptors

T14 extends that same ownership boundary with:

- `ctx.science.registerArtifact(sessionId, request, signal?)`
- `ctx.science.traceProvenance(sessionId, request, signal?)`
- generic immutable objects addressed by streamed SHA256 content digests
- metadata-only Journal events and a replayable artifact projection
- bounded artifact → source entity → project provenance traversal

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

T27 closes the first user-visible Files-to-Notebook loop:

- `ctx.science.importArtifact(sessionId, request, signal?)` accepts one canonical-base64 browser
  file up to 8 MiB, infers kind/MIME from a path-free supported filename, and returns metadata only
- imported bytes enter the same owner-only content-addressed object store; encoded bytes and host
  paths enter neither the idempotency record, Journal event, projection, nor Remote response
- `executeNotebookCell` may address up to four workspace-local input artifacts totaling 32 MiB;
  verified owner-only temporary copies are exposed only to controller-owned kernel setup as
  `DSH_SCIENCE_INPUT_<n>` and removed after success, failure, cancellation, or disposal
- replayable Notebook cells retain exact input artifact ids, and provenance reports Notebook
  `uses` edges without exposing a materialized path

T16 adds the first Writing Studio revision path:

- `ctx.science.createDocument(sessionId, request, signal?)`
- `ctx.science.modifyDocument(sessionId, request, signal?)`
- logical `.typ`, `.tex`, `.md`, and `.bib` documents with bounded client-safe source
- textarea-compatible UTF-16 source selections and exact-revision conflict rejection
- append-only propose/accept/reject facts preserving original text, proposed text, reasoning summary,
  resolution, source hashes, and provenance receipts
- deterministic structural, missing-figure-reference, and unsupported-claim diagnostics

Writing diagnostics are local heuristics, not compiler results. Every document explicitly reports
`structural: checked` and `compilation: not-run`; callers must run a real Typst/LaTeX compiler before
claiming compilability or rendered validity.

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

- `ctx.science.createQuestion/createHypothesis/recordClaim/linkEvidence` for bounded Research Map
  facts and typed scientific relations
- `ctx.science.defineExperiment/startRun/finishRun/compareRuns` for exact-revision Experiment and
  Run lifecycle records, redacted environments, metrics, and artifact association
- `ctx.science.exportProject` for deterministic JSON stored as a content-addressed object while the
  Journal retains only digest, byte count, entity counts, and provenance
- exactly seven aggregate tools from `@swarmx/dsh-science/tools`: `science_notebook`,
  `science_write`, `science_figure`, `science_experiment`, `science_record`, `science_query`, and
  `science_export`
- `runScienceDemo(ctx.science, sessionId, signal?)`, an executable local-only tour covering Notebook,
  Experiment, Figure, Writing, Research Map, provenance, and export

Every successful tool result distinguishes `fact`, `inference`, or `proposal` and carries a locator
with the DSH session, tool call, Science entity, and Journal sequence. The ordinary DSH tool events
therefore associate agent operations with scientific facts in Trajectory without reading private
reasoning or modifying the agent loop. Harness rc.8 does not publish a Trajectory anchor-routing
extension, so the locator is searchable in Science Workspace but is not rendered as a custom
clickable Trajectory anchor. Oversized textual results use the Web profile's existing
session-scoped spill policy.

The Host defaults to the configured local `jupymcp` executable, 256 KiB per cell output, a 5 MiB
Notebook resource bound, and a 60-second MCP request timeout. It uses the same official SDK and
scrubbed child environment precedent as the Harness MCP client, additionally removes Python
injection and proxy variables, and never returns an executable, Notebook, or kernel path.
The controller is validated against JupyMCP 0.3.1's native dependency set (`mcp>=2,<3` plus
`ipykernel`); Science adds no MCP downgrade or kernel-package compatibility pin.

The legacy `isolated` mode defaults to `python3` and a two-second process termination grace. It is
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

Storage migration v1 owns project/notebook facts and projections. Migration v2 adds the artifact
projection and digest index. Migration v3 adds the document projection, migration v4 adds the
figure projection, and migration v5 adds Research Map, Experiment/Run, relation, and export
projections; every mount rebuilds all projections from the append-only Journal.
Provenance traversal follows explicit artifact sources and notebook ownership, is restricted to the
live session workspace, includes document ownership and revision events, and caps depth at 20,
entities at 200, and relations at 400.

The Host derives an opaque workspace key from the addressed live DSH session. Requests and
responses never carry a filesystem path. A caller-provided UUID is the idempotency key; reusing it
with different input is rejected.

The DSH session log remains the interaction truth. Science Journal events contain scientific
facts only and never enter Chat, Trajectory, or model history.

No `dsh-memory` package or stable memory service is present in the rc.8 profile. Science therefore
does not invent a memory adapter: its stable integration surfaces are the client-safe contracts,
strict Typert Remote descriptors, `ctx.science`, DSH session ids, `ctx.subprocess`, `ctx.tools`, and
the profile spill policy.
