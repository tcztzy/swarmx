# `@swarmx/dsh-ui-science`

Browser half of the T13 Science Workspace slice. It mounts the strict Science Remote contribution
and registers one additive `conversation.view` tab. It does not read or replace Chat or Trajectory
snapshots.

The view registration lives inside a Cordis `remote.science` injection scope, so every workspace
action runs with an active Remote namespace and the view withdraws with that capability.

The initial view exposes Notebook, Writing, Figures, Research Map, and Experiments, with working
project/notebook creation backed by the Host Science Journal. Later tasks fill the other studios.

T14 adds read-only artifact metadata to the populated workspace. File capture stays on the Host
through `ctx.science.registerArtifact`; the browser receives digest, media type, size, scientific
metadata, and provenance only—never a host path or artifact bytes.

T15 makes Notebook the first executable studio: once a notebook exists, the view exposes a labeled
Python source editor, runs the cell through the strict `executeNotebookCell` Remote method, reloads
the durable projection, and renders code/output cells with execution numbers. An in-flight cell is
aborted if the view unmounts; controller ownership and output bounds remain on the Host.

T16 makes Writing the second working studio. A project can create one of the supported logical
source documents, select source with native UTF-16 textarea offsets, prepare a proposed replacement
with a bounded reasoning summary, and accept or reject pending proposals. The view renders durable
revision/provenance metadata plus structural and scientific warnings. It labels compilation as not
run so local heuristics cannot be mistaken for a successful Typst or LaTeX build.

T17 makes Figures the third working studio. It creates code figures for the four supported plotting
libraries, renders the durable semantic object map as a keyboard-accessible local canvas, supports
single selection and additive Shift/Command/Ctrl brush selection, and sends object-linked code patch
proposals through strict Remote methods. The browser displays patch provenance and accept/reject
controls; it neither executes plotting code nor receives host artifact paths or bytes.

T18 makes Research Map and Experiments working destinations. Research Map renders bounded facts and
typed relations and provides a keyboard-accessible entity id/title/tag search field that accepts the
entity id from a Science tool locator. Experiments can be defined, started, finished, and inspected;
the project can be downloaded as deterministic JSON. All failures enter the same visible retryable
error state, and the browser receives no host path.

T19 connects artifact locators to the generic Side View. The artifact list opens a deduplicated,
Session-local right-column tab through `ctx.sideView`; Science owns only the keyed
`science-artifact` renderer. The entry carries bounded metadata, digest, revision, Run association,
source entity ids, and Journal sequence—never bytes or a host path—so the Science Workspace remains
mounted and keeps its editor state while the user inspects provenance.

T20 adds Files preview without widening the rc.8 concession. The Host authorizes artifact ids against
the requesting Session workspace, verifies immutable bytes, and returns only bounded text, bounded
PNG/JPEG/GIF/WebP data URLs, or a typed unavailable reason. Tool Details recognizes strict Science
artifact locators and opens the same tab. “Open in Science” retains a per-Session fullscreen target:
it focuses immediately when Science is mounted, otherwise waits for the user to select the Science
tab because rc.8 publishes no programmatic conversation-view switch. Chat stores, drafts, and scroll
memory remain untouched.

T24 replaces the dashboard with a compact Claude-Science-style project rail and full workbench.
T27 makes its Files section operational: keyboard file selection and drag/drop admit only supported
files up to 8 MiB, show independent importing/error state, and reuse the existing artifact Side View
preview. “Analyze” switches only the Science destination, creates a Notebook when needed, runs one
deterministic extension-aware inspection against the selected immutable artifact, and captures the
JSON analysis back into Files. Chat, Trajectory, composer draft, scroll memory, and the agent loop
remain unchanged.

T28 parses bounded CSV/TSV and scalar-record JSON on the Host with Papa Parse/native JSON, preserving
blank and duplicate headings in positional rows. Side View renders the typed result with AG Grid
Community, including sorting, filtering, column resizing, and bounded pagination.

T29 makes configured local JupyMCP the default Notebook Controller: each workspace Notebook owns one
MCP stdio process and persistent Jupyter kernel, while the legacy fresh-process runtime is available
only as an explicit isolated mode. T30 adapts the ordered client-safe MIME contract into JupyterLab's
untrusted `OutputArea`/`RenderMimeRegistry`, with canonical `.ipynb` output types and a visible
plain-text fallback if the rich renderer cannot mount.
