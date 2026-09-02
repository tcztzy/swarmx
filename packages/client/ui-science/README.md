# `@swarmx/dsh-ui-science`

Browser half of the local-first Science product. It mounts the strict Science Remote contribution
but deliberately registers no `conversation.view`: Chat and Trajectory remain the only peer views.
Science project, Notebook, Writing, Figure, operational research facts, Experiment, and export work
stays agent-mediated through the seven aggregate tools. Project reads and exports use RO-Crate 1.3.
Local publication discovery uses the separate
`literature_search` tool and its Zotero-to-BibTeX Host boundary; the browser owns no literature API.

A completed Chat turn containing valid same-Session Science artifact results renders one Claude-style
`GENERATED · n` card group beneath the closing answer. Image cards load bounded Host-authorized
thumbnails; other assets retain a complete filename/type identity. The entire card is keyboard
operable and opens one deduplicated artifact tab in the existing DetailsPanel.

The DetailsPanel follows the Claude Science-style hierarchy: a Side View tab strip, compact filename and
actions bar, large edge-to-edge preview, on-demand RO-Crate provenance from More, and collapsed file details.
Science workbench entries request an 880px initial panel width. User drag and the existing viewport
concession solver remain authoritative. The Host verifies every artifact digest and returns only
bounded text, table, or image previews; host paths and raw unbounded bytes never reach the browser.

Image previews support Claude Science-style point annotations. Clicking the rendered image opens a
bounded comment editor; saved comments become numbered pins and are inserted into the existing Chat
composer as the generic `annotation` reference. Its `comment.target.type=image_point` payload
preserves the user's draft and contains only a verified artifact locator, normalized point, and
comment. When the model needs pixels it calls the existing aggregate `science_query` tool with that
complete comment object; the Host re-authorizes the artifact and returns a durable image attachment
rather than exposing a browser data URL or host path.

CSV/TSV and scalar-record JSON previews are parsed on the Host into a bounded typed table and rendered
with AG Grid Community. Text and image previews remain bounded. No fullscreen Science route, delayed
view handoff, duplicate project shell, or browser Notebook renderer exists.

Typst papers are a first-class artifact route rather than a generic text preview. A safe
workspace-relative `.typ` or `.typst` path produced or explicitly referenced in a completed Chat turn
appears in the same `Files` row. Activating it opens the same DetailsPanel and starts an authorized
Host-side semantic Typst watcher. The paper header switches between an optimistic, conflict-checked
source editor and a multi-page PDF.js viewer. The viewer retains the last successful PDF while a
newer source revision is compiling or invalid and exposes the current compiler diagnostics.
Each paper path owns a separate workbench lifecycle, so switching between Typst tabs cannot carry an
unsaved draft, imported-source caret, conflict, or inverse-search request into another paper.

The Host-global deliverable contract asks every preset to cite completed output files as Markdown
inline code, using the exact workspace-relative path or a unique basename. The plugin resolves those
tokens only against the closing turn's mutation evidence or explicit safe Typst references in
Assistant prose and Tool arguments. History replay rebuilds the same ordered deduplicated row, so a
Bash-authored paper does not depend on the model choosing a particular final-answer path spelling.
Absolute and traversal candidates remain inert, and the Host authorizes the file when the workbench
opens. The Host watcher owns initial compilation and every later source change. Only the
`dsh-science` preset adds annotation/literature guidance and denies a model-requested Bash
`typst compile` or `typst watch`; sibling presets receive neither Science tools nor that
Science-specific contract.

PDF.js text layers make rendered prose selectable. Adding a selection to Chat preserves the current
draft and inserts the same structured `annotation` reference with a
`comment.target.type=document_text` payload carrying only the paper-relative path, page, normalized
rectangle, selected text, comment, and source/PDF revisions. Figure regions use
`document_region` and open as deduplicated Side View tabs on double activation so the same
conversation-led image editing flow can operate without browser popups or unbounded PDF bytes.

An ordinary single click on PDF.js text performs Typst inverse search. The browser sends only the
page, normalized click point, and exact displayed PDF revision; the Host resolves it against the same
semantic snapshot that produced the PDF. On success the existing workbench switches to Source,
opens the owning workspace-relative `.typ` file (including imported files), focuses its exact UTF-16
caret, and reveals it. A stale PDF/source revision is shown as an error and never moves the caret.
Dragging or double-clicking a text selection still opens the annotation flow, while figure clicks and
double-clicks retain their annotation and dedicated-tab behavior.

The alpha.4 conversation projection and file-mention resolver are used directly. This plugin
registers its deliverables node through `uiConversation.events`, reconstructs mutating file-tool paths
from validated call arguments, and delegates ordinary file opens to
`remote.session.openWorkspacePath`. Only safe `.typ`/`.typst` references extend the vocabulary and
enter the paper workbench.
