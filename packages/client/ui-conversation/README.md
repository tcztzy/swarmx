# `@swarmx/dsh-ui-conversation`

Browser conversation extensions for SwarmX: non-destructive Retry/Edit, annotation-aware user
messages, and the generic Side View.

First-turn Retry/Edit uses alpha.2's session object layer to create an immediately addressable
sibling in the source Workspace, preserving its agent preset and adopting its cwd when ungrouped.

User messages shadow the upstream renderer only at a lower slot priority. The renderer extracts
model-only `<dsh-annotation>` transport, presents bounded annotation cards, and renders the remaining
body with DSH's safe Markdown renderer. Copy produces readable Markdown rather than JSON. Persisted
`<annotation>` and legacy Science tags are recovered only for display; session history is never
rewritten.

Selecting bounded text inside a durable user, steering, or finalized assistant Chat row opens one
compact localized `Add to chat` action beside the selection. The first selection is captured
immediately; later selections open a PDF-style optional-note editor and Enter confirms. Detached
references leave visible draft text untouched and render as one count control above the composer.
Its popup lists numbered selections in insertion order; row hover/focus reveals functional Edit and
Remove controls. The annotation records the source Session and durable message locator separately
from the destination composer Session, so today's same-Session action and a future cross-Session
picker use the same persisted payload. Selections spanning rows, Tool/context rows, and
streaming-only assistant text remain inert. All annotation UI copy comes from complete Chinese and
English dictionaries.

Side View reuses the installed DeepSeek Harness baseline's published `details` column and
`ctx.layout` geometry. Its
client-only `ctx.sideView` service stores JSON-serializable entry descriptors per Session, while the
keyed `side-view.content` slot keeps React ownership in the contributing UI plugin. Opening the same
entry id updates and activates its existing tab; closing the visible column preserves its tabs.

Science registers keyed artifact and paper renderers that pass metadata/provenance only, never
artifact bytes or host paths. These routes replace neither Chat, the composer, session state, nor
Trajectory.

The generic completed-turn contribution declares `conversation.chat.turnTail.items` as an additive
list because DSH's upstream turn-tail is a first-match chain. It renders only explicitly registered
children. Science registers its Generated card group there; SwarmX derives no generic Tool action
from Bash or any other completed call.

`inspect` opens at the upstream 360px default; `workbench` requests 880px only when the details
column is closed. SwarmX's exact-baseline package-manager patch adds that optional preferred-width
argument and removes the fixed 520px ceiling from both the drag store and concession solver. A user
drag while the panel is open remains authoritative. On narrow windows the existing 300px details
floor, 640px center floor, and shrink/auto-close chain still win.
