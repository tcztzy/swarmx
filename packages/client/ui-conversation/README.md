# `@swarmx/dsh-ui-conversation`

Browser conversation extensions for SwarmX: non-destructive Retry/Edit, annotation-aware user
messages, and the generic Side View.

User messages shadow the upstream renderer only at a lower slot priority. The renderer extracts the
single model-only `<annotation>` transport, presents bounded annotation cards, and renders the
remaining body with DSH's safe Markdown renderer. Copy produces readable Markdown rather than JSON.
Persisted legacy Science annotation tags are recovered only for display; session history is never
rewritten.

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
