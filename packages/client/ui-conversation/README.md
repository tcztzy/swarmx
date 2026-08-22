# `@swarmx/dsh-ui-conversation`

Browser conversation extensions for SwarmX: non-destructive Retry/Edit plus the generic Side View.

Side View reuses DeepSeek Harness rc.8's published `details` column and `ctx.layout` geometry. Its
client-only `ctx.sideView` service stores JSON-serializable entry descriptors per Session, while the
keyed `side-view.content` slot keeps React ownership in the contributing UI plugin. Opening the same
entry id updates and activates its existing tab; closing the visible column preserves its tabs.

The first content route finds Tool calls through the public Conversation snapshot and delegates Tool
output to the existing `conversation.details.tool` slot. Science registers a separate
`science-artifact` renderer and passes metadata/provenance only, never artifact bytes or host paths.
Neither route replaces Chat, the composer, session state, or Trajectory.

Tool Details also declares the additive `side-view.tool.actions` list. Science uses it to recognize
only strict, same-Session artifact locators emitted by its seven aggregate tools and open the same
deduplicated artifact tab; malformed or cross-Session results render no action.

`inspect` and `workbench` open at the upstream 360px default and keep the 300px floor. SwarmX's exact
rc.8 package-manager patch removes the fixed 520px ceiling from both the drag store and concession
solver, so the panel can consume all viewport space left after the sidebar and 640px center floor.
On narrow windows the original shrink/auto-close chain still wins, and the preferred width returns
when the window widens.

The remaining upstream boundary is an optional preferred-width policy on the existing verb, for example
`openDetails({ mode: "inspect" | "workbench" })`. Layout keeps today's inspect defaults; workbench
can select a wider initial preference while the same center-minimum/narrow-concession solver remains
authoritative. No new panel component or parallel layout service is required.
