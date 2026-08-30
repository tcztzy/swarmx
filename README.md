# SwarmX

An Electron desktop application that keeps [DeepSeek Harness Web](https://github.com/deepseek-ai/deepseek-harness) as its only UI and plugin host, with DSH and Codex App Server as peer conversation runtimes below it.

Each runtime owns its native transcript, agent loop, tools, approvals, credentials, and persistence.
SwarmX contributes the desktop lifecycle, a narrow runtime protocol inside DSH Web,
runtime-native Retry/Edit actions, a generic Side View, and local-first Science tools whose
artifacts stay inside the conversation flow.

## How it works

DSH's browser surface is a profile bundle that serves the complete application UI over HTTP.
SwarmX composes its extensions after DSH's bundles, boots the result **inside the Electron main
process**, registers DSH/Codex adapters as a Host Cordis service on that same listener, suppresses
the profile's normal system-browser handoff, and points one Electron window at the bound port:

```
Electron main ──boot()──> DSH Web + SwarmX Cordis layers ──HTTP/127.0.0.1──> BrowserWindow
                              │
                              ├── DSH Session/Agent runtime
                              └── Codex app-server over stdio
```

Consequences of that shape:

- **No replacement renderer.** Published `dsh-client-ui-*` packages remain the
  baseline; local extensions enter through public conversation and details slots,
  with two exact-version package patches kept explicit under `patches/`.
  One exact Web-server route supplies the trusted Markdown file-link seam absent
  from rc.2's prebuilt frontend and refuses to start if that upstream seam drifts.
- **No second browser transport.** Runtime lifecycle routes are registered under
  `/api/swarmx/conversation-runtimes` on DSH Web; the browser never connects to app-server directly.
- **No double harness.** Codex runs as its own app-server process below the UI, never inside the DSH Agent loop.
- **One Conversation seat.** DSH selection keeps the published DSH Conversation unchanged. Codex
  selection shadows only that DSH Web slot with the runtime-neutral protocol client; the DSH app
  frame and plugin host remain the sole application UI.

The desktop host keeps the UI and runtime composition explicit:

| File | Role |
| --- | --- |
| `src/harness.ts` | Compose the profile's patch layers, boot, report the bound URL |
| `src/markdown-file-links.ts` | Add trusted file-link resolution to the one rc.2 frontend asset |
| `src/window.ts` | Create the window, fence navigation to the harness origin |
| `src/main.ts` | Sequence startup and shutdown |
| `src/runtime/platform.ts` | Register peer runtimes under the DSH Web host |
| `src/runtime/web-plugin.ts` | Mount the bounded runtime protocol on the existing listener |

`packages/client/ui-conversation` owns the Retry/Edit extension. Edit is an icon on
each eligible user message, following its timestamp and Copy action in one
non-overlapping visual sequence. A failed turn keeps its error text visible and adds
a Retry icon even when no final assistant message exists. Retry prepares a
separate session before the selected turn and sends the original text; Edit
opens that session with the original prompt as a draft. Later turns use DSH's
fork primitive; the first turn uses a fresh session in the same Workspace
(adopting the source cwd when it was ungrouped) because DSH cannot fork an empty
completed-turn prefix. Neither action mutates the source history.

The same client plugin owns a per-Session Side View that reuses DSH's draggable right details
column. Science artifacts and Typst papers open as deduplicated tabs through serializable locators;
their React content remains owned by keyed slots. Opening or closing it does not replace Chat,
Trajectory, the composer, or message scroll state. Science registers no peer tab beside Chat and
Trajectory. Instead, successful artifact-producing Science calls render a Claude-style `GENERATED`
card group beneath the closing answer; clicking a card opens that asset in DetailsPanel.

Artifact tabs request only Host-authorized bounded previews. A Science workbench entry requests an
880px initial panel width while `inspect` entries keep the 360px default; user drag, the
300px details floor, the 640px conversation floor, and narrow-screen concession remain authoritative.
The panel uses the Claude Science-style hierarchy: Side View tabs, filename/action bar, large edge-to-edge
preview, on-demand Provenance, and collapsed file details. Image clicks create numbered point
comments; saving one inserts a structured reference into the existing Chat composer, and the model
can resolve the verified image plus normalized point through `science_query.inspect_annotation`.

Produced or explicitly referenced `.typ` and `.typst` papers appear in one completed-turn `Files` row
and open a live dsh-science paper workbench in that same right column. A header toggle switches
between conflict-checked source editing and a selectable multi-page PDF.js preview compiled by a
managed local `typst watch`. PDF text selections and figure points can be added to the current Chat as
structured annotations; double-activating a figure opens a deduplicated image tab for
conversation-led editing.

Final answers prefer standard workspace-relative Markdown links such as
`[paper.typ](./papers/paper.typ)`. The Science plugin also recovers safe Typst references from closing
prose and Tool arguments during history replay, so Bash writes and legacy inline-code paths retain a
stable entry even when the model omits Markdown syntax. Opening a Typst entry registers it with the
Host watcher. In the `dsh-science` preset, the model edits source but never spends a tool call on
`typst compile` or `typst watch`.

DSH Retry/Edit remain append-only branch operations. Codex threads created by
SwarmX use legacy history by default because standalone App Server builds may
advertise paginated schemas before implementing their reads. Setting
`SWARMX_CODEX_PAGINATED_HISTORY=1` explicitly enables paginated history: editing
or retrying the latest terminal turn reverts that turn in the same Codex thread
and then starts the replacement; older turns and legacy Codex threads still
branch. The Edit icon only fills the composer—no native history changes until the revised draft is submitted.
Codex history revert does not roll back file edits or commands already applied
to the workspace.

`@swarmx/dsh-science` and `@swarmx/dsh-ui-science` add an AI-native scientific IDE through
published Harness seams without introducing a second workspace. Project, Notebook, Writing, Figure,
operational research facts, Experiment, and export operations are agent-mediated through the seven
aggregate Science tools in the system-owned `dsh-science` preset. That mode carries the complete
locked `standard` composition plus Science tools, annotation/literature guidance, and managed-Typst
protection; other presets keep the Host service and artifact UI but do not expose Science model
capabilities. Project research graphs and exports use RO-Crate 1.3 rather than a SwarmX
vocabulary. A separate `literature_search` tool searches the running local Zotero library through
an owner-only BibTeX exchange snapshot; it neither contacts a cloud index nor reads attachment
paths. Notebook work executes through one persistent Jupyter kernel per document in the
configured local JupyMCP controller. Science Journal stores domain facts separately from the DSH
session log, while results return through ordinary Tool/Chat rows and generated artifact cards; Chat,
Trajectory, the composer, and the agent loop remain the product's single interaction surface.

`science_query` can progressively read local entities through canonical typed `sx:` IDs: start with
one bounded `head`, reuse its revision-guarded `exactId`, and request only metadata, a verified
Artifact preview window, or typed neighbors when needed. See
[`docs/resource-addressing.md`](docs/resource-addressing.md).

## Requirements

Node `^22.19.0 || >=24.0.0`.

## Running

```shell
pnpm install
pnpm dev
```

`pnpm dev` reuses the canonical `pnpm start` chain: it builds the current source,
boots Harness, and opens the Electron app.

DSH is the default conversation runtime. Select the Codex peer explicitly with either form:

```shell
pnpm start -- --runtime codex
SWARMX_RUNTIME=codex pnpm start
```

Codex mode requires an installed, authenticated `codex` CLI. SwarmX still boots the same DSH Web
application, starts `codex app-server` over stdio behind the Host, and loads no second URL.

First run creates `~/.dsh/profiles/web/` from DSH's shipped template. A model
provider must be configured before the harness can answer; do that in the app's
own settings, or in `~/.dsh/settings.yaml`. Select **Science mode / 科学模式** (`dsh-science`)
for the Science tool surface; standard, PTC, minimal, and creation modes remain unchanged.

## Customizing the UI

### Conversation action icons

`packages/client/ui-conversation/src/client/icons.ts` is the only source-level
mapping for SwarmX conversation icons. It maps the semantic `edit` and `retry`
actions to 16px React icon components. The defaults reuse DSH primitives so
size, `currentColor`, hover, and accessibility behavior stay native; replacing
either mapping value does not require changing the action components, CSS, or
client bundler.

### Harness profile

Every adjustment happens in the profile's patch layer —
`~/.dsh/profiles/web/cordis.patch.yml` — without touching this repository. A
patch replaces the targeted row's whole `config`, so restate the fields you
keep.

Remove a UI domain:

```yaml
- id: ui-workflow-run
  disabled: true
```

Add your own plugin row:

```yaml
- insert:
    - id: my-panel
      name: '@me/dsh-client-ui-my-panel'
```

Profile bundles can also be installed through DSH's published plugin command.
SwarmX resolves the installed bundle from the active profile before falling
back to its own dependency graph, including in Electron where Node's internal
ESM loader is unavailable. For example, after cloning and building
[dsh-cowork](https://github.com/Jesse-njx/dsh-cowork):

```shell
pnpm --filter @swarmx/desktop exec dsh plugin --profile web add /absolute/path/to/dsh-cowork/packages/dsh
```

Restart SwarmX and the model receives Cowork's bounded `doc_read` for xlsx,
ipynb, PDF, docx, and pptx plus atomic `doc_write` for xlsx and ipynb.

Within a client plugin, `ctx.slots.register` contributes a component into a
declared slot, and a lower `priority` shadows an existing occupant without
modifying the package that registered it. Recolor by overriding the `--dsw-*`
alias tokens `dsh-client-ui-theme` publishes.

See DSH's own documentation for the slot contract and the cookbooks.

## Security boundary

The renderer is remote content: no preload, no Node integration, context
isolation and sandbox on. The server binds loopback on an OS-assigned port.
Navigation away from that origin is cancelled. Only `http:` and `https:` links
are handed to the OS browser; local files, scripts, and custom protocols are
blocked. Chromium permission requests are denied unless SwarmX explicitly adds
and tests a capability, so model-rendered content cannot acquire camera,
microphone, location, notification, or similar browser authority.

## License

MIT
