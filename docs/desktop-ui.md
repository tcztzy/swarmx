# Desktop interface

SwarmX uses a Codex-inspired monochrome workspace: a light task sidebar, compact header,
readable conversation column and rounded composer. The sidebar can be hidden and remains
available in narrow windows. Search filters the selected Agent's native session titles.
New task creates a native session; refresh reloads native titles without replacing the active
conversation. Agent changes load that Agent's sessions; late responses cannot replace the
current selection. Connection, session creation and history failures are visible in the UI.

assistant-ui owns messages, Markdown (including GFM tables and task lists), composer state,
draft suggestions, copy feedback, scroll-to-latest, streaming and Stop. Starter prompts only fill the draft; sending remains
explicit. Send and Stop are mutually exclusive. History loading and pending interactions
disable new sends. Approval and question forms use the existing AG-UI interrupt/resume path;
required text and numeric fields are validated before submission. A boolean permission can
be explicitly declined. Failed and cancelled tool calls must not appear successful.
Completed tool-only messages show neither an ongoing-work indicator nor an empty copy action.
Switching tasks or Agents closes the active stream and stops that run, as in the existing Host
contract. The running-state hint makes this visible; background execution is not implied.

Swarm delegations appear in an expandable sub-Agent list following assistant-ui's
[SubagentList](https://www.assistant-ui.com/elements/subagent-list) and
[SpeakerIdentity](https://www.assistant-ui.com/elements/speaker-identity) patterns. Each execution
shows its dispatched task, parent, Harness, requested/reported model and independent status;
parallel children may finish in any order. Expanding a card shows the observed conversation for
that run and its original execution records. This is a projection of the persistent journal,
including nested delegations, and remains readable after restart. Missing terminal records are
unknown unless the Host still owns that active run. No percentage or successful outcome is inferred.
Active children accept additional instructions and Stop through their native Agent, targeted by
execution ID. Pending child interactions use the main conversation's existing confirmation form;
concurrent requests are queued. Child controls remain disabled until confirmation is answered or
cancelled. Reading child details preserves the parent draft and does not switch native sessions.
Only delegations with recorded causal links are shown; hidden Harness-internal agents are not inferred.

The execution trace is an optional side pane, closed initially, with the existing react-o11y
waterfall. It is a transient view. Native transcripts own history hydration and resume; the Host
execution journal independently retains observed operations (see `execution-log.md`). No browser
transcript store or new transport is introduced. Search and sidebar/trace visibility are local
presentation state.

The composer provides a Harness menu and a combined model/Thinking popover following the
[assistant-ui model selector](https://www.assistant-ui.com/elements/model-selector): model rows,
an active-model check, and a separated Thinking row of effort segments. The trigger shows the
model and current effort together. Selecting a model closes the popover; selecting an effort
keeps it open. Keyboard navigation uses cmdk for models and Radix RadioGroup for effort levels.
The list scrolls independently and the effort row wraps when needed on narrow windows.
Harness selects
the native runtime (Codex, Claude, Hermes or OpenClaw); the default Swarm shows its lead's
Harness. Switching Harness retains the existing native-session ownership rule. Model and
effort changes preserve the current conversation and draft, and apply on the next send.
Controls remain disabled while streaming or waiting for an interaction response.
Harness is also locked while creating a session. History failures keep Harness available
so another native runtime can be selected.

The Host reads the selected Harness's model catalog lazily. Effort segments reflect supported
levels such as Low, Med, High, XHigh and Max, including additional native levels when advertised.
A selected effort is retained across model switches when supported; otherwise that model's
advertised default applies. Models without configurable thinking hide the row and send no effort.
Switching back restores a previously selected compatible effort. Empty catalogs and failures
remain visible and never become fabricated choices.
assistant-ui's model context forwards `modelName` and `reasoningEffort` through AG-UI
`forwardedProps`; the Host validates these fields and maps them to native run settings.
Native configuration stays authoritative, and global vendor settings are not rewritten.

Codex uses `model/list` and `turn/start`; Claude uses SDK `supportedModels()` and query
options; OpenClaw uses `models.list` and session-scoped `sessions.patch` before `chat.send`.
When OpenClaw omits `thinkingLevels`, the Host looks up the exact provider/model in
[`@earendil-works/pi-ai`](https://github.com/earendil-works/pi/tree/main/packages/ai) and uses its
supported thinking levels. An explicit native list, including an empty list, takes precedence.
Unknown or non-reasoning models receive no invented levels. The catalog lookup is local and
does not load pi-ai in the Renderer or change the Harness used for inference.
Hermes uses `model.options` and `config.set` with explicit session scope. Its catalog does
not enumerate supported reasoning levels or expose the corresponding setting, so the Thinking row stays hidden; native
model-change confirmations use the existing interaction form. A failed setting change must
not send the prompt, and Stop during a pending settings change must not start a new run.

Composer choices represent the next run's requested settings, not proof of the model used by
a previous response. Native transcripts and native event metadata remain the factual record;
the Host snapshots dispatched settings in its execution journal. Unsent menu selections are not logged.

The existing Host session endpoints and AG-UI adapter provide the interface capabilities.
Rename/archive, attachments, file review,
terminal panels and concurrent background tasks need additional end-to-end contracts and
are not exposed as nonfunctional controls.

The conversation is the primary workspace. The header always exposes Assets and Observe;
both open a side view without replacing the conversation, its draft or active stream.
Switching between Assets and Observe also retains the open figure editor and its running job.
Assets shows scientific files and figures at useful preview size; manual source editing is
optional. Users can ask the assistant to work on a selected artifact from the same composer.
Observe groups the react-o11y waterfall, recorded scientific runs and the RO-Crate graph/JSON-LD.
Its entry remains visible before the first run, with an explicit empty state. Tool result cards
open their referenced artifact or project in Assets. Narrow windows show a closable side view.
The sidebar lists saved projects, each with its own canonical directory and native tasks. Add project
registers an existing directory; duplicate canonical directories reuse the existing project. Opening
a project uses a project-specific URL. Other windows and active executions retain their original project.
Project settings are opened beside the selected project; its directory is an identity, not a mutable global preference.
The bottom-left profile row opens global Settings for language and user preferences; Back
restores the same mounted conversation. Research collections, research-question records, bounded data/image imports,
artifact previews, execution history and RO-Crate export use existing Science contracts.
Notebook history reads the latest 100 execution facts from the Science journal, including failed
cells, exit status and output. A code cell's source is selected explicitly; the following output
cell is never presented as editable Python or counted as another execution.

The side view presents an artifact gallery, a relation graph, runs and the JSON-LD
document. Search and selected-node neighborhoods keep the graph readable; at most 200 matched
entities render at once. React Flow owns pan/zoom/fit and keyboard navigation. The details pane
shows the original identifier, content hash, source code, environment and relations. The graph
projects recorded references and never invents scientific relationships or permits visual edits
that would bypass the journal.

The figure workbench edits Python beside a result preview and execution output. The initial
example is explicitly illustrative. Users can select up to four immutable inputs, execute,
stop, correct errors and generate a new artifact while retaining earlier outputs. Imported
raster images use Pillow; plots use the recorded source. Existing source without a recorded
output path requires an explicit path before Run,
so an unrelated pre-existing `figure.png` cannot be silently chosen by the editor's default.
The initial preview shows the selected immutable artifact.
This is executable scientific figure generation/editing, not a configured generative-image API.
PNG/SVG preview in the workspace; PDF
and larger files download for inspection. RO-Crate exports are metadata documents; users download
the artifact files separately. The UI does not imply that a metadata export is a self-contained
research-data archive.

The project settings page exposes its canonical work directory,
Host tool/delegation grants and research-container file/resource limits. Native modes are chosen
in each conversation using upstream labels; model/effort changes preserve that choice. Environment setup streams bounded build logs,
supports cancellation and verification, and exports the resolved manifest and dependency lock.
Setup is an explicit networked operation; figure execution uses the configured image offline.
Failures remain actionable. Policy changes apply only after current executions have ended.

English and Simplified Chinese cover navigation, chat controls, approvals, asset/editor views,
observability and settings, including accessible labels and empty states. i18next owns lookup
and interpolation. The first launch follows the browser language (English for unsupported
languages); Settings persists an explicit choice in the private product home and updates the UI
immediately. Bootstrap restores it across Host restarts, loopback port changes and workspaces.
Document language, dates and numbers follow that choice. Scientific data, code, native model
names, protocol identifiers and original execution/error payloads remain untranslated.

General Settings edit shared user preferences. Project Settings include Memory and knowledge:
bounded project notes, automatic review settings,
current-conversation review, persistent pending approvals, and a searchable Vault dependency graph.
Selecting a concept loads its prerequisite closure in order; changed prerequisite revisions are
marked for review. Conflicting edits stay pending and display the error. The graph uses the same
React Flow view as research provenance, with read-only edges. See `memory.md` for storage and authority.

Acceptance is covered by renderer interaction tests plus the existing gateway and observability
tests. Browser checks cover the empty state, a populated conversation and a narrow viewport.
