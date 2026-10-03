# Desktop interface

SwarmX uses assistant-ui's [homepage Demo](https://github.com/assistant-ui/assistant-ui/tree/1a5da0f272668cf313e5213e49aa70e0f987de6d/apps/docs/components/pages/home/demo)
message, thread and composer layout within its light interface. Messages have no avatars:
user text sits in a right-aligned neutral bubble (at most 80% of the column), while assistant
text has no enclosing card, border or fill. Reply actions sit below the text, aligned left.
Their row reserves its height while hidden, so hovering a reply never shifts later messages.
The thread and composer share the Demo's 42rem maximum width, 15px message text, 24px message
spacing and 16px corner radius. The Demo's Public Sans and JetBrains Mono fonts are bundled
locally, with system fallbacks for other scripts. The composer uses a 16px input, a bottom control row and the
square rounded arrow-up Send button. Turn work uses the Demo's unboxed Reasoning disclosure.
The light task sidebar and compact header retain SwarmX navigation. The sidebar can be hidden and remains
available in narrow windows. Search filters the selected Agent's native session titles.
New task creates a native session; refresh reloads native titles without replacing the active
conversation. Agent changes load that Agent's sessions; late responses cannot replace the
current selection. Connection, session creation and history failures are visible in the UI.
History failures identify the failed operation, show the native error detail, and offer an
explicit retry for the same session. Retry preserves the mounted chat, draft and side view;
it only reloads native history, never sends a prompt or reruns an analysis. Sending remains
disabled until supported history loads successfully. A failed retry leaves the recovery action available.
If the Agent advertises no history replay, the Host validates the native session and returns an
explicit unsupported status, without a transcript. The UI labels that limitation and allows
continued prompts in the same session; it displays only messages received since opening the
view. This state does not offer a history retry or imply that the native transcript is empty.
The internal history IPC response is `{supported:true,messages}` or `{supported:false}`.
Codex history and model choices remain readable while Codex App holds the same conversation.
A conflicting send reports that the other instance holds write access; viewing history does
not require closing Codex App.
The Host preserves textual native error details across IPC instead of reducing them to
the generic JSON-RPC message.

UI changes start with assistant-ui's existing primitives and [Tool UI examples](https://www.assistant-ui.com/docs/tools/tool-ui).
Generic presentation uses the official component sources and their Radix/shadcn dependencies
in `renderer/components/`, with upstream provenance recorded there. Each component's local
adaptation is limited to ten changed source lines; props and slots supply application labels and
data. Replacements remove the old implementation and unused styles together. Official sources
retain their formatting for direct comparison; application code remains Biome-formatted.
Model Selector, Reasoning, Tool Fallback, Tool Group, MarkdownText, Tabs, Collapsible and
CodeBlock own their generic interactions. Shared buttons, inputs and badges use the same
official foundation. The renderer uses React's ref-as-prop support rather than local ref wrappers.
Native catalogs and run settings remain in `agent-controls.tsx`; tool outcomes, Memory
actions and cross-message grouping remain in the conversation renderer. Generic tool cards
separate arguments, results and errors; expanding them never submits an approval or retries a call.
The Reasoning component displays commentary Markdown and tool calls as one turn's work.
Native reasoning text is hidden in both streaming and restored conversations; it creates
neither a nested disclosure nor empty message spacing. Native records remain unchanged.
Codex's native `contextCompaction` maintenance tool is likewise hidden from the chat in
history and streaming, including between turns. It remains in the native records.
Consecutive tool calls share one expandable group, including calls in separate native history
messages. Visible commentary and user messages delimit groups. Native tool kinds supply
summaries such as `Read files, ran commands`; unknown tools retain a generic tool summary.
The renderer uses assistant-ui's [Tool Group](https://www.assistant-ui.com/elements/tool-group)
`MessagePrimitive.GroupedParts` and backend Tool UI render props. Shell results follow its
[Terminal Block](https://www.assistant-ui.com/elements/terminal-block) layout: command, scrollable plain-text output
and actual exit status, rather than a JSON argument/result dump. Failed, cancelled and
unfinished calls cannot show Success. Expanding a group never executes a tool or sends a prompt.
Claude shell stdout/stderr and Hermes terminal output remain visible, with native exit codes
when provided. Unrecognized command results retain the generic result view instead of blank output.
assistant-ui owns messages, Markdown (including GFM tables and task lists), composer state,
draft suggestions, copy feedback, scroll-to-latest, streaming and Stop. Starter prompts only fill the draft; sending remains
explicit. Send and Stop are mutually exclusive. History loading and pending interactions
disable new sends. Approval and question forms use the existing AG-UI interrupt/resume path;
required text and numeric fields are validated before submission. A boolean permission can
be explicitly declined. Failed and cancelled tool calls must not appear successful.
Completed tool-only messages show neither an ongoing-work indicator nor an empty copy action.
Messages marked `commentary` never offer a copy action, including on hover and inside expanded
work. Final answers and ordinary unmarked assistant replies retain assistant-ui's copy action.
Live Codex commentary uses the Reasoning component's open streaming preview. When Codex
supplies a `final_answer`, the commentary and tools in that turn default to a collapsed
`Worked for {time}` row using the same component; the final text stays outside it. Tool-only
work can also be expanded. A final answer with native timing but no visible work shows a
plain `Worked for {time}` row above the answer, without a button, chevron or hidden panel.
Hidden reasoning and context maintenance never create an empty work disclosure. The duration
still comes from the whole native turn; hiding maintenance does not change it or attach it
to a neighbouring turn. A turn without visible work or native timing has no work row.
Each turn expands independently, and a manual toggle persists across streaming updates and
the final-answer transition. Unfinished history without a Codex final-answer marker remains
visible. Native turn duration supplies the completed label for both
history and streaming; while the final answer streams, elapsed time uses the native start time.
Unavailable timing shows `Worked` without inventing a duration. Reloading history never resets
the recorded work time. Expanding or collapsing work changes presentation only.
Switching tasks or Agents closes the active stream and stops that run, as in the existing Host
contract. The running-state hint makes this visible; background execution is not implied.

Swarm delegations appear in an expandable sub-Agent list following assistant-ui's
[SubagentList](https://www.assistant-ui.com/elements/subagent-list) and
[SpeakerIdentity](https://www.assistant-ui.com/elements/speaker-identity) patterns. Each execution
shows its dispatched task, parent, Harness, requested model and independent status;
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
execution journal independently retains observed operations (see `execution-log.md`). No renderer
transcript store or new transport is introduced. Search and sidebar/trace visibility are local
presentation state.

The composer provides a Harness menu and a combined model/Thinking popover using the
[assistant-ui model selector](https://www.assistant-ui.com/elements/model-selector): model rows,
an active-model check, and a separated Thinking row of effort segments. The trigger shows the
model and current effort together. Selecting a model closes the popover; selecting an effort
keeps it open. Keyboard navigation uses cmdk for models and Radix RadioGroup for effort levels.
The list scrolls independently and the effort row wraps when needed on narrow windows.
Harness selects
the native runtime (external ACP, Codex, Claude, DSH, Hermes or OpenClaw); the default Swarm shows its lead's
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
DSH exposes no model catalog. Each DSH task runs once; completed output is replayed from Host
logs and the composer directs the user to create a new task. Stopping closes the owned SDK runtime.

Codex uses `model/list` and `turn/start`; Claude uses SDK `supportedModels()` and query
options; OpenClaw uses `models.list` and session-scoped `sessions.patch` before `chat.send`.
OpenClaw uses the Gateway's advertised thinking levels without a separate model registry.
Hermes uses `model.options`, `config.get` and `config.set` with explicit session scope. Its native
provider capabilities constrain the exposed reasoning settings; native
model-change confirmations use the existing interaction form. A failed setting change must
not send the prompt, and Stop during a pending settings change must not start a new run.

Composer choices represent the next run's requested settings, not proof of the model used by
a previous response. Native transcripts and native event metadata remain the factual record;
the Host snapshots dispatched settings in its execution journal. Unsent menu selections are not logged.

The existing Host session endpoints and AG-UI adapter provide the interface capabilities.
Rename/archive, attachments, file review,
terminal panels and concurrent background tasks need additional end-to-end contracts and
are not exposed as nonfunctional controls.

The conversation is the primary view. The header exposes Long-term work and Observe;
these open a side view without replacing the conversation, its draft or active stream.
Chat and split view share the same mounted conversation, message styles, typography and
composer controls. Closing the side view restores the full conversation area; history,
draft, model choices and streaming remain intact. Source-specific composer context and
its placeholder appear only while the source-inspection view is open. On wide screens,
the conversation occupies 40% of the split view; narrow screens retain a closable inspector.
Observe shows the existing react-o11y execution waterfall. Its entry remains visible before
the first run, with an explicit empty state. Scientific asset galleries, notebooks, figure
editing, scientific computation history and scientific runtime setup belong to the external
domain application and are not offered by SwarmX.
Successful Memory reads expose saved-concept metadata and exact source references beneath
the current answer. Historical tool results remain readable through generic tool cards,
without reinstating domain-specific actions. External source references stay readable and
copyable; opening one explains that its originating application owns inspection.
Execution source references open the same side view directly through the read-only `logs.evidence`
bridge operation; they do not require a domain workspace. The view shows statistics only for
the cited executions, with completion, error and cancellation counts kept separate. Elapsed time
is Host wall-clock time including tool execution and waiting, and missing usage, cost or route
fields remain explicitly unknown. Requested settings identify the Agent; original native reports
remain available in raw records.
Original cited inputs, outputs, terminal events and review snapshots are expandable records.
The same review attempt's saved plan exposes its conclusion and requested reviewer identity
separately from the frozen execution evidence and its statistics.
Unavailable or foreign-directory references show an error rather than substituting other history.
Saved selection concepts distinguish observations, AI judgments and user preferences, and display
their task, criteria and limitations. Older selection concepts without structured evaluation
metadata are marked unverified. Structured evaluations whose original references cannot resolve
also display the Host's unverified status and reason; citations do not certify a conclusion.
Structured evaluation cards can export their exact displayed concept revision as a local
RO-Crate ZIP. A loaded review snapshot offers the same export without requiring a saved
Memory concept, including reviews with no proposed changes. The ZIP contains root
`ro-crate-metadata.json` and the Host-selected evidence files with their original text intact.
The UI states that selected private source text is included; it only downloads locally.
ZIP entry timestamps are fixed, so an unchanged exported payload produces identical ZIP bytes.
Export stays disabled while pending, and unavailable evidence or revision conflicts remain
visible errors without downloading a partial package.
Opening, copying or inspecting evidence never reruns work. The composer can add the
selected source reference as text context; existing Harness, model and thinking controls
remain available. The sidebar lists native tasks. The bottom-left settings control opens
Settings; Back restores the same mounted conversation.

The reusable generic RO-Crate graph projection preserves recorded identifiers and relation
names. Search and selected-node neighborhoods keep it readable; at most 200 matched entities
render at once. React Flow owns pan/zoom/fit and keyboard navigation. This shared graph view
also renders Memory dependencies; edges are read-only and no domain relationships are inferred.

Settings shows the Host's canonical execution directory and explicit tool/delegation grants.
The existing domain-reference read grant remains visible for externally owned pinned references.
Legacy policy and environment metadata remain persisted but have no local scientific-runtime
controls. Native modes are chosen per conversation using upstream labels. Permission-save
failures remain actionable; policy changes apply only after current executions have ended.

English and Simplified Chinese cover navigation, chat controls, approvals, execution evidence,
observability and settings, including accessible labels and empty states. i18next owns lookup
and interpolation. The first launch follows the browser language (English for unsupported
languages); Settings persists an explicit choice in the private product home and updates the UI
immediately. Bootstrap restores it across Host restarts.
Document language, dates and numbers follow that choice. Original data, code, native model
names, protocol identifiers and original execution/error payloads remain untranslated.

Settings includes shared user preferences, Memory and knowledge: the bounded user note, automatic review settings,
current-conversation review, persistent pending approvals, and a searchable Vault dependency graph.
Selecting a concept loads its prerequisite closure in order; changed prerequisite revisions are
marked for review. Conflicting edits stay pending and display the error. The graph uses the same
generic RO-Crate React Flow view, with read-only edges. See `memory.md` for storage and authority.

Long-term work uses the existing Host work manager. Users create cycles with shared budgets and
registered execution configurations, then add goals with written acceptance criteria. The side
view shows execution and acceptance separately, actual reported spending and still-held reserves,
original output and opaque pinned artifact IDs/revisions. Artifact references can be copied
without constructing domain-specific resource URLs. Start and Stop are explicit; opening or closing
the view never dispatches or cancels work. Native confirmations reuse the conversation's form
fields. User acceptance and corrections preserve their displayed attempt and criteria revision;
goal revisions retain work identity, budget and historical evidence. Errors remain visible and
do not discard form edits. See `work-management.md` for lifecycle and accounting rules.

Acceptance is covered by renderer interaction tests plus the existing gateway and observability
tests. Streaming fixtures wait for the IPC run-start call before delivering events: the AG-UI
client initializes asynchronously before subscribing, and normal streams begin with `RUN_STARTED`.
Send controls follow the active UI language. Desktop checks cover the empty state, a populated
conversation and a narrow viewport.

Settings retains the project prompt/skill file-access selector. Read-only blocks resource learning
and updates; workspace-write permits the registered, validated approval flow. Users can enable
or tighten it independently of native Agent filesystem permissions and domain reference grants.
