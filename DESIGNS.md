# SwarmX design

The Host owns ProductServices and lazily loaded native Agents. It resolves `SWARMX_CWD` or the
process's starting directory once; all execution uses that canonical directory. Science and execution
journal records are isolated by a hash of the directory. Settings and Memory are shared in the private
product home. Each Swarm borrows its lead Agent and forwards
method calls and Observer callbacks in process. The owner disposes the lead; disposing a Swarm
does not close a borrowed native runtime. MCP's `swarm` tool enters the same direct delegation tree.

| Data | Owner |
| --- | --- |
| Native messages, configuration, history, approvals | Pi, Codex, Claude, DSH, Hermes or OpenClaw |
| Observed execution events and causal links | Host execution journal, append-only SQLite |
| Swarm membership | ProductServices, in memory |
| Research entities, journal and artifacts | Science |
| Execution permissions and resolved Python environment | Host, private validated settings |
| Shared semantic memory in OKF Markdown | Memory |
| Repository and data versions | Git and DVC |
| External ACP connections; A2A communication Tasks | Official protocol SDKs |
| Rendering, interaction forms and trace waterfall | assistant-ui, transient |

The native/gateway boundary interface contains list/create/read/start/steer/interrupt/dispose. Host-only Observer
callbacks expose native text, tools, raw events and interaction requests. The Host records these
before projecting them to a caller, including callers that ignore raw events. Prefixed session IDs
enforce ownership at the Host boundary. See `docs/execution-log.md` for the persistence contract.

The Agent contract carries capabilities, run results and cancellation directly. Native integrations
use official SDKs or native runtime APIs and project their events into the Host Observer. Native
terminal outcomes confirm completion; the Host does not add title-generation requests. MCP
credentials are bound to their Host execution and revoked at the execution boundary.
Pi supplies the default lead through its embedded SDK. Its native session manager and resource
loader own persistence and skill discovery; SwarmX maintains no second Agent loop or skill loader.
Pi custom tools call ProductServices in process under the active Host execution, retaining the
same grants, cancellation and journal as the external product MCP boundary.
See `docs/native-agents.md` for lifecycle contracts and `docs/runtime-platform.md` for
selectable integrations and setup.
ACP remains an external stdio boundary. Its official SDK translates the Agent contract and its
negotiated permission extension into the public wire format; internal delegation and memory
reviews do not open ACP connections. See `docs/acp.md`.
A2A JSON-RPC is an external entry point into the same Swarm. A2A stores SDK-owned
ingress messages and final artifacts, never hydrates native history, and rejects interactive
requests that this text-only entry point cannot answer. ACP permission requests/form elicitation and desktop AG-UI
interrupt/resume answer the pending native callback.

AG-UI events travel between Renderer and Host over Electron IPC. Product MCP calls reach ProductServices
through the Host stdio bridge socket. Codex, Claude, DSH and Hermes receive its command and
a per-run credential; OpenClaw retains its own native tool configuration. Swarm nesting does not inspect these provider differences.

The react-o11y waterfall derives IDs, hierarchy, status and observed timing from assistant-ui
messages. It is not a durable audit log or evidence of an action's correctness. Science and
Memory keep their existing domain records. The execution journal records observed operations and
their returned domain locators; it does not replace native resume state or scientific facts.

Memory persists curated research knowledge across sessions and Agents through one shared store.
A bounded user note supplies frozen session context; the OKF concept pool supplies knowledge and
explicit revision-pinned prerequisites on demand. A trigram FTS5 index derives searchable original
messages from the execution journal, including Chinese substring recall. Native resume histories
and skills remain owned by their runtimes. Post-turn reviews use an isolated native conversation
with product tools unavailable; validated proposals use the same Memory writes as foreground calls.
Pending approvals and review outcomes are durable journal events. Only the user through the desktop can approve
pending writes. Deterministic checks establish structure, scope and revisions, not factual truth.
The execution journal also supplies the learning backlog: eligible terminal events remain pending
until an exact batch is acknowledged. One Host consumer persists a review plan before writing,
then records each applied or staged operation. Restart replays this plan with original grants
intersected with current policy; idempotent operations cover interrupted receipts. There is no
separate scheduler or scoring service. Registered project Markdown resources use the same plan
and approval path, with pinned file/configuration revisions and project-owned validation commands.
Native runtimes still load their own prompts and skills; see `docs/learning-resources.md`.

The sandboxed Electron preload exposes named operations. IPC validates payloads and accepts calls
only from the application's top-level window. The external A2A Host binds a random loopback port
and authenticates calls with a bearer token. Native session ownership is checked at the Host boundary.
The Host enforces its API grants: tool access, harness/model admission and delegation. Native
harnesses independently own execution modes, sandboxing and approvals. The Host intersects
Host policy, authenticated caller and captured Swarm grants before dispatch. Bound MCP calls
recover the execution snapshot; wire parent IDs and tool arguments cannot increase authority.
Ordinary modes, native tools/hooks/delegation and ambient MCP are preserved. A native approval or
YOLO selection never widens Host grants. A native process may still access data directly outside
these APIs; SwarmX does not promise one inherited filesystem ceiling across harnesses.
Memory reviews use a separate restricted configuration with no tools or approval grants.
See `docs/permissions.md` for the boundary and `docs/swarm.md` for the delegation API.
Conversation grants are derived from creation and run-start events for the execution directory/native session
ID. Every turn intersects the saved grants before dispatch; creation records empty-session grants.
This preserves restrictions through restart and entry-point changes without another permissions store.

Python notebook/figure execution uses the Host-owned Docker runtime. The image is addressed by
immutable ID, the execution directory and declared inputs are the only mounts, and code receives no host
credentials. Permission changes require idle execution. Science's internal Project entities group
research objects and appear as research collections in the UI. Shutdown cancels and settles active
execution before closing journals.
Native Agents use native permission APIs. The existing trusted Typst compiler and Git/DVC remain
host processes; they are not described as container-isolated. Science's optional
`documentSubprocess` separates the bundled platform compiler from the Python execution boundary.

Research graph nodes and edges are derived from the existing RO-Crate document using React Flow.
The graph has no independent persistence or editable relations. Graph, artifact list and
inspector select the same entity identifiers. Figure editing appends notebook execution records
and immutable artifacts; the renderer does not own a second notebook history.
