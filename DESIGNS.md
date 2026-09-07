# SwarmX design

The Host owns a persisted project catalog and one ProductServices per loaded project, with lazily
loaded native Agents. Each project binds one canonical directory; existing workspace identifiers
and stored research/memory data remain unchanged. Each Swarm relays ACP requests to its lead and
relays updates, approvals and forms back to its caller using official SDK connections.
MCP's `swarm` tool is the native tool carrier; its delegation operations enter the same ACP tree.

| Data | Owner |
| --- | --- |
| Native messages, configuration, history, approvals | Codex, Claude, Hermes or OpenClaw |
| Observed execution events and causal links | Host execution journal, append-only SQLite |
| Swarm membership | ProductServices, in memory |
| Research entities, journal and artifacts | Science |
| Workspace permissions and resolved Python environment | Host, private validated settings |
| Shared semantic memory in OKF Markdown | Memory |
| Repository and data versions | Git and DVC |
| ACP connections; A2A communication Tasks | Official protocol SDKs |
| Rendering, interaction forms and trace waterfall | assistant-ui, transient |

The native/gateway boundary interface contains list/create/read/start/steer/interrupt/dispose. Host-only Observer
callbacks expose native text, tools, raw events and interaction requests. The Host records these
before projecting them to a caller, including callers that ignore raw events. Prefixed session IDs
enforce ownership at the Host boundary. See `docs/execution-log.md` for the persistence contract.

ACP supplies the shared capability, prompt-result and cancellation semantics. Its official SDK
supports in-process connections as well as stdio; SwarmX-specific permissions and trusted delegation
metadata use negotiated ACP extensions, not a second network protocol. See `docs/acp.md`.
Harness translation belongs to upstream ACP adapters. SwarmX starts their stdio servers and
maps standard ACP sessions/events to Host state; third-party servers need no SwarmX handshake.
Temporary upstream fixes live in `patches/`, applied by pnpm rather than a parallel native implementation.
Each prompted adapter process receives a unique MCP endpoint bound to its Host run; the endpoint is
revoked and the process closed on completion. Later turns resume the same native conversation.
A2A JSON-RPC is an external entry point into the same Swarm. A2A stores SDK-owned
ingress messages and final artifacts, never hydrates native history, and rejects interactive
requests that this text-only entry point cannot answer. ACP permission requests/form elicitation and browser AG-UI
interrupt/resume answer the pending native callback.

AG-UI belongs only between Renderer and Host. MCP calls the ProductServices owner selected by its project URL.
Codex and Claude receive its authenticated MCP endpoint; Hermes and OpenClaw retain their own
native tool configuration. Swarm nesting does not inspect these provider differences.

The react-o11y waterfall derives IDs, hierarchy, status and observed timing from assistant-ui
messages. It is not a durable audit log or evidence of an action's correctness. Science and
Memory keep their existing domain records. The execution journal records observed operations and
their returned domain locators; it does not replace native resume state or scientific facts.

Memory persists curated research knowledge across sessions and Agents through one scoped store.
Bounded USER/workspace notes supply frozen session context; the OKF vault supplies knowledge and
explicit revision-pinned prerequisites on demand. A trigram FTS5 index derives searchable original
messages from the execution journal, including Chinese substring recall. Native resume histories
and skills remain owned by their runtimes. Post-turn reviews use an isolated native conversation
with product tools unavailable; validated proposals use the same Memory writes as foreground calls.
Pending approvals and review outcomes are durable journal events. Only the browser can approve
pending writes. Deterministic checks establish structure, scope and revisions, not factual truth.

The Host binds a random loopback port. One-use launch tokens become HttpOnly Strict cookies.
Host, Origin, session ownership, bearer authentication and canonical static paths are checked.
The Host enforces its API grants: tool access, harness/model admission and delegation. Native
harnesses independently own execution modes, sandboxing and approvals. The Host intersects
project policy, authenticated caller and captured Swarm grants before dispatch. Bound MCP calls
recover the execution snapshot; wire parent IDs and tool arguments cannot increase authority.
Ordinary modes, native tools/hooks/delegation and ambient MCP are preserved. A native approval or
YOLO selection never widens Host grants. A native process may still access data directly outside
these APIs; SwarmX does not promise one inherited filesystem ceiling across harnesses.
Memory reviews use a separate restricted configuration with no tools or approval grants.
See `docs/permissions.md` for the boundary and `docs/swarm.md` for the delegation API.
Conversation grants are derived from creation and run-start events for the workspace/native session
ID. Every turn intersects the saved grants before dispatch; creation records empty-session grants.
This preserves restrictions through restart and entry-point changes without another permissions store.

Python notebook/figure execution uses the Host-owned Docker runtime. The image is addressed by
immutable ID, workspace and declared inputs are the only mounts, and code receives no host
credentials. Permission changes require idle execution within the affected project. Browser, MCP
and A2A URLs carry the project ID; selecting another project cannot retarget an existing request or
dispose another project's runtime. The last opened project is navigation state in `projects.json`,
not a global working-directory setting. Science's internal Project entities group research objects
inside a filesystem project and appear as research collections in the UI. Shutdown cancels and
settles all loaded project owners before closing journals.
Native Agents use native permission APIs. The existing trusted Typst compiler and Git/DVC remain
host processes; they are not described as container-isolated. Science's optional
`documentSubprocess` separates the bundled platform compiler from the Python execution boundary.

Research graph nodes and edges are derived from the existing RO-Crate document using React Flow.
The graph has no independent persistence or editable relations. Graph, artifact list and
inspector select the same entity identifiers. Figure editing appends notebook execution records
and immutable artifacts; the renderer does not own a second notebook history.
