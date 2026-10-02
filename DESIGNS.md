# SwarmX design

[Product direction](docs/product-direction.md) defines the long-term research objective and
[ROADMAP](ROADMAP.md) tracks unfinished milestones. This document defines data and execution ownership.

The Host owns ProductServices and lazily loaded native Agents. It resolves `SWARMX_CWD` or the
process's starting directory once; all execution uses that canonical directory. Science and execution
journal records are isolated by a hash of the directory. Settings and Memory are shared in the private
product home. Each Swarm borrows its lead Agent and forwards
method calls and Observer callbacks in process. The owner disposes the lead; disposing a Swarm
does not close a borrowed native runtime. MCP's `swarm` tool enters the same direct delegation tree.

| Data | Owner |
| --- | --- |
| Native messages, configuration, history, approvals | External ACP agent, Codex, Claude, DSH, Hermes or OpenClaw |
| Observed execution events and causal links | Host execution journal, append-only SQLite |
| Swarm membership | ProductServices, in memory |
| Work cycles/items, resource reservations and acceptance | Host work management; references native sessions, execution records and Science revisions |
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
terminal outcomes confirm that execution ended, independently of work acceptance; the Host does not add title-generation requests. MCP
credentials are bound to their Host execution and revoked at the execution boundary.
SwarmX owns orchestration and Host API authorization, with no built-in Agent loop or
provider runtime. Codex is the default external integration. Configured ACP endpoints remain
separately admitted by Host policy. The retired Pi adapter and its historical tests are retained
under `examples/legacy-pi/`, outside production builds, dependencies and default tests.
See `docs/native-agents.md` for lifecycle contracts and `docs/runtime-platform.md` for
selectable integrations and setup.
ACP remains an external stdio boundary. The inbound gateway projects the Agent contract and
its negotiated permission extension. The opt-in outbound integration adapts one explicitly
configured external process to that same Agent interface; Swarm composition stays in process.
The remote agent owns its native runtime and session files. Host observation logs never replace
remote history or resume state. An external agent may change its internal framework without
changing this boundary. SwarmX does not instantiate Pi or directly depend on its SDK. An
external native SDK may bring its own backend dependencies (currently DSH brings `pi-ai`);
those do not implement a SwarmX-owned Agent loop.
The first outbound ACP client supplies no Host MCP credential, automatic Memory context or
post-turn review; unsupported Host-tool access fails explicitly. See `docs/acp.md`.
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

[Memory](docs/memory.md) owns one shared knowledge store; native runtimes own resume histories
and skill loading. The Memory tool serves the bundled authoring guide on demand; background
reviews read the same guide directly. New session snapshots contain only short usage guidance
and the bounded user note. Its searchable conversation index and review backlog derive from execution
records. A single Host consumer persists a review plan before writes, then records each applied
or staged operation so restart can replay with original grants intersected with current policy.
Pending approvals are durable; only the user can approve them. Registered project prompts and
skills use this same plan and approval path, plus revision checks and project-owned validators;
see [learning resources](docs/learning-resources.md). Structural validation does not establish
factual truth or a behavior change's effectiveness.

The built-in Memory and delegation skills belong to the existing `@swarmx/memory` and
`@swarmx/swarm` packages. Their Markdown resources are exported and included in those packages;
application entry points resolve the package exports. They do not belong to the Desktop UI.

[Work management](docs/work-management.md) owns Agent selection, managed continuations, runtime
limits and acceptance above the Swarm layer. It derives cost from the execution journal's existing
parent/child links and independent charges. Per-Agent totals include descendants; cycle spending
counts each charge once. No second aggregate ledger or general-purpose billing layer is introduced.

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
