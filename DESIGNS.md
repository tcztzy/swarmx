# SwarmX design

The Host owns one ProductServices and lazily loaded native Agents. A Swarm delegates the same
in-process Agent interface to its lead, whether that lead is another Swarm or a native Agent.
MCP's `swarm` tool provides composition and delegation; no internal protocol connection is needed.

| Data | Owner |
| --- | --- |
| Native messages, configuration, history, approvals | Codex, Claude, Hermes or OpenClaw |
| Swarm membership | ProductServices, in memory |
| Research entities, journal and artifacts | Science |
| Shared semantic memory in OKF Markdown | Memory |
| Repository and data versions | Git and DVC |
| ACP connections; A2A communication Tasks | Official protocol SDKs |
| Rendering, interaction forms and trace waterfall | assistant-ui, transient |

The Agent interface contains list/create/read/start/steer/interrupt/dispose. Host-only Observer
callbacks project native text, tools, raw events and interaction requests. They are not a wire
protocol or another transcript. Prefixed session IDs enforce ownership at the Host boundary.

ACP stdio and A2A JSON-RPC are external entry points into the same Swarm. A2A stores SDK-owned
ingress messages and final artifacts, never hydrates native history, and rejects interactive
requests that this text-only entry point cannot answer. ACP form elicitation and browser AG-UI
interrupt/resume answer the pending native callback.

AG-UI belongs only between Renderer and Host. MCP calls the single ProductServices owner.
Codex and Claude receive its authenticated MCP endpoint; Hermes and OpenClaw retain their own
native tool configuration. Swarm nesting does not inspect these provider differences.

The react-o11y waterfall derives IDs, hierarchy, status and observed timing from assistant-ui
messages. It is not a durable audit log or evidence of an action's correctness. Science and
Memory keep their existing domain records. No new transcript or logging database.

Memory persists curated research knowledge across sessions and Agents through one scoped store.
The Host exposes explicit search, read, create, update, deprecate, and lint operations. OKF is
the storage format; semantic memory describes the store's role. Native runtimes retain their
own history and context management. Deterministic validation checks structure and references,
not factual truth or automatic learning.

The Host binds a random loopback port. One-use launch tokens become HttpOnly Strict cookies.
Host, Origin, session ownership, bearer authentication and canonical static paths are checked.
Native permission settings remain authoritative; SwarmX does not grant blanket tool approval.
