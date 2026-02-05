# Design Notes

Supplemental guidance for coding agents. Use this alongside `AGENTS.md` for deeper architectural context.

## Agent-Specific Notes
- **Design principle:** Build focused, single-purpose agents; avoid overloading a single agent with unrelated concerns.
- **Hooks:** Implement cross-cutting behavior with `on_llm_start`, `on_handoff`, or other hooks rather than embedding ad hoc logic in core flows.
- **Routing:** Use function-based edge transfers for routing between agents; keep routing logic explicit and type-checked.
- **Tools:** Expose tool capabilities explicitly so routing and orchestration can select them dynamically.
- **MCP integration:** Configure MCP servers via environment variables before use; interact through `src/swarmx/mcp_manager.py`. Validate tool schemas and authentication flows when adding or modifying MCP integrations.

## Graph Architecture
- **Swarm Graph (workflow definition):**
  - Structure: `nx.DiGraph` DAG of agent nodes with conditional transitions.
  - Edge targets: agent names, CEL expressions (e.g., `"score > 0.5 ? 'agent_b' : 'agent_c'"`), or MCP tool calls that compute destinations.
  - Cycle rule: Any cycle must include at least one conditional edge that can break the loop; unconditional cycles are invalid.
  - Validation: Enforce DAG property after accounting for conditional escape edges; reject graphs that would create unavoidable cycles.
- **Messages Graph (conversation DAG):**
  - Structure: `nx.DiGraph` DAG of message nodes, wrapping an `Iterable[ChatCompletionMessageParam]` so that agents can treat it like a simple sequence while the framework maintains a richer graph.
  - Default shape: a straight single chain built from the initial iterable (numeric IDs `0..N-1` for external messages, connected in order).
  - Context rewriting: some agents may rewrite context (including the user query); downstream sub‑agents consume only from the rewritten segment, so the “main” conversation line may not be fully connected.
  - Branches (Git‑style):
    - Each branch has: `name` (e.g. `main`, `origin`), `description` (e.g. “main thread of conversation”, “isolated conversation history for [specific task]”), and `start` / `stop` message IDs.
    - `origin` stores the full, uncompressed history: `start` is always `0`, `stop` tracks the latest message in the raw log.
    - `main` is the active working branch for LLM calls; its `start` can move forward when history is compressed.
  - Message IDs:
    - External messages entering the system get numeric IDs (`0..N-1`) in insertion order.
    - New LLM messages use a UUID derived from the underlying completion request (e.g. `completion._request_id`).
    - New tool messages use the tool call identifier (e.g. `tool_call_id`) as their node ID.
  - Per‑message metadata:
    - Every node stores the `ChatCompletionMessageParam` as `message`.
    - LLM‑generated nodes additionally carry the raw `ChatCompletion` object as `completion`.
    - Tool‑generated nodes additionally carry the raw `mcp.types.CallToolResult` as `result`.
  - Purpose:
    - Agents remain “atomic”: they see `Messages` as just an iterable of messages.
    - Only swarms' queen can see the `Messages` graph and handle with it in **Context Engineering** ways (such as Retrieve/Compress/Isolated/Remember)
    - The framework uses the Messages DAG to track which agent/tool produced which message, on which branch, with which raw completion/result, providing an execution trace over the conversation itself.
