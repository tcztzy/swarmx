import * as acp from "@agentclientprotocol/sdk";
import type { NativeAgent, Observer } from "../agents/types.js";

/** ACP is an external stdio entry point, never an internal Swarm transport. */
export function acpAgent(agent: NativeAgent, cwd: string): acp.AgentApp {
  const cancelled = new Set<string>();
  let forms = false;
  function workspace(request: { cwd: string; mcpServers: acp.McpServer[] }) {
    if (request.cwd !== cwd) throw new Error("This SwarmX Host owns a different workspace.");
    if (request.mcpServers.length)
      throw new Error("Configure MCP in the native Agent; SwarmX owns its product carrier.");
  }
  return acp
    .agent({ name: "swarmx" })
    .onRequest(acp.methods.agent.initialize, ({ params }) => {
      forms = params.clientCapabilities?.elicitation?.form !== undefined;
      return {
        protocolVersion: acp.PROTOCOL_VERSION,
        agentInfo: { name: agent.name, version: "0.1.0" },
        agentCapabilities: { loadSession: true, sessionCapabilities: { list: {} } },
      };
    })
    .onRequest(acp.methods.agent.session.list, async () => ({
      sessions: (await agent.list()).map((session) => ({ ...session, cwd })),
    }))
    .onRequest(acp.methods.agent.session.new, async ({ params }) => {
      workspace(params);
      return { sessionId: await agent.create() };
    })
    .onRequest(acp.methods.agent.session.load, async ({ params, client }) => {
      workspace(params);
      const projection = observe(params.sessionId, client, forms);
      await agent.read(params.sessionId, projection.observer);
      await projection.flush();
      return {};
    })
    .onRequest(acp.methods.agent.session.prompt, async ({ params, client }) => {
      if (!params.prompt.every((part) => part.type === "text"))
        throw new Error("SwarmX accepts text-only ACP prompts.");
      const projection = observe(params.sessionId, client, forms);
      await agent.start(
        params.sessionId,
        params.prompt.map((part) => part.text).join(""),
        projection.observer,
      );
      await projection.flush();
      return { stopReason: cancelled.delete(params.sessionId) ? "cancelled" : "end_turn" };
    })
    .onNotification(acp.methods.agent.session.cancel, async ({ params }) => {
      cancelled.add(params.sessionId);
      await agent.interrupt(params.sessionId);
    });
}

function observe(sessionId: string, client: acp.AgentContext, forms: boolean) {
  const pending: Promise<void>[] = [];
  const tools = new Set<string>();
  const update = (update: acp.SessionUpdate, metadata?: Record<string, unknown>) => {
    pending.push(
      client.notify(acp.methods.client.session.update, {
        sessionId,
        update,
        ...(metadata ? { _meta: metadata } : {}),
      }),
    );
  };
  const observer: Observer = {
    text(id, text, role = "assistant") {
      update({
        sessionUpdate:
          role === "user"
            ? "user_message_chunk"
            : role === "reasoning"
              ? "agent_thought_chunk"
              : "agent_message_chunk",
        content: { type: "text", text },
        messageId: id,
      });
    },
    tool(id, name, input, output) {
      if (!tools.has(id)) {
        tools.add(id);
        update({
          sessionUpdate: "tool_call",
          toolCallId: id,
          title: name,
          status: "in_progress",
          rawInput: input,
        });
      }
      if (output !== undefined)
        update({
          sessionUpdate: "tool_call_update",
          toolCallId: id,
          status: "completed",
          rawOutput: output,
        });
    },
    raw(event) {
      update({ sessionUpdate: "session_info_update" }, { "swarmx/native": event });
    },
    async interact(request) {
      if (!forms) throw new Error("Native interactions require ACP form elicitation support.");
      const answer = await client.request(acp.methods.client.elicitation.create, {
        mode: "form",
        sessionId,
        message: request.title,
        requestedSchema: request.schema as Extract<
          acp.CreateElicitationRequest,
          { requestedSchema: unknown }
        >["requestedSchema"],
      });
      return answer.action === "accept" ? answer.content : undefined;
    },
  };
  return { observer, flush: () => Promise.all(pending) };
}
