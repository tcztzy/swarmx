import * as acp from "@agentclientprotocol/sdk";
import { expect, it, vi } from "vitest";
import { createSwarm } from "../src/index.js";

function nested(leaf: acp.AgentApp) {
  return createSwarm("parent", (client) =>
    client.connect(
      createSwarm("child", (child) =>
        child.connect(createSwarm("grandchild", (grandchild) => grandchild.connect(leaf))),
      ),
    ),
  );
}

it.each(["end_turn", "cancelled", "max_tokens", "max_turn_requests", "refusal"] as const)(
  "routes ACP %s through three real connections",
  async (stopReason) => {
    const connected = vi.fn();
    const updates: acp.SessionNotification[] = [];
    const prompt = vi.fn<acp.AgentRequestHandler<acp.PromptRequest, acp.PromptResponse>>(
      async ({ params, client }) => {
        await client.notify(acp.methods.client.session.update, {
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "answer" },
          },
        });
        const answer = await client.request(acp.methods.client.session.requestPermission, {
          sessionId: params.sessionId,
          toolCall: { toolCallId: "write", title: "Write?" },
          options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }],
        });
        expect(answer.outcome).toEqual({ outcome: "selected", optionId: "allow" });
        return { stopReason, _meta: { opaque: "preserved" } };
      },
    );
    const leaf = acp
      .agent({ name: "leaf" })
      .onConnect(connected)
      .onRequest(acp.methods.agent.initialize, () => ({
        protocolVersion: acp.PROTOCOL_VERSION,
        agentCapabilities: { loadSession: true },
      }))
      .onRequest(acp.methods.agent.session.new, () => ({ sessionId: "native" }))
      .onRequest(acp.methods.agent.session.prompt, prompt);
    const connection = acp
      .client()
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        updates.push(params);
      })
      .onRequest(acp.methods.client.session.requestPermission, () => ({
        outcome: { outcome: "selected", optionId: "allow" },
      }))
      .connect(nested(leaf));
    try {
      expect(
        await connection.agent.request(acp.methods.agent.initialize, {
          protocolVersion: acp.PROTOCOL_VERSION,
        }),
      ).toMatchObject({ agentInfo: { name: "parent" }, agentCapabilities: { loadSession: true } });
      const session = await connection.agent.request(acp.methods.agent.session.new, {
        cwd: "/workspace",
        mcpServers: [],
      });
      const request = {
        sessionId: session.sessionId,
        prompt: [{ type: "text" as const, text: "work" }],
        _meta: { opaque: { parent: "untrusted" } },
      };
      expect(await connection.agent.request(acp.methods.agent.session.prompt, request)).toEqual({
        stopReason,
        _meta: { opaque: "preserved" },
      });
      expect(prompt.mock.calls[0]?.[0].params).toEqual(request);
      expect(updates).toHaveLength(1);
      expect(connected).toHaveBeenCalledOnce();
    } finally {
      connection.close();
    }
  },
);

it("forwards cancellation while a prompt is pending and preserves protocol errors", async () => {
  const started = Promise.withResolvers<void>();
  const result = Promise.withResolvers<acp.PromptResponse>();
  const leaf = acp
    .agent()
    .onRequest(acp.methods.agent.session.prompt, () => {
      started.resolve();
      return result.promise;
    })
    .onNotification(acp.methods.agent.session.cancel, () => {
      result.resolve({ stopReason: "cancelled" });
    })
    .onRequest(acp.methods.agent.session.load, () => {
      throw new acp.RequestError(-32000, "missing native session");
    });
  const connection = acp.client().connect(nested(leaf));
  try {
    const running = connection.agent.request(acp.methods.agent.session.prompt, {
      sessionId: "native",
      prompt: [],
    });
    await started.promise;
    await connection.agent.notify(acp.methods.agent.session.cancel, { sessionId: "native" });
    expect(await running).toEqual({ stopReason: "cancelled" });
    await expect(
      connection.agent.request(acp.methods.agent.session.load, {
        sessionId: "native",
        cwd: "/workspace",
        mcpServers: [],
      }),
    ).rejects.toThrow("missing native session");
  } finally {
    connection.close();
  }
});
