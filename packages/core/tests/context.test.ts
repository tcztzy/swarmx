import { describe, expect, it } from "vitest";
import {
  buildContextPacket,
  contextPromptSha256,
  parseContextPacket,
  parseSummaryCheckpoint,
  resolveContextStrategy,
} from "../src/context.js";

function contextObject(overrides: Record<string, unknown>) {
  const content = String(overrides.content ?? "content");
  return {
    objectId: "ctxo_default",
    kind: "message_tail",
    title: "Default",
    content,
    sourceIds: [],
    messageIds: [],
    priority: 0,
    originalBytes: Buffer.byteLength(content, "utf8"),
    renderedBytes: Buffer.byteLength(content, "utf8"),
    compressed: false,
    truncated: false,
    ...overrides,
  };
}

describe("context packet primitives", () => {
  it("resolves auto strategy from checkpoint availability", () => {
    expect(resolveContextStrategy("auto", { hasUsableCheckpoint: true })).toBe("checkpoint_tail");
    expect(resolveContextStrategy("auto", { hasUsableCheckpoint: false })).toBe("microcompact");
    expect(resolveContextStrategy("full_tail", { hasUsableCheckpoint: true })).toBe("full_tail");
  });

  it("builds auditable thread packets with included, dropped, and truncated ids", () => {
    const packet = buildContextPacket({
      packetId: "ctxp_analysis_delegate",
      conversationId: "conv_1",
      triggerMessageId: "msg_4",
      requestedStrategy: "auto",
      latestSummaryCheckpointId: "chk_latest",
      promptBudgetBytes: 260,
      objects: [
        contextObject({
          objectId: "ctxo_instructions",
          kind: "instructions",
          title: "Instructions",
          content: "Keep claims separate from observations.",
          messageIds: ["msg_0"],
          priority: 10,
        }),
        contextObject({
          objectId: "ctxo_checkpoint",
          kind: "summary_checkpoint",
          title: "Checkpoint",
          content: "The earlier plan selected deterministic validation.",
          messageIds: ["msg_1", "msg_2"],
          priority: 8,
          compressed: true,
          truncated: true,
        }),
        contextObject({
          objectId: "ctxo_invocations",
          kind: "agent_invocations",
          title: "Prior Invocation",
          content: "x".repeat(500),
          messageIds: ["msg_3"],
          priority: 1,
        }),
        contextObject({
          objectId: "ctxo_request",
          kind: "delegated_request",
          title: "Delegated Request",
          content: "Run the next validation step.",
          messageIds: ["msg_4"],
          priority: 100,
        }),
      ],
    });

    expect(packet.metadata.mode).toBe("thread_packet");
    expect(packet.metadata.resolvedStrategy).toBe("checkpoint_tail");
    expect(packet.metadata.includedObjectIds).toEqual([
      "ctxo_instructions",
      "ctxo_checkpoint",
      "ctxo_request",
    ]);
    expect(packet.metadata.droppedObjectIds).toEqual(["ctxo_invocations"]);
    expect(packet.metadata.truncatedObjectIds).toEqual(["ctxo_checkpoint"]);
    expect(packet.metadata.includedMessageIds).toEqual(["msg_0", "msg_1", "msg_2", "msg_4"]);
    expect(packet.metadata.promptSha256).toBe(contextPromptSha256(packet.prompt));
    expect(parseContextPacket(packet)).toEqual(packet);
  });

  it("keeps delegated requests in isolated packets and refuses packets without one", () => {
    const packet = buildContextPacket({
      packetId: "ctxp_isolated",
      conversationId: "conv_1",
      requestedStrategy: "isolated",
      objects: [
        contextObject({
          objectId: "ctxo_instructions",
          kind: "instructions",
          title: "Instructions",
          content: "Do not include me.",
        }),
        contextObject({
          objectId: "ctxo_request",
          kind: "delegated_request",
          title: "Delegated Request",
          content: "Only this request should remain.",
        }),
      ],
    });

    expect(packet.metadata.mode).toBe("isolated");
    expect(packet.metadata.includedObjectIds).toEqual(["ctxo_request"]);
    expect(packet.metadata.droppedObjectIds).toEqual(["ctxo_instructions"]);

    expect(() =>
      buildContextPacket({
        packetId: "ctxp_missing_request",
        conversationId: "conv_1",
        requestedStrategy: "microcompact",
        objects: [contextObject({ objectId: "ctxo_tail" })],
      }),
    ).toThrow(/delegated_request/);
  });

  it("validates checkpoint secret boundaries and packet prompt hashes", () => {
    const compressionPrompt = "Summarize without inventing facts.";
    const checkpoint = parseSummaryCheckpoint({
      checkpointId: "chk_latest",
      conversationId: "conv_1",
      createdAt: "2026-07-03T00:00:00.000Z",
      source: "provider",
      requestedStrategy: "auto",
      resolvedStrategy: "microcompact",
      provider: { providerId: "openai-prod", model: "gpt-5" },
      compressionPromptBytes: Buffer.byteLength(compressionPrompt, "utf8"),
      compressionPromptSha256: contextPromptSha256(compressionPrompt),
      summary: "Checkpoint summary.",
    });
    expect(checkpoint.provider?.model).toBe("gpt-5");

    expect(() =>
      parseSummaryCheckpoint({
        checkpointId: "chk_bad",
        conversationId: "conv_1",
        createdAt: "2026-07-03T00:00:00.000Z",
        source: "provider",
        requestedStrategy: "auto",
        resolvedStrategy: "microcompact",
        provider: { apiKey: "sk-test" },
        compressionPromptBytes: 0,
        compressionPromptSha256: "0".repeat(64),
        summary: "Bad checkpoint.",
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(() =>
      parseContextPacket({
        metadata: {
          packetId: "ctxp_bad_hash",
          mode: "thread_packet",
          conversationId: "conv_1",
          requestedStrategy: "microcompact",
          resolvedStrategy: "microcompact",
          promptBytes: 0,
          promptSha256: "0".repeat(64),
          includedObjectIds: [],
          droppedObjectIds: [],
          truncatedObjectIds: [],
          includedMessageIds: [],
        },
        objects: [],
        prompt: "not empty",
      }),
    ).toThrow(/promptBytes/);
  });
});
