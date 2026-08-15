import { describe, expect, it, vi } from "vitest";
import { agentChunkPublisher } from "./agent-chunk-publisher.js";

describe("agentChunkPublisher", () => {
  it("records content-free observations and rejects chunks after close", () => {
    const send = vi.fn();
    const onLateChunk = vi.fn();
    const publish = agentChunkPublisher({ isDestroyed: () => false, send }, "request-settled", {
      sessionId: "session-settled",
      adapter: "acp",
      onLateChunk,
    });

    publish.close();
    publish({
      role: "tool",
      kind: "tool_result",
      content: "secret terminal output",
      structuredContent: { parameters: { command: "cat secret" } },
      render: { invocationId: "call-1", status: "succeeded" },
    });

    expect(send).not.toHaveBeenCalled();
    expect(onLateChunk).toHaveBeenCalledOnce();
    expect(onLateChunk).toHaveBeenCalledWith({
      requestId: "request-settled",
      sessionId: "session-settled",
      adapter: "acp",
      chunkKind: "tool_result",
      boundary: "closed",
      observationCount: 1,
    });
    const observation = onLateChunk.mock.calls[0]?.[0] as Record<string, unknown>;
    expect(observation).not.toHaveProperty("content");
    expect(observation).not.toHaveProperty("structuredContent");
    expect(observation).not.toHaveProperty("render");
  });
});
