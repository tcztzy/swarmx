import type { ThreadMessage } from "@assistant-ui/react";
import { describe, expect, it } from "vitest";
import { traceSpans } from "../src/renderer/trace.js";

describe("Renderer observability", () => {
  it("projects hydrated runs and Swarm calls into stable parented spans", () => {
    const startedAt = Date.parse("2026-09-04T12:00:00.000Z");
    const messages: ThreadMessage[] = [
      {
        id: "message-1",
        role: "assistant",
        createdAt: new Date(startedAt),
        status: { type: "complete", reason: "stop" },
        content: [
          { type: "text", text: "done" },
          {
            type: "tool-call",
            toolCallId: "agent-call-1",
            toolName: "swarm",
            args: { action: "send_message" },
            argsText: '{"action":"send_message"}',
            result: { response: { stopReason: "end_turn" } },
            timing: { startedAt: startedAt + 10, completedAt: startedAt + 40 },
          },
        ],
        metadata: {
          unstable_state: null,
          unstable_annotations: [],
          unstable_data: [],
          steps: [],
          timing: {
            streamStartTime: startedAt,
            totalStreamTime: 50,
            totalChunks: 2,
            toolCallCount: 1,
          },
          custom: {},
        },
      },
    ];

    expect(traceSpans(messages)).toEqual([
      {
        id: "run:message-1",
        parentSpanId: null,
        name: "Swarm response",
        type: "run",
        status: "completed",
        startedAt,
        endedAt: startedAt + 50,
        latencyMs: 50,
      },
      {
        id: "tool:agent-call-1",
        parentSpanId: "run:message-1",
        name: "swarm",
        type: "agent",
        status: "completed",
        startedAt: startedAt + 10,
        endedAt: startedAt + 40,
        latencyMs: 30,
      },
    ]);
  });
});
