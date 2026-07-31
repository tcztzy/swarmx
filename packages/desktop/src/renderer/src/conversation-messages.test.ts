import { describe, expect, it } from "vitest";
import type { DesktopMessageChunk as MessageChunk } from "../../shared/desktop-api.js";
import { mergeStreamingMessage, withRequestTiming } from "./conversation-messages.js";

describe("conversation message helpers", () => {
  it("merges append-mode terminal progress for the same invocation", () => {
    const first: MessageChunk = {
      role: "tool",
      kind: "tool_progress",
      toolName: "exec_command",
      content: "one\n",
      structuredContent: { output: "one\n", stream: "stdout", mode: "append" },
      render: { invocationId: "call-1", status: "running" },
    };
    const second: MessageChunk = {
      ...first,
      content: "two\n",
      structuredContent: { output: "two\n", stream: "stdout", mode: "append" },
    };

    expect(mergeStreamingMessage([first], second)).toEqual([
      {
        ...second,
        content: "one\ntwo\n",
        structuredContent: { output: "one\ntwo\n", stream: "stdout", mode: "append" },
      },
    ]);
  });

  it("adds request timing without replacing message-specific timing", () => {
    const messages: MessageChunk[] = [
      {
        role: "assistant",
        kind: "message",
        content: "done",
        render: { durationMs: 250 },
      },
    ];

    expect(
      withRequestTiming(messages, "2026-07-31T00:00:00.000Z", "2026-07-31T00:00:01.000Z"),
    ).toEqual([
      {
        ...messages[0],
        render: {
          startedAt: "2026-07-31T00:00:00.000Z",
          endedAt: "2026-07-31T00:00:01.000Z",
          durationMs: 250,
        },
      },
    ]);
  });
});
