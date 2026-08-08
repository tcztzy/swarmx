import type { ChatMessage, MessageChunk } from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import { PermissionAutoReviewer } from "./permission-review.js";

const request = {
  source: "direct" as const,
  toolName: "exec_command",
  toolKind: "execute",
  userMessages: ["Run the focused test suite."],
  toolInput: { cmd: "pnpm test -- permission-review" },
  options: [
    { optionId: "reject_once", kind: "reject_once" as const },
    { optionId: "allow_once", kind: "allow_once" as const },
  ],
};

describe("PermissionAutoReviewer", () => {
  it("approves only a strict low or controlled-risk allow_once verdict", async () => {
    const generate = vi.fn(
      async (_messages: ChatMessage[]): Promise<MessageChunk[]> => [
        {
          role: "assistant",
          kind: "message",
          content: JSON.stringify({ decision: "allow", risk: "controlled" }),
        },
      ],
    );
    const reviewer = new PermissionAutoReviewer({ generate });

    await expect(reviewer.review(request)).resolves.toEqual({
      decision: "allow",
      optionId: "allow_once",
      risk: "controlled",
    });
    const serializedPrompt = JSON.stringify(generate.mock.calls[0]?.[0]);
    expect(serializedPrompt).toContain("Run the focused test suite.");
    expect(serializedPrompt).toContain("pnpm test -- permission-review");
    expect(serializedPrompt).not.toContain("tool result");
  });

  it("defers high-risk, malformed, failed, and non-one-call verdicts to a human", async () => {
    const responses: Array<MessageChunk[] | Error> = [
      [{ role: "assistant", kind: "message", content: '{"decision":"allow","risk":"high"}' }],
      [{ role: "assistant", kind: "message", content: "allow" }],
      new Error("classifier unavailable"),
    ];
    const reviewer = new PermissionAutoReviewer({
      generate: async () => {
        const response = responses.shift();
        if (response instanceof Error) throw response;
        return response ?? [];
      },
    });

    await expect(reviewer.review(request)).resolves.toEqual({ decision: "defer" });
    await expect(reviewer.review(request)).resolves.toEqual({ decision: "defer" });
    await expect(reviewer.review(request)).resolves.toEqual({ decision: "defer" });
    await expect(
      reviewer.review({
        ...request,
        options: [{ optionId: "allow", kind: "allow_always" }],
      }),
    ).resolves.toEqual({ decision: "defer" });
  });
});
