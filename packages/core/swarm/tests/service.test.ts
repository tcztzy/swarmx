import type { Agent } from "@deepseek-ai/dsh-agent";
import { SessionId } from "@deepseek-ai/dsh-session";
import { describe, expect, it, vi } from "vitest";
import type { SwarmSnapshot } from "../src/contracts.js";
import { SwarmService } from "../src/index.js";

describe("Swarm service Remote projection", () => {
  it("V169: strips task descriptions, attempts, and write paths before strict validation", async () => {
    const sessionId = SessionId("session-lead");
    const lead = { id: sessionId } as Agent;
    const snapshot: SwarmSnapshot = {
      kind: "active",
      memberName: "lead",
      members: [{ description: "Team lead", name: "lead", role: "lead", status: "idle" }],
      name: "Research team",
      pendingMessages: 0,
      revision: 2,
      role: "lead",
      tasks: [
        {
          attemptId: "private-attempt",
          blockedBy: [],
          description: "private task detail",
          id: "task-1",
          kind: "write",
          ownerName: "lead",
          ready: true,
          revision: 2,
          status: "in_progress",
          subject: "Implement scheduler",
          writeScopes: ["private/workspace/path"],
        },
      ],
      updatedAt: 100,
    };
    const service = Object.create(SwarmService.prototype) as SwarmService;
    Object.defineProperty(service, "ctx", {
      value: { agents: { get: () => lead } },
    });
    vi.spyOn(service, "snapshot").mockResolvedValue(snapshot);

    const result = await service.uiSnapshot(sessionId);

    expect(result.kind).toBe("active");
    expect(JSON.stringify(result)).not.toMatch(
      /private task detail|private-attempt|private\/workspace/u,
    );
  });
});
