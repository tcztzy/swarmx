import { EventEmitter } from "node:events";
import { cancelAcpRequest, withAcpRequest } from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import { AgentInteractionBroker, type AgentInteractionOwner } from "./agent-interactions.js";

class FakeOwner extends EventEmitter implements AgentInteractionOwner {
  readonly send = vi.fn();
  destroyed = false;

  constructor(readonly id: number) {
    super();
  }

  isDestroyed(): boolean {
    return this.destroyed;
  }

  destroy(): void {
    this.destroyed = true;
    this.emit("destroyed");
  }
}

describe("AgentInteractionBroker", () => {
  it("V445-V446 accepts only an offered tool approval option", async () => {
    const broker = new AgentInteractionBroker();
    const owner = new FakeOwner(1);
    const pending = broker.request(owner, "permission-request", {
      kind: "tool_approval",
      title: "Allow Bash?",
      toolKind: "execute",
      summary: "Project-sandboxed shell command",
      options: [
        { optionId: "reject", name: "Reject", kind: "reject_once" },
        { optionId: "allow", name: "Allow once", kind: "allow_once" },
      ],
    });
    const event = owner.send.mock.calls[0]?.[1] as {
      requestId: string;
      interactionId: string;
    };

    expect(() =>
      broker.resolve(owner, {
        ...event,
        response: { kind: "tool_approval", optionId: "forged" },
      }),
    ).toThrow(/not offered/i);
    expect(broker.size).toBe(1);
    expect(
      broker.resolve(owner, {
        ...event,
        response: { kind: "tool_approval", optionId: "allow" },
      }),
    ).toBe(true);
    await expect(pending).resolves.toEqual({ kind: "tool_approval", optionId: "allow" });
  });

  it("V387 resolves only a matching owner, request, id, and response kind", async () => {
    const broker = new AgentInteractionBroker();
    const owner = new FakeOwner(1);
    const stranger = new FakeOwner(2);
    const pending = broker.request(owner, "request-1", {
      kind: "questions",
      questions: [
        {
          question: "Which runtime?",
          header: "Runtime",
          options: [
            { label: "Node", description: "Use Node.js" },
            { label: "Bun", description: "Use Bun" },
          ],
          multiSelect: false,
        },
      ],
    });
    const event = owner.send.mock.calls[0]?.[1] as {
      requestId: string;
      interactionId: string;
    };

    expect(
      broker.resolve(stranger, { ...event, response: { kind: "questions", answers: {} } }),
    ).toBe(false);
    expect(
      broker.resolve(owner, {
        ...event,
        requestId: "other-request",
        response: { kind: "questions", answers: {} },
      }),
    ).toBe(false);
    expect(() =>
      broker.resolve(owner, {
        ...event,
        response: { kind: "plan_approval", approved: true },
      }),
    ).toThrow(/kind does not match/i);
    expect(broker.size).toBe(1);
    expect(
      broker.resolve(owner, {
        ...event,
        response: { kind: "questions", answers: { "Which runtime?": "Node" } },
      }),
    ).toBe(true);
    await expect(pending).resolves.toEqual({
      kind: "questions",
      answers: { "Which runtime?": "Node" },
    });
    expect(broker.size).toBe(0);
    expect(owner.listenerCount("destroyed")).toBe(0);
  });

  it("V387 rejects and cleans a pending prompt when its request is cancelled", async () => {
    const broker = new AgentInteractionBroker();
    const owner = new FakeOwner(1);
    const requestId = "cancel-interaction";
    const running = withAcpRequest(requestId, () =>
      broker.request(owner, requestId, {
        kind: "plan_approval",
        plan: "# Plan",
        filePath: "/private/plan.md",
      }),
    );
    await vi.waitFor(() => expect(owner.send).toHaveBeenCalledOnce());

    await expect(cancelAcpRequest(requestId)).resolves.toBe(true);
    await expect(running).rejects.toThrow(/cancelled/i);
    expect(broker.size).toBe(0);
    expect(owner.listenerCount("destroyed")).toBe(0);
  });

  it("V387 rejects every owner-bound prompt when its renderer is destroyed", async () => {
    const broker = new AgentInteractionBroker();
    const owner = new FakeOwner(1);
    const pending = broker.request(owner, "destroyed-interaction", {
      kind: "plan_approval",
      plan: "# Plan",
      filePath: "/private/plan.md",
    });

    owner.destroy();
    await expect(pending).rejects.toThrow(/window was closed/i);
    expect(broker.size).toBe(0);
  });

  it("V387 lets tool-manager close cancel only its own request interactions", async () => {
    const broker = new AgentInteractionBroker();
    const owner = new FakeOwner(1);
    const first = broker.request(owner, "request-first", {
      kind: "plan_approval",
      plan: "# First",
      filePath: "/private/first.md",
    });
    const second = broker.request(owner, "request-second", {
      kind: "plan_approval",
      plan: "# Second",
      filePath: "/private/second.md",
    });

    expect(broker.cancelRequest(owner, "request-first")).toBe(1);
    await expect(first).rejects.toThrow(/request-first.*closed/i);
    expect(broker.size).toBe(1);
    owner.destroy();
    await expect(second).rejects.toThrow(/window was closed/i);
  });
});
