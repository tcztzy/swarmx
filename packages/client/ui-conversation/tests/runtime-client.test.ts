import { describe, expect, it, vi } from "vitest";
import {
  approvalAnswerKey,
  approvalIdentityKey,
  approvalResponseFields,
  ConversationRuntimeClient,
  projectApprovalEvent,
  projectRuntimeEvent,
  type RuntimeEventSource,
  removeApprovalAnswers,
  removeApprovalRequest,
} from "../src/client/runtime-client.js";

class FakeEventSource implements RuntimeEventSource {
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  close = vi.fn();
}

describe("conversation runtime browser client", () => {
  it("qualifies reads and mutations by runtime without exposing native transports", async () => {
    const requests: Array<{ input: string; init?: RequestInit }> = [];
    const fetcher = vi.fn(async (input: string, init?: RequestInit) => {
      requests.push({ input, ...(init === undefined ? {} : { init }) });
      return new Response(
        JSON.stringify(input.includes("/start") ? { turnId: "codex:turn-1" } : []),
        {
          headers: { "content-type": "application/json" },
          status: 200,
        },
      );
    });
    const client = new ConversationRuntimeClient(fetcher);

    await client.list("codex");
    await client.start("codex", "codex:thread-1", "hello");
    await client.edit("codex", "codex:thread-1", "codex:user-1", "replacement");

    expect(requests[0]?.input).toBe(
      "/api/swarmx/conversation-runtimes/conversations?runtimeKind=codex",
    );
    expect(requests[1]).toMatchObject({
      input: "/api/swarmx/conversation-runtimes/start",
      init: {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "codex:thread-1",
          text: "hello",
        }),
      },
    });
    expect(requests[2]).toMatchObject({
      input: "/api/swarmx/conversation-runtimes/edit",
      init: {
        method: "POST",
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "codex:thread-1",
          userItemId: "codex:user-1",
          text: "replacement",
        }),
      },
    });
  });

  it("delivers runtime-qualified SSE events and closes the owned stream", () => {
    const source = new FakeEventSource();
    const events: unknown[] = [];
    const client = new ConversationRuntimeClient(undefined, () => source);

    const unsubscribe = client.subscribe((event) => events.push(event));
    source.onmessage?.(
      new MessageEvent("message", {
        data: JSON.stringify({
          seq: 1,
          runtime: "codex",
          conversationId: "codex:thread-1",
          type: "turn_status",
          turnId: "codex:turn-1",
          status: "running",
        }),
      }),
    );
    unsubscribe();

    expect(events).toEqual([
      expect.objectContaining({ runtime: "codex", conversationId: "codex:thread-1" }),
    ]);
    expect(source.close).toHaveBeenCalledOnce();
  });

  it("rejects duplicate or regressing SSE sequence numbers before projection", () => {
    const source = new FakeEventSource();
    const events: unknown[] = [];
    const onError = vi.fn();
    const client = new ConversationRuntimeClient(undefined, () => source);
    client.subscribe((event) => events.push(event), onError);
    const event = {
      seq: 2,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "turn_status",
      turnId: "codex:turn-1",
      status: "running",
    };

    source.onmessage?.(new MessageEvent("message", { data: JSON.stringify(event) }));
    source.onmessage?.(new MessageEvent("message", { data: JSON.stringify({ ...event, seq: 1 }) }));

    expect(events).toEqual([event]);
    expect(onError).toHaveBeenCalledOnce();
  });

  it("surfaces bounded Host errors", async () => {
    const client = new ConversationRuntimeClient(
      async () =>
        new Response(JSON.stringify({ error: "Codex runtime is unavailable." }), {
          headers: { "content-type": "application/json" },
          status: 503,
        }),
    );

    await expect(client.create("codex")).rejects.toThrow("Codex runtime is unavailable.");
  });

  it("submits typed elicitation values as form content", async () => {
    const approval = {
      seq: 1,
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      type: "approval_requested" as const,
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      approvalId: "codex:approval-1",
      kind: "elicitation" as const,
      prompt: "Confirm",
      choices: ["accept", "decline"] as const,
      questions: [
        {
          id: "confirm",
          type: "boolean" as const,
          prompt: "Confirm",
          required: true,
        },
        {
          id: "count",
          type: "integer" as const,
          prompt: "Count",
          required: true,
        },
        {
          id: "tags",
          type: "string_array" as const,
          prompt: "Tags",
          defaultValue: ["safe"],
        },
      ],
    };
    const fields = approvalResponseFields(approval, "accept", {
      confirm: true,
      count: "2",
    });
    const requests: RequestInit[] = [];
    const client = new ConversationRuntimeClient(async (_input, init) => {
      if (init !== undefined) requests.push(init);
      return new Response("null", { status: 200 });
    });

    expect(fields).toEqual({ form: { confirm: true, count: 2, tags: ["safe"] } });
    expect(approvalResponseFields(approval, "accept", { count: "2" })).toEqual({
      form: { confirm: false, count: 2, tags: ["safe"] },
    });
    await client.approve(approval, "accept", fields);
    expect(JSON.parse(String(requests[0]?.body))).toMatchObject({
      decision: "accept",
      form: { confirm: true, count: 2, tags: ["safe"] },
    });
    expect(() => approvalResponseFields(approval, "accept", { confirm: true })).toThrow(
      '"Count" is required.',
    );
  });

  it("preserves legal inherited-name form fields as own JSON properties", () => {
    const approval = {
      seq: 1,
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      type: "approval_requested" as const,
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      approvalId: "codex:approval-1",
      kind: "elicitation" as const,
      prompt: "Confirm",
      choices: ["accept", "decline"] as const,
      questions: [{ id: "toString", type: "string" as const, prompt: "Value", required: true }],
    };

    const result = approvalResponseFields(
      approval,
      "accept",
      Object.fromEntries([["toString", "safe"]]),
    );
    expect(result).toEqual({ form: { toString: "safe" } });
    expect(Object.hasOwn(result?.form ?? {}, "toString")).toBe(true);
  });

  it("projects streamed deltas and authoritative completed items without another transcript", () => {
    const snapshot = {
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      workspace: { id: "workspace-1", label: "workspace" },
      title: "Thread",
      archived: false,
      turns: [],
    };
    const partial = projectRuntimeEvent(snapshot, {
      seq: 1,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_delta",
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      itemType: "assistant_message",
      delta: "hel",
    });
    const completed = projectRuntimeEvent(partial, {
      seq: 2,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_completed",
      turnId: "codex:turn-1",
      item: {
        type: "assistant_message",
        id: "codex:item-1",
        turnId: "codex:turn-1",
        text: "hello",
        createdAt: 1,
      },
    });
    const afterLateDelta = projectRuntimeEvent(completed, {
      seq: 3,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_delta",
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      itemType: "assistant_message",
      delta: " ignored",
    });

    expect(afterLateDelta.turns).toEqual([
      {
        id: "codex:turn-1",
        status: "running",
        items: [
          {
            type: "assistant_message",
            id: "codex:item-1",
            turnId: "codex:turn-1",
            text: "hello",
            createdAt: 1,
          },
        ],
      },
    ]);
    expect(snapshot.turns).toEqual([]);
  });

  it("keeps a durable completed item when a contradictory completion arrives", () => {
    const snapshot = {
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      workspace: { id: "workspace-1", label: "workspace" },
      title: "Thread",
      archived: false,
      turns: [
        {
          id: "codex:turn-1",
          status: "completed" as const,
          items: [
            {
              type: "assistant_message" as const,
              id: "codex:item-1",
              turnId: "codex:turn-1",
              text: "durable",
              createdAt: 1,
            },
          ],
        },
      ],
    };

    const projected = projectRuntimeEvent(snapshot, {
      seq: 3,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_completed",
      turnId: "codex:turn-1",
      item: {
        type: "assistant_message",
        id: "codex:item-1",
        turnId: "codex:turn-1",
        text: "contradiction",
        createdAt: 2,
      },
    });

    expect(projected).toBe(snapshot);
    expect(projected.turns[0]?.items[0]).toMatchObject({ text: "durable" });
  });

  it("continues a running native snapshot and completes its running tool", () => {
    const snapshot = {
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      workspace: { id: "workspace-1", label: "workspace" },
      title: "Thread",
      archived: false,
      turns: [
        {
          id: "codex:turn-1",
          status: "running" as const,
          items: [
            {
              type: "assistant_message" as const,
              id: "codex:item-1",
              turnId: "codex:turn-1",
              text: "hel",
              createdAt: 1,
              provisional: true as const,
            },
            {
              type: "tool" as const,
              id: "codex:tool-1",
              turnId: "codex:turn-1",
              name: "shell",
              status: "running" as const,
              createdAt: 1,
            },
          ],
        },
      ],
    };
    const afterDelta = projectRuntimeEvent(snapshot, {
      seq: 3,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_delta",
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      delta: "lo",
    });
    const afterTool = projectRuntimeEvent(afterDelta, {
      seq: 4,
      runtime: "codex",
      conversationId: "codex:thread-1",
      type: "item_completed",
      turnId: "codex:turn-1",
      item: {
        type: "tool",
        id: "codex:tool-1",
        turnId: "codex:turn-1",
        name: "shell",
        status: "completed",
        summary: "done",
        createdAt: 1,
      },
    });

    expect(afterTool.turns[0]?.items).toEqual([
      expect.objectContaining({ id: "codex:item-1", text: "hello", provisional: true }),
      expect.objectContaining({ id: "codex:tool-1", status: "completed", summary: "done" }),
    ]);
  });

  it("removes only the exact server-resolved approval", () => {
    const first = {
      seq: 1,
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      type: "approval_requested" as const,
      turnId: "codex:turn-1",
      itemId: "codex:item-1",
      approvalId: "codex:approval-1",
      kind: "command" as const,
      prompt: "Run?",
      choices: ["accept", "decline"] as const,
    };
    const second = { ...first, itemId: "codex:item-2" };

    expect(approvalIdentityKey(first)).not.toBe(approvalIdentityKey(second));
    expect(removeApprovalRequest([first, second], first)).toEqual([second]);
    const firstAnswer = approvalAnswerKey(first, "confirm");
    const secondAnswer = approvalAnswerKey(second, "confirm");
    expect(firstAnswer).not.toBe(secondAnswer);
    expect(removeApprovalAnswers({ [firstAnswer]: true, [secondAnswer]: false }, first)).toEqual({
      [secondAnswer]: false,
    });

    expect(
      projectApprovalEvent([first, second], {
        seq: 2,
        runtime: "codex",
        conversationId: "codex:thread-1",
        type: "approval_resolved",
        turnId: "codex:turn-1",
        itemId: "codex:item-1",
        approvalId: "codex:approval-1",
      }),
    ).toEqual([second]);
  });
});
