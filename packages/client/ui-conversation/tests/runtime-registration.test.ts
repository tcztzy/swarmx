import type { Context } from "@deepseek-ai/cordis";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RuntimeEvent } from "../src/client/runtime-client.js";
import {
  ConversationListFence,
  ConversationOpenFence,
  initialConversationId,
  PendingOperationCounter,
  registerPeerRuntimeConversation,
  runtimeControlState,
  terminalTurnAction,
} from "../src/client/runtime-conversation.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

function context() {
  const register = vi.fn(() => vi.fn());
  const inject = vi.fn((_name: string, install: () => () => void) => install());
  const disposers: Array<() => void> = [];
  const ctx = {
    effect: vi.fn((effect: () => () => void) => disposers.push(effect())),
    slots: { inject, register },
  } as unknown as Context;
  return { ctx, disposers, inject, register };
}

function metadata(defaultRuntimeKind: "dsh" | "codex") {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            defaultRuntimeKind,
            runtimeKinds: defaultRuntimeKind === "dsh" ? ["dsh"] : ["dsh", "codex"],
          }),
          { status: 200 },
        ),
    ),
  );
}

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function turnStatusEvent(conversationId: string, status: "running" | "completed"): RuntimeEvent {
  return {
    seq: 1,
    runtime: "codex",
    conversationId,
    type: "turn_status",
    turnId: "turn-1",
    status,
  };
}

describe("peer runtime Conversation registration", () => {
  it("leaves the published DSH Conversation occupant unchanged for DSH", async () => {
    metadata("dsh");
    const fixture = context();

    registerPeerRuntimeConversation(fixture.ctx);
    await vi.waitFor(() => expect(fetch).toHaveBeenCalledOnce());

    expect(fixture.register).not.toHaveBeenCalled();
    for (const dispose of fixture.disposers) dispose();
  });

  it("shadows only the Conversation slot for the selected Codex peer", async () => {
    metadata("codex");
    const fixture = context();

    registerPeerRuntimeConversation(fixture.ctx);
    await vi.waitFor(() => expect(fixture.register).toHaveBeenCalledOnce());

    expect(fixture.inject).toHaveBeenCalledWith("conversation", expect.any(Function));
    expect(fixture.register.mock.calls[0]?.[0]).toEqual({
      name: "conversation",
      priority: -10,
    });
    for (const dispose of fixture.disposers) dispose();
  });

  it("V227 surfaces metadata failure instead of silently showing DSH", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new Error("metadata unavailable"))),
    );
    const fixture = context();

    registerPeerRuntimeConversation(fixture.ctx);
    await vi.waitFor(() => expect(fixture.register).toHaveBeenCalledOnce());

    expect(fixture.inject).toHaveBeenCalledWith("conversation", expect.any(Function));
    expect(fixture.register.mock.calls[0]?.[0]).toEqual({
      name: "conversation",
      priority: -10,
    });
    expect(fixture.register.mock.calls[0]?.[1]).toBeTypeOf("function");
    for (const dispose of fixture.disposers) dispose();
  });
});

describe("V200 peer runtime Conversation selection", () => {
  it("refreshes the list for a background terminal turn without reloading that conversation", () => {
    expect(terminalTurnAction(turnStatusEvent("background", "completed"), "selected")).toEqual({
      refreshList: true,
    });
    expect(terminalTurnAction(turnStatusEvent("selected", "completed"), "selected")).toEqual({
      refreshList: true,
      reloadConversationId: "selected",
    });
    expect(terminalTurnAction(turnStatusEvent("selected", "running"), "selected")).toBeUndefined();
  });

  it("aborts the superseded read and prevents its late response from replacing the new choice", async () => {
    const fence = new ConversationOpenFence();
    const first = deferred<string>();
    const second = deferred<string>();
    const applied: string[] = [];
    let firstSignal: AbortSignal | undefined;

    const firstOpen = fence.open(
      "first",
      (signal) => {
        firstSignal = signal;
        return first.promise;
      },
      (value) => applied.push(value),
    );
    const secondOpen = fence.open(
      "second",
      () => second.promise,
      (value) => applied.push(value),
    );

    expect(firstSignal?.aborted).toBe(true);
    expect(fence.selectedConversationId).toBe("second");

    second.resolve("second");
    await expect(secondOpen).resolves.toBe(true);
    first.resolve("first");
    await expect(firstOpen).resolves.toBe(false);
    expect(applied).toEqual(["second"]);
  });

  it("restores the committed selection when the current read fails", async () => {
    const fence = new ConversationOpenFence();
    const applied: string[] = [];
    await fence.open(
      "first",
      async () => "first",
      (value) => applied.push(value),
    );

    await expect(
      fence.open(
        "second",
        async () => {
          throw new Error("read failed");
        },
        (value) => applied.push(value),
      ),
    ).rejects.toThrow("read failed");

    expect(fence.selectedConversationId).toBe("first");
    expect(applied).toEqual(["first"]);
  });

  it("aborts a superseded list refresh and ignores its late response", async () => {
    const fence = new ConversationListFence();
    const first = deferred<readonly string[]>();
    const second = deferred<readonly string[]>();
    const applied: readonly string[][] = [];
    let firstSignal: AbortSignal | undefined;

    const firstRefresh = fence.refresh(
      (signal) => {
        firstSignal = signal;
        return first.promise;
      },
      (value) => applied.push(value),
    );
    const secondRefresh = fence.refresh(
      () => second.promise,
      (value) => applied.push(value),
    );

    expect(firstSignal?.aborted).toBe(true);
    second.resolve(["second"]);
    await expect(secondRefresh).resolves.toEqual(["second"]);
    first.resolve(["first"]);
    await expect(firstRefresh).resolves.toBeUndefined();
    expect(applied).toEqual([["second"]]);
  });

  it("does not auto-select a stale first item after a successful manual choice", async () => {
    const conversations = [
      {
        runtime: "codex" as const,
        conversationId: "first",
        workspace: { id: "workspace", label: "workspace" },
        title: "First",
        archived: false,
        updatedAt: 1,
      },
    ];
    const listFence = new ConversationListFence();
    const slowList = deferred<typeof conversations>();
    const refresh = listFence.refresh(
      () => slowList.promise,
      () => {},
    );
    const openFence = new ConversationOpenFence();
    await openFence.open(
      "selected",
      async () => "selected",
      () => {},
    );
    slowList.resolve(conversations);
    const items = await refresh;

    expect(initialConversationId(items, openFence.selectedConversationId)).toBeUndefined();
    expect(initialConversationId(conversations, undefined)).toBe("first");
  });

  it("keeps selection and archive controls unavailable for conflicting operations", () => {
    expect(runtimeControlState(true, false)).toEqual({
      archiveDisabled: true,
      selectConversationDisabled: true,
    });
    expect(runtimeControlState(false, true)).toEqual({
      archiveDisabled: true,
      selectConversationDisabled: false,
    });
    expect(runtimeControlState(false, false)).toEqual({
      archiveDisabled: false,
      selectConversationDisabled: false,
    });
  });

  it("keeps busy while any interleaved operation remains pending", () => {
    const pending = new PendingOperationCounter();
    pending.begin();
    pending.begin();

    expect(pending.finish()).toBe(true);
    expect(pending.busy).toBe(true);
    expect(pending.finish()).toBe(false);
    expect(pending.busy).toBe(false);
  });
});
