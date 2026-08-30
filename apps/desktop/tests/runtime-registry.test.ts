import { describe, expect, it, vi } from "vitest";
import type {
  ConversationRuntime,
  ConversationSummary,
  RuntimeEvent,
  RuntimeKind,
} from "../src/runtime/contracts.js";
import { ConversationRuntimeRegistry } from "../src/runtime/registry.js";

function runtime(kind: RuntimeKind, updatedAt: number): ConversationRuntime {
  const listeners = new Set<(event: RuntimeEvent) => void>();
  const summary: ConversationSummary = {
    runtime: kind,
    conversationId: `${kind}-native-1`,
    workspace: { id: "workspace", label: "workspace" },
    title: `${kind} conversation`,
    archived: false,
    updatedAt,
  };
  return {
    kind,
    list: vi.fn(async () => [summary]),
    create: vi.fn(async () => summary),
    read: vi.fn(async () => ({ ...summary, turns: [] })),
    start: vi.fn(async () => ({ turnId: `${kind}-turn-1` })),
    steer: vi.fn(async () => {}),
    interrupt: vi.fn(async () => {}),
    revise: vi.fn(async () => summary),
    fork: vi.fn(async () => summary),
    archive: vi.fn(async () => {}),
    subscribe: vi.fn((listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    }),
    respondToApproval: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
  };
}

describe("conversation runtime registry", () => {
  it("registers peer adapters without falling back across runtime kinds", async () => {
    const dsh = runtime("dsh", 10);
    const codex = runtime("codex", 20);
    const registry = new ConversationRuntimeRegistry([dsh, codex], "codex");

    expect(registry.defaultKind).toBe("codex");
    expect(registry.kinds()).toEqual(["dsh", "codex"]);
    await expect(registry.runtime("codex").list()).resolves.toMatchObject([
      { runtime: "codex", conversationId: "codex-native-1" },
    ]);
    expect(dsh.list).not.toHaveBeenCalled();
    expect(() => registry.runtime("missing" as RuntimeKind)).toThrow(
      'Conversation runtime "missing" is not registered',
    );
  });

  it("multiplexes qualified native events and disposes each adapter once", async () => {
    const dsh = runtime("dsh", 10);
    const codex = runtime("codex", 20);
    const registry = new ConversationRuntimeRegistry([dsh, codex], "dsh");
    const listener = vi.fn();
    const secondListener = vi.fn();
    const unsubscribe = registry.subscribe(listener);
    const unsubscribeSecond = registry.subscribe(secondListener);
    const dshListener = vi.mocked(dsh.subscribe).mock.calls[0]?.[0];
    const codexListener = vi.mocked(codex.subscribe).mock.calls[0]?.[0];
    dshListener?.({
      type: "turn_status",
      seq: 1,
      runtime: "dsh",
      conversationId: "dsh-native-1",
      turnId: "dsh-turn-1",
      status: "running",
    });
    codexListener?.({
      type: "turn_status",
      seq: 1,
      runtime: "codex",
      conversationId: "codex-native-1",
      turnId: "codex-turn-1",
      status: "running",
    });

    expect(listener.mock.calls.map(([event]) => event.runtime)).toEqual(["dsh", "codex"]);
    expect(listener.mock.calls.map(([event]) => event.seq)).toEqual([1, 2]);
    expect(secondListener.mock.calls.map(([event]) => event.seq)).toEqual([1, 2]);
    expect(dsh.subscribe).toHaveBeenCalledTimes(1);
    expect(codex.subscribe).toHaveBeenCalledTimes(1);
    unsubscribe();
    unsubscribeSecond();
    await registry.dispose();
    await registry.dispose();
    expect(dsh.dispose).toHaveBeenCalledTimes(1);
    expect(codex.dispose).toHaveBeenCalledTimes(1);
  });

  it("enforces adapter order and completed-item authority on the production fanout", () => {
    const dsh = runtime("dsh", 10);
    const codex = runtime("codex", 20);
    const registry = new ConversationRuntimeRegistry([dsh, codex], "dsh");
    const listener = vi.fn();
    registry.subscribe(listener);
    const dshListener = vi.mocked(dsh.subscribe).mock.calls[0]?.[0];
    const codexListener = vi.mocked(codex.subscribe).mock.calls[0]?.[0];
    const completed: RuntimeEvent = {
      type: "item_completed",
      seq: 1,
      runtime: "dsh",
      conversationId: "dsh-native-1",
      turnId: "dsh-turn-1",
      item: {
        type: "assistant_message",
        id: "dsh-item-1",
        turnId: "dsh-turn-1",
        text: "authoritative",
        createdAt: 1,
      },
    };

    dshListener?.(completed);
    dshListener?.({
      type: "item_delta",
      seq: 2,
      runtime: "dsh",
      conversationId: "dsh-native-1",
      turnId: "dsh-turn-1",
      itemId: "dsh-item-1",
      delta: " ignored",
    });
    dshListener?.({ ...completed, seq: 3 });
    codexListener?.({
      type: "turn_status",
      seq: 1,
      runtime: "codex",
      conversationId: "codex-native-1",
      turnId: "codex-turn-1",
      status: "running",
    });

    expect(listener.mock.calls.map(([event]) => event.seq)).toEqual([1, 2]);
    expect(listener.mock.calls.map(([event]) => event.type)).toEqual([
      "item_completed",
      "turn_status",
    ]);
    expect(() =>
      dshListener?.({
        ...completed,
        seq: 4,
        item: { ...completed.item, text: "contradiction" },
      }),
    ).toThrow('Contradictory completed item "dsh-item-1"');
    expect(() => dshListener?.({ ...completed, seq: 4 })).toThrow("event sequence 4");
  });

  it("fails closed at the bounded active completed-item projection limit", () => {
    const dsh = runtime("dsh", 10);
    const registry = new ConversationRuntimeRegistry([dsh], "dsh");
    registry.subscribe(() => undefined);
    const emit = vi.mocked(dsh.subscribe).mock.calls[0]?.[0];

    for (let index = 1; index <= 10_000; index += 1) {
      emit?.({
        type: "item_completed",
        seq: index,
        runtime: "dsh",
        conversationId: "dsh-native-1",
        turnId: "dsh-turn-1",
        item: {
          type: "assistant_message",
          id: `dsh-item-${String(index)}`,
          turnId: "dsh-turn-1",
          text: "bounded",
          createdAt: index,
        },
      });
    }
    expect(() =>
      emit?.({
        type: "item_completed",
        seq: 10_001,
        runtime: "dsh",
        conversationId: "dsh-native-1",
        turnId: "dsh-turn-1",
        item: {
          type: "assistant_message",
          id: "dsh-item-overflow",
          turnId: "dsh-turn-1",
          text: "rejected",
          createdAt: 10_001,
        },
      }),
    ).toThrow("active completed-item limit 10000 exceeded");
  });
});
