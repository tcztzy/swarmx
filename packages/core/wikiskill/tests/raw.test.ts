import { describe, expect, it, vi } from "vitest";
import { DshRawTraceReader } from "../src/index.js";

function event(seq: number) {
  return { data: { value: seq }, seq, time: 1_000 + seq, type: "fixture/event" };
}

describe("V233 DSH Raw trace reader", () => {
  it("returns one bounded contiguous native event window without creating a Raw store", async () => {
    const readSession = vi.fn(async () => ({
      session: { createdAt: 1, id: "session-1" },
      events: [event(0), event(1), event(2), event(3)],
    }));
    const reader = new DshRawTraceReader({ readSession } as never, { maxEvents: 3 });

    await expect(reader.read({ endSeq: 3, sessionId: "session-1", startSeq: 1 })).resolves.toEqual({
      events: [event(1), event(2), event(3)],
      locator: { endSeq: 3, sessionId: "session-1", startSeq: 1 },
      trust: "untrusted-execution-trace",
    });
    expect(readSession).toHaveBeenCalledWith("session-1");
  });

  it("fails closed on gaps, oversized windows, and cancellation after a native read", async () => {
    const controller = new AbortController();
    const readSession = vi.fn(async () => {
      controller.abort();
      return {
        session: { createdAt: 1, id: "session-1" },
        events: [event(0), event(2), event(3)],
      };
    });
    const reader = new DshRawTraceReader({ readSession } as never, { maxEvents: 2 });

    await expect(
      reader.read({ endSeq: 1, sessionId: "session-1", startSeq: 0 }, controller.signal),
    ).rejects.toMatchObject({ name: "AbortError" });

    const activeReader = new DshRawTraceReader(
      {
        readSession: vi.fn(async () => ({
          session: { createdAt: 1, id: "session-1" },
          events: [event(0), event(2), event(3)],
        })),
      } as never,
      { maxEvents: 3 },
    );
    await expect(
      activeReader.read({ endSeq: 2, sessionId: "session-1", startSeq: 0 }),
    ).rejects.toMatchObject({ code: "WIKISKILL_RAW_GAP" });
    await expect(
      activeReader.read({ endSeq: 3, sessionId: "session-1", startSeq: 0 }),
    ).rejects.toMatchObject({ code: "WIKISKILL_INVALID_REQUEST" });
  });
});
