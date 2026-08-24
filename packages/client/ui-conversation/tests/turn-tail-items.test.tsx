import { describe, expect, it, vi } from "vitest";
import { TurnTailItems } from "../src/client/turn-tail-items.js";

describe("V99 turn-tail item bridge", () => {
  it("renders only explicitly registered child contributions", () => {
    const child = <span>Generated paper</span>;
    const renderSlot = vi.fn(() => child);

    expect(TurnTailItems({ matched: 7, renderSlot } as never)).toBe(child);
    expect(renderSlot).toHaveBeenCalledWith("conversation.chat.turnTail.items", { turn: 7 });
  });
});
