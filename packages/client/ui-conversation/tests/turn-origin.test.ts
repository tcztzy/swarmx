import { describe, expect, it } from "vitest";
import { type LookupNode, turnTextOf } from "../src/turn-origin.js";

/** A user message node. */
function user(seq: number, text: string): LookupNode {
  return { kind: "user", seq, content: [{ type: "text", text }] };
}

const TURN = { turn: 2, status: "closed" as const, start: { seq: 2 }, end: { seq: 5 } };

describe("turnTextOf", () => {
  it("resolves the user text inside the requested turn boundaries", () => {
    expect(turnTextOf([user(1, "first"), user(3, "second")], TURN)).toBe("second");
  });

  it("joins multi-block text and trims it", () => {
    const nodes: LookupNode[] = [
      {
        kind: "user",
        seq: 1,
        content: [
          { type: "text", text: " a" },
          { type: "text", text: "b " },
        ],
      },
    ];
    expect(turnTextOf(nodes, { ...TURN, start: { seq: 0 } })).toBe("ab");
  });

  it("ignores non-text blocks such as images", () => {
    const nodes: LookupNode[] = [
      { kind: "user", seq: 1, content: [{ type: "image" }, { type: "text", text: "caption" }] },
    ];
    expect(turnTextOf(nodes, { ...TURN, start: { seq: 0 } })).toBe("caption");
  });

  it("returns undefined when the opening message is outside the window", () => {
    expect(turnTextOf([user(1, "older")], TURN)).toBeUndefined();
  });

  it("ignores steering messages inside the turn", () => {
    const steering: LookupNode = {
      kind: "steering",
      seq: 2,
      messageId: "m1",
      content: [{ type: "text", text: "queued" }],
    };
    expect(turnTextOf([user(1, "older"), steering], TURN)).toBeUndefined();
  });

  it("returns undefined when the opening message has no text", () => {
    const imageOnly: LookupNode = { kind: "user", seq: 1, content: [{ type: "image" }] };
    expect(turnTextOf([imageOnly], { ...TURN, start: { seq: 0 } })).toBeUndefined();
  });
});
