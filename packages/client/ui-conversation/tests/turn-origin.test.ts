import { describe, expect, it } from "vitest";
import { turnTextOf } from "../src/turn-origin.js";

function chat(nodes: Record<string, unknown>, turnKeys = Object.keys(nodes)) {
  return {
    nodes: {
      get: (key: string) => nodes[key],
      values: () => Object.values(nodes),
    },
    locations: {
      getTurn: () => turnKeys,
      getStep: () => [],
    },
  } as never;
}

function user(text: string) {
  return { kind: "user", data: { content: [{ type: "text", text }] } };
}

describe("turnTextOf", () => {
  it("resolves the user text indexed under the requested turn", () => {
    expect(turnTextOf(chat({ first: user("first"), second: user("second") }, ["second"]), 2)).toBe(
      "second",
    );
  });

  it("joins multi-block text and trims it", () => {
    expect(
      turnTextOf(
        chat({
          user: {
            kind: "user",
            data: {
              content: [
                { type: "text", text: " a" },
                { type: "text", text: "b " },
              ],
            },
          },
        }),
        2,
      ),
    ).toBe("ab");
  });

  it("ignores non-text blocks such as images", () => {
    const user = {
      kind: "user",
      data: { content: [{ type: "image" }, { type: "text", text: "caption" }] },
    };
    expect(turnTextOf(chat({ user }), 2)).toBe("caption");
  });

  it("returns undefined when the opening message is outside the window", () => {
    expect(turnTextOf(chat({ older: user("older") }, []), 2)).toBeUndefined();
  });

  it("ignores steering messages inside the turn", () => {
    const steering = {
      kind: "steering",
      data: { content: [{ type: "text", text: "queued" }] },
    };
    expect(turnTextOf(chat({ steering }), 2)).toBeUndefined();
  });

  it("returns undefined when the opening message has no text", () => {
    const imageOnly = { kind: "user", data: { content: [{ type: "image" }] } };
    expect(turnTextOf(chat({ imageOnly }), 2)).toBeUndefined();
  });
});
