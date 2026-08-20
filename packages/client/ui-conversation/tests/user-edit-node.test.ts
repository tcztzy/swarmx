import { describe, expect, it } from "vitest";
import { userEditDefinition } from "../src/user-edit-node.js";

const event = {
  type: "user/message",
  seq: 4,
  time: 100,
  data: {
    id: "user-1",
    role: "user",
    content: [{ type: "text", text: " revise me " }],
    source: { kind: "user" },
  },
  surfaceOp: "append",
};

describe("userEditDefinition", () => {
  it("publishes an edit node immediately after a user-authored message", () => {
    expect(userEditDefinition.match(event as never)).toEqual({ id: "user-1", role: "start" });
    const location = {
      kind: "turn",
      turn: { turn: 2, status: "closed", steps: [], data: { get: () => undefined } },
    };
    const match = { event, role: "start", location };
    const state = userEditDefinition.start(
      { matches: [match], start: match } as never,
      match as never,
      {
        previous: () => undefined,
      },
    );
    const node = userEditDefinition.buildViewNode?.({
      key: "swarmx-user-edit:user-1",
      kind: "swarmx-user-edit",
      id: "user-1",
      matches: [match],
      start: match,
      state,
      current: new Map(),
    } as never);
    expect(node).toMatchObject({
      kind: "swarmx-user-edit",
      anchorSeq: 4.01,
      data: { turn: 2, text: "revise me" },
    });
  });

  it("ignores plugin-authored context messages", () => {
    expect(
      userEditDefinition.match({
        ...event,
        data: { ...event.data, source: { kind: "plugin", plugin: "test" } },
      } as never),
    ).toBeNull();
  });
});
