import { describe, expect, it } from "vitest";
import { findToolCall, latestToolCallInTurn } from "../src/client/tool-side-view.js";

const child = {
  callId: "child-call",
  name: "read",
  argsRaw: '{"path":"report.md"}',
  turn: 1,
  step: 2,
  time: 2,
  callView: null,
  subCalls: [],
};

const root = {
  callId: "root-call",
  name: "science_query",
  argsRaw: "{}",
  turn: 1,
  step: 1,
  time: 1,
  callView: null,
  subCalls: [child],
};

function snapshot() {
  return {
    chat: {
      locations: { getTurn: () => ["tool:root-call"] },
      nodes: {
        get: () => ({ kind: "tool-call", data: { root } }),
        values: () => [{ kind: "tool-call", data: { root } }],
      },
    },
    runningCalls: [root],
  };
}

describe("V49 Tool Details routing", () => {
  it("finds root and nested calls through the public conversation snapshot", () => {
    expect(findToolCall(snapshot() as never, "root-call")).toBe(root);
    expect(findToolCall(snapshot() as never, "child-call")).toBe(child);
    expect(findToolCall(snapshot() as never, "missing")).toBeUndefined();
  });

  it("selects the latest Tool root in a completed turn", () => {
    expect(latestToolCallInTurn(snapshot() as never, 1)).toBe(root);
  });
});
