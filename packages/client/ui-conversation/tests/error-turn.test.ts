import { describe, expect, it } from "vitest";
import { selectFailedTurn } from "../src/error-turn.js";

describe("selectFailedTurn", () => {
  it("exposes a terminal connection failure for the retry row", () => {
    const owner = {
      turn: {
        turn: 3,
        end: {
          data: {
            reason: {
              kind: "error",
              error: { name: "Error", message: "Connection error", code: "connection-error" },
            },
          },
        },
      },
    };
    expect(selectFailedTurn(owner as never)).toEqual({
      turn: 3,
      message: "Connection error",
      code: "connection-error",
    });
  });

  it("declines a completed turn", () => {
    const owner = { turn: { turn: 1, end: { data: { reason: { kind: "completed" } } } } };
    expect(selectFailedTurn(owner as never)).toBeNull();
  });
});
