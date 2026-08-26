import { describe, expect, it } from "vitest";
import { SWARM_INVOCATIONS } from "../src/remote-contract.js";

describe("V169/V171 strict Swarm Remote", () => {
  it("publishes only bounded read projection and wait methods", () => {
    expect(SWARM_INVOCATIONS.map((descriptor) => descriptor.method)).toEqual([
      "uiSnapshot",
      "waitUi",
    ]);
    expect(SWARM_INVOCATIONS.every((descriptor) => descriptor.service === "swarm")).toBe(true);
    expect(
      SWARM_INVOCATIONS.every(
        (descriptor) =>
          descriptor.result.mode === "strict" && descriptor.cancellation.parameter === "signal",
      ),
    ).toBe(true);
    expect(SWARM_INVOCATIONS.map((descriptor) => descriptor.method).join(" ")).not.toMatch(
      /createTask|sendMessage|archive|reassign|messageBody|workspacePath/u,
    );
  });
});
