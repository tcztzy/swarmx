import { describe, expect, it } from "vitest";
import { GIT_UI_REMOTE } from "../src/remote.js";
import { GIT_UI_INVOCATIONS } from "../src/remote-contract.js";
import { TYPERT } from "../src/typert.js";

describe("Git UI Typert contract", () => {
  it("keeps Host and Client descriptors identical and read-only", () => {
    expect(TYPERT.invocations).toBe(GIT_UI_INVOCATIONS);
    expect(GIT_UI_REMOTE.descriptors).toBe(GIT_UI_INVOCATIONS);
    expect(GIT_UI_INVOCATIONS.map(({ method }) => method)).toEqual(["snapshot"]);
  });
});
