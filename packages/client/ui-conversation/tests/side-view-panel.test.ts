import { describe, expect, it } from "vitest";
import { sideViewTabTarget } from "../src/client/side-view-panel.js";

const entryIds = ["one", "two", "three"];

describe("V49 Side View keyboard navigation", () => {
  it("wraps directional navigation and supports Home/End", () => {
    expect(sideViewTabTarget(entryIds, "two", "ArrowRight")).toBe("three");
    expect(sideViewTabTarget(entryIds, "three", "ArrowRight")).toBe("one");
    expect(sideViewTabTarget(entryIds, "one", "ArrowLeft")).toBe("three");
    expect(sideViewTabTarget(entryIds, "two", "Home")).toBe("one");
    expect(sideViewTabTarget(entryIds, "two", "End")).toBe("three");
    expect(sideViewTabTarget(entryIds, "two", "Enter")).toBeNull();
  });
});
