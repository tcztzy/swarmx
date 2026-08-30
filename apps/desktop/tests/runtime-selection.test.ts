import { describe, expect, it } from "vitest";
import { resolveRuntimeSelection } from "../src/runtime/selection.js";

describe("runtime selection", () => {
  it("defaults to DSH and accepts one explicit CLI or environment selection", () => {
    expect(resolveRuntimeSelection([], {})).toBe("dsh");
    expect(resolveRuntimeSelection([], { SWARMX_RUNTIME: "codex" })).toBe("codex");
    expect(resolveRuntimeSelection(["--runtime", "codex"], { SWARMX_RUNTIME: "dsh" })).toBe(
      "codex",
    );
    expect(resolveRuntimeSelection(["--runtime=dsh"], {})).toBe("dsh");
  });

  it("rejects unknown, missing, and repeated runtime arguments", () => {
    expect(() => resolveRuntimeSelection(["--runtime"], {})).toThrow("requires dsh or codex");
    expect(() => resolveRuntimeSelection(["--runtime", "other"], {})).toThrow(
      'Unknown runtime "other"',
    );
    expect(() => resolveRuntimeSelection(["--runtime=dsh", "--runtime=codex"], {})).toThrow(
      "specified more than once",
    );
  });
});
