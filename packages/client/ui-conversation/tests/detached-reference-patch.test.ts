import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const runtime = readFileSync(
  new URL("../node_modules/@deepseek-ai/dsh-client-ui-conversation/lib/client.js", import.meta.url),
  "utf8",
);

describe("V128 detached reference seam", () => {
  it("keeps detached occurrences out of visible draft geometry and admits annotation-only submit", () => {
    expect(runtime).toContain('reference.placement === "detached"');
    expect(runtime).toContain('o.placement !== "detached"');
    expect(runtime).toContain("occurrences.length === 0");
    expect(runtime).toContain("input?.occurrences.length ?? 0");
  });
});
