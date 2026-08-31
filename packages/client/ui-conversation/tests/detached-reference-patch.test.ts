import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const packageRoot = new URL(
  "../node_modules/@deepseek-ai/dsh-client-ui-conversation/lib/",
  import.meta.url,
);
const runtime = readFileSync(new URL("client.js", packageRoot), "utf8");
const contract = readFileSync(new URL("types/client/contract/input.d.ts", packageRoot), "utf8");
const facade = readFileSync(new URL("types/client/input/facade.d.ts", packageRoot), "utf8");

describe("V128 detached reference seam", () => {
  it("keeps detached occurrences outside Lexical geometry and admits annotation-only submit", () => {
    expect(runtime).toContain('ref.placement === "detached"');
    expect(runtime).toContain('placement: "detached"');
    expect(runtime).toContain("this.detachedOccurrences = [...this.detachedOccurrences");
    expect(runtime).toContain('applied = $replaceDetectSpanWithText(span, "")');
    expect(runtime).toContain("const occurrences = this.occurrences");
    expect(runtime).toContain("this.detachedOccurrences = []");
    expect(runtime).toContain('if (trimmed === "" && !hasReferences) return []');
    expect(runtime).toContain("hasReferences: this.occurrences.length > 0");
    expect(runtime).toContain("input?.occurrences.length ?? 0");
  });

  it("publishes the detached placement and occurrence mutation contract", () => {
    expect(contract).toContain("readonly placement?: 'inline' | 'detached'");
    expect(contract).toContain(
      "replaceReference(occurrenceId: number, ref: ReferenceInsert): boolean",
    );
    expect(contract).toContain("removeReference(occurrenceId: number): boolean");
    expect(facade).toContain(
      "replaceReference(occurrenceId: number, ref: ReferenceInsert): boolean",
    );
    expect(facade).toContain("removeReference(occurrenceId: number): boolean");
  });
});
