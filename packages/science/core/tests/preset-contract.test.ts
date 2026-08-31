import { describe, expect, it, vi } from "vitest";
import { apply, name } from "../src/preset.js";

describe("V103/V111/V122 Science preset contract", () => {
  it("registers only Science guidance and the managed Typst guard", () => {
    const sections: Array<{ name: string; order: number; text: string }> = [];
    let guard:
      | ((execution: { readonly name: string; readonly arguments: unknown }) => string | undefined)
      | undefined;
    const disposeGuard = vi.fn();
    const getSectionOrder = vi.fn(() => 5_000);

    const dispose = apply({
      systemPrompt: { getSectionOrder, section: (section) => sections.push(section) },
      tools: {
        guard: (candidate) => {
          guard = candidate;
          return disposeGuard;
        },
      },
    } as never) as unknown as () => void;

    expect(name).toBe("swarmx-science-preset-contract");
    expect(sections.map((section) => section.name)).toEqual([
      "science:typst-workflow",
      "science:annotations",
      "science:literature-search",
      "science:resource-addressing",
    ]);
    expect(getSectionOrder).toHaveBeenCalledWith("TOOLS_SDK");
    expect(sections.map((section) => section.order)).toEqual([5_100, 5_101, 5_102, 5_103]);
    expect(sections[0]?.text).toContain("Do not call `typst compile` or `typst watch`");
    expect(sections[1]?.text).toContain("science_query");
    expect(sections[2]?.text).toContain("`literature_search`");
    expect(sections[2]?.text).toContain("local Zotero");
    expect(sections[3]?.text).toContain("`head`");
    expect(sections[3]?.text).toContain("`exactId`");
    expect(sections[3]?.text).toContain("RESOURCE_REVISION_MISMATCH");
    expect(guard?.({ name: "bash", arguments: { command: "typst compile paper.typ" } })).toContain(
      "managed automatically",
    );
    expect(
      guard?.({ name: "bash", arguments: { command: 'rg -n "typst compile" docs' } }),
    ).toBeUndefined();
    expect(guard?.({ name: "science_write", arguments: {} })).toBeUndefined();

    dispose();
    expect(disposeGuard).toHaveBeenCalledOnce();
  });
});
