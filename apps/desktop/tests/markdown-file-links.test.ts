import { describe, expect, it } from "vitest";
import { transformMarkdownFileLinks } from "../src/markdown-file-links.js";

const RC2_LINK_BRANCH = 'case"link":return oa(n.url,Rt(n.children,{...l,inLink:!0}),i);';

describe("V105 Markdown file-link frontend seam", () => {
  it("adds a trusted relative-link resolver while preserving the native link fallback", () => {
    const transformed = transformMarkdownFileLinks(`before${RC2_LINK_BRANCH}after`);

    expect(transformed).toContain("fileMentions?.resolveLink?.(n.url)");
    expect(transformed).toContain(":oa(n.url,u,i)");
    expect(transformed).not.toContain(RC2_LINK_BRANCH);
  });

  it("rejects a missing or duplicated upstream seam", () => {
    expect(() => transformMarkdownFileLinks("no Markdown branch")).toThrow("Markdown link seam");
    expect(() => transformMarkdownFileLinks(`${RC2_LINK_BRANCH}${RC2_LINK_BRANCH}`)).toThrow(
      "exactly once",
    );
  });
});
