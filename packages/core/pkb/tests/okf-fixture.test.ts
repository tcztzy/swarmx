import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";

const testDirectory = dirname(fileURLToPath(import.meta.url));

function splitFrontmatter(source: string): { body: string; frontmatter: unknown } {
  const match = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/u.exec(source);
  if (match?.[1] === undefined || match[2] === undefined) {
    throw new Error("fixture must contain one leading YAML frontmatter block");
  }
  return { frontmatter: parse(match[1]), body: match[2] };
}

describe("PKB OKF fixture", () => {
  it("V130 V131 V134: stays valid OKF-shaped portable Markdown", async () => {
    const source = await readFile(join(testDirectory, "fixtures", "decision.md"), "utf8");
    const { body, frontmatter } = splitFrontmatter(source);

    expect(frontmatter).toMatchObject({
      type: "Decision",
      status: "draft",
      swarmx_scope: "workspace",
      sources: [
        {
          id: "chat-a19f",
          resource: "../references/conversations/a19f-42-47.md",
        },
      ],
      "x-fixture-field": "preserve-me",
    });
    expect(body).toContain("[^chat-a19f]");
    expect(body).toContain("[^chat-a19f]:");
    expect(body).toContain("](../references/conversations/a19f-42-47.md)");
    expect(body).not.toMatch(/\[\[[^\]]+\]\]/u);
    expect(body).not.toMatch(/\^[a-z0-9-]+$/imu);
  });
});
