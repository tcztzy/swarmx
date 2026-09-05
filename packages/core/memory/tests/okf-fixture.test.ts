import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parseConcept } from "../src/markdown.js";

const testDirectory = dirname(fileURLToPath(import.meta.url));

describe("memory OKF fixture", () => {
  it("V130 V131 V134: stays valid OKF-shaped portable Markdown", async () => {
    const source = await readFile(join(testDirectory, "fixtures", "decision.md"), "utf8");
    const { body, metadata: frontmatter } = parseConcept(source);

    expect(frontmatter).toMatchObject({
      type: "Decision",
      status: "draft",
      swarmx_scope: "workspace",
      sources: [
        {
          id: "okf-spec",
          resource:
            "https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md",
        },
      ],
      "x-fixture-field": "preserve-me",
    });
    expect(body).toContain("[^okf-spec]");
    expect(body).toContain("[^okf-spec]:");
    expect(body).toContain(
      "](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)",
    );
    expect(body).not.toMatch(/\[\[[^\]]+\]\]/u);
    expect(body).not.toMatch(/\^[a-z0-9-]+$/imu);
  });
});
