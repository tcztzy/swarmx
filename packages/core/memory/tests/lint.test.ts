import { describe, expect, it } from "vitest";
import { lintMemory } from "../src/lint.js";
import { conceptRevision, parseConcept, renderConcept } from "../src/markdown.js";

const workspaceDirectory = "workspaces/project--83bd29e31a4c";
const path = `${workspaceDirectory}/concepts/finding.md`;
const now = "2026-09-05T00:00:00Z";
const metadata = {
  type: "Finding",
  title: "Finding",
  description: "A reproducible finding.",
  generated: { by: "swarmx/test", at: now },
  swarmx_scope: "workspace" as const,
  swarmx_workspace: "83bd29e31a4c",
  status: "draft" as const,
  sources: [{ id: "paper", resource: "https://example.org/paper" }],
  tags: [],
};

function source(body = "# Finding\n\nResult.[^paper]\n\n[^paper]: Source paper.") {
  return renderConcept(metadata, body);
}

function lint(text: string | Uint8Array, extra: ReadonlyMap<string, string> = new Map()) {
  return lintMemory(new Map([[path, text], ...extra]), { workspaceDirectory, now });
}

describe("memory validation", () => {
  it("accepts the production format and preserves unknown types and fields", () => {
    const text = renderConcept({ ...metadata, type: "Custom fact", "x-owner": "me" }, "# Fact");
    expect(parseConcept(text).metadata["x-owner"]).toBe("me");
    expect(lint(text).filter((item) => item.severity === "error")).toEqual([]);
  });

  it.each([
    ["type: Finding", "type: 123"],
    ["title: Finding", 'title: "   "'],
    ["2026-09-05T00:00:00Z", "2026-02-30T00:00:00Z"],
    ["2026-09-05T00:00:00Z", "2026-09-05T00:00:00"],
    ["status: draft", "status: unknown"],
    ["type: Finding", "type: Finding\ntype: Other"],
  ])("rejects malformed metadata: %s → %s", (before, after) => {
    expect(lint(source().replace(before, after))).toContainEqual(
      expect.objectContaining({ severity: "error", path }),
    );
  });

  it("reports invalid UTF-8, empty bodies, and malformed lifecycle metadata", () => {
    expect(lint(new Uint8Array([0xff]))).toContainEqual(
      expect.objectContaining({ ruleId: "document.encoding", severity: "error" }),
    );
    const header = source().split("\n---\n")[0];
    for (const text of [
      `${header}\n---\n \n`,
      source().replace("status: draft", "status: draft\nverified: false"),
      source().replace("status: draft", "status: draft\nstale_after: 2026-02-30T00:00:00Z"),
    ])
      expect(lint(text).some((item) => item.severity === "error")).toBe(true);
  });

  it("ignores literal examples while detecting actual undefined and duplicate footnotes", () => {
    const body =
      "# Examples\n\n`[^missing] [[Wiki]]`\n\n```md\n[^missing]\n[[Wiki]]\n<script>\n```\n\n\\[^escaped]\n";
    expect(lint(source(body)).filter((item) => item.severity === "error")).toEqual([]);
    const missing = source().replace("Result.[^paper]", "Result.[^missing]");
    expect(lint(missing)).toContainEqual(
      expect.objectContaining({
        ruleId: "footnote.undefined",
        severity: "error",
        revision: conceptRevision(missing),
      }),
    );
    const duplicate = `${source()}\n[^paper]: Another source.\n`;
    expect(lint(duplicate)).toContainEqual(
      expect.objectContaining({
        ruleId: "footnote.duplicate",
        severity: "error",
      }),
    );
  });

  it("allows explanatory footnotes and warns about unassociated sources", () => {
    const text = source("# Note\n\nExplanation.[^aside]\n\n[^aside]: An explanatory note.");
    const diagnostics = lint(text);
    expect(diagnostics.filter((item) => item.severity === "error")).toEqual([]);
    expect(diagnostics).toContainEqual(expect.objectContaining({ ruleId: "source.unassociated" }));
    expect(diagnostics).toContainEqual(expect.objectContaining({ ruleId: "source.unused" }));
  });

  it("reports duplicate YAML and source IDs at their source locations", () => {
    const duplicateKey = "---\ntype: Finding\ntype: Other\n---\n\n# Invalid\n";
    expect(lint(duplicateKey)).toContainEqual(
      expect.objectContaining({
        ruleId: "document.yaml",
        line: 3,
        severity: "error",
      }),
    );
    const duplicateSource = source().replace(
      "sources:\n",
      "sources:\n  - id: paper\n    resource: https://example.org/duplicate\n",
    );
    const issue = lint(duplicateSource).find((item) => item.ruleId === "source.duplicate");
    expect(issue?.severity).toBe("error");
    expect(issue?.line).toBeGreaterThan(3);
    expect(issue?.revision).toBe(conceptRevision(duplicateSource));
  });

  it("rejects invalid reserved YAML and executable HTML inside index entries", () => {
    for (const index of [
      "---\nokf_version: *unknown\n---\n\n# Index\n",
      "---\n- invalid\n---\n\n# Index\n",
    ]) {
      expect(
        lintMemory(new Map([["index.md", index]]), { workspaceDirectory, now }),
      ).toContainEqual(
        expect.objectContaining({ ruleId: "reserved.frontmatter", severity: "error" }),
      );
    }
    expect(
      lintMemory(new Map([["index.md", "# Index\n\n* [Link](global/) <script>bad()</script>\n"]]), {
        workspaceDirectory,
        now,
      }),
    ).toContainEqual(expect.objectContaining({ ruleId: "markdown.executable", severity: "error" }));
  });

  it("rejects forged scope metadata and foreign or escaping local links", () => {
    expect(
      lint(source().replace("swarmx_workspace: 83bd29e31a4c", "swarmx_workspace: aaaaaaaaaaaa")),
    ).toContainEqual(expect.objectContaining({ ruleId: "scope.mismatch", severity: "error" }));
    for (const url of [
      "/workspaces/other--aaaaaaaaaaaa/concepts/secret.md",
      "../../../../secret.md",
    ]) {
      expect(lint(source(`# Link\n\n[Secret](${url})`))).toContainEqual(
        expect.objectContaining({ ruleId: "link.scope", severity: "error" }),
      );
    }
  });

  it("checks links and index descriptions against one file snapshot", () => {
    const text = source("# Finding\n\n[Missing](./missing.md)\n\n`[Example](./literal.md)`");
    const index = `${workspaceDirectory}/index.md`;
    const diagnostics = lint(
      text,
      new Map([[index, "# Project\n\n* [Old](./concepts/finding.md) - Old summary\n"]]),
    );
    expect(diagnostics.filter((item) => item.ruleId === "link.broken")).toHaveLength(1);
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "index.stale", path: index }),
    );
    expect(lint(text)).toContainEqual(expect.objectContaining({ ruleId: "index.missing" }));
  });

  it("validates reserved documents and excludes internal history and foreign workspaces", () => {
    const files = new Map([
      [path, source()],
      ["index.md", '---\nokf_version: "0.2"\n---\n\n# Knowledge\n'],
      ["log.md", "# Log\n\n## 2026-02-30\n\n* Updated.\n"],
      [".swarmx/history/old.md", "invalid"],
      ["workspaces/other--aaaaaaaaaaaa/concepts/private.md", "invalid"],
    ]);
    const diagnostics = lintMemory(files, { workspaceDirectory, now });
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "log.date", severity: "error" }),
    );
    expect(
      diagnostics.some((item) => item.path.includes("private") || item.path.includes("history")),
    ).toBe(false);
  });

  it("uses an explicit clock and reports stable positions and content revisions", () => {
    const text = renderConcept({ ...metadata, stale_after: now }, "# Finding");
    const files = new Map([[path, text]]);
    const diagnostics = lintMemory(files, { workspaceDirectory, now });
    expect(diagnostics).toEqual(lintMemory(files, { workspaceDirectory, now }));
    expect(diagnostics).toContainEqual(
      expect.objectContaining({
        ruleId: "lifecycle.stale",
        severity: "warning",
        path,
        revision: conceptRevision(text),
        line: expect.any(Number),
        column: expect.any(Number),
      }),
    );
    expect(
      lintMemory(files, { workspaceDirectory, now: "2026-09-04T23:59:59Z" }).some(
        (item) => item.ruleId === "lifecycle.stale",
      ),
    ).toBe(false);
  });
});
