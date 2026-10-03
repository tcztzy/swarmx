import { describe, expect, it } from "vitest";
import { lintMemory, parseMemoryConcept } from "../src/lint.js";
import { conceptRevision, evaluationSchema, parseConcept, renderConcept } from "../src/markdown.js";

const path = "finding.md";
const now = "2026-09-05T00:00:00Z";
const metadata = {
  type: "Finding",
  title: "Finding",
  description: "A reproducible finding.",
  generated: { by: "swarmx/test", at: now },
  status: "draft" as const,
  sources: [{ id: "paper", resource: "https://example.org/paper" }],
  tags: [],
};

function source(body = "# Finding\n\nResult.[^paper]\n\n[^paper]: Source paper.") {
  return renderConcept(metadata, body);
}

function lint(text: string | Uint8Array, extra: ReadonlyMap<string, string> = new Map()) {
  return lintMemory(new Map([[path, text], ...extra]), { now });
}

describe("memory validation", () => {
  it("validates structured evaluation evidence and checks references omitted from hand-edited sources", () => {
    const evidence = "urn:swarmx:execution:10000000-0000-4000-8000-000000000001";
    const counterEvidence = "urn:swarmx:execution:10000000-0000-4000-8000-000000000002";
    const review = "urn:swarmx:execution:10000000-0000-4000-8000-000000000003";
    const evaluation = {
      kind: "preference",
      task: "Draft prose",
      criteria: "Use the user's preferred provider",
      evidence: [evidence],
      limitations: "Preference is not a measured quality ranking",
    };
    expect(evaluationSchema.parse(evaluation)).toMatchObject({ counterEvidence: [] });
    for (const invalid of [
      { ...evaluation, task: " " },
      { ...evaluation, evidence: [] },
      { ...evaluation, evidence: [evidence, evidence] },
      { ...evaluation, counterEvidence: [evidence, evidence] },
      { ...evaluation, evidence: ["urn:swarmx:execution:invalid"] },
      { ...evaluation, extra: true },
    ])
      expect(evaluationSchema.safeParse(invalid).success).toBe(false);
    const text = renderConcept(
      {
        ...metadata,
        swarmx_evaluation: evaluationSchema.parse({
          ...evaluation,
          counterEvidence: [counterEvidence],
          review,
        }),
        sources: [],
      },
      "# Preference",
    );
    expect(parseMemoryConcept(path, text).metadata.sources).toEqual([]);
    expect(lint(text)).toContainEqual(
      expect.objectContaining({ ruleId: "source.unchecked", severity: "warning" }),
    );
    const checked: string[] = [];
    const diagnostics = lintMemory(new Map([[path, text]]), {
      now,
      checkResource: (resource) => {
        checked.push(resource);
        return { ruleId: "source.unresolved", severity: "warning", message: "Another directory" };
      },
    });
    expect(new Set(checked)).toEqual(new Set([evidence, counterEvidence, review]));
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "source.unresolved", severity: "warning" }),
    );
    expect(diagnostics.some(({ severity }) => severity === "error")).toBe(false);
    expect(diagnostics.some(({ ruleId }) => ruleId === "source.missing")).toBe(false);
    expect(() =>
      parseMemoryConcept(path, text, undefined, () => ({
        ruleId: "source.foreign",
        severity: "error",
        message: "Foreign execution",
      })),
    ).toThrow("Invalid memory concept reference");
  });

  it("accepts the production format and preserves unknown types and fields", () => {
    const text = renderConcept({ ...metadata, type: "Custom fact", "x-owner": "me" }, "# Fact");
    expect(parseConcept(text).metadata["x-owner"]).toBe("me");
    expect(lint(text).filter((item) => item.severity === "error")).toEqual([]);
  });

  it("offers references to the supplied resolver without owning their domain syntax", () => {
    const execution = "urn:swarmx:execution:10000000-0000-4000-8000-000000000001";
    const science = "sx:artifact:test";
    const domain = "bio:record/one@1";
    const checked: string[] = [];
    const text = source(
      `# Finding\n\n[Run](${execution}) [Artifact](${science}) [Domain](${domain})`,
    );
    parseMemoryConcept(path, text, undefined, (resource) => {
      checked.push(resource);
      return undefined;
    });
    expect(checked).toEqual([execution, science, domain, "https://example.org/paper"]);
    const invalid = source("# Finding\n\n[Run](urn:swarmx:execution:invalid)");
    expect(() => parseMemoryConcept(path, invalid)).toThrow("Invalid memory concept reference");
    expect(lint(invalid)).toContainEqual(
      expect.objectContaining({ ruleId: "source.invalid", severity: "error" }),
    );
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
      expect(lintMemory(new Map([["index.md", index]]), { now })).toContainEqual(
        expect.objectContaining({ ruleId: "reserved.frontmatter", severity: "error" }),
      );
    }
    expect(
      lintMemory(
        new Map([["index.md", "# Index\n\n* [Link](./example.md) <script>bad()</script>\n"]]),
        { now },
      ),
    ).toContainEqual(expect.objectContaining({ ruleId: "markdown.executable", severity: "error" }));
  });

  it("rejects escaping and hidden local links", () => {
    for (const url of [".private/hidden.md", "../../../../secret.md"]) {
      expect(lint(source(`# Link\n\n[Secret](${url})`))).toContainEqual(
        expect.objectContaining({ ruleId: "link.path", severity: "error" }),
      );
    }
  });

  it("checks links and index descriptions against one file snapshot", () => {
    const text = source("# Finding\n\n[Missing](./missing.md)\n\n`[Example](./literal.md)`");
    const diagnostics = lint(
      text,
      new Map([["index.md", "# SwarmX Memory\n\n* [Old](./finding.md) - Old summary\n"]]),
    );
    expect(diagnostics.filter((item) => item.ruleId === "link.broken")).toHaveLength(1);
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "index.stale", path: "index.md" }),
    );
    expect(lint(text)).toContainEqual(expect.objectContaining({ ruleId: "index.missing" }));
  });

  it("validates the reserved index and excludes documents outside the root", () => {
    const files = new Map([
      [path, source()],
      ["index.md", '---\nokf_version: "0.2"\n---\n\n# Knowledge\n\nInvalid index paragraph.\n'],
      [".private/hidden.md", "invalid"],
      ["nested/private.md", "invalid"],
      ["README.md", "not a concept"],
      ["USER.md", "not a concept"],
    ]);
    const diagnostics = lintMemory(files, { now });
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "index.structure", severity: "error" }),
    );
    expect(diagnostics.some((item) => item.path !== "index.md" && item.path !== path)).toBe(false);
  });

  it("uses an explicit clock and reports stable positions and content revisions", () => {
    const text = renderConcept({ ...metadata, stale_after: now }, "# Finding");
    const files = new Map([[path, text]]);
    const diagnostics = lintMemory(files, { now });
    expect(diagnostics).toEqual(lintMemory(files, { now }));
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
      lintMemory(files, { now: "2026-09-04T23:59:59Z" }).some(
        (item) => item.ruleId === "lifecycle.stale",
      ),
    ).toBe(false);
  });
});
