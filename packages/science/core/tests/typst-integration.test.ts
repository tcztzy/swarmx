import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { NodeScienceProcessRuntime } from "../../../../apps/desktop/src/host/process-runner.js";
import { TypstPreviewRuntime } from "../src/typst-preview.js";

const typstAvailable = spawnSync("typst", ["--version"], { stdio: "ignore" }).status === 0;
const writingRuntimeCommand = fileURLToPath(
  new URL(
    `../bin/${process.platform}-${process.arch}/${
      process.platform === "win32"
        ? "swarmx-writing-preview-runtime.exe"
        : "swarmx-writing-preview-runtime"
    }`,
    import.meta.url,
  ),
);
const writingRuntimeAvailable = existsSync(writingRuntimeCommand);
const scratchDirectories: string[] = [];

afterAll(() => {
  for (const directory of scratchDirectories) rmSync(directory, { recursive: true, force: true });
});

describe("V89 real Typst integration", () => {
  it.runIf(typstAvailable)("compiles and watches a real workspace paper", async () => {
    const scratch = mkdtempSync(join(tmpdir(), "swarmx-real-typst-"));
    scratchDirectories.push(scratch);
    const workspace = join(scratch, "workspace");
    mkdirSync(workspace);
    writeFileSync(
      join(workspace, "paper.typ"),
      `#set page(width: 120mm, height: 90mm, margin: 10mm)
= Live Paper
PDF.js can select this sentence.
`,
    );
    const runtime = new TypstPreviewRuntime(new NodeScienceProcessRuntime(), {
      command: "typst",
      graceMs: 1_000,
      initialCompileTimeoutMs: 10_000,
      maxDiagnosticsBytes: 64 * 1_024,
      maxPdfBytes: 4 * 1_024 * 1_024,
      maxSourceBytes: 1_024 * 1_024,
    });
    try {
      const preview = await runtime.preview({
        workspaceKey: "real-workspace",
        workspaceRoot: workspace,
        relativePath: "paper.typ",
      });
      expect(preview.status, preview.diagnostics.join("\n")).toBe("ready");
      expect(preview.pdfBase64).not.toBeNull();
      expect(
        Buffer.from(preview.pdfBase64 ?? "", "base64")
          .subarray(0, 5)
          .toString(),
      ).toBe("%PDF-");
      expect(preview.pdfSize).toBeGreaterThan(1_000);
    } finally {
      await runtime.close();
    }
  });

  it.runIf(writingRuntimeAvailable)(
    "V104/V107 compiles and resolves a PDF click through the writing runtime snapshot",
    async () => {
      const scratch = mkdtempSync(join(tmpdir(), "swarmx-semantic-typst-"));
      scratchDirectories.push(scratch);
      const workspace = join(scratch, "workspace");
      mkdirSync(workspace);
      writeFileSync(
        join(workspace, "paper.typ"),
        `#set page(width: 120mm, height: 90mm, margin: 10mm)
#include "section.typ"
`,
      );
      writeFileSync(join(workspace, "section.typ"), "= Included heading\n\nIncluded body text.\n");
      const runtime = new TypstPreviewRuntime(new NodeScienceProcessRuntime(), {
        command: "typst",
        runtimeCommand: writingRuntimeCommand,
        graceMs: 1_000,
        initialCompileTimeoutMs: 15_000,
        maxDiagnosticsBytes: 64 * 1_024,
        maxPdfBytes: 4 * 1_024 * 1_024,
        maxSourceBytes: 1_024 * 1_024,
      });
      try {
        const identity = {
          workspaceKey: "semantic-workspace",
          workspaceRoot: workspace,
          relativePath: "paper.typ",
        };
        const preview = await runtime.preview(identity);
        expect(preview.status, preview.diagnostics.join("\n")).toBe("ready");
        expect(preview.pdfSourceRevision).toBe(preview.sourceRevision);
        expect(preview.pdfRevision).not.toBeNull();

        let target = null;
        for (let y = 0.1; y <= 0.9 && target === null; y += 0.1) {
          for (let x = 0.1; x <= 0.9 && target === null; x += 0.1) {
            target = await runtime.resolveSourceAtPoint({
              ...identity,
              pdfRevision: preview.pdfRevision ?? "",
              page: 1,
              x,
              y,
            });
          }
        }
        expect(target).toMatchObject({
          relativePath: "section.typ",
          title: "section.typ",
          source: "= Included heading\n\nIncluded body text.\n",
        });
        expect(target?.offset).toBeLessThanOrEqual(target?.source.length ?? 0);

        writeFileSync(join(workspace, "section.typ"), "#broken(\n");
        let stale = preview;
        const deadline = Date.now() + 5_000;
        while (Date.now() < deadline && stale.status !== "stale") {
          await new Promise((resolve) => setTimeout(resolve, 50));
          stale = await runtime.preview(identity);
        }
        expect(stale.status).toBe("stale");
        expect(stale.pdfRevision).toBe(preview.pdfRevision);
        await expect(
          runtime.resolveSourceAtPoint({
            ...identity,
            pdfRevision: preview.pdfRevision ?? "",
            page: 1,
            x: 0.2,
            y: 0.2,
          }),
        ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });
      } finally {
        await runtime.close();
      }
    },
    20_000,
  );
});
