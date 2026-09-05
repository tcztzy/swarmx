import { chmodSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { NodeScienceProcessRuntime } from "../../../../apps/desktop/src/host/process-runner.js";
import {
  resolveTypstSourceAtPointRequestSchema,
  typstSourceTargetSchema,
} from "../src/contracts.js";
import type { ScienceError } from "../src/errors.js";
import { TypstPreviewRuntime } from "../src/typst-preview.js";

const scratchDirectories: string[] = [];

afterEach(() => {
  for (const directory of scratchDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

async function setup() {
  const scratch = mkdtempSync(join(tmpdir(), "swarmx-typst-preview-"));
  scratchDirectories.push(scratch);
  const workspace = join(scratch, "workspace");
  mkdirSync(workspace);
  const command = join(scratch, "fake-typst.mjs");
  writeFileSync(
    command,
    `#!/usr/bin/env node
import { readFileSync, statSync, watchFile, writeFileSync } from "node:fs";
const args = process.argv.slice(2);
const source = args.at(-2);
const output = args.at(-1);
function compile() {
  const text = readFileSync(source, "utf8");
  if (text.includes("#broken")) {
    process.stderr.write("error: broken source\\n");
    return;
  }
  writeFileSync(output, Buffer.from("%PDF-1.7\\n" + text));
}
compile();
watchFile(source, { interval: 25 }, (next, previous) => {
  if (next.mtimeMs !== previous.mtimeMs || next.size !== previous.size) compile();
});
process.on("SIGTERM", () => process.exit(0));
`,
  );
  chmodSync(command, 0o700);
  const runtime = new TypstPreviewRuntime(new NodeScienceProcessRuntime(), {
    command,
    graceMs: 200,
    initialCompileTimeoutMs: 3_000,
    maxDiagnosticsBytes: 16_384,
    maxPdfBytes: 2 * 1024 * 1024,
    maxSourceBytes: 512 * 1024,
  });
  return { runtime, scratch, workspace };
}

async function waitForRevision(
  runtime: TypstPreviewRuntime,
  request: { workspaceKey: string; workspaceRoot: string; relativePath: string },
  revision: string,
) {
  const deadline = Date.now() + 3_000;
  while (Date.now() < deadline) {
    const preview = await runtime.preview(request);
    if (preview.pdfRevision !== revision && preview.status === "ready") return preview;
    await new Promise((resolve) => setTimeout(resolve, 30));
  }
  throw new Error("Typst watcher did not publish a new PDF revision");
}

describe("V89/V91 managed Typst preview runtime", () => {
  it("V104 accepts only revision-bound normalized inverse-search requests and UTF-16 targets", () => {
    expect(
      resolveTypstSourceAtPointRequestSchema.parse({
        relativePath: "paper.typ",
        pdfRevision: `sha256:${"a".repeat(64)}`,
        page: 2,
        x: 0.25,
        y: 0.75,
      }),
    ).toMatchObject({ page: 2, x: 0.25, y: 0.75 });
    expect(() =>
      resolveTypstSourceAtPointRequestSchema.parse({
        relativePath: "paper.typ",
        pdfRevision: `sha256:${"a".repeat(64)}`,
        page: 0,
        x: 1.1,
        y: 0.5,
      }),
    ).toThrow();
    expect(
      typstSourceTargetSchema.parse({
        relativePath: "sections/intro.typ",
        title: "intro.typ",
        source: "😀= Intro",
        sourceRevision: `sha256:${"b".repeat(64)}`,
        offset: 2,
      }).offset,
    ).toBe(2);
  });

  it("watches one authorized paper, returns bounded PDF bytes, and atomically saves by revision", async () => {
    const { runtime, workspace } = await setup();
    writeFileSync(join(workspace, "paper.typ"), "= First draft\n");

    const first = await runtime.preview({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
    });

    expect(first).toMatchObject({
      relativePath: "paper.typ",
      title: "paper.typ",
      source: "= First draft\n",
      status: "ready",
      diagnostics: [],
    });
    expect(Buffer.from(first.pdfBase64 ?? "", "base64").toString()).toContain("First draft");
    expect(first.sourceRevision).toMatch(/^sha256:[0-9a-f]{64}$/u);
    expect(first.pdfRevision).toMatch(/^sha256:[0-9a-f]{64}$/u);
    expect(first.pdfSourceRevision).toBe(first.sourceRevision);

    const saved = await runtime.updateSource({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
      expectedSourceRevision: first.sourceRevision,
      source: "= Revised paper\n",
    });
    expect(saved.source).toBe("= Revised paper\n");
    expect(saved.sourceRevision).not.toBe(first.sourceRevision);

    const second = await waitForRevision(
      runtime,
      { workspaceKey: "workspace-a", workspaceRoot: workspace, relativePath: "paper.typ" },
      first.pdfRevision ?? "",
    );
    expect(Buffer.from(second.pdfBase64 ?? "", "base64").toString()).toContain("Revised paper");
    expect(runtime.controllerCount()).toBe(1);

    await runtime.close();
    expect(runtime.controllerCount()).toBe(0);
  });

  it("retains the last successful PDF with diagnostics after a broken source revision", async () => {
    const { runtime, workspace } = await setup();
    writeFileSync(join(workspace, "paper.typ"), "= Valid\n");
    const first = await runtime.preview({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
    });
    await runtime.updateSource({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
      expectedSourceRevision: first.sourceRevision,
      source: "#broken\n",
    });
    await new Promise((resolve) => setTimeout(resolve, 100));

    const stale = await runtime.preview({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
    });
    expect(stale.status).toBe("stale");
    expect(stale.pdfRevision).toBe(first.pdfRevision);
    expect(stale.pdfBase64).toBe(first.pdfBase64);
    expect(stale.diagnostics.join("\n")).toContain("broken source");

    await runtime.close();
  });

  it("rejects traversal, symlink escape, unsupported extensions, and stale source writes", async () => {
    const { runtime, scratch, workspace } = await setup();
    writeFileSync(join(workspace, "paper.typ"), "= Safe\n");
    writeFileSync(join(scratch, "outside.typ"), "= Outside\n");
    symlinkSync(join(scratch, "outside.typ"), join(workspace, "escape.typ"));
    const preview = await runtime.preview({
      workspaceKey: "workspace-a",
      workspaceRoot: workspace,
      relativePath: "paper.typ",
    });

    for (const relativePath of ["../outside.typ", "paper.tex", "escape.typ"]) {
      await expect(
        runtime.preview({ workspaceKey: "workspace-a", workspaceRoot: workspace, relativePath }),
      ).rejects.toMatchObject<Partial<ScienceError>>({ code: "INVALID_REQUEST" });
    }
    await expect(
      runtime.updateSource({
        workspaceKey: "workspace-a",
        workspaceRoot: workspace,
        relativePath: "paper.typ",
        expectedSourceRevision: `sha256:${"0".repeat(64)}`,
        source: "= Overwrite\n",
      }),
    ).rejects.toMatchObject<Partial<ScienceError>>({ code: "REVISION_CONFLICT" });
    expect(preview.source).toBe("= Safe\n");

    await runtime.close();
  });
});
