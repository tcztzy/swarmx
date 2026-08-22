import { randomUUID } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

async function fixture(
  config: Parameters<typeof createScienceFixture>[0] = {},
): Promise<ScienceFixture> {
  const created = await createScienceFixture(config);
  fixtures.push(created);
  return created;
}

function notebook(current: ScienceFixture) {
  const project = current.context.science.createProject(current.sessionA, {
    requestId: randomUUID(),
    title: "Executable project",
  });
  const created = current.context.science.createNotebook(current.sessionA, {
    requestId: randomUUID(),
    projectId: project.id,
    title: "Python analysis",
  });
  return { notebook: created, project };
}

describe("T15 Python notebook execution", () => {
  it("V72 reports a missing configured JupyMCP server without an execution fact", async () => {
    const current = await fixture({
      jupymcpCommand: "swarmx-test-missing-jupymcp-command",
      jupymcpRequestTimeoutMs: 1_000,
      notebookRuntime: "jupymcp",
    });
    const { notebook: created } = notebook(current);

    await expect(
      current.context.science.executeNotebookCell(current.sessionA, {
        requestId: randomUUID(),
        notebookId: created.id,
        source: "print('must not run')",
        outputArtifact: null,
      }),
    ).rejects.toMatchObject({ code: "JUPYMCP_UNAVAILABLE" });
    expect(current.context.science.journalCount()).toBe(2);
    expect(current.context.science.getWorkspace(current.sessionA).notebooks).toEqual([created]);
  });

  it("V67 materializes a verified artifact for one replayable Notebook analysis", async () => {
    const current = await fixture();
    const { notebook: created, project } = notebook(current);
    const imported = current.context.science.importArtifact(current.sessionA, {
      requestId: randomUUID(),
      projectId: project.id,
      name: "input.csv",
      dataBase64: Buffer.from("sample,value\nA,42\n").toString("base64"),
    });

    const execution = await current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: created.id,
      inputArtifactIds: [imported.id],
      source: [
        "import json, os",
        "from pathlib import Path",
        'text = Path(os.environ["DSH_SCIENCE_INPUT_0"]).read_text()',
        'Path("analysis.json").write_text(json.dumps({"lines": len(text.splitlines())}))',
        "print(text, end='')",
      ].join("\n"),
      outputArtifact: {
        relativePath: "analysis.json",
        kind: "dataset",
        title: "input-analysis.json",
        mime: "application/json",
        license: null,
      },
    });

    expect(execution.status).toBe("succeeded");
    expect(execution.stdout.text).toBe("sample,value\nA,42\n");
    expect(execution.inputArtifactIds).toEqual([imported.id]);
    expect(execution.notebook.cells[0]).toMatchObject({
      inputArtifactIds: [imported.id],
      outputArtifactIds: [execution.artifact?.id],
    });
    expect(execution.artifact).toMatchObject({ title: "input-analysis.json" });
    expect(JSON.stringify(execution)).not.toContain(current.root);
    expect(JSON.stringify(execution)).not.toContain(current.workspaceA);
    expect(readdirSync(join(current.root, "artifacts", "v1", "staging"))).toEqual([]);

    const trace = current.context.science.traceProvenance(current.sessionA, {
      entityId: created.id,
      maxDepth: 20,
    });
    expect(trace.relations).toContainEqual({
      fromId: created.id,
      toId: imported.id,
      type: "uses",
    });

    await current.scienceFiber.dispose();
    await current.remount();
    expect(
      current.context.science.getWorkspace(current.sessionA).notebooks[0]?.cells[0],
    ).toMatchObject({ inputArtifactIds: [imported.id] });
  });

  it("V67 rejects foreign or excessive Notebook artifact inputs before running code", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);
    const foreignProject = current.context.science.createProject(current.sessionB, {
      requestId: randomUUID(),
      title: "Foreign project",
    });
    const foreign = current.context.science.importArtifact(current.sessionB, {
      requestId: randomUUID(),
      projectId: foreignProject.id,
      name: "foreign.csv",
      dataBase64: Buffer.from("secret\n").toString("base64"),
    });

    await expect(
      current.context.science.executeNotebookCell(current.sessionA, {
        requestId: randomUUID(),
        notebookId: created.id,
        inputArtifactIds: [foreign.id],
        source: 'raise RuntimeError("must not run")',
        outputArtifact: null,
      }),
    ).rejects.toMatchObject({ code: "ARTIFACT_NOT_FOUND" });
    expect(() =>
      current.context.science.executeNotebookCell(current.sessionA, {
        requestId: randomUUID(),
        notebookId: created.id,
        inputArtifactIds: ["a", "b", "c", "d", "e"],
        source: 'raise RuntimeError("must not run")',
        outputArtifact: null,
      }),
    ).toThrowError(expect.objectContaining({ code: "INVALID_REQUEST" }));
    expect(current.context.science.getWorkspace(current.sessionA).notebooks[0]?.cells).toEqual([]);
  });

  it("executes one cell through the managed Python runtime and records bounded evidence", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);

    const execution = await current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: created.id,
      source: "print(6 * 7)",
      outputArtifact: null,
    });

    expect(execution).toMatchObject({
      notebookId: created.id,
      executionCount: 1,
      status: "succeeded",
      stdout: { text: "42\n", truncated: false },
      stderr: { text: "", truncated: false },
      exitCode: 0,
      signal: null,
      artifact: null,
    });
    expect(execution.environment).toEqual({
      packageSetHash: expect.stringMatching(/^sha256:[0-9a-f]{64}$/u),
      pythonImplementation: expect.any(String),
      pythonVersion: expect.stringMatching(/^\d+\.\d+\.\d+/u),
    });
    expect(execution.notebook).toMatchObject({ id: created.id, revision: 2 });
    expect(execution.notebook.cells).toHaveLength(2);
    expect(execution.notebook.cells[0]).toMatchObject({
      id: execution.cellId,
      kind: "code",
      source: "print(6 * 7)",
      executionCount: 1,
      outputArtifactIds: [],
    });
    expect(execution.notebook.cells[1]).toMatchObject({
      kind: "output",
      source: "42\n",
      executionCount: 1,
    });
    expect(JSON.stringify(execution.environment)).not.toContain("/Users/");
    expect(current.context.science.journalCount()).toBe(3);
  });

  it("captures one declared output artifact in the same replayable execution fact", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);

    const execution = await current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: created.id,
      source:
        'from pathlib import Path\nPath("result.csv").write_text("sample,value\\nA,42\\n")\nprint("saved")',
      outputArtifact: {
        relativePath: "result.csv",
        kind: "dataset",
        title: "Cell result",
        mime: "text/csv",
        license: null,
      },
    });

    expect(execution.status).toBe("succeeded");
    expect(execution.artifact).toMatchObject({
      projectId: created.projectId,
      kind: "dataset",
      title: "Cell result",
      mime: "text/csv",
      runId: execution.id,
      sourceEntityIds: [created.id],
      environment: execution.environment,
    });
    expect(execution.notebook.cells[0]?.outputArtifactIds).toEqual([execution.artifact?.id]);
    expect(current.context.science.journalCount()).toBe(3);
    const trace = current.context.science.traceProvenance(current.sessionA, {
      entityId: execution.artifact?.id ?? "missing-artifact",
      maxDepth: 20,
    });
    expect(trace.events.map((event) => event.operation)).toEqual([
      "project/created",
      "notebook/created",
      "notebook/cell-executed",
    ]);
    expect(new Set(trace.events.map((event) => event.journalSeq)).size).toBe(trace.events.length);

    await current.scienceFiber.dispose();
    const database = new DatabaseSync(current.databasePath);
    database.exec("DELETE FROM science_artifacts; DELETE FROM science_notebooks;");
    database.close();
    await current.remount();

    const workspace = current.context.science.getWorkspace(current.sessionA);
    expect(workspace.notebooks).toEqual([execution.notebook]);
    expect(workspace.artifacts).toEqual([execution.artifact]);
  });

  it("records a nonzero exit as scientific evidence and bounds both output streams", async () => {
    const current = await fixture({ maxCellOutputBytes: 96 });
    const { notebook: created } = notebook(current);

    const execution = await current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: created.id,
      source: 'print("x" * 500)\nraise ValueError("expected failure")',
      outputArtifact: null,
    });

    expect(execution.status).toBe("failed");
    expect(execution.exitCode).not.toBe(0);
    expect(execution.stdout.truncated).toBe(true);
    expect(Buffer.byteLength(execution.stdout.text)).toBeLessThanOrEqual(96);
    expect(execution.stderr.text).toContain("ValueError: expected failure");
    expect(Buffer.byteLength(execution.stderr.text)).toBeLessThanOrEqual(96);
    expect(execution.notebook.revision).toBe(2);
  });

  it("resolves an idempotent retry before running user code again", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);
    const request = {
      requestId: randomUUID(),
      notebookId: created.id,
      source: [
        "from pathlib import Path",
        'path = Path("execution-count.txt")',
        "count = int(path.read_text()) if path.exists() else 0",
        "path.write_text(str(count + 1))",
      ].join("\n"),
      outputArtifact: null,
    };

    const first = await current.context.science.executeNotebookCell(current.sessionA, request);
    const repeated = await current.context.science.executeNotebookCell(current.sessionA, request);

    expect(repeated).toEqual(first);
    expect(readFileSync(join(current.workspaceA, "execution-count.txt"), "utf8")).toBe("1");
    expect(current.context.science.journalCount()).toBe(3);
  });

  it("terminates a cancelled process and appends no execution fact", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);
    const controller = new AbortController();
    const pending = current.context.science.executeNotebookCell(
      current.sessionA,
      {
        requestId: randomUUID(),
        notebookId: created.id,
        source: "import time\ntime.sleep(30)",
        outputArtifact: null,
      },
      controller.signal,
    );
    setTimeout(() => controller.abort(), 50);

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(current.context.science.journalCount()).toBe(2);
    expect(current.context.science.getWorkspace(current.sessionA).notebooks).toEqual([created]);
  });

  it("terminates an active cell before closing the journal on plugin disposal", async () => {
    const current = await fixture();
    const { notebook: created } = notebook(current);
    const pending = current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: created.id,
      source: "import time\ntime.sleep(30)",
      outputArtifact: null,
    });
    const rejected = expect(pending).rejects.toMatchObject({ code: "SCIENCE_CLOSED" });
    await new Promise((resolve) => setTimeout(resolve, 50));

    await current.scienceFiber.dispose();

    await rejected;
    await current.remount();
    expect(current.context.science.journalCount()).toBe(2);
  });
});
