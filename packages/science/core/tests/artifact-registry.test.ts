import { createHash, randomUUID } from "node:crypto";
import { existsSync, readdirSync, statSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { ScienceError } from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const fixtures: ScienceFixture[] = [];

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

function artifactRequest(projectId: string, relativePath: string) {
  return {
    requestId: randomUUID(),
    projectId,
    relativePath,
    kind: "dataset" as const,
    title: "Measurements",
    mime: "text/csv",
    runId: null,
    environment: { python: "3.12" },
    license: "MIT",
    sourceEntityIds: [],
  };
}

function objectPath(root: string, digest: string): string {
  const hash = digest.slice("sha256:".length);
  return join(root, "artifacts", "v1", "objects", hash.slice(0, 2), hash);
}

function filesBelow(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesBelow(path) : [path];
  });
}

async function fixture(maxArtifactBytes?: number): Promise<ScienceFixture> {
  const created = await createScienceFixture(maxArtifactBytes);
  fixtures.push(created);
  return created;
}

describe("T14 content-addressed Artifact Registry", () => {
  it("V66 imports one canonical browser file without persisting its encoded bytes", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Browser import",
    });
    const content = "sample,value\nA,42\n";
    const request = {
      requestId: randomUUID(),
      projectId: project.id,
      name: "measurements.csv",
      dataBase64: Buffer.from(content).toString("base64"),
    };

    const artifact = current.context.science.importArtifact(current.sessionA, request);

    expect(artifact).toMatchObject({
      projectId: project.id,
      kind: "dataset",
      title: "measurements.csv",
      mime: "text/csv",
      size: Buffer.byteLength(content),
      sourceEntityIds: [],
    });
    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: artifact.id }),
    ).toMatchObject({
      kind: "table",
      columns: [
        { id: "column-0", name: "sample", type: "string" },
        { id: "column-1", name: "value", type: "number" },
      ],
      rows: [["A", 42]],
      rowCount: 1,
      truncated: false,
    });
    expect(current.context.science.importArtifact(current.sessionA, request)).toEqual(artifact);
    expect(() =>
      current.context.science.importArtifact(current.sessionA, {
        ...request,
        dataBase64: Buffer.from("different bytes").toString("base64"),
      }),
    ).toThrowError(expect.objectContaining({ code: "IDEMPOTENCY_CONFLICT" }));

    const database = new DatabaseSync(current.databasePath, { readOnly: true });
    const event = database
      .prepare("SELECT payload_json FROM science_journal WHERE type = 'artifact/registered'")
      .get() as { payload_json: string };
    database.close();
    expect(event.payload_json).not.toContain(request.dataBase64);
    expect(event.payload_json).not.toContain(content);
    expect(event.payload_json).not.toContain(current.workspaceA);
  });

  it("V66 rejects unsafe names, malformed bytes, oversize input, cancellation, and foreign projects", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Browser import boundary",
    });
    const base = {
      requestId: randomUUID(),
      projectId: project.id,
      name: "safe.csv",
      dataBase64: Buffer.from("a,b\n1,2\n").toString("base64"),
    };

    for (const name of ["../safe.csv", "folder/safe.csv", "safe.exe", " safe.csv"] as const) {
      expect(() =>
        current.context.science.importArtifact(current.sessionA, { ...base, name }),
      ).toThrowError(expect.objectContaining({ code: "INVALID_REQUEST" }));
    }
    for (const dataBase64 of ["", "not base64", "YR=="] as const) {
      expect(() =>
        current.context.science.importArtifact(current.sessionA, {
          ...base,
          requestId: randomUUID(),
          dataBase64,
        }),
      ).toThrowError(expect.objectContaining({ code: "INVALID_REQUEST" }));
    }
    expect(() =>
      current.context.science.importArtifact(current.sessionA, {
        ...base,
        requestId: randomUUID(),
        dataBase64: Buffer.alloc(8 * 1024 * 1024 + 1).toString("base64"),
      }),
    ).toThrowError(expect.objectContaining({ code: "INVALID_REQUEST" }));
    expect(() => current.context.science.importArtifact(current.sessionB, base)).toThrowError(
      expect.objectContaining({ code: "PROJECT_NOT_FOUND" }),
    );

    const controller = new AbortController();
    controller.abort();
    expect(() =>
      current.context.science.importArtifact(current.sessionA, base, controller.signal),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(current.context.science.journalCount()).toBe(1);
  });

  it("publishes immutable bytes before appending metadata-only provenance", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Artifact project",
    });
    const content = "sample,value\nA,42\n";
    const relativePath = "measurements-secret-name.csv";
    writeFileSync(join(current.workspaceA, relativePath), content);

    const artifact = current.context.science.registerArtifact(
      current.sessionA,
      artifactRequest(project.id, relativePath),
    );
    const expectedHash = createHash("sha256").update(content).digest("hex");

    expect(artifact).toMatchObject({
      projectId: project.id,
      kind: "dataset",
      title: "Measurements",
      digest: `sha256:${expectedHash}`,
      mime: "text/csv",
      size: Buffer.byteLength(content),
      creator: { kind: "session", sessionId: current.sessionA },
      runId: null,
      environment: { python: "3.12" },
      license: "MIT",
      sourceEntityIds: [],
    });
    expect(current.context.science.getWorkspace(current.sessionA).artifacts).toEqual([artifact]);

    const stored = objectPath(current.root, artifact.digest);
    expect(statSync(stored).mode & 0o777).toBe(0o600);
    expect(statSync(join(current.root, "artifacts", "v1")).mode & 0o777).toBe(0o700);
    const database = new DatabaseSync(current.databasePath, { readOnly: true });
    const event = database
      .prepare("SELECT payload_json FROM science_journal WHERE type = 'artifact/registered'")
      .get() as { payload_json: string };
    database.close();
    expect(event.payload_json).not.toContain(relativePath);
    expect(event.payload_json).not.toContain(content);
    expect(event.payload_json).toContain(artifact.digest);
  });

  it("V52 returns a bounded verified text preview and rejects cross-workspace access", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Preview project",
    });
    const content = "sample,value\nA,42\n";
    writeFileSync(join(current.workspaceA, "preview.csv"), content);
    const artifact = current.context.science.registerArtifact(
      current.sessionA,
      artifactRequest(project.id, "preview.csv"),
    );

    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: artifact.id }),
    ).toEqual({
      kind: "table",
      artifactId: artifact.id,
      digest: artifact.digest,
      mime: "text/csv",
      size: Buffer.byteLength(content),
      columns: [
        { id: "column-0", name: "sample", type: "string" },
        { id: "column-1", name: "value", type: "number" },
      ],
      rows: [["A", 42]],
      rowCount: 1,
      truncated: false,
    });
    expect(() =>
      current.context.science.previewArtifact(current.sessionB, { artifactId: artifact.id }),
    ).toThrowError(expect.objectContaining({ code: "ARTIFACT_NOT_FOUND" }));

    const controller = new AbortController();
    controller.abort();
    expect(() =>
      current.context.science.previewArtifact(
        current.sessionA,
        { artifactId: artifact.id },
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
  });

  it("V70 returns typed scalar-record JSON without leaking nested values", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "JSON table preview",
    });
    const content = JSON.stringify([
      { sample: "A", value: 42, passed: true, note: null, nested: { private: "value" } },
      { sample: "B", value: 40.5, passed: false, note: "review", nested: [1, 2] },
    ]);
    writeFileSync(join(current.workspaceA, "preview.json"), content);
    const artifact = current.context.science.registerArtifact(current.sessionA, {
      ...artifactRequest(project.id, "preview.json"),
      mime: "application/json",
    });

    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: artifact.id }),
    ).toMatchObject({
      kind: "table",
      columns: [
        { name: "sample", type: "string" },
        { name: "value", type: "number" },
        { name: "passed", type: "boolean" },
        { name: "note", type: "string" },
        { name: "nested", type: "string" },
      ],
      rows: [
        ["A", 42, true, null, '{"private":"value"}'],
        ["B", 40.5, false, "review", "[1,2]"],
      ],
      rowCount: 2,
      truncated: false,
    });
  });

  it("V70 preserves blank and duplicate CSV headings while bounding rows", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Wide CSV preview",
    });
    const content = [
      ",value,value",
      ...Array.from({ length: 501 }, (_, index) => `A${index},${index},${index + 1}`),
    ].join("\n");
    writeFileSync(join(current.workspaceA, "duplicates.csv"), content);
    const artifact = current.context.science.registerArtifact(
      current.sessionA,
      artifactRequest(project.id, "duplicates.csv"),
    );

    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: artifact.id }),
    ).toMatchObject({
      kind: "table",
      columns: [
        { id: "column-0", name: "", type: "string" },
        { id: "column-1", name: "value", type: "number" },
        { id: "column-2", name: "value", type: "number" },
      ],
      rowCount: 501,
      truncated: true,
    });
    const preview = current.context.science.previewArtifact(current.sessionA, {
      artifactId: artifact.id,
    });
    expect(preview.kind === "table" ? preview.rows : []).toHaveLength(500);
  });

  it("V52 returns a safe image data URL and refuses unsupported artifact bytes", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Image preview",
    });
    const image = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
    writeFileSync(join(current.workspaceA, "figure.png"), image);
    const figure = current.context.science.registerArtifact(current.sessionA, {
      ...artifactRequest(project.id, "figure.png"),
      kind: "figure",
      mime: "image/png",
      title: "Figure",
    });
    writeFileSync(join(current.workspaceA, "model.bin"), "binary model");
    const model = current.context.science.registerArtifact(current.sessionA, {
      ...artifactRequest(project.id, "model.bin"),
      kind: "model",
      mime: "application/octet-stream",
      title: "Model",
    });

    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: figure.id }),
    ).toMatchObject({
      kind: "image",
      artifactId: figure.id,
      dataUrl: `data:image/png;base64,${image.toString("base64")}`,
    });
    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: model.id }),
    ).toMatchObject({
      kind: "unavailable",
      artifactId: model.id,
      reason: "unsupported",
    });

    writeFileSync(join(current.workspaceA, "large.txt"), "x".repeat(64 * 1024 + 1));
    const large = current.context.science.registerArtifact(current.sessionA, {
      ...artifactRequest(project.id, "large.txt"),
      mime: "text/plain",
      title: "Large text",
    });
    expect(
      current.context.science.previewArtifact(current.sessionA, { artifactId: large.id }),
    ).toMatchObject({
      kind: "unavailable",
      artifactId: large.id,
      reason: "too-large",
    });
  });

  it("V28 redacts secret values and absolute paths before persistence", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Redacted metadata",
    });
    writeFileSync(join(current.workspaceA, "safe.bin"), "safe bytes");
    const request = artifactRequest(project.id, "safe.bin");

    const artifact = current.context.science.registerArtifact(current.sessionA, {
      ...request,
      environment: {
        API_TOKEN: "do-not-store-this-token",
        interpreter: join(current.workspaceA, ".venv", "bin", "python"),
        python: "3.12",
      },
    });

    expect(artifact.environment).toEqual({
      API_TOKEN: "[redacted]",
      interpreter: "[redacted]",
      python: "3.12",
    });
    const serialized = JSON.stringify(current.context.science.getWorkspace(current.sessionA));
    expect(serialized).not.toContain("do-not-store-this-token");
    expect(serialized).not.toContain(current.workspaceA);
  });

  it("deduplicates object bytes and resolves an idempotent retry before rereading its source", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Dedup project",
    });
    const content = "identical artifact bytes";
    writeFileSync(join(current.workspaceA, "first.bin"), content);
    writeFileSync(join(current.workspaceA, "second.bin"), content);
    const firstRequest = artifactRequest(project.id, "first.bin");
    const first = current.context.science.registerArtifact(current.sessionA, firstRequest);
    const second = current.context.science.registerArtifact(
      current.sessionA,
      artifactRequest(project.id, "second.bin"),
    );

    expect(second.id).not.toBe(first.id);
    expect(second.digest).toBe(first.digest);
    expect(filesBelow(join(current.root, "artifacts", "v1", "objects"))).toEqual([
      objectPath(current.root, first.digest),
    ]);

    unlinkSync(join(current.workspaceA, "first.bin"));
    expect(current.context.science.registerArtifact(current.sessionA, firstRequest)).toEqual(first);
    expect(current.context.science.journalCount()).toBe(3);
  });

  it("rejects cancellation before publishing an object or journal fact", async () => {
    const current = await fixture();
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Cancelled artifact",
    });
    writeFileSync(join(current.workspaceA, "cancelled.bin"), "must not persist");
    const controller = new AbortController();
    controller.abort();

    expect(() =>
      current.context.science.registerArtifact(
        current.sessionA,
        artifactRequest(project.id, "cancelled.bin"),
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(current.context.science.journalCount()).toBe(1);
    expect(filesBelow(join(current.root, "artifacts", "v1", "objects"))).toEqual([]);
  });

  it("clears an abandoned staging file on a replacement mount", async () => {
    const current = await fixture();
    const staging = join(current.root, "artifacts", "v1", "staging");
    writeFileSync(join(staging, "abandoned-capture"), "partial bytes");
    await current.scienceFiber.dispose();

    await current.remount();

    expect(filesBelow(staging)).toEqual([]);
    expect(current.context.science.journalCount()).toBe(0);
  });

  it("rejects absolute, traversal, symlink-escape, directory, and oversized sources", async () => {
    const current = await fixture(4);
    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Path security",
    });
    const outside = join(current.scratch, "outside.bin");
    writeFileSync(outside, "data");
    symlinkSync(outside, join(current.workspaceA, "escape.bin"));
    writeFileSync(join(current.workspaceA, "large.bin"), "12345");

    for (const relativePath of [outside, "../outside.bin", "escape.bin", "."]) {
      expect(() =>
        current.context.science.registerArtifact(
          current.sessionA,
          artifactRequest(project.id, relativePath),
        ),
      ).toThrowError(
        expect.objectContaining<Partial<ScienceError>>({ code: "ARTIFACT_PATH_INVALID" }),
      );
    }
    expect(() =>
      current.context.science.registerArtifact(
        current.sessionA,
        artifactRequest(project.id, "large.bin"),
      ),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "ARTIFACT_TOO_LARGE" }));
    expect(current.context.science.journalCount()).toBe(1);
    expect(filesBelow(join(current.root, "artifacts", "v1", "objects"))).toEqual([]);
  });

  it("migrates v1 storage and replays a deleted artifact projection", async () => {
    const current = await fixture();
    await current.scienceFiber.dispose();
    const v1 = new DatabaseSync(current.databasePath);
    v1.exec(
      "DELETE FROM science_migrations WHERE version >= 2; DROP TABLE science_figures; DROP TABLE science_documents; DROP TABLE science_artifacts;",
    );
    v1.close();

    await current.remount();
    const migrated = new DatabaseSync(current.databasePath, { readOnly: true });
    expect(
      (
        migrated.prepare("SELECT MAX(version) AS version FROM science_migrations").get() as {
          version: number;
        }
      ).version,
    ).toBe(5);
    migrated.close();

    const project = current.context.science.createProject(current.sessionA, {
      requestId: randomUUID(),
      title: "Replay artifacts",
    });
    writeFileSync(join(current.workspaceA, "replay.bin"), "replay me");
    const artifact = current.context.science.registerArtifact(
      current.sessionA,
      artifactRequest(project.id, "replay.bin"),
    );
    await current.scienceFiber.dispose();
    const missingProjection = new DatabaseSync(current.databasePath);
    missingProjection.exec("DELETE FROM science_artifacts");
    missingProjection.close();

    await current.remount();
    expect(current.context.science.getWorkspace(current.sessionA).artifacts).toEqual([artifact]);
  });
});
