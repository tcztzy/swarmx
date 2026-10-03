import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  realpath,
  rm,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { afterEach, expect, it, vi } from "vitest";
import { renderConcept } from "../src/markdown.js";
import { importModelObservation, queryModelExperience } from "../src/model-experience.js";
import { MemoryVault } from "../src/vault.js";

const roots: string[] = [];
const observation = {
  schemaVersion: 1,
  kind: "observation",
  observedAt: "2026-01-02T03:04:05Z",
  observer: "synthetic-test",
  task: "Parse a synthetic JSON object",
  criteria: "Return the two expected keys",
  outcome: "success",
  limitations: "Synthetic fixture; no model was called; not a quality ranking.",
  confidence: null,
  requested: {
    model: "synthetic-model",
    effort: "high",
    provider: null,
    harness: "test",
    runtimeVersion: null,
  },
  actual: null,
  retries: null,
  elapsed: { value: 42, unit: "ms", source: "synthetic test clock" },
  tokens: null,
  cost: null,
};
async function setup() {
  const root = await realpath(await mkdtemp(join(tmpdir(), "shared-model-experience-")));
  roots.push(root);
  return { root, vault: new MemoryVault({ root: join(root, "memory") }) };
}
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});
async function snapshot(root: string, includeDirectoryMtime = true): Promise<unknown> {
  const stat = await lstat(root);
  return {
    mode: stat.mode,
    mtime: includeDirectoryMtime ? stat.mtimeMs : null,
    entries: await Promise.all(
      (await readdir(root)).sort().map(async (name) => {
        const path = join(root, name);
        const info = await lstat(path);
        return {
          name,
          mode: info.mode,
          mtime: info.mtimeMs,
          data: info.isDirectory()
            ? await snapshot(path, includeDirectoryMtime)
            : (await readFile(path)).toString("base64"),
        };
      }),
    ),
  };
}
it("queries missing, empty and populated vaults without content, mode or mtime mutations", async () => {
  const { root, vault } = await setup();
  const before = await snapshot(root);
  expect((await queryModelExperience(vault)).concepts).toEqual([]);
  await expect(vault.readConcept("missing.md")).rejects.toMatchObject({
    code: "CONCEPT_NOT_FOUND",
  });
  expect(await snapshot(root)).toEqual(before);
  await mkdir(vault.root, { mode: 0o755 });
  const empty = await snapshot(root);
  expect((await queryModelExperience(vault)).concepts).toEqual([]);
  expect(await snapshot(root)).toEqual(empty);
  const concept = await vault.createConcept({
    title: "Legacy experience",
    type: "Finding",
    description: "Relevant historical lesson",
    body: "PRIVATE PROMPT /private/user/file",
    tags: ["agent-selection"],
  });
  await chmod(vault.root, 0o755);
  const populated = await snapshot(root);
  const result = await queryModelExperience(vault);
  expect(result.concepts[0]).toMatchObject({
    id: concept.id,
    revision: concept.revision,
    detail: "needs-detailed-read",
    kind: "unknown",
  });
  expect(JSON.stringify(result)).not.toContain("PRIVATE PROMPT");
  expect((await queryModelExperience(vault, { includeBody: true })).concepts[0]?.body).toContain(
    "PRIVATE PROMPT",
  );
  await vault.snapshotConcept(concept.id, concept.revision);
  expect(await snapshot(root)).toEqual(populated);
});
it("imports evidence-backed observations and preserves unknowns in another process", async () => {
  const { root, vault } = await setup();
  const artifact = join(root, "observation.json");
  await writeFile(artifact, JSON.stringify(observation));
  const request = { artifact, title: "Synthetic parser trial", requestId: randomUUID() };
  const saved = await importModelObservation(vault, request);
  expect((await importModelObservation(vault, request)).revision).toBe(saved.revision);
  const before = await snapshot(root);
  const { stdout } = await promisify(execFile)(
    process.execPath,
    ["--import", "tsx", resolve("packages/core/memory/src/cli.ts"), "query", "--vault", vault.root],
    { cwd: process.cwd() },
  );
  const result = JSON.parse(stdout);
  expect(result.concepts[0]).toMatchObject({
    revision: saved.revision,
    kind: "observation",
    provenance: "external-self-asserted",
    observation: { actual: null, cost: null, tokens: null, elapsed: { value: 42, unit: "ms" } },
  });
  expect(stdout).not.toContain(root);
  expect(stdout).not.toContain("urn:swarmx:execution:");
  expect(await snapshot(root)).toEqual(before);
  await writeFile(artifact, JSON.stringify({ ...observation, outcome: "failure" }));
  await expect(importModelObservation(vault, request)).rejects.toMatchObject({
    code: "REVISION_CONFLICT",
  });
  const updated = await vault.updateConcept({
    id: saved.id,
    expectedRevision: saved.revision,
    body: "A corrected observation.",
  });
  await expect(
    vault.updateConcept({
      id: saved.id,
      expectedRevision: saved.revision,
      body: "Stale correction",
    }),
  ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });
  expect(updated.revision).not.toBe(saved.revision);
  await writeFile(artifact, JSON.stringify(observation));
  await expect(importModelObservation(vault, request)).rejects.toMatchObject({
    code: "REVISION_CONFLICT",
  });
  expect((await vault.readConcept(saved.id)).revision).toBe(updated.revision);
});
it("rejects malformed evidence and forged authority before vault initialization", async () => {
  const { root, vault } = await setup();
  const artifact = join(root, "observation.json");
  for (const invalid of [
    "{",
    JSON.stringify({ ...observation, approved: true }),
    JSON.stringify({ ...observation, kind: "preference" }),
    JSON.stringify({
      ...observation,
      evaluation: { evidence: [`urn:swarmx:execution:${randomUUID()}`] },
    }),
    JSON.stringify({ ...observation, cost: { value: 1, unit: "credits", source: "estimated" } }),
    JSON.stringify({ ...observation, elapsed: { value: -1, unit: "ms", source: "clock" } }),
  ]) {
    await writeFile(artifact, invalid);
    await expect(
      importModelObservation(vault, { artifact, title: "Bad trial", requestId: randomUUID() }),
    ).rejects.toBeDefined();
    await expect(lstat(vault.root)).rejects.toMatchObject({ code: "ENOENT" });
  }
});

it("rejects invalid encoding, oversize, unexpected nested data and incomplete unknowns", async () => {
  const { root, vault } = await setup();
  const artifact = join(root, "observation.json");
  const { actual: _actual, ...missing } = observation;
  for (const bytes of [
    Buffer.from([0xff]),
    Buffer.alloc(32 * 1024 + 1, 32),
    Buffer.from(JSON.stringify(missing)),
    Buffer.from(
      JSON.stringify({ ...observation, requested: { ...observation.requested, approved: true } }),
    ),
    Buffer.from(
      JSON.stringify({
        ...observation,
        tokens: {
          input: 1.5,
          output: null,
          cacheRead: null,
          cacheWrite: null,
          unit: "tokens",
          source: "fake",
        },
      }),
    ),
  ]) {
    await writeFile(artifact, bytes);
    await expect(
      importModelObservation(vault, { artifact, title: "Bad trial", requestId: randomUUID() }),
    ).rejects.toBeDefined();
    await expect(lstat(vault.root)).rejects.toMatchObject({ code: "ENOENT" });
  }
});

it("filters exact tags, preserves evaluation kinds and withholds private extensions", async () => {
  const { vault } = await setup();
  const request = {
    title: "Body only",
    type: "Finding",
    description: "No selection tag",
    body: "agent-selection",
  };
  await vault.createConcept(request);
  const concept = await vault.createConcept({
    ...request,
    title: "Preference",
    tags: ["agent-selection"],
    evaluation: {
      kind: "preference",
      task: "Synthetic parsing",
      criteria: "Owner preference",
      evidence: [`urn:swarmx:execution:${randomUUID()}`],
      limitations: "Unresolved external to Host",
    },
    sources: [{ resource: "https://example.test/private-token", secret: "PRIVATE SOURCE" }],
  });
  const result = await queryModelExperience(vault);
  expect(result.concepts).toHaveLength(1);
  expect(result.concepts[0]).toMatchObject({
    kind: "preference",
    provenance: "host-execution-references-unchecked",
    dependencyState: "unchecked",
  });
  expect(JSON.stringify(result)).not.toContain("private-token");
  expect(JSON.stringify(result)).not.toContain("PRIVATE SOURCE");
  await vault.deprecateConcept({ id: concept.id, expectedRevision: concept.revision });
  expect((await queryModelExperience(vault)).concepts).toEqual([]);
});

it("rejects revision races between search and snapshot", async () => {
  const { vault } = await setup();
  const saved = await vault.createConcept({
    title: "Race trial",
    type: "Finding",
    description: "Synthetic",
    body: "Before",
    tags: ["agent-selection"],
  });
  const search = vault.search.bind(vault);
  vi.spyOn(vault, "search").mockImplementationOnce(async (request) => {
    const result = await search(request);
    await vault.updateConcept({ id: saved.id, expectedRevision: saved.revision, body: "After" });
    return result;
  });
  await expect(queryModelExperience(vault)).rejects.toMatchObject({ code: "REVISION_CONFLICT" });
});

it("withholds measurement sources, keeps values and distinguishes cancellation", async () => {
  const { root, vault } = await setup();
  const artifact = join(root, "observation.json");
  await writeFile(
    artifact,
    JSON.stringify({
      ...observation,
      outcome: "cancelled",
      actual: {
        model: "reported-synthetic-model",
        provider: null,
        version: null,
        source: "/private/report",
      },
      elapsed: { value: 42, unit: "ms", source: "/private/clock" },
      cost: {
        value: 0.001,
        unit: "USD",
        source: "/private/receipt",
        coverage: "This synthetic attempt only",
      },
    }),
  );
  await importModelObservation(vault, {
    artifact,
    title: "Cancelled trial",
    requestId: randomUUID(),
  });
  const result = await queryModelExperience(vault);
  expect(result.concepts[0]?.observation).toMatchObject({
    outcome: "cancelled",
    requested: { model: "synthetic-model" },
    actual: { model: "reported-synthetic-model", source: null, sourceWithheld: true },
    cost: { value: 0.001, unit: "USD", source: null },
  });
  expect(JSON.stringify(result)).not.toContain("/private/");
  expect((await queryModelExperience(vault, { includeBody: true })).concepts[0]?.body).toContain(
    "/private/receipt",
  );
});

it("rejects aliased vaults and symlinked artifacts without mutation", async () => {
  const { root, vault } = await setup();
  await mkdir(vault.root);
  const alias = join(root, "alias");
  await symlink(vault.root, alias);
  await expect(queryModelExperience(new MemoryVault({ root: alias }))).rejects.toMatchObject({
    code: "UNSAFE_PATH",
  });
  await expect(
    queryModelExperience(new MemoryVault({ root: join(alias, "missing") })),
  ).rejects.toMatchObject({ code: "UNSAFE_PATH" });
  const file = join(root, "real.json");
  await writeFile(file, JSON.stringify(observation));
  const artifact = join(root, "link.json");
  await symlink(file, artifact);
  await expect(
    importModelObservation(vault, { artifact, title: "Symlink trial", requestId: randomUUID() }),
  ).rejects.toBeDefined();
  expect(await readdir(vault.root)).toEqual([]);
  const dangling = join(root, "dangling");
  await symlink(join(root, "absent"), dangling);
  for (const path of [dangling, join(dangling, "descendant")])
    await expect(queryModelExperience(new MemoryVault({ root: path }))).rejects.toMatchObject({
      code: "UNSAFE_PATH",
    });
});

it("rejects a title collision and malformed stored observations", async () => {
  const { root, vault } = await setup();
  const artifact = join(root, "observation.json");
  await writeFile(artifact, JSON.stringify(observation));
  const request = { artifact, title: "Collision trial", requestId: randomUUID() };
  const saved = await importModelObservation(vault, request);
  await expect(
    importModelObservation(vault, { ...request, requestId: randomUUID() }),
  ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });
  await vault.updateConcept({
    id: saved.id,
    expectedRevision: saved.revision,
    sources: [{ resource: "urn:sha256:bad", swarmx_model_observation: observation }],
  });
  await expect(queryModelExperience(vault)).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
});

it("rejects replay after metadata-only or direct-byte corrections", async () => {
  for (const edit of ["aliases", "generated", "formatting"]) {
    const { root, vault } = await setup();
    const artifact = join(root, "observation.json");
    await writeFile(artifact, JSON.stringify(observation));
    const request = { artifact, title: "Replay integrity", requestId: randomUUID() };
    const saved = await importModelObservation(vault, request);
    if (edit === "aliases") {
      await vault.updateConcept({
        id: saved.id,
        expectedRevision: saved.revision,
        aliases: ["Owner correction"],
      });
    } else {
      const path = join(vault.root, saved.id);
      const original = await readFile(path, "utf8");
      await writeFile(
        path,
        edit === "generated"
          ? original.replace(/by: swarmx-memory\/[^\n]+/u, "by: owner-correction")
          : original.replace("---\n", "---\n# Owner formatting edit\n"),
      );
    }
    // Import initializes the vault and can create/remove an index temp file; only reads promise
    // zero directory mutations. A replay conflict must preserve the concept and index bytes/mtimes.
    const before = await snapshot(vault.root, false);
    await expect(importModelObservation(vault, request)).rejects.toMatchObject({
      code: "REVISION_CONFLICT",
    });
    expect(await snapshot(vault.root, false)).toEqual(before);
  }
});

it("does not print private paths or artifact contents in CLI failures", async () => {
  const { root, vault } = await setup();
  await expect(
    promisify(execFile)(
      process.execPath,
      [
        "--import",
        "tsx",
        resolve("packages/core/memory/src/cli.ts"),
        "import",
        "--vault",
        vault.root,
        "--artifact",
        join(root, "private-missing.json"),
        "--request-id",
        randomUUID(),
        "--title",
        "Synthetic",
      ],
      { cwd: process.cwd() },
    ),
  ).rejects.toMatchObject({ stderr: expect.not.stringContaining(root), code: 1 });
});

it("bounds the exact CLI snapshot output, including detailed bodies", async () => {
  const { vault } = await setup();
  await mkdir(vault.root);
  await Promise.all(
    Array.from({ length: 20 }, (_, i) =>
      writeFile(
        join(vault.root, `trial-${i}.md`),
        renderConcept(
          {
            title: `Trial ${i}`,
            type: "Finding",
            description: "Synthetic",
            tags: ["agent-selection"],
            generated: { at: "2026-01-02T03:04:05Z", by: "synthetic-fixture" },
            status: "draft",
            sources: [],
          },
          "x".repeat(12500),
        ),
      ),
    ),
  );
  const { stdout } = await promisify(execFile)(
    process.execPath,
    [
      "--import",
      "tsx",
      resolve("packages/core/memory/src/cli.ts"),
      "query",
      "--vault",
      vault.root,
      "--include-body",
    ],
    { cwd: process.cwd() },
  );
  expect(Buffer.byteLength(stdout)).toBeLessThanOrEqual(256 * 1024);
  const first = (await vault.search({ query: "Trial 0" })).items[0];
  if (!first) throw new Error("Fixture missing");
  await vault.updateConcept({
    id: first.id,
    expectedRevision: first.revision,
    body: "x".repeat(65000),
  });
  await expect(queryModelExperience(vault, { includeBody: true })).rejects.toMatchObject({
    code: "INVALID_REQUEST",
  });
});
