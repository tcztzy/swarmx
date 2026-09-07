import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it } from "vitest";
import { type MemoryConcept, MemoryVault } from "../src/vault.js";

let root: string;
let vault: MemoryVault;
beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "swarmx-memory-graph-"));
  vault = new MemoryVault({ root: join(root, "vault") });
});
afterEach(() => rm(root, { recursive: true, force: true }));
const dependency = ({ id, revision }: MemoryConcept) => ({ id, revision });
const create = (title: string, dependencies: ReturnType<typeof dependency>[] = []) =>
  vault.createConcept(root, {
    title,
    description: title,
    type: "Playbook",
    body: title,
    dependencies,
  });

it("loads shared prerequisites once in dependency order and propagates stale revisions", async () => {
  const base = await create("Environment");
  const analysis = await create("Analysis", [dependency(base)]);
  const plotting = await create("Plot", [dependency(base), dependency(analysis)]);
  expect((await vault.load(root, plotting.id)).concepts.map(({ id }) => id)).toEqual([
    base.id,
    analysis.id,
    plotting.id,
  ]);
  await vault.updateConcept(root, {
    id: base.id,
    expectedRevision: base.revision,
    body: "Updated environment",
  });
  const graph = await vault.graph(root);
  expect(graph.nodes.filter(({ stale }) => stale).map(({ id }) => id)).toEqual([
    analysis.id,
    plotting.id,
  ]);
  expect(graph.edges.every(({ stale }) => stale)).toBe(true);
});

it("rejects cycles without publishing a partial revision", async () => {
  const base = await create("Base");
  const child = await create("Child", [dependency(base)]);
  await expect(
    vault.updateConcept(root, {
      id: base.id,
      expectedRevision: base.revision,
      dependencies: [dependency(child)],
    }),
  ).rejects.toThrow("cycle");
  expect((await vault.readConcept(root, base.id)).revision).toBe(base.revision);
});

it("rejects inaccessible, missing, deprecated, and outdated dependency targets", async () => {
  const base = await create("Private");
  await expect(
    vault.createConcept(root, {
      title: "Global",
      description: "Global",
      body: "Global",
      type: "Playbook",
      scope: "global",
      dependencies: [dependency(base)],
    }),
  ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
  const other = join(root, "other");
  await mkdir(other);
  await expect(
    vault.createConcept(other, {
      title: "Other",
      description: "Other",
      body: "Other",
      type: "Playbook",
      dependencies: [dependency(base)],
    }),
  ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
  await expect(
    create("Missing", [{ id: "global/concepts/missing.md", revision: base.revision }]),
  ).rejects.toThrow("not found");
  const updated = await vault.updateConcept(root, {
    id: base.id,
    expectedRevision: base.revision,
    body: "Changed",
  });
  await expect(create("Stale", [dependency(base)])).rejects.toThrow("dependency");
  const deprecated = await vault.deprecateConcept(root, {
    id: base.id,
    expectedRevision: updated.revision,
  });
  await expect(create("Deprecated", [dependency(deprecated)])).rejects.toThrow("dependency");
});
