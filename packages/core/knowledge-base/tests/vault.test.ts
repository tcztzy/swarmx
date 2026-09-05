import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parseDocument } from "yaml";
import { KnowledgeBaseError, KnowledgeBaseVault } from "../src/index.js";

const roots: string[] = [];

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-knowledge-base-test-"));
  roots.push(root);
  return root;
}

afterEach(async () => {
  const { rm } = await import("node:fs/promises");
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

async function fixture() {
  const root = await temporaryRoot();
  const vaultRoot = join(root, "vault");
  const firstParent = join(root, "first");
  const secondParent = join(root, "second");
  const workspace = join(firstParent, "project");
  const otherWorkspace = join(secondParent, "project");
  const alias = join(root, "project-alias");
  await Promise.all([
    mkdir(workspace, { recursive: true }),
    mkdir(otherWorkspace, { recursive: true }),
  ]);
  await symlink(workspace, alias);
  const vault = new KnowledgeBaseVault({ root: vaultRoot });
  await vault.initialize();
  return { alias, otherWorkspace, root, vault, vaultRoot, workspace };
}

function mode(statMode: number): number {
  return statMode & 0o777;
}

describe("KnowledgeBaseVault", () => {
  it("V130 V137: initializes one owner-only OKF bundle", async () => {
    const { vaultRoot } = await fixture();

    expect(mode((await lstat(vaultRoot)).mode)).toBe(0o700);
    expect(mode((await lstat(join(vaultRoot, "index.md"))).mode)).toBe(0o600);
    expect(mode((await lstat(join(vaultRoot, "log.md"))).mode)).toBe(0o600);
    expect(await readFile(join(vaultRoot, "index.md"), "utf8")).toContain('okf_version: "0.2"');
  });

  it("V132: canonicalizes aliases and separates same-basename workspaces", async () => {
    const { alias, otherWorkspace, vault, workspace } = await fixture();

    const direct = await vault.resolveWorkspace(workspace);
    const throughAlias = await vault.resolveWorkspace(alias);
    const other = await vault.resolveWorkspace(otherWorkspace);

    expect(direct).toEqual(throughAlias);
    expect(other.key).not.toBe(direct.key);
    expect(other.directory).not.toBe(direct.directory);
    expect(JSON.stringify([direct, other])).not.toContain(await realpath(workspace));
  });

  it("V130 V131 V136 V137: creates, revises, indexes, logs, and deprecates a concept", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const created = await vault.createConcept(workspace, {
      body: "# 决定\n\n知识库使用 Markdown。",
      description: "知识库使用开放 Markdown 作为持久知识。",
      scope: "workspace",
      tags: ["knowledge-base", "架构"],
      title: "知识库使用 Markdown",
      type: "Decision",
    });

    expect(created.id).toMatch(/^workspaces\/project--[a-f0-9]{12}\/concepts\//u);
    expect(created.revision).toMatch(/^sha256:[a-f0-9]{64}$/u);
    expect(JSON.stringify(created)).not.toContain(workspace);
    const createdPath = join(vaultRoot, created.id);
    expect(mode((await lstat(createdPath)).mode)).toBe(0o600);

    const source = await readFile(createdPath, "utf8");
    const document = parseDocument(source.slice(4, source.indexOf("\n---\n", 4)));
    document.set("x-obsidian-field", "preserve-me");
    const externallyEdited = `---\n${document.toString({ lineWidth: 0 })}---${source.slice(source.indexOf("\n---\n", 4) + 4)}`;
    await writeFile(createdPath, externallyEdited, { mode: 0o600 });
    const observed = await vault.readConcept(workspace, created.id);

    await expect(
      vault.updateConcept(workspace, {
        body: "# stale",
        description: "stale",
        expectedRevision: created.revision,
        id: created.id,
        title: created.metadata.title,
      }),
    ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });

    const updated = await vault.updateConcept(workspace, {
      body: "# 决定\n\n知识库使用 OKF v0.2 Markdown。",
      description: "知识库使用 OKF v0.2 Markdown 作为持久知识。",
      expectedRevision: observed.revision,
      id: observed.id,
      title: observed.metadata.title,
    });
    expect(updated.metadata["x-obsidian-field"]).toBe("preserve-me");
    expect(updated.revision).not.toBe(observed.revision);

    const deprecated = await vault.deprecateConcept(workspace, {
      expectedRevision: updated.revision,
      id: updated.id,
    });
    expect(deprecated.metadata.status).toBe("deprecated");
    expect((await lstat(createdPath)).isFile()).toBe(true);

    const workspaceDirectory = join(vaultRoot, updated.id.split("/concepts/")[0] ?? "");
    expect(await readFile(join(workspaceDirectory, "index.md"), "utf8")).toContain(
      `](./concepts/${updated.id.split("/concepts/")[1]})`,
    );
    const log = await readFile(join(vaultRoot, "log.md"), "utf8");
    expect(log).toContain("**Creation**");
    expect(log).toContain("**Update**");
    expect(log).toContain("**Deprecation**");

    const historyDirectory = join(vaultRoot, ".swarmx", "history");
    const historyEntries = await import("node:fs/promises").then(({ readdir }) =>
      readdir(historyDirectory, { recursive: true }),
    );
    expect(historyEntries.filter((entry) => entry.endsWith(".md"))).toHaveLength(2);
  });

  it("V178: makes owner-side concept creation idempotent and rejects changed reuse", async () => {
    const { vault, workspace } = await fixture();
    const request = {
      requestId: "10000000-0000-4000-8000-000000000001",
      body: "# Verified\n\nOne admitted synthesis.",
      description: "One verified admitted synthesis.",
      scope: "workspace" as const,
      sources: [{ resource: "urn:uuid:20000000-0000-4000-8000-000000000001" }],
      title: "Verified synthesis",
      type: "Finding",
    };

    const first = await vault.createConcept(workspace, request);
    const repeated = await vault.createConcept(workspace, request);
    expect(repeated).toEqual(first);
    expect(repeated.metadata.swarmx_request_id).toBe(request.requestId);
    await expect(
      vault.createConcept(workspace, { ...request, body: "# Changed" }),
    ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });
  });

  it("V221: normalizes omitted create scope before idempotency hashing", async () => {
    const { vault, workspace } = await fixture();
    const request = {
      requestId: "10000000-0000-4000-8000-000000000221",
      body: "# Default scope\n\nWorkspace knowledge.",
      description: "Workspace knowledge with an omitted scope.",
      title: "Default workspace scope",
      type: "Finding",
    };

    const omitted = await vault.createConcept(workspace, request);
    const explicit = await vault.createConcept(workspace, { ...request, scope: "workspace" });
    expect(explicit).toEqual(omitted);
    expect(omitted.id).toMatch(/^workspaces\/project--[a-f0-9]{12}\/concepts\//u);

    const global = await vault.createConcept(workspace, {
      ...request,
      requestId: "10000000-0000-4000-8000-000000000222",
      scope: "global",
      title: "Explicit global scope",
    });
    expect(global.id).toMatch(/^global\/concepts\//u);
  });

  it("V131 V139: preserves malformed hand edits and reports relative diagnostics", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const resolved = await vault.resolveWorkspace(workspace);
    const concepts = join(vaultRoot, resolved.directory, "concepts");
    await mkdir(concepts, { mode: 0o700, recursive: true });
    const malformedPath = join(concepts, "broken.md");
    const malformed = "---\ntitle: Missing type\n---\n\n[[Non portable]].\n";
    await writeFile(malformedPath, malformed, { mode: 0o600 });

    const result = await vault.search(workspace, { query: "portable" });

    expect(result.items).toEqual([]);
    expect(result.diagnostics).toEqual([
      expect.objectContaining({
        path: `${resolved.directory}/concepts/broken.md`,
      }),
    ]);
    expect(JSON.stringify(result.diagnostics)).not.toContain(vaultRoot);
    expect(await readFile(malformedPath, "utf8")).toBe(malformed);
  });

  it("V132: rejects foreign workspace ids and symlinked concept files", async () => {
    const { otherWorkspace, root, vault, vaultRoot, workspace } = await fixture();
    const concept = await vault.createConcept(workspace, {
      body: "# Private",
      description: "Current workspace only.",
      scope: "workspace",
      title: "Private",
      type: "Reference",
    });

    await expect(vault.readConcept(otherWorkspace, concept.id)).rejects.toBeInstanceOf(
      KnowledgeBaseError,
    );

    const resolved = await vault.resolveWorkspace(workspace);
    const external = join(root, "external.md");
    await writeFile(external, "secret", { mode: 0o600 });
    const link = join(vaultRoot, resolved.directory, "concepts", "linked.md");
    await symlink(external, link);
    await expect(
      vault.readConcept(workspace, `${resolved.directory}/concepts/linked.md`),
    ).rejects.toMatchObject({ code: "UNSAFE_PATH" });

    await chmod(external, 0o600);
  });

  it("V131: rejects nonportable generated body syntax", async () => {
    const { vault, workspace } = await fixture();

    await expect(
      vault.createConcept(workspace, {
        body: "# Bad\n\n[[Wikilink]]",
        description: "Must remain portable.",
        scope: "workspace",
        title: "Bad",
        type: "Reference",
      }),
    ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
  });
});
