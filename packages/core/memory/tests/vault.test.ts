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
import { executeMemoryOperation, MemoryError, MemoryVault } from "../src/index.js";

const roots: string[] = [];

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-test-"));
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
  const vault = new MemoryVault({ root: vaultRoot });
  await vault.initialize();
  return { alias, otherWorkspace, root, vault, vaultRoot, workspace };
}

function mode(statMode: number): number {
  return statMode & 0o777;
}

describe("MemoryVault", () => {
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
      tags: ["memory", "架构"],
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

    await expect(vault.readConcept(otherWorkspace, concept.id)).rejects.toBeInstanceOf(MemoryError);

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

  it("excludes forged scope metadata and invalid UTF-8 from reads and recall", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const concept = await vault.createConcept(workspace, {
      type: "Reference",
      title: "Boundary",
      description: "Scope boundary",
      body: "# Boundary",
    });
    const path = join(vaultRoot, concept.id);
    const source = await readFile(path, "utf8");
    await writeFile(
      path,
      source.replace(
        `swarmx_workspace: ${concept.metadata.swarmx_workspace}`,
        "swarmx_workspace: aaaaaaaaaaaa",
      ),
    );
    await expect(vault.readConcept(workspace, concept.id)).rejects.toMatchObject({
      code: "INVALID_CONCEPT",
    });
    expect((await vault.search(workspace, { query: "Boundary" })).items).toEqual([]);
    expect(await vault.lint(workspace, { id: concept.id })).toContainEqual(
      expect.objectContaining({
        ruleId: "scope.mismatch",
        path: concept.id,
        severity: "error",
      }),
    );
    await writeFile(path, Buffer.concat([Buffer.from(source), Buffer.from([0xff])]));
    await expect(vault.readConcept(workspace, concept.id)).rejects.toMatchObject({
      code: "INVALID_CONCEPT",
    });
    expect(await vault.lint(workspace, { id: concept.id })).toContainEqual(
      expect.objectContaining({ ruleId: "document.encoding" }),
    );
  });

  it("excludes deprecated exact matches by default and exposes expiry", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const concept = await vault.createConcept(workspace, {
      type: "Reference",
      title: "Lifecycle",
      description: "Lifecycle",
      body: "# Lifecycle",
    });
    const path = join(vaultRoot, concept.id);
    await writeFile(
      path,
      (await readFile(path, "utf8")).replace(
        "status: draft",
        "status: draft\nstale_after: 2000-01-01T00:00:00Z",
      ),
    );
    expect((await vault.search(workspace, { query: "Lifecycle" })).items).toMatchObject([
      { stale: true },
    ]);
    const current = await vault.readConcept(workspace, concept.id);
    await vault.deprecateConcept(workspace, { id: current.id, expectedRevision: current.revision });
    expect((await vault.search(workspace, { query: "Lifecycle" })).items).toEqual([]);
    expect(
      (await vault.search(workspace, { query: "Lifecycle", includeDeprecated: true })).items,
    ).toMatchObject([{ id: concept.id, status: "deprecated", stale: true }]);
  });

  it("lints without writes, refuses symlinks, and hides foreign workspaces", async () => {
    const { vault, vaultRoot, workspace, otherWorkspace, root } = await fixture();
    const request = {
      type: "Finding",
      title: "Evidence",
      description: "Needs evidence",
      body: "# Evidence",
    };
    const current = await vault.createConcept(workspace, request);
    const foreign = await vault.createConcept(otherWorkspace, {
      ...request,
      title: "Private foreign",
    });
    await writeFile(join(vaultRoot, foreign.id), "malformed foreign content");
    const external = join(root, "outside.md");
    await writeFile(external, "external secret");
    const linked = `${current.id.slice(0, current.id.lastIndexOf("/"))}/linked.md`;
    await symlink(external, join(vaultRoot, linked));
    await chmod(vaultRoot, 0o750);
    const paths = [
      vaultRoot,
      join(vaultRoot, current.id),
      join(vaultRoot, "index.md"),
      join(vaultRoot, "log.md"),
    ];
    const before = await Promise.all(paths.map((path) => lstat(path)));
    const diagnostics = await vault.lint(workspace, { now: "2026-09-05T00:00:00Z" });
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ path: linked, revision: null, severity: "error" }),
    );
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ path: current.id, ruleId: "source.missing" }),
    );
    expect(JSON.stringify(diagnostics)).not.toContain(foreign.id);
    expect(JSON.stringify(diagnostics)).not.toContain("external secret");
    const after = await Promise.all(paths.map((path) => lstat(path)));
    expect(after.map(({ mtimeMs, mode }) => ({ mtimeMs, mode }))).toEqual(
      before.map(({ mtimeMs, mode }) => ({ mtimeMs, mode })),
    );
    await expect(vault.lint(workspace, { id: foreign.id })).rejects.toMatchObject({
      code: "UNSAFE_PATH",
    });
    expect(await vault.lint(workspace, { id: "global/concepts/missing.md" })).toContainEqual(
      expect.objectContaining({ path: "global/concepts/missing.md", severity: "error" }),
    );
  });

  it("returns post-edit diagnostics and keeps approval and cancellation enforced", async () => {
    const { vault, workspace } = await fixture();
    const context = {
      workspaceRoot: workspace,
      actorId: "test",
      callId: "test",
      signal: new AbortController().signal,
      approve: async () => "allowed-once",
    };
    const call = {
      action: "create_memory",
      request: {
        type: "Finding",
        title: "Review me",
        description: "Review me",
        body: "# Review me",
      },
    };
    await expect(
      executeMemoryOperation(vault, call, { ...context, approve: async () => "rejected" }),
    ).rejects.toMatchObject({ code: "AUTHORIZATION_REQUIRED" });
    const aborted = new AbortController();
    aborted.abort(new Error("Cancelled"));
    await expect(
      executeMemoryOperation(vault, call, { ...context, signal: aborted.signal }),
    ).rejects.toThrow("Cancelled");
    const created = await executeMemoryOperation(vault, call, context);
    expect(created.diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "source.missing", severity: "warning" }),
    );
    expect(created.diagnostics?.filter((issue) => issue.severity === "error")).toEqual([]);
    const [item] = (await vault.search(workspace, { query: "Review me" })).items;
    if (!item) throw new Error("Expected created concept");
    const updated = await executeMemoryOperation(
      vault,
      {
        action: "update_memory",
        request: {
          id: item.id,
          expectedRevision: item.revision,
          body: "# Review me\n\n[Missing](./missing.md)",
        },
      },
      context,
    );
    expect(updated.diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "link.broken", path: item.id }),
    );
    const read = await vault.readConcept(workspace, item.id);
    const deprecated = await executeMemoryOperation(
      vault,
      {
        action: "deprecate_memory",
        request: {
          id: read.id,
          expectedRevision: read.revision,
        },
      },
      context,
    );
    expect(deprecated.diagnostics).toContainEqual(
      expect.objectContaining({ ruleId: "link.broken" }),
    );
    const linted = await executeMemoryOperation(
      vault,
      { action: "lint_memory", request: { id: read.id } },
      {
        ...context,
        approve: async () => {
          throw new Error("Read-only lint requested approval");
        },
      },
    );
    expect(linted.data).toContainEqual(expect.objectContaining({ ruleId: "link.broken" }));
  });

  it("rejects invalid updates before replacing content or recording history", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const concept = await vault.createConcept(workspace, {
      type: "Reference",
      title: "Untouched",
      description: "Untouched",
      body: "# Untouched",
    });
    const source = await readFile(join(vaultRoot, concept.id));
    await expect(
      vault.updateConcept(workspace, {
        id: concept.id,
        expectedRevision: concept.revision,
        body: "# Invalid\n\n[^undefined]",
      }),
    ).rejects.toThrow("footnote.undefined");
    for (const body of ["# Invalid\n\n[^undefined]", "[Escape](../../../../secret.md)"]) {
      await expect(
        vault.updateConcept(workspace, {
          id: concept.id,
          expectedRevision: concept.revision,
          body,
        }),
      ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
    }
    expect(await readFile(join(vaultRoot, concept.id))).toEqual(source);
    await expect(lstat(join(vaultRoot, ".swarmx", "history"))).rejects.toMatchObject({
      code: "ENOENT",
    });
  });

  it("reports incomplete and oversized scans without claiming an inspected revision", async () => {
    const { vault, vaultRoot, workspace } = await fixture();
    const concept = await vault.createConcept(workspace, {
      type: "Reference",
      title: "A",
      description: "A",
      body: "# A",
    });
    await vault.createConcept(workspace, {
      type: "Reference",
      title: "B",
      description: "B",
      body: "# B",
    });
    const limited = new MemoryVault({ root: vaultRoot, maxSearchPages: 1 });
    expect(await limited.lint(workspace)).toContainEqual(
      expect.objectContaining({ ruleId: "scan.limit", revision: null }),
    );
    await writeFile(join(vaultRoot, concept.id), "x".repeat(128 * 1024 + 1));
    expect(await vault.lint(workspace, { id: concept.id })).toContainEqual(
      expect.objectContaining({ ruleId: "document.size", path: concept.id, revision: null }),
    );
  });
});
