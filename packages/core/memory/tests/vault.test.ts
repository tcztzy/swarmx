import { chmod, lstat, mkdtemp, readdir, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { parseDocument } from "yaml";
import manifest from "../package.json";
import { executeMemoryOperation, MemoryVault } from "../src/index.js";

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
  const vault = new MemoryVault({ root: vaultRoot });
  await vault.initialize();
  return { root, vault, vaultRoot };
}

function mode(statMode: number): number {
  return statMode & 0o777;
}

describe("MemoryVault", () => {
  it("V130 V137: initializes one owner-only OKF bundle", async () => {
    const { vaultRoot } = await fixture();

    expect(mode((await lstat(vaultRoot)).mode)).toBe(0o700);
    expect(mode((await lstat(join(vaultRoot, "index.md"))).mode)).toBe(0o600);
    expect((await readdir(vaultRoot)).sort()).toEqual(["index.md"]);
    expect(await readFile(join(vaultRoot, "index.md"), "utf8")).toContain('okf_version: "0.2"');
  });

  it("V130 V131 V136 V137: creates, revises, indexes, and deprecates a concept", async () => {
    const { vault, vaultRoot } = await fixture();
    const created = await vault.createConcept({
      body: "# 决定\n\n知识库使用 Markdown。",
      description: "知识库使用开放 Markdown 作为持久知识。",
      tags: ["memory", "架构"],
      title: "知识库使用 Markdown",
      type: "Decision",
    });

    expect(created.id).toMatch(/^[^/]+\.md$/u);
    expect(created.revision).toMatch(/^sha256:[a-f0-9]{64}$/u);
    expect(created.metadata.generated.by).toBe(`swarmx-memory/${manifest.version}`);
    expect(JSON.stringify(created)).not.toContain(vaultRoot);
    const createdPath = join(vaultRoot, created.id);
    expect(mode((await lstat(createdPath)).mode)).toBe(0o600);

    const source = await readFile(createdPath, "utf8");
    const document = parseDocument(source.slice(4, source.indexOf("\n---\n", 4)));
    document.set("x-obsidian-field", "preserve-me");
    const externallyEdited = `---\n${document.toString({ lineWidth: 0 })}---${source.slice(source.indexOf("\n---\n", 4) + 4)}`;
    await writeFile(createdPath, externallyEdited, { mode: 0o600 });
    const observed = await vault.readConcept(created.id);

    await expect(
      vault.updateConcept({
        body: "# stale",
        description: "stale",
        expectedRevision: created.revision,
        id: created.id,
        title: created.metadata.title,
      }),
    ).rejects.toMatchObject({ code: "REVISION_CONFLICT" });

    const updated = await vault.updateConcept({
      body: "# 决定\n\n知识库使用 OKF v0.2 Markdown。",
      description: "知识库使用 OKF v0.2 Markdown 作为持久知识。",
      expectedRevision: observed.revision,
      id: observed.id,
      title: observed.metadata.title,
    });
    expect(updated.metadata["x-obsidian-field"]).toBe("preserve-me");
    expect(updated.revision).not.toBe(observed.revision);

    const deprecated = await vault.deprecateConcept({
      expectedRevision: updated.revision,
      id: updated.id,
    });
    expect(deprecated.metadata.status).toBe("deprecated");
    expect((await lstat(createdPath)).isFile()).toBe(true);

    expect(await readFile(join(vaultRoot, "index.md"), "utf8")).toContain(`](./${updated.id})`);
    expect((await readdir(vaultRoot)).sort()).toEqual(["index.md", updated.id]);
  });

  it("uses canonical concept names and rejects duplicate creation without overwriting", async () => {
    const { vault, vaultRoot } = await fixture();
    const request = {
      title: "Agent Protocols",
      description: "Protocols for agents.",
      body: "# Agent Protocols\n\nVerified research.",
      type: "Reference",
    };
    const created = await vault.createConcept(request);
    expect(created.id).toBe("agent-protocols.md");
    const paths = [created.id, "index.md"];
    const before = await Promise.all(paths.map((path) => readFile(join(vaultRoot, path))));

    for (const title of ["Agent Protocols", "agent-protocols"]) {
      await expect(
        vault.createConcept({ ...request, title, body: "# Replacement" }),
      ).rejects.toMatchObject({
        code: "REVISION_CONFLICT",
        message: expect.stringContaining(created.id),
      });
    }
    expect(await Promise.all(paths.map((path) => readFile(join(vaultRoot, path))))).toEqual(before);
    expect((await vault.search({ query: "Agent Protocols" })).items).toHaveLength(1);
    const updated = await vault.updateConcept({
      id: created.id,
      expectedRevision: created.revision,
      body: "# Agent Protocols\n\nUpdated research.",
    });
    expect(updated.id).toBe(created.id);
    expect(updated.body).toContain("Updated research.");
  });

  it("keeps one root index, reserves its name, and generates no change logs", async () => {
    const { vault, vaultRoot } = await fixture();
    const request = {
      title: "Shared knowledge",
      description: "Shared knowledge.",
      body: "# Shared knowledge",
      type: "Reference",
    };
    const shared = await vault.createConcept(request);
    const rootIndex = await readFile(join(vaultRoot, "index.md"), "utf8");
    expect(rootIndex).toContain("[Shared knowledge](./shared-knowledge.md)");
    for (const title of ["Index", "README", "User"]) {
      await expect(vault.createConcept({ ...request, title })).rejects.toMatchObject({
        code: "INVALID_REQUEST",
      });
    }
    await expect(vault.readConcept("index.md")).rejects.toMatchObject({
      code: "UNSAFE_PATH",
    });
    for (const id of ["README.md", "USER.md", "nested/private.md"]) {
      await expect(vault.readConcept(id)).rejects.toMatchObject({ code: "UNSAFE_PATH" });
    }
    const reopened = new MemoryVault({ root: vaultRoot });
    expect(await reopened.readConcept(shared.id)).toEqual(shared);
    expect(await readFile(join(vaultRoot, "index.md"), "utf8")).toBe(rootIndex);
    expect((await readdir(vaultRoot)).sort()).toEqual(["index.md", "shared-knowledge.md"]);
    expect(reopened.indexSnapshot()).toContain("[Shared knowledge](./shared-knowledge.md)");
    expect(await reopened.lint()).toEqual([]);
  });

  it("V178: makes owner-side concept creation idempotent and rejects changed reuse", async () => {
    const { vault } = await fixture();
    const request = {
      requestId: "10000000-0000-4000-8000-000000000001",
      body: "# Verified\n\nOne admitted synthesis.",
      description: "One verified admitted synthesis.",
      sources: [{ resource: "urn:uuid:20000000-0000-4000-8000-000000000001" }],
      title: "Verified synthesis",
      type: "Finding",
    };

    const first = await vault.createConcept(request);
    const repeated = await vault.createConcept(request);
    expect(repeated).toEqual(first);
    expect(repeated.metadata.swarmx_request_id).toBe(request.requestId);
    await expect(vault.createConcept({ ...request, body: "# Changed" })).rejects.toMatchObject({
      code: "REVISION_CONFLICT",
    });
  });

  it("replays the last update across restart without changing its revision and repairs the index", async () => {
    const { vault, vaultRoot } = await fixture();
    const created = await vault.createConcept({
      requestId: "10000000-0000-4000-8000-000000000001",
      title: "Provider experience",
      description: "Observed provider behavior.",
      type: "Finding",
      body: "# Provider experience\n\nOriginal observation.",
    });
    const request = {
      id: created.id,
      expectedRevision: created.revision,
      requestId: "10000000-0000-4000-8000-000000000002",
      description: "Updated provider behavior.",
      body: "# Provider experience\n\nVerified observation.",
    };
    const updated = await vault.updateConcept(request);
    await writeFile(join(vaultRoot, "index.md"), '---\nokf_version: "0.2"\n---\n\n# Stale index\n');
    const reopened = new MemoryVault({ root: vaultRoot });
    expect(await reopened.updateConcept(request)).toEqual(updated);
    expect(await readFile(join(vaultRoot, "index.md"), "utf8")).toContain(request.description);
    expect(updated.metadata.swarmx_request_id).toBe(created.metadata.swarmx_request_id);
    expect(updated.metadata.swarmx_request_hash).toBe(created.metadata.swarmx_request_hash);
    expect(updated.metadata.swarmx_update_request_id).toBe(request.requestId);
    expect(updated.metadata.swarmx_update_request_hash).toMatch(/^sha256:[a-f0-9]{64}$/u);
    for (const changed of [
      { ...request, body: "# Different observation" },
      { ...request, expectedRevision: updated.revision },
    ])
      await expect(reopened.updateConcept(changed)).rejects.toMatchObject({
        code: "REVISION_CONFLICT",
      });
    expect(await reopened.readConcept(created.id)).toEqual(updated);
    const path = join(vaultRoot, created.id);
    await writeFile(
      path,
      (await readFile(path, "utf8")).replace("Verified observation.", "Hand edited observation."),
    );
    await expect(reopened.updateConcept(request)).rejects.toMatchObject({
      code: "REVISION_CONFLICT",
    });
    expect((await reopened.readConcept(created.id)).body).toContain("Hand edited observation.");
  });

  it.each([undefined, "10000000-0000-4000-8000-000000000003"])(
    "rejects stale replay after an intervening update with requestId %s",
    async (requestId) => {
      const { vault } = await fixture();
      const created = await vault.createConcept({
        title: "Provider experience",
        description: "Observed provider behavior.",
        type: "Finding",
        body: "# Original observation",
      });
      const request = {
        id: created.id,
        expectedRevision: created.revision,
        requestId: "10000000-0000-4000-8000-000000000002",
        body: "# Reviewed observation",
      };
      const reviewed = await vault.updateConcept(request);
      const later = await vault.updateConcept({
        id: created.id,
        expectedRevision: reviewed.revision,
        ...(requestId === undefined ? {} : { requestId }),
        body: "# User correction",
      });
      expect(later.metadata.swarmx_update_request_id).toBe(requestId);
      if (requestId === undefined)
        expect(later.metadata.swarmx_update_request_hash).toBeUndefined();
      await expect(vault.updateConcept(request)).rejects.toMatchObject({
        code: "REVISION_CONFLICT",
      });
      expect(await vault.readConcept(created.id)).toEqual(later);
    },
  );

  it("V131 V139: preserves malformed hand edits and reports relative diagnostics", async () => {
    const { vault, vaultRoot } = await fixture();
    const malformed = "---\ntitle: Missing type\n---\n\n[[Non portable]].\n";
    await writeFile(join(vaultRoot, "broken.md"), malformed, { mode: 0o600 });

    const result = await vault.search({ query: "portable" });

    expect(result.items).toEqual([]);
    expect(result.diagnostics).toEqual([expect.objectContaining({ path: "broken.md" })]);
    expect(JSON.stringify(result.diagnostics)).not.toContain(vaultRoot);
    expect(await readFile(join(vaultRoot, "broken.md"), "utf8")).toBe(malformed);
  });

  it("V132: rejects symlinked concept files", async () => {
    const { root, vault, vaultRoot } = await fixture();
    const external = join(root, "external.md");
    await writeFile(external, "secret", { mode: 0o600 });
    await symlink(external, join(vaultRoot, "linked.md"));
    await expect(vault.readConcept("linked.md")).rejects.toMatchObject({ code: "UNSAFE_PATH" });

    await chmod(external, 0o600);
  });

  it("V131: rejects nonportable generated body syntax", async () => {
    const { vault } = await fixture();

    await expect(
      vault.createConcept({
        body: "# Bad\n\n[[Wikilink]]",
        description: "Must remain portable.",
        title: "Bad",
        type: "Reference",
      }),
    ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
  });

  it("preserves custom metadata while rejecting invalid UTF-8", async () => {
    const { vault, vaultRoot } = await fixture();
    const concept = await vault.createConcept({
      type: "Reference",
      title: "Boundary",
      description: "Document boundary",
      body: "# Boundary",
    });
    const path = join(vaultRoot, concept.id);
    const source = await readFile(path, "utf8");
    const customized = source.replace("status: draft", "status: draft\nx-reviewer: researcher");
    await writeFile(path, customized);
    const observed = await vault.readConcept(concept.id);
    expect(observed.metadata["x-reviewer"]).toBe("researcher");
    await writeFile(path, Buffer.concat([Buffer.from(customized), Buffer.from([0xff])]));
    await expect(vault.readConcept(concept.id)).rejects.toMatchObject({
      code: "INVALID_CONCEPT",
    });
    expect(await vault.lint({ id: concept.id })).toContainEqual(
      expect.objectContaining({ ruleId: "document.encoding" }),
    );
  });

  it("excludes deprecated exact matches by default and exposes expiry", async () => {
    const { vault, vaultRoot } = await fixture();
    const concept = await vault.createConcept({
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
    expect((await vault.search({ query: "Lifecycle" })).items).toMatchObject([{ stale: true }]);
    const current = await vault.readConcept(concept.id);
    await vault.deprecateConcept({ id: current.id, expectedRevision: current.revision });
    expect((await vault.search({ query: "Lifecycle" })).items).toEqual([]);
    expect(
      (await vault.search({ query: "Lifecycle", includeDeprecated: true })).items,
    ).toMatchObject([{ id: concept.id, status: "deprecated", stale: true }]);
  });

  it("lints without writes, refuses symlinks, and hides documents outside the root", async () => {
    const { root, vault, vaultRoot } = await fixture();
    const request = {
      type: "Finding",
      title: "Evidence",
      description: "Needs evidence",
      body: "# Evidence",
    };
    const current = await vault.createConcept(request);
    const external = join(root, "outside.md");
    await writeFile(external, "external secret");
    await symlink(external, join(vaultRoot, "linked.md"));
    await chmod(vaultRoot, 0o750);
    const paths = [vaultRoot, join(vaultRoot, current.id), join(vaultRoot, "index.md")];
    const before = await Promise.all(paths.map((path) => lstat(path)));
    const diagnostics = await vault.lint({ now: "2026-09-05T00:00:00Z" });
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ path: "linked.md", revision: null, severity: "error" }),
    );
    expect(diagnostics).toContainEqual(
      expect.objectContaining({ path: current.id, ruleId: "source.missing" }),
    );
    expect(JSON.stringify(diagnostics)).not.toContain("external secret");
    const after = await Promise.all(paths.map((path) => lstat(path)));
    expect(after.map(({ mtimeMs, mode }) => ({ mtimeMs, mode }))).toEqual(
      before.map(({ mtimeMs, mode }) => ({ mtimeMs, mode })),
    );
    await expect(vault.lint({ id: "nested/private.md" })).rejects.toMatchObject({
      code: "UNSAFE_PATH",
    });
    expect(await vault.lint({ id: "missing.md" })).toContainEqual(
      expect.objectContaining({ path: "missing.md", severity: "error" }),
    );
  });

  it("returns post-edit diagnostics and keeps approval and cancellation enforced", async () => {
    const { vault } = await fixture();
    const context = {
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
    const [item] = (await vault.search({ query: "Review me" })).items;
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
    const read = await vault.readConcept(item.id);
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

  it("rejects invalid updates before replacing content", async () => {
    const { vault, vaultRoot } = await fixture();
    const concept = await vault.createConcept({
      type: "Reference",
      title: "Untouched",
      description: "Untouched",
      body: "# Untouched",
    });
    const source = await readFile(join(vaultRoot, concept.id));
    await expect(
      vault.updateConcept({
        id: concept.id,
        expectedRevision: concept.revision,
        body: "# Invalid\n\n[^undefined]",
      }),
    ).rejects.toThrow("footnote.undefined");
    for (const body of ["# Invalid\n\n[^undefined]", "[Escape](../../../../secret.md)"]) {
      await expect(
        vault.updateConcept({
          id: concept.id,
          expectedRevision: concept.revision,
          body,
        }),
      ).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
    }
    expect(await readFile(join(vaultRoot, concept.id))).toEqual(source);
    await expect(lstat(join(vaultRoot, ".swarmx"))).rejects.toMatchObject({
      code: "ENOENT",
    });
  });

  it("reports incomplete and oversized scans without claiming an inspected revision", async () => {
    const { vault, vaultRoot } = await fixture();
    const concept = await vault.createConcept({
      type: "Reference",
      title: "A",
      description: "A",
      body: "# A",
    });
    await vault.createConcept({
      type: "Reference",
      title: "B",
      description: "B",
      body: "# B",
    });
    const limited = new MemoryVault({ root: vaultRoot, maxSearchPages: 1 });
    expect(await limited.lint()).toContainEqual(
      expect.objectContaining({ ruleId: "scan.limit", revision: null }),
    );
    await writeFile(join(vaultRoot, concept.id), "x".repeat(128 * 1024 + 1));
    expect(await vault.lint({ id: concept.id })).toContainEqual(
      expect.objectContaining({ ruleId: "document.size", path: concept.id, revision: null }),
    );
  });
});
