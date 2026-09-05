import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { MemoryVault } from "@swarmx/memory";
import { formatScienceResourceId } from "@swarmx/science";
import { expect, it } from "vitest";
import { ProductServices } from "../src/host/product-services.js";

it("exposes the Memory tool and persists its six operations across Host restarts", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-host-"));
  const options = {
    productHome: join(root, "product"),
    workspace: { id: "current", label: "Current", root },
  };
  const context = { actorId: "actor", callId: "memory-test", signal: new AbortController().signal };
  const products = await ProductServices.create(options);
  try {
    expect(products.toolManifest.filter((tool) => tool.name === "memory")).toMatchObject([
      {
        inputSchema: {
          properties: {
            action: {
              enum: [
                "search_memory",
                "read_memory",
                "create_memory",
                "update_memory",
                "deprecate_memory",
                "lint_memory",
              ],
            },
          },
        },
      },
    ]);
    const request = {
      title: "Research decision",
      type: "Decision",
      description: "A decision for later sessions.",
      body: "# Decision\n\nUse the recorded protocol.",
    };
    await expect(
      products.callTool(
        "knowledge-base",
        { action: "search_memory", request: { query: "decision" } },
        context,
      ),
    ).rejects.toThrow("Unknown SwarmX product tool");
    await expect(
      products.callTool("memory", { action: "create_memory", request }, context),
    ).rejects.toMatchObject({ code: "AUTHORIZATION_REQUIRED" });
    await expect(
      products.callTool("memory", { action: "create_memory", request, approved: true }, context),
    ).resolves.toMatchObject({
      action: "create_memory",
      data: { metadata: { title: request.title } },
    });
    const [entry] = (await products.memory.vault.search(root, { query: request.title })).items;
    expect(entry).toBeDefined();
    if (!entry) throw new Error("Created memory is missing.");
    await expect(
      products.callTool(
        "memory",
        { action: "search_memory", request: { query: request.title } },
        context,
      ),
    ).resolves.toMatchObject({ data: { items: [{ id: entry.id }] } });
    await expect(
      products.callTool("memory", { action: "read_memory", request: { id: entry.id } }, context),
    ).resolves.toMatchObject({ data: { revision: entry.revision } });
    await products.callTool(
      "memory",
      {
        action: "update_memory",
        approved: true,
        request: {
          id: entry.id,
          expectedRevision: entry.revision,
          body: "# Decision\n\nUse the revised protocol.",
        },
      },
      context,
    );
    const updated = await products.memory.vault.readConcept(root, entry.id);
    await products.callTool(
      "memory",
      {
        action: "deprecate_memory",
        approved: true,
        request: { id: entry.id, expectedRevision: updated.revision },
      },
      context,
    );
    await expect(
      products.callTool("memory", { action: "lint_memory", request: {} }, context),
    ).resolves.toMatchObject({ action: "lint_memory", data: [] });
    expect(existsSync(join(options.productHome, "memory", "vault", entry.id))).toBe(true);
  } finally {
    await products.dispose();
  }
  const reopened = await ProductServices.create(options);
  try {
    const saved = await reopened.memory.vault.search(root, {
      query: "Research decision",
      includeDeprecated: true,
    });
    expect(saved.items).toMatchObject([{ status: "deprecated" }]);
  } finally {
    await reopened.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it("moves the previous vault once without changing revisions or neighboring native memory", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-upgrade-"));
  const previous = join(root, "knowledge-base", "vault");
  const current = join(root, "memory", "vault");
  const vault = new MemoryVault({ root: previous, actor: "swarmx/previous-release" });
  const concept = await vault.createConcept(root, {
    title: "Retained decision",
    type: "Decision",
    description: "Keep the existing revision.",
    body: "# Decision\n\nRetain the source bytes.",
  });
  const bytes = await readFile(join(previous, concept.id));
  const salt = await readFile(join(previous, ".swarmx", "salt"));
  await mkdir(join(root, "memory"));
  await writeFile(join(root, "memory", "README.md"), "Native memory stays here.\n");
  const options = { productHome: root, workspace: { id: "current", label: "Current", root } };
  const products = await ProductServices.create(options);
  try {
    expect(await products.memory.vault.readConcept(root, concept.id)).toEqual(concept);
    expect(await readFile(join(current, concept.id))).toEqual(bytes);
    expect(await readFile(join(current, ".swarmx", "salt"))).toEqual(salt);
    expect(await readFile(join(root, "memory", "README.md"), "utf8")).toBe(
      "Native memory stays here.\n",
    );
    expect(existsSync(join(root, "knowledge-base"))).toBe(false);
  } finally {
    await products.dispose();
  }
  const reopened = await ProductServices.create(options);
  try {
    expect((await reopened.memory.vault.readConcept(root, concept.id)).revision).toBe(
      concept.revision,
    );
  } finally {
    await reopened.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it("rejects conflicting vaults without overwriting either", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-conflict-"));
  try {
    for (const directory of ["knowledge-base", "memory"]) {
      await mkdir(join(root, directory, "vault"), { recursive: true });
      await writeFile(join(root, directory, "vault", "index.md"), directory);
    }
    await expect(
      ProductServices.create({
        productHome: root,
        workspace: { id: "current", label: "Current", root },
      }),
    ).rejects.toThrow("Both previous and current Memory vaults exist");
    for (const directory of ["knowledge-base", "memory"]) {
      expect(await readFile(join(root, directory, "vault", "index.md"), "utf8")).toBe(directory);
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

it("checks Memory Science sources with the current workspace resolver", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-science-"));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: { id: "current", label: "Current", root },
  });
  const context = { actorId: "actor", callId: "lint-test", signal: new AbortController().signal };
  const create = (resource: string) =>
    products.callTool(
      "memory",
      {
        action: "create_memory",
        approved: true,
        request: {
          title: "Evidence",
          type: "Finding",
          description: "Evidence reference",
          body: "# Evidence\n\nResult.[^evidence]\n\n[^evidence]: Science resource.",
          sources: [{ id: "evidence", resource }],
          scope: "global",
        },
      },
      context,
    );
  try {
    const project = products.science.createProject("actor", {
      requestId: randomUUID(),
      title: "Evidence",
    });
    const exact = formatScienceResourceId("project", project.id, project.revision);
    await expect(create("sx:invalid")).rejects.toMatchObject({ code: "INVALID_CONCEPT" });
    expect((await products.memory.vault.search(root, { query: "Evidence" })).items).toEqual([]);
    await expect(create(exact)).resolves.toMatchObject({ diagnostics: [] });
    for (const resource of [
      formatScienceResourceId("project", project.id, project.revision + 1),
      "sx:p/missing@1",
    ]) {
      await expect(create(resource)).resolves.toMatchObject({
        diagnostics: expect.arrayContaining([
          expect.objectContaining({ ruleId: "source.unresolved", severity: "warning" }),
        ]),
      });
    }
    const linted = await products.callTool(
      "memory",
      { action: "lint_memory", request: {} },
      context,
    );
    expect(linted).toMatchObject({
      action: "lint_memory",
      data: expect.arrayContaining([
        expect.objectContaining({ ruleId: "source.unresolved", severity: "warning" }),
      ]),
    });
  } finally {
    await products.dispose();
  }
  const other = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: { id: "other", label: "Other", root },
  });
  try {
    const issues = await other.memory.vault.lint(root);
    expect(issues.filter((issue) => issue.ruleId === "source.unresolved")).toHaveLength(3);
  } finally {
    await other.dispose();
    await rm(root, { recursive: true, force: true });
  }
});
