import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { formatScienceResourceId } from "@swarmx/science";
import { expect, it } from "vitest";
import { ProductServices } from "../src/host/product-services.js";

it("exposes Memory operations, rejects model approvals and persists across Host restarts", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-host-"));
  const options = {
    productHome: join(root, "product"),
    cwd: root,
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
                "graph_memory",
                "load_memory",
                "read_memory_guide",
                "read_core_memory",
                "update_core_memory",
                "search_sessions",
                "memory_status",
                "memory_configure",
                "memory_review",
                "memory_decide",
                "export_evaluation",
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
      products.callTool("memory", { action: "create_memory", request, approved: true }, context),
    ).rejects.toThrow();
    await expect(
      products.callTool("memory", { action: "create_memory", request }, context),
    ).resolves.toMatchObject({
      action: "create_memory",
      data: { metadata: { title: request.title } },
    });
    const [entry] = (await products.memory.vault.search({ query: request.title })).items;
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
        request: {
          id: entry.id,
          expectedRevision: entry.revision,
          body: "# Decision\n\nUse the revised protocol.",
        },
      },
      context,
    );
    const updated = await products.memory.vault.readConcept(entry.id);
    await products.callTool(
      "memory",
      {
        action: "deprecate_memory",
        request: { id: entry.id, expectedRevision: updated.revision },
      },
      context,
    );
    await expect(
      products.callTool("memory", { action: "lint_memory", request: {} }, context),
    ).resolves.toMatchObject({ action: "lint_memory", data: [] });
    expect(existsSync(join(options.productHome, "memory", entry.id))).toBe(true);
  } finally {
    await products.dispose();
  }
  const reopened = await ProductServices.create(options);
  try {
    const saved = await reopened.memory.vault.search({
      query: "Research decision",
      includeDeprecated: true,
    });
    expect(saved.items).toMatchObject([{ status: "deprecated" }]);
  } finally {
    await reopened.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it("checks Memory Science sources with the current directory resolver", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-science-"));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
  });
  const context = { actorId: "actor", callId: "lint-test", signal: new AbortController().signal };
  const create = (resource: string) =>
    products.callTool(
      "memory",
      {
        action: "create_memory",
        request: {
          title: `Evidence for ${resource}`,
          type: "Finding",
          description: "Evidence reference",
          body: "# Evidence\n\nResult.[^evidence]\n\n[^evidence]: Science resource.",
          sources: [{ id: "evidence", resource }],
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
    expect((await products.memory.vault.search({ query: "Evidence" })).items).toEqual([]);
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
  const otherDirectory = join(root, "other");
  await mkdir(otherDirectory);
  const other = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: otherDirectory,
  });
  try {
    const issues = await other.memory.vault.lint();
    expect(issues.filter((issue) => issue.ruleId === "source.unresolved")).toHaveLength(3);
  } finally {
    await other.dispose();
    await rm(root, { recursive: true, force: true });
  }
});
