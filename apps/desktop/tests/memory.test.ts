import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { formatScienceResourceId } from "@swarmx/science";
import { expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { currentAgentBinding } from "../src/host/agent-registry.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { bridgeClient } from "./mcp-bridge-support.js";

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

it.each(["memory_status", "memory_configure", "memory_review", "memory_decide"] as const)(
  "rejects Agent %s through an active MCP binding while preserving desktop management",
  async (action) => {
    const root = await mkdtemp("/tmp/sx-memory-auth-");
    const products = await ProductServices.create({ productHome: join(root, "home"), cwd: root });
    products.settings.writeMemory({ writeApproval: true, autoReview: false });
    const settings = products.settings.readMemory();
    const review = vi.spyOn(products.learning, "review").mockImplementation(() => {});
    const sessionId = "codex:memory-management";
    let request: Record<string, unknown> = {};
    const native: NativeAgent = {
      name: "memory authorization fixture",
      capabilities: HARNESS_CAPABILITIES.codex,
      models: async () => ({ models: [], current: {} }),
      list: async () => [],
      create: async () => sessionId,
      read: async () => {},
      start: async (id) => {
        const execution = products.journal.activeSession(id);
        expect(execution?.permissions?.tools).toEqual(["memory.read", "memory.write"]);
        if (!execution) throw new Error("Missing active execution.");
        const token = randomUUID();
        const lease = currentAgentBinding().registerMcp?.(token);
        if (!lease) throw new Error("Missing MCP registration.");
        lease.bind(id, execution.runId);
        const client = await bridgeClient(products.mcpSocket, token);
        try {
          const note = await products.learning.core.read();
          const saved = await client.callTool({
            name: "memory",
            arguments: {
              action: "update_core_memory",
              request: {
                content: "Prefer reproducible environments",
                expectedRevision: note.revision,
              },
            },
          });
          expect(saved.isError).not.toBe(true);
          expect(saved.structuredContent).toMatchObject({ staged: true });
          const pending = products.journal.pendingMemories()[0];
          if (!pending) throw new Error("Missing staged memory change.");
          request = {
            memory_status: {},
            memory_configure: { writeApproval: false, autoReview: false },
            memory_review: { sessionId: id },
            memory_decide: { id: pending.id, decision: "approve" },
          }[action];
          const result = await client.callTool({ name: "memory", arguments: { action, request } });
          expect(result.isError).toBe(true);
          expect(JSON.stringify(result)).toContain(
            "Memory management requires a trusted Host caller",
          );
          await expect(
            products.callTool(
              "memory",
              { action, request },
              {
                actorId: "renderer",
                callId: randomUUID(),
                signal: new AbortController().signal,
              },
            ),
          ).rejects.toThrow("Memory management requires a trusted Host caller");
          const read = await client.callTool({
            name: "memory",
            arguments: { action: "read_core_memory", request: {} },
          });
          expect(read.isError).not.toBe(true);
          expect(read.structuredContent).toMatchObject({ data: { content: "" } });
        } finally {
          await client.close();
          lease.dispose();
        }
        return { stopReason: "end_turn" };
      },
      steer: async () => {},
      interrupt: async () => {},
      dispose: async () => {},
    };
    const host = await startHost({ products, agent: native, agentId: "codex" });
    try {
      await products.rootAgent.start(
        sessionId,
        "Save a preference",
        {
          text() {},
          tool() {},
          raw() {},
          interact: async () => undefined,
        },
        { permissions: { tools: ["memory.read", "memory.write"], delegation: false } },
      );
      expect(products.settings.readMemory()).toEqual(settings);
      expect((await products.learning.core.read()).content).toBe("");
      expect(products.journal.pendingMemories()).toHaveLength(1);
      expect(review).not.toHaveBeenCalled();
      const desktop = new HostOperations(host);
      await expect(
        desktop.callTool("memory", { action, request }, randomUUID(), host.signal),
      ).resolves.toMatchObject({ action });
      if (action === "memory_decide") {
        expect((await products.learning.core.read()).content).toBe(
          "Prefer reproducible environments",
        );
        expect(products.journal.pendingMemories()).toEqual([]);
      } else if (action === "memory_configure") {
        expect(products.settings.readMemory().writeApproval).toBe(false);
      } else if (action === "memory_review") {
        expect(review).toHaveBeenCalledWith(sessionId, "");
      }
    } finally {
      await host.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);
