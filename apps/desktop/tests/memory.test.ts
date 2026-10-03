import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { currentAgentBinding } from "../src/host/agent-registry.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { policyPermissions } from "../src/permissions.js";
import type { ReferenceProvider } from "../src/reference-provider.js";
import { bridgeClient } from "./mcp-bridge-support.js";

const unresolvedSource = {
  ruleId: "source.unresolved",
  severity: "warning",
  message: "Reference is unavailable in this directory or its revision has changed.",
} as const;

function createSourceMemory(products: ProductServices, resource: string) {
  return products.callTool(
    "memory",
    {
      action: "create_memory",
      request: {
        title: `Evidence for ${resource}`,
        type: "Finding",
        description: "Evidence reference",
        body: "# Evidence\n\nResult.[^evidence]\n\n[^evidence]: Domain resource.",
        sources: [{ id: "evidence", resource }],
      },
    },
    { actorId: "actor", callId: randomUUID(), signal: new AbortController().signal },
  );
}

function trustedProvider(scheme = "sx:") {
  return {
    scheme,
    requiredPermissions: ["science.read", "science.write"] as const,
    resolve: vi.fn<ReferenceProvider["resolve"]>((id) => ({ id, exactId: id, revision: "1" })),
    checkResource: vi.fn<ReferenceProvider["checkResource"]>(() => undefined),
  };
}

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

it.each(["sx:", "bio:"])(
  "checks Memory %s sources with the trusted directory provider",
  async (scheme) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-memory-provider-"));
    const exact = `${scheme}p/evidence@1`;
    const referenceProvider = trustedProvider(scheme);
    referenceProvider.checkResource.mockImplementation((resource) => {
      if (resource === `${scheme}invalid`)
        return { ruleId: "source.invalid", severity: "error", message: "Invalid domain resource." };
      return resource === exact ? undefined : unresolvedSource;
    });
    const products = await ProductServices.create({
      productHome: join(root, "product"),
      cwd: root,
      referenceProvider,
    });
    const context = { actorId: "actor", callId: "lint-test", signal: new AbortController().signal };
    try {
      await expect(createSourceMemory(products, `${scheme}invalid`)).rejects.toMatchObject({
        code: "INVALID_CONCEPT",
      });
      expect((await products.memory.vault.search({ query: "Evidence" })).items).toEqual([]);
      await expect(createSourceMemory(products, exact)).resolves.toMatchObject({ diagnostics: [] });
      for (const resource of [`${scheme}p/evidence@2`, `${scheme}p/missing@1`]) {
        await expect(createSourceMemory(products, resource)).resolves.toMatchObject({
          diagnostics: expect.arrayContaining([expect.objectContaining(unresolvedSource)]),
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
      expect(referenceProvider.checkResource).toHaveBeenCalledWith(exact);
      expect(referenceProvider.resolve).not.toHaveBeenCalled();
    } finally {
      await products.dispose();
    }
    const otherDirectory = join(root, "other");
    await mkdir(otherDirectory);
    const otherProvider = trustedProvider(scheme);
    otherProvider.checkResource.mockReturnValue(unresolvedSource);
    const other = await ProductServices.create({
      productHome: join(root, "product"),
      cwd: otherDirectory,
      referenceProvider: otherProvider,
    });
    try {
      const issues = await other.memory.vault.lint();
      expect(issues.filter((issue) => issue.ruleId === "source.unresolved")).toHaveLength(3);
      expect(otherProvider.checkResource).toHaveBeenCalledWith(exact);
    } finally {
      await other.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);

it.each([
  ["science.read", "policy"],
  ["science.read", "execution"],
  ["science.write", "policy"],
  ["science.write", "execution"],
] as const)(
  "does not invoke the Memory provider without %s permission in the %s",
  async (deniedPermission, authority) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-memory-provider-auth-"));
    const referenceProvider = {
      ...trustedProvider(),
      requiredPermissions:
        deniedPermission === "science.read" ? [] : (["science.read", "science.write"] as const),
    };
    const products = await ProductServices.create({
      productHome: join(root, "product"),
      cwd: root,
      referenceProvider,
    });
    try {
      const { policy } = products.settings.read();
      const restricted = {
        ...policy,
        tools: policy.tools.filter((permission) => permission !== deniedPermission),
      };
      if (authority === "policy") products.updatePolicy(restricted);
      const create = () => createSourceMemory(products, "sx:p/evidence@1");
      const result =
        authority === "policy"
          ? create()
          : products.journal.scope.run(
              {
                sessionId: null,
                runId: randomUUID(),
                causedBy: null,
                attributes: {},
                permissions: policyPermissions(restricted),
              },
              create,
            );
      await expect(result).resolves.toMatchObject({
        diagnostics: expect.arrayContaining([
          expect.objectContaining({ ruleId: "source.unresolved", severity: "warning" }),
        ]),
      });
      expect(referenceProvider.resolve).not.toHaveBeenCalled();
      expect(referenceProvider.checkResource).not.toHaveBeenCalled();
    } finally {
      await products.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);

it.each(["missing", "different-scheme"])(
  "preserves unresolved Memory diagnostics with a %s provider",
  async (configuration) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-memory-provider-missing-"));
    const referenceProvider = trustedProvider("bio:");
    const products = await ProductServices.create({
      productHome: join(root, "product"),
      cwd: root,
      ...(configuration === "missing" ? {} : { referenceProvider }),
    });
    try {
      for (const resource of ["https://example.org/evidence", "other:reference"])
        await expect(createSourceMemory(products, resource)).resolves.toMatchObject({
          diagnostics: [],
        });
      await expect(createSourceMemory(products, "sx:p/evidence@1")).resolves.toMatchObject({
        diagnostics: expect.arrayContaining([
          expect.objectContaining({ ruleId: "source.unresolved", severity: "warning" }),
        ]),
      });
      expect(referenceProvider.resolve).not.toHaveBeenCalled();
      expect(referenceProvider.checkResource).not.toHaveBeenCalled();
    } finally {
      await products.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);

it("propagates unexpected Memory provider failures without saving an unchecked concept", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-provider-failure-"));
  const referenceProvider = trustedProvider();
  const failure = new Error("Trusted reference store is unavailable.");
  referenceProvider.checkResource.mockImplementation(() => {
    throw failure;
  });
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
    referenceProvider,
  });
  try {
    await expect(createSourceMemory(products, "sx:p/evidence@1")).rejects.toBe(failure);
    expect((await products.memory.vault.search({ query: "Evidence" })).items).toEqual([]);
    expect(referenceProvider.resolve).not.toHaveBeenCalled();
  } finally {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it("keeps execution-source validation in the directory journal without calling a domain provider", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-execution-source-"));
  const referenceProvider = trustedProvider("urn:swarmx:execution:");
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
    referenceProvider,
  });
  try {
    const { policy } = products.settings.read();
    products.updatePolicy({
      ...policy,
      tools: policy.tools.filter((permission) => permission !== "science.read"),
    });
    const record = products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "evidence",
      value: { result: "Recorded evidence" },
    });
    await expect(
      createSourceMemory(products, `urn:swarmx:execution:${record.id}`),
    ).resolves.toMatchObject({ diagnostics: [] });
    await expect(
      createSourceMemory(products, `urn:swarmx:execution:${randomUUID()}`),
    ).rejects.toThrow("Execution source is missing or belongs to another directory.");
    await expect(createSourceMemory(products, "urn:swarmx:execution:invalid")).rejects.toThrow(
      "Invalid execution source",
    );
    expect(referenceProvider.resolve).not.toHaveBeenCalled();
    expect(referenceProvider.checkResource).not.toHaveBeenCalled();
  } finally {
    await products.dispose();
  }
  const otherDirectory = join(root, "other");
  await mkdir(otherDirectory);
  const other = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: otherDirectory,
    referenceProvider,
  });
  try {
    const issues = await other.memory.vault.lint();
    expect(issues.filter((issue) => issue.ruleId === "source.unresolved")).toHaveLength(1);
    expect(referenceProvider.resolve).not.toHaveBeenCalled();
    expect(referenceProvider.checkResource).not.toHaveBeenCalled();
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
