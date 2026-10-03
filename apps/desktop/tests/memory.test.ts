import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { importModelObservation, queryModelExperience } from "@swarmx/memory";
import { expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { currentAgentBinding } from "../src/host/agent-registry.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { policyPermissions } from "../src/permissions.js";
import { bridgeClient } from "./mcp-bridge-support.js";

const unverifiedSource = {
  ruleId: "source.unverified",
  severity: "warning",
  message:
    "External reference is recorded, not verified by SwarmX. Use the domain Agent or tool to assess it.",
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

it.each([
  "sx:p/evidence@1",
  "sx:invalid-domain-claim",
  "bio:dataset/evidence@v3",
  "http://example.org/evidence",
  "https://example.org/evidence",
  "other:reference",
])("records external Memory source %s as unverified without domain lookup", async (resource) => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-external-source-"));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
  });
  try {
    await expect(createSourceMemory(products, resource)).resolves.toMatchObject({
      data: { metadata: { sources: [{ id: "evidence", resource }] } },
      diagnostics: expect.arrayContaining([expect.objectContaining(unverifiedSource)]),
    });
    const issues = await products.memory.vault.lint();
    expect(issues.filter((issue) => issue.ruleId === "source.unverified")).toEqual([
      expect.objectContaining(unverifiedSource),
    ]);
    expect(issues.some((issue) => issue.ruleId === "source.unresolved")).toBe(false);
  } finally {
    await products.dispose();
  }
  const otherDirectory = join(root, "other");
  await mkdir(otherDirectory);
  const reopened = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: otherDirectory,
  });
  try {
    const issues = await reopened.memory.vault.lint();
    expect(issues.filter((issue) => issue.ruleId === "source.unverified")).toEqual([
      expect.objectContaining(unverifiedSource),
    ]);
  } finally {
    await reopened.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it.each([
  ["science.read", "policy"],
  ["science.read", "execution"],
  ["science.write", "policy"],
  ["science.write", "execution"],
] as const)(
  "retains an unverified external source without %s permission in the %s",
  async (deniedPermission, authority) => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-memory-external-source-auth-"));
    const products = await ProductServices.create({
      productHome: join(root, "product"),
      cwd: root,
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
        diagnostics: expect.arrayContaining([expect.objectContaining(unverifiedSource)]),
      });
    } finally {
      await products.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
);

it("preserves ordinary local Memory link validation alongside external-source warnings", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-local-source-"));
  const products = await ProductServices.create({ productHome: join(root, "product"), cwd: root });
  const context = { actorId: "actor", callId: randomUUID(), signal: new AbortController().signal };
  try {
    await products.callTool(
      "memory",
      {
        action: "create_memory",
        request: {
          title: "Local decision",
          type: "Decision",
          description: "An existing local concept.",
          body: "# Local decision\n\nUse reproducible tests.",
        },
      },
      context,
    );
    await expect(createSourceMemory(products, "local-decision.md")).resolves.toMatchObject({
      diagnostics: [],
    });
    await expect(createSourceMemory(products, "missing-concept.md")).resolves.toMatchObject({
      diagnostics: expect.arrayContaining([
        expect.objectContaining({ ruleId: "link.broken", severity: "warning" }),
      ]),
    });
    for (const resource of ["../outside.md", ".hidden.md"])
      await expect(createSourceMemory(products, resource)).rejects.toMatchObject({
        code: "INVALID_CONCEPT",
      });
    const issues = await products.memory.vault.lint();
    expect(issues.some((issue) => issue.ruleId === "source.unverified")).toBe(false);
    expect(issues.filter((issue) => issue.ruleId === "link.broken")).toHaveLength(1);
  } finally {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  }
});

it("keeps execution-source validation in the directory journal without science.read", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-memory-execution-source-"));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
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
    expect(issues.filter((issue) => issue.ruleId === "source.unresolved")).toHaveLength(1);
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

it("keeps external observation provenance after accepted Host evaluation updates", async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), "swarmx-shared-experience-")));
  const products = await ProductServices.create({ productHome: join(root, "product"), cwd: root });
  try {
    const artifact = join(root, "observation.json");
    await writeFile(
      artifact,
      JSON.stringify({
        schemaVersion: 1,
        kind: "observation",
        observedAt: "2026-01-02T03:04:05Z",
        observer: "synthetic-external-observer",
        task: "Synthetic parser trial",
        criteria: "Two keys returned",
        outcome: "success",
        limitations: "Synthetic external assertion; not provider verified",
        confidence: "Only this synthetic fixture",
        requested: {
          model: "synthetic-model",
          effort: null,
          provider: null,
          harness: null,
          runtimeVersion: null,
        },
        actual: null,
        retries: null,
        elapsed: null,
        tokens: null,
        cost: null,
      }),
    );
    let saved = await importModelObservation(products.memory.vault, {
      artifact,
      title: "Shared synthetic experience",
      requestId: randomUUID(),
    });
    const context = {
      actorId: "actor",
      callId: randomUUID(),
      signal: new AbortController().signal,
    };
    const evaluation = {
      kind: "judgment" as const,
      task: "Host synthetic trial",
      criteria: "Recorded validator accepts the output",
      evidence: [`urn:swarmx:execution:${randomUUID()}`],
      counterEvidence: [],
      limitations: "One synthetic Host event; no model quality claim",
    };
    await expect(
      products.callTool(
        "memory",
        {
          action: "update_memory",
          request: {
            id: saved.id,
            expectedRevision: saved.revision,
            body: "Host judgment",
            evaluation,
          },
        },
        context,
      ),
    ).rejects.toThrow("Execution source is missing or belongs to another directory.");
    const record = products.journal.append(null, {
      type: EventType.CUSTOM,
      name: "synthetic.validator.result",
      value: { accepted: true, fixtureOnly: true },
    });
    for (const kind of ["judgment", "preference"] as const) {
      await products.callTool(
        "memory",
        {
          action: "update_memory",
          request: {
            id: saved.id,
            expectedRevision: saved.revision,
            body: `Host ${kind} about the recorded fixture`,
            evaluation: { ...evaluation, kind, evidence: [`urn:swarmx:execution:${record.id}`] },
          },
        },
        context,
      );
      saved = await products.memory.vault.readConcept(saved.id);
      const result = await queryModelExperience(products.memory.vault);
      expect(result.concepts[0]).toMatchObject({
        revision: saved.revision,
        kind,
        provenance: "mixed-unchecked",
        confidence: null,
        observation: {
          kind: "observation",
          provenance: "external-self-asserted",
          observer: "synthetic-external-observer",
          confidence: "Only this synthetic fixture",
          actual: null,
          cost: null,
        },
        evaluation: {
          kind,
          provenance: "host-execution-references-unchecked",
          evidence: [`urn:swarmx:execution:${record.id}`],
        },
      });
      expect(saved.metadata.sources).toHaveLength(2);
    }
  } finally {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  }
});
