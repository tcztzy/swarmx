import { describe, expect, it, vi } from "vitest";
import {
  buildMemoryGraph,
  createMemoryAgentTool,
  MAX_MEMORY_PAGE_CHARS,
  type MemoryBackend,
  MemoryCreateInputSchema,
  MemoryDocumentSchema,
  type MemoryPage,
} from "../src/memory.js";
import { MAX_MEMORY_LINK_MARKERS } from "../src/memory-links.js";

const CREATED_AT = "2026-08-09T00:00:00.000Z";

function page(input: Partial<MemoryPage> & Pick<MemoryPage, "id" | "title">): MemoryPage {
  return {
    aliases: [],
    content: "",
    revision: 1,
    createdAt: CREATED_AT,
    updatedAt: CREATED_AT,
    ...input,
  };
}

function memoryFixture() {
  const pages = new Map<string, MemoryPage>();
  let nextId = 0;
  let generation = 0;
  const backend: MemoryBackend = {
    create: vi.fn(async (input) => {
      const created = page({ id: `mem_${++nextId}`, ...input });
      pages.set(created.id, created);
      generation += 1;
      return created;
    }),
    get: vi.fn(async (id) => pages.get(id) ?? null),
    list: vi.fn(async () =>
      [...pages.values()].map(({ content: _content, ...summary }) => summary),
    ),
    search: vi.fn(async ({ query, limit = 20 }) =>
      [...pages.values()]
        .filter((candidate) =>
          [candidate.title, ...candidate.aliases, candidate.content].some((value) =>
            value.toLocaleLowerCase("en-US").includes(query.toLocaleLowerCase("en-US")),
          ),
        )
        .slice(0, limit),
    ),
    update: vi.fn(async (input) => {
      const current = pages.get(input.id);
      if (!current) throw new Error("missing");
      const updated = page({
        ...current,
        ...input,
        revision: current.revision + 1,
        updatedAt: "2026-08-09T00:01:00.000Z",
      });
      pages.set(updated.id, updated);
      generation += 1;
      return updated;
    }),
    delete: vi.fn(async ({ id }) => {
      const current = pages.get(id);
      if (!current) throw new Error("missing");
      pages.delete(id);
      generation += 1;
      return current;
    }),
    graph: vi.fn(async () => buildMemoryGraph(generation, [...pages.values()])),
    history: vi.fn(async () => []),
    getVersion: vi.fn(async () => {
      throw new Error("missing");
    }),
    diff: vi.fn(async () => {
      throw new Error("missing");
    }),
    restore: vi.fn(async () => {
      throw new Error("missing");
    }),
  };
  return { backend, pages };
}

describe("Memory contracts", () => {
  it("rejects conflicting, illegal, and unbounded linked Markdown pages", () => {
    expect(
      MemoryCreateInputSchema.safeParse({
        title: "Oversized",
        content: "x".repeat(MAX_MEMORY_PAGE_CHARS + 1),
      }).success,
    ).toBe(false);
    expect(
      MemoryCreateInputSchema.safeParse({ title: "Illegal|Title", content: "x" }).success,
    ).toBe(false);
    expect(
      MemoryCreateInputSchema.safeParse({ title: "Strict", content: "x", extra: true }).success,
    ).toBe(false);
    expect(
      MemoryDocumentSchema.safeParse({
        schemaVersion: 1,
        generation: 2,
        pages: [
          page({ id: "mem_1", title: "Mercury", content: "planet" }),
          page({ id: "mem_2", title: "Element", aliases: ["Mercury"], content: "metal" }),
        ],
      }).success,
    ).toBe(false);
    expect(
      MemoryDocumentSchema.safeParse({
        schemaVersion: 1,
        generation: 1,
        pages: [
          page({
            id: "mem_1",
            title: "Too many links",
            content: "[[Target]]".repeat(MAX_MEMORY_LINK_MARKERS + 1),
          }),
        ],
      }).success,
    ).toBe(false);
  });

  it("derives memory_link edges without turning the organization into product identity", () => {
    const graph = buildMemoryGraph(2, [
      page({ id: "mem_target", title: "Hermes Agent", aliases: ["Hermes"] }),
      page({
        id: "mem_source",
        title: "SwarmX",
        content: "Related to [[Hermes Agent|Hermes]].",
      }),
    ]);

    expect(graph).toMatchObject({
      generation: 2,
      edges: [
        {
          kind: "memory_link",
          source: "mem_source",
          target: "mem_target",
          occurrences: [{ targetText: "Hermes Agent", alias: "Hermes" }],
        },
      ],
      diagnostics: [],
    });
  });
});

describe("Memory Agent tool", () => {
  it("uses the generic Memory name and applies only confirmed CRUD mutations", async () => {
    const { backend, pages } = memoryFixture();
    let approved = false;
    const confirmations: unknown[] = [];
    const audit: unknown[] = [];
    const tool = createMemoryAgentTool(backend, {
      confirm: async (mutation) => {
        confirmations.push(mutation);
        return approved;
      },
      audit: (event) => audit.push(event),
    });
    if (tool.kind === "text") throw new Error("Memory must be a function tool");
    expect(tool.name).toBe("Memory");

    await expect(
      tool.call({ operation: "create", title: "SwarmX", content: "Uses [[Memory]]." }),
    ).resolves.toMatchObject({ structuredContent: { status: "denied", operation: "create" } });
    expect(pages.size).toBe(0);

    approved = true;
    await expect(
      tool.call({
        operation: "create",
        title: "SwarmX",
        aliases: ["Swarm X"],
        content: "Uses [[Memory]].",
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        status: "applied",
        operation: "create",
        page: { id: "mem_1", revision: 1 },
      },
    });
    await expect(tool.call({ operation: "search", query: "Memory" })).resolves.toMatchObject({
      structuredContent: {
        status: "ok",
        operation: "search",
        pages: [{ id: "mem_1", content: "Uses [[Memory]]." }],
      },
    });
    await expect(
      tool.call({ operation: "update", id: "mem_1", expectedRevision: 1, content: "Durable" }),
    ).resolves.toMatchObject({
      structuredContent: { status: "applied", page: { id: "mem_1", revision: 2 } },
    });
    await expect(
      tool.call({ operation: "delete", id: "mem_1", expectedRevision: 2 }),
    ).resolves.toMatchObject({
      structuredContent: { status: "applied", operation: "delete" },
    });

    expect(confirmations).toHaveLength(4);
    expect(JSON.stringify(audit)).not.toContain("Uses [[Memory]].");
    expect(audit).toContainEqual({
      operation: "create",
      outcome: "completed",
      pageId: "mem_1",
      characterCount: 16,
    });
  });

  it("fails closed when mutation confirmation is unavailable", async () => {
    const { backend, pages } = memoryFixture();
    const audit: unknown[] = [];
    const tool = createMemoryAgentTool(backend, {
      confirm: async () => {
        throw new Error("owner unavailable");
      },
      audit: (event) => audit.push(event),
    });
    if (tool.kind === "text") throw new Error("Memory must be a function tool");

    await expect(
      tool.call({ operation: "create", title: "Private", content: "secret body" }),
    ).rejects.toThrow("owner unavailable");
    expect(pages.size).toBe(0);
    expect(audit).toEqual([{ operation: "create", outcome: "failed", characterCount: 11 }]);
  });

  it("exposes bounded version reads and confirms restore without auditing Markdown", async () => {
    const oldVersion = "a".repeat(40);
    const currentVersion = "b".repeat(40);
    const versioned = page({
      id: "mem_versioned",
      title: "Versioned memory",
      content: "private historical Markdown",
      revision: 3,
    });
    const backend = {
      ...memoryFixture().backend,
      history: vi.fn(async () => [
        {
          version: currentVersion,
          revision: 2,
          operation: "update" as const,
          committedAt: "2026-08-09T00:01:00.000Z",
        },
      ]),
      getVersion: vi.fn(async () => ({
        version: oldVersion,
        revision: 1,
        operation: "create" as const,
        committedAt: CREATED_AT,
        page: { ...versioned, revision: 1 },
        deleted: false,
      })),
      diff: vi.fn(async () => ({
        id: versioned.id,
        fromVersion: oldVersion,
        toVersion: currentVersion,
        unifiedDiff: "-old\n+new",
        truncated: false,
      })),
      restore: vi.fn(async () => versioned),
    } satisfies MemoryBackend;
    const audit: unknown[] = [];
    const tool = createMemoryAgentTool(backend, {
      confirm: async () => true,
      audit: (event) => audit.push(event),
    });
    if (tool.kind === "text") throw new Error("Memory must be a function tool");

    await expect(
      tool.call({ operation: "get_version", id: versioned.id, version: oldVersion }),
    ).resolves.toMatchObject({
      structuredContent: { version: { page: { content: "private historical Markdown" } } },
    });
    await expect(
      tool.call({
        operation: "restore",
        id: versioned.id,
        expectedRevision: 2,
        version: oldVersion,
      }),
    ).resolves.toMatchObject({ structuredContent: { status: "applied", operation: "restore" } });
    expect(JSON.stringify(audit)).not.toContain("private historical Markdown");
  });
});
