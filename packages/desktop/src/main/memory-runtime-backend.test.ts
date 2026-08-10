import type {
  MemoryRuntimeRequest,
  MemoryRuntimeResult,
} from "@swarmx/core/memory-runtime-protocol";
import { describe, expect, it, vi } from "vitest";
import { MemoryRuntimeBackend } from "./memory-runtime-backend.js";

const page = {
  id: "mem_alpha",
  title: "Alpha",
  aliases: ["A"],
  content: "Links to [[Beta]].",
  revision: 2,
  createdAt: "2026-08-10T00:00:00.000Z",
  updatedAt: "2026-08-10T01:00:00.000Z",
};
const beta = {
  ...page,
  id: "mem_beta",
  title: "Beta",
  aliases: [],
  content: "",
};
const version = "a".repeat(40);

describe("MemoryRuntimeBackend", () => {
  it("maps CRUD, graph, and version operations onto the private runtime protocol", async () => {
    const request = vi.fn(async (input: MemoryRuntimeRequest): Promise<unknown> => {
      switch (input.operation) {
        case "list":
          return { pages: [{ ...page, content: undefined }] };
        case "get":
          return { page };
        case "search":
          return { pages: [page] };
        case "snapshot":
          return { generation: 4, pages: [page, beta] };
        case "history":
          return {
            versions: [
              {
                version,
                revision: 2,
                operation: "update",
                committedAt: "2026-08-10T01:00:00.000Z",
              },
            ],
          };
        case "get_version":
          return {
            version: {
              version,
              revision: 2,
              operation: "update",
              committedAt: "2026-08-10T01:00:00.000Z",
              page,
              deleted: false,
            },
          };
        case "diff":
          return {
            diff: {
              id: page.id,
              fromVersion: "b".repeat(40),
              toVersion: version,
              unifiedDiff: "@@ -1 +1 @@",
              truncated: false,
            },
          };
        case "create":
        case "update":
        case "delete":
        case "restore":
          return { page, version };
      }
    });
    const backend = new MemoryRuntimeBackend({ request: typedRequest(request) });

    await expect(backend.get(page.id)).resolves.toEqual(page);
    await expect(backend.search({ query: "alpha" })).resolves.toEqual([page]);
    await expect(backend.create({ title: "Alpha", content: page.content })).resolves.toEqual(page);
    await expect(
      backend.update({ id: page.id, expectedRevision: 1, content: page.content }),
    ).resolves.toEqual(page);
    await expect(backend.delete({ id: page.id, expectedRevision: 2 })).resolves.toEqual(page);
    await expect(backend.history({ id: page.id })).resolves.toHaveLength(1);
    await expect(backend.getVersion({ id: page.id, version })).resolves.toMatchObject({ page });
    await expect(
      backend.diff({ id: page.id, fromVersion: "b".repeat(40), toVersion: version }),
    ).resolves.toMatchObject({ id: page.id });
    await expect(backend.restore({ id: page.id, expectedRevision: 3, version })).resolves.toEqual(
      page,
    );
    await expect(backend.graph()).resolves.toMatchObject({
      generation: 4,
      edges: [{ source: "mem_alpha", target: "mem_beta", kind: "memory_link" }],
    });

    expect(request).toHaveBeenCalledWith({
      protocolVersion: 1,
      operation: "history",
      id: page.id,
      limit: 20,
    });
  });
});

function typedRequest(
  request: (input: MemoryRuntimeRequest) => Promise<unknown>,
): <Request extends MemoryRuntimeRequest>(input: Request) => Promise<MemoryRuntimeResult<Request>> {
  return request as <Request extends MemoryRuntimeRequest>(
    input: Request,
  ) => Promise<MemoryRuntimeResult<Request>>;
}
