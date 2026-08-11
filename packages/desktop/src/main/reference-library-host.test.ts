import path from "node:path";
import { describe, expect, it, vi } from "vitest";
import { type ReferenceLibraryConnection, ReferenceLibraryHost } from "./reference-library-host.js";

const launch = {
  pythonPath: "/opt/swarmx/python",
  zimPath: "/Users/example/wikipedia.zim",
};

function connection(
  overrides: Partial<ReferenceLibraryConnection> = {},
): ReferenceLibraryConnection {
  const structuredContent = {
    operation: "search",
    source: "zim",
    query: "SwarmX",
    mode: "full_text",
    estimatedMatches: 1,
    matches: [{ source: "zim", path: "A/SwarmX", title: "SwarmX" }],
  };
  return {
    serverInfo: () => ({ name: "swarmx-ref", version: "3.2.0" }),
    listTools: async () => [{ name: "swarmx_reference" }],
    callTool: async () => ({
      content: [{ type: "text", text: JSON.stringify(structuredContent) }],
      structuredContent,
    }),
    close: vi.fn(async () => undefined),
    ...overrides,
  };
}

describe("ReferenceLibraryHost", () => {
  it("validates absolute launch paths", () => {
    expect(() => new ReferenceLibraryHost({ pythonPath: "python", zimPath: "wiki.zim" })).toThrow();
    expect(path.isAbsolute(launch.zimPath)).toBe(true);
    expect(() => new ReferenceLibraryHost({ pythonPath: "/opt/swarmx/python" })).toThrow();
    expect(
      () => new ReferenceLibraryHost({ pythonPath: "/opt/swarmx/python", zotero: true }),
    ).not.toThrow();
  });

  it("verifies the exact MCP surface and result boundary", async () => {
    const remote = connection();
    const host = new ReferenceLibraryHost({ ...launch, connect: async () => remote });
    await expect(
      host.request({ operation: "search", query: "SwarmX", limit: 1 }),
    ).resolves.toMatchObject({ operation: "search", matches: [{ title: "SwarmX" }] });
    await host.close();
    expect(remote.close).toHaveBeenCalledOnce();
  });

  it("fails closed on an unexpected server or contradictory text", async () => {
    const wrong = new ReferenceLibraryHost({
      ...launch,
      connect: async () => connection({ serverInfo: () => ({ name: "other", version: "1" }) }),
    });
    await expect(wrong.request({ operation: "status" })).rejects.toThrow(/unavailable/);

    const contradictory = new ReferenceLibraryHost({
      ...launch,
      connect: async () =>
        connection({
          callTool: async () => ({
            content: [{ type: "text", text: '{"operation":"status"}' }],
            structuredContent: { operation: "search" },
          }),
        }),
    });
    await expect(contradictory.request({ operation: "status" })).rejects.toThrow(/contradictory/);

    const wrongSource = new ReferenceLibraryHost({
      ...launch,
      connect: async () => connection(),
    });
    await expect(
      wrongSource.request({ operation: "search", source: "zotero", query: "SwarmX", limit: 1 }),
    ).rejects.toThrow(/source mismatch/);
  });
});
