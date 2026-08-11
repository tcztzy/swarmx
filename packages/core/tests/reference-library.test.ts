import { describe, expect, it, vi } from "vitest";
import {
  createReferenceLibraryAgentTool,
  ReferenceLibraryRequestSchema,
  ReferenceLibraryResultSchema,
  ReferenceLibraryUnavailableError,
} from "../src/reference-library.js";

describe("ReferenceLibrary", () => {
  it("exposes only strict bounded read requests", () => {
    expect(() =>
      ReferenceLibraryRequestSchema.parse({ operation: "create", title: "Subjective claim" }),
    ).toThrow();
    expect(() =>
      ReferenceLibraryRequestSchema.parse({ operation: "search", query: "x".repeat(257) }),
    ).toThrow();
    expect(() =>
      ReferenceLibraryRequestSchema.parse({ operation: "get", path: "A", maxChars: 32_001 }),
    ).toThrow();
    expect(() =>
      ReferenceLibraryRequestSchema.parse({
        operation: "search",
        source: "web",
        query: "current research",
      }),
    ).toThrow();
    expect(
      ReferenceLibraryResultSchema.parse({
        operation: "status",
        sources: [
          {
            id: "zotero",
            kind: "zotero",
            name: "Zotero",
            endpoint: "http://127.0.0.1:23119/api/",
          },
        ],
      }),
    ).toMatchObject({ sources: [{ id: "zotero" }] });
  });

  it("returns the source-qualified backend result", async () => {
    const request = vi.fn(async () => ({
      operation: "search" as const,
      source: "zotero" as const,
      query: "SwarmX",
      mode: "zotero" as const,
      estimatedMatches: 1,
      matches: [
        {
          source: "zotero" as const,
          path: "ABCD2345",
          title: "SwarmX",
          url: "https://example.com/swarmx",
          snippet: "Bibliographic result",
        },
      ],
    }));
    const tool = createReferenceLibraryAgentTool({ request });
    const result = await tool.call({
      operation: "search",
      source: "zotero",
      query: "SwarmX",
      limit: 1,
    });
    expect(request).toHaveBeenCalledWith({
      operation: "search",
      source: "zotero",
      query: "SwarmX",
      limit: 1,
    });
    expect(result).toMatchObject({
      structuredContent: {
        status: "ok",
        operation: "search",
        source: "zotero",
        matches: [{ source: "zotero", title: "SwarmX" }],
      },
    });
  });

  it("reports an unavailable module without claiming a reference was used", async () => {
    const tool = createReferenceLibraryAgentTool({
      request: async () => {
        throw new ReferenceLibraryUnavailableError();
      },
    });
    await expect(tool.call({ operation: "status" })).resolves.toMatchObject({
      structuredContent: { status: "unsupported", operation: "status" },
    });
  });

  it("rejects a backend response from a different selected source", async () => {
    const tool = createReferenceLibraryAgentTool({
      request: async () => ({
        operation: "search",
        source: "zotero",
        query: "paper",
        mode: "zotero",
        estimatedMatches: 0,
        matches: [],
      }),
    });
    await expect(tool.call({ operation: "search", source: "zim", query: "paper" })).rejects.toThrow(
      /source mismatch/,
    );
  });
});
