import { describe, expect, it, vi } from "vitest";
import {
  createReferenceLibraryAgentTool,
  ReferenceLibraryRequestSchema,
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
  });

  it("returns the source-qualified backend result", async () => {
    const request = vi.fn(async () => ({
      operation: "search" as const,
      query: "SwarmX",
      mode: "full_text" as const,
      estimatedMatches: 1,
      matches: [{ path: "A/SwarmX", title: "SwarmX" }],
    }));
    const tool = createReferenceLibraryAgentTool({ request });
    const result = await tool.call({ operation: "search", query: "SwarmX", limit: 1 });
    expect(request).toHaveBeenCalledWith({ operation: "search", query: "SwarmX", limit: 1 });
    expect(result).toMatchObject({
      structuredContent: {
        status: "ok",
        operation: "search",
        matches: [{ path: "A/SwarmX", title: "SwarmX" }],
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
});
