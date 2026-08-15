import { describe, expect, it } from "vitest";
import { MAX_MEMORY_SEARCH_RESULTS } from "../src/memory.js";
import {
  MEMORY_RUNTIME_PROTOCOL_VERSION,
  MemoryRuntimeRequestSchema,
  MemoryRuntimeToolResponseSchema,
} from "../src/memory-runtime-protocol.js";

const PAGE_TIMESTAMP = "2026-08-10T00:00:00.000Z";

function page(index: number, content = "body") {
  return {
    id: `mem_page_${index}`,
    title: `Page ${index}`,
    aliases: [],
    content,
    revision: 1,
    createdAt: PAGE_TIMESTAMP,
    updatedAt: PAGE_TIMESTAMP,
  };
}

describe("Memory runtime protocol", () => {
  it("accepts bounded global Markdown file operations", () => {
    expect(
      MemoryRuntimeRequestSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "global_save",
        target: "user",
        expectedRevision: 0,
        content: "Prefers concise answers.",
      }),
    ).toMatchObject({ operation: "global_save", target: "user" });
    expect(
      MemoryRuntimeToolResponseSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "global_get",
        ok: true,
        result: {
          user: {
            target: "user",
            fileName: "USER.md",
            content: "Prefers concise answers.",
            revision: 1,
            updatedAt: "2026-08-12T08:00:00.000Z",
          },
          memory: {
            target: "memory",
            fileName: "MEMORY.md",
            content: null,
            revision: 0,
            updatedAt: null,
          },
        },
      }),
    ).toMatchObject({ operation: "global_get", ok: true });
  });

  it("accepts bounded CRUD and version requests", () => {
    expect(
      MemoryRuntimeRequestSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "update",
        id: "mem_page",
        expectedRevision: 2,
        content: "Links to [[Hermes Agent]].",
      }),
    ).toMatchObject({ operation: "update", expectedRevision: 2 });

    expect(
      MemoryRuntimeRequestSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "restore",
        id: "mem_page",
        expectedRevision: 3,
        version: "a".repeat(40),
      }),
    ).toMatchObject({ operation: "restore", version: "a".repeat(40) });

    expect(
      MemoryRuntimeRequestSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "create",
        title: "Mercury",
        kind: "technology",
        summary: "A protocol.",
        sources: ["https://example.test/protocol"],
        scope: "Project Orion",
        content: "Protocol note.",
      }),
    ).toMatchObject({ operation: "create", kind: "technology" });
  });

  it("rejects unknown operations, extra fields, oversized input, and invalid commits", () => {
    expect(
      MemoryRuntimeRequestSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "execute",
      }).success,
    ).toBe(false);
    expect(
      MemoryRuntimeRequestSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "list",
        path: "/private/memory",
      }).success,
    ).toBe(false);
    expect(
      MemoryRuntimeRequestSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "create",
        title: "Oversized",
        content: "x".repeat(64_001),
      }).success,
    ).toBe(false);
    expect(
      MemoryRuntimeRequestSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "get_version",
        id: "mem_page",
        version: "HEAD",
      }).success,
    ).toBe(false);
  });

  it("requires operation-matched strict success or bounded content-free errors", () => {
    expect(
      MemoryRuntimeToolResponseSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "list",
        ok: true,
        result: { pages: [] },
      }),
    ).toMatchObject({ ok: true, operation: "list" });

    expect(
      MemoryRuntimeToolResponseSchema.parse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "search",
        ok: true,
        result: {
          pages: [page(1)],
          results: [
            {
              title: "Page 1",
              summary: "Human summary.",
              kind: "note",
              sources: [],
              relatedPages: ["Page 2"],
              id: "mem_page_1",
            },
          ],
        },
      }),
    ).toMatchObject({ result: { results: [{ title: "Page 1" }] } });
    expect(
      MemoryRuntimeToolResponseSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "list",
        ok: false,
        error: { code: "internal", message: "failed", markdown: "private body" },
      }).success,
    ).toBe(false);
    expect(
      MemoryRuntimeToolResponseSchema.safeParse({
        protocolVersion: 2,
        operation: "list",
        ok: true,
        result: { pages: [] },
      }).success,
    ).toBe(false);
    expect(
      MemoryRuntimeToolResponseSchema.safeParse({
        protocolVersion: MEMORY_RUNTIME_PROTOCOL_VERSION,
        operation: "search",
        ok: true,
        result: {
          pages: Array.from({ length: MAX_MEMORY_SEARCH_RESULTS + 1 }, (_, index) => page(index)),
        },
      }).success,
    ).toBe(false);
  });
});
