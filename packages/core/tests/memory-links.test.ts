import { describe, expect, it } from "vitest";
import {
  buildMemoryLinkEdges,
  MAX_MEMORY_GRAPH_ENTITIES,
  MAX_MEMORY_LINK_MARKERS,
  MAX_MEMORY_MARKDOWN_CHARS,
  MemoryGraphBuildInputSchema,
  MemoryGraphBuildResultSchema,
} from "../src/memory-links.js";

const entities = [
  { id: "entity:swarmx", title: "SwarmX" },
  { id: "entity:hermes", title: "Hermes Agent", aliases: ["Hermes"] },
  { id: "entity:memory", title: "Memory" },
];

describe("buildMemoryLinkEdges", () => {
  it("projects Obsidian-style links into deduplicated directed entity edges", () => {
    const markdown =
      "Uses [[Hermes Agent|Hermes]], [[Memory.md#Capacity]], and ![[Memory]]. Repeats [[hermes]].";

    const result = buildMemoryLinkEdges({
      sourceEntityId: "entity:swarmx",
      markdown,
      entities,
    });

    expect(result.edges).toEqual([
      {
        kind: "memory_link",
        source: "entity:swarmx",
        target: "entity:hermes",
        occurrences: [
          {
            targetText: "Hermes Agent",
            alias: "Hermes",
            embedded: false,
            start: markdown.indexOf("[[Hermes Agent|Hermes]]"),
            end: markdown.indexOf("[[Hermes Agent|Hermes]]") + "[[Hermes Agent|Hermes]]".length,
          },
          {
            targetText: "hermes",
            embedded: false,
            start: markdown.indexOf("[[hermes]]"),
            end: markdown.indexOf("[[hermes]]") + "[[hermes]]".length,
          },
        ],
      },
      {
        kind: "memory_link",
        source: "entity:swarmx",
        target: "entity:memory",
        occurrences: [
          {
            targetText: "Memory.md",
            heading: "Capacity",
            embedded: false,
            start: markdown.indexOf("[[Memory.md#Capacity]]"),
            end: markdown.indexOf("[[Memory.md#Capacity]]") + "[[Memory.md#Capacity]]".length,
          },
          {
            targetText: "Memory",
            embedded: true,
            start: markdown.indexOf("![[Memory]]"),
            end: markdown.indexOf("![[Memory]]") + "![[Memory]]".length,
          },
        ],
      },
    ]);
    expect(result.diagnostics).toEqual([]);
    expect(MemoryGraphBuildResultSchema.parse(result)).toEqual(result);
  });

  it("resolves titles and aliases with case-insensitive Unicode NFC keys", () => {
    const result = buildMemoryLinkEdges({
      sourceEntityId: "source",
      markdown: "See [[CAFE\u0301]] and [[short name]].",
      entities: [
        { id: "source", title: "Source" },
        { id: "cafe", title: "Café" },
        { id: "long", title: "A Long Entity Name", aliases: ["Short Name"] },
      ],
    });

    expect(result.edges.map(({ source, target }) => ({ source, target }))).toEqual([
      { source: "source", target: "cafe" },
      { source: "source", target: "long" },
    ]);
  });

  it("returns explicit diagnostics instead of inventing unsafe edges", () => {
    const markdown = "[[Unknown]] [[Mercury]] [[Source]] [[|blank]] [[Unclosed";
    const result = buildMemoryLinkEdges({
      sourceEntityId: "source",
      markdown,
      entities: [
        { id: "source", title: "Source" },
        { id: "planet", title: "Mercury (planet)", aliases: ["Mercury"] },
        { id: "element", title: "Mercury (element)", aliases: ["Mercury"] },
      ],
    });

    expect(result.edges).toEqual([]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.kind)).toEqual([
      "unresolved",
      "ambiguous",
      "self_reference",
      "malformed",
      "malformed",
    ]);
    expect(result.diagnostics[1]).toMatchObject({
      targetText: "Mercury",
      candidateEntityIds: ["element", "planet"],
    });
  });

  it("ignores double-bracket links inside inline and fenced code", () => {
    const markdown = ["`[[Inline]]`", "```ts", "[[Fenced]]", "```", "Real [[Target]]."].join("\n");
    const result = buildMemoryLinkEdges({
      sourceEntityId: "source",
      markdown,
      entities: [
        { id: "source", title: "Source" },
        { id: "inline", title: "Inline" },
        { id: "fenced", title: "Fenced" },
        { id: "target", title: "Target" },
      ],
    });

    expect(result.edges).toHaveLength(1);
    expect(result.edges[0]?.target).toBe("target");
  });

  it("enforces strict, bounded public input", () => {
    expect(
      MemoryGraphBuildInputSchema.safeParse({
        sourceEntityId: "source",
        markdown: "[[Target]]",
        entities: [
          { id: "source", title: "Source" },
          { id: "source", title: "Duplicate" },
        ],
      }).success,
    ).toBe(false);
    expect(
      MemoryGraphBuildInputSchema.safeParse({
        sourceEntityId: "missing",
        markdown: "",
        entities: [{ id: "source", title: "Source" }],
      }).success,
    ).toBe(false);
    expect(
      MemoryGraphBuildInputSchema.safeParse({
        sourceEntityId: "source",
        markdown: "x".repeat(MAX_MEMORY_MARKDOWN_CHARS + 1),
        entities: [{ id: "source", title: "Source" }],
      }).success,
    ).toBe(false);
    expect(
      MemoryGraphBuildInputSchema.safeParse({
        sourceEntityId: "source",
        markdown: "[[Target]]".repeat(MAX_MEMORY_LINK_MARKERS + 1),
        entities: [{ id: "source", title: "Source" }],
      }).success,
    ).toBe(false);
    expect(
      MemoryGraphBuildInputSchema.safeParse({
        sourceEntityId: "source",
        markdown: "",
        entities: Array.from({ length: MAX_MEMORY_GRAPH_ENTITIES + 1 }, (_, index) => ({
          id: `entity:${index}`,
          title: `Entity ${index}`,
        })),
        extra: true,
      }).success,
    ).toBe(false);
  });
});
