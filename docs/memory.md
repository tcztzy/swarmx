# Memory

SwarmX Core provides a local, bounded collection of Markdown entity pages that
an authorized host can create, read, list, search, update, and delete. Double-bracket links
between the current pages are projected into directed knowledge-graph edges. It
is the currently shipped linked-page organization of Memory, alongside the
compact Personal Memory Settings surface. Neither surface turns Memory into
Session history, a graph database, or a workflow runtime.

SwarmX-owned Agent execution exposes this store as one `Memory` tool. An
Agent can list, get, search, and inspect the derived graph. It can also propose
creating, updating, or deleting a page, but every mutation pauses for explicit
user approval. External ACP Harnesses do not receive this local tool.

## Engine choice

The LLM Wiki pattern treats raw sources as immutable input, compiles durable
knowledge into Markdown pages, and connects those pages with double-bracket links.
[geronimo-iia/llm-wiki](https://github.com/geronimo-iia/llm-wiki) provides a
headless Git-backed engine, BM25 search, graph projection, and MCP without a
second Desktop application or LLM Provider. SwarmX therefore uses its exactly
pinned Rust crate behind a narrow product-owned sidecar for persistence, BM25,
and Git history instead of rebuilding those engine capabilities in TypeScript
or exposing the upstream MCP mutation surface directly to Agents.

The maintained
[`@flowershow/remark-wiki-link`](https://github.com/flowershow/remark-wiki-link)
supports the relevant Obsidian syntax and is a good fit when an application
already owns a Unified/Remark Markdown AST and rendering pipeline. SwarmX Core
does not have that pipeline, and this feature neither renders nor rewrites
Markdown, so adding the AST stack would make a two-entity projection larger
than its contract. The older
[`remark-wiki-link`](https://github.com/landakram/remark-wiki-link) has the same
AST orientation and defaults to a different alias divider.

[Graphology](https://graphology.github.io/) remains an appropriate Renderer-side
container if interactive layout becomes a real requirement. It is not a
persistence authority. The browser-safe TypeScript scanner remains available
for callers that only need deterministic edge projection. Production Memory
search, persistence, and Git history live in the managed sidecar; the
host rebuilds the bounded graph projection from the current MCP snapshot.

## Persistent CRUD and versions

Production Memory uses one Git-backed Markdown repository at
`~/.swarmx/memory/`. Each page has a stable generated id, unique title and
aliases, Markdown content, optimistic revision, and creation/update timestamps.
The Tantivy search index and TypeScript double-bracket link graph projection are
rebuildable and are not independent authorities. There is no JSON fallback,
legacy importer, or second persistence authority.

Authorized TypeScript hosts depend on the `MemoryBackend` contract and do
not open the repository directly. Desktop implements that contract through the
private `swarmx-mem` MCP server. The Agent-facing `Memory` tool accepts
the following strict operations:

```json
{ "operation": "create", "title": "SwarmX", "aliases": ["Swarm X"], "content": "SwarmX learns from [[Hermes Agent]]." }
{ "operation": "search", "query": "Hermes", "limit": 10 }
{ "operation": "update", "id": "mem_...", "expectedRevision": 1, "content": "Updated Markdown" }
{ "operation": "history", "id": "mem_...", "limit": 20 }
{ "operation": "restore", "id": "mem_...", "expectedRevision": 2, "version": "<40-hex Git commit>" }
{ "operation": "delete", "id": "mem_...", "expectedRevision": 3 }
```

Updates, deletes, and restores require the currently observed revision. A stale
revision returns a conflict instead of overwriting newer knowledge. Each
mutation validates the bounded page, writes it, commits it, and refreshes the
engine index before reporting success. Delete creates a content-free tombstone;
history, historical reads, diffs, and restore use immutable Git commit ids.

Reads are bounded:

- `get` returns one full page or `null`;
- `list` returns metadata without duplicating page content;
- `search` performs bounded BM25 retrieval over explicit current pages and
  returns at most the requested limit;
- `graph` returns the current document generation, page summaries, derived
  edges, and source-qualified link diagnostics.

## Agent CRUD

The Agent tool has bounded read operations for list, get, search, graph,
history, historical version reads, and diffs. Create, update, delete, and
restore require owner confirmation on every call; losing the confirmation
channel fails closed. Updates, deletes, and restores include the current page
revision, so an Agent cannot silently overwrite a concurrent edit.

Tool results return page content only to the requesting SwarmX-owned execution.
Mutation audit records contain the operation, outcome, page id, expected
revision, and content character count where applicable. Page titles, aliases,
and Markdown bodies are excluded from audit records.

## Managed Memory MCP server

The packaged Rust executable and crate are named `swarmx-mem`
with exactly one tool, `swarmx_memory`. Desktop verifies the executable digest,
runtime/protocol version, MCP server identity, and exact tool list before the
first call. Calls and structured results are size-bounded and validated again
with Core zod schemas; the Renderer receives neither filesystem nor raw MCP
authority.

The executable is built from the repository crate with a locked Cargo graph and
the exact `llm-wiki-engine` dependency, then shipped inside the Desktop app.
SwarmX never invokes `cargo install` and never compiles or downloads the engine
while handling a Memory request. The private MCP tool is not registered with an
Agent or external Harness; Desktop projects it through the confirmed,
content-free-audit `Memory` tool.

## Contract

The lower-level browser-safe `buildMemoryLinkEdges` accepts:

- the id of the entity whose Markdown is being scanned;
- bounded Markdown text;
- a bounded registry of entities with stable ids, titles, and optional aliases.

It recognizes the compact subset documented by
[Obsidian internal links](https://obsidian.md/help/links):

- `[[Target]]` and `[[Target.md]]`;
- `[[Target#Heading]]`;
- `[[Target|Visible label]]`;
- `![[Target]]` embeds, which still express a source-to-target relationship.

Resolution is case-insensitive after trimming and Unicode NFC normalization.
The terminal `.md` suffix is optional. A folder-qualified target is just a
title or alias; the caller must declare shorter aliases explicitly, so the
resolver never guesses between duplicate basenames.

One edge is returned for each unique source-target pair. Repeated references
are retained as occurrences with offsets and alias/heading/embed metadata.
Links inside inline code or fenced code are ignored. Unknown, ambiguous,
malformed, and self references do not become edges and are returned as explicit
diagnostics.

```ts
import { buildMemoryLinkEdges } from "@swarmx/core/memory-links";

const result = buildMemoryLinkEdges({
  sourceEntityId: "entity:swarmx",
  markdown: "SwarmX learns from [[Hermes Agent|Hermes]].",
  entities: [
    { id: "entity:swarmx", title: "SwarmX" },
    { id: "entity:hermes", title: "Hermes Agent", aliases: ["Hermes"] },
  ],
});

result.edges[0];
// { kind: "memory_link", source: "entity:swarmx", target: "entity:hermes", ... }
```

## Boundaries

Memory does not call an LLM, ingest sources, infer or admit claims,
mutate itself without explicit approval, provide vector retrieval, or render a
graph. It also does not convert knowledge edges into
`SwarmConfig` edges. Personal preferences still belong to Personal Memory;
hosts retain ownership of Agent authorization, approvals, and UI. The current
Desktop surface exposes CRUD through the Agent tool rather than a standalone
Memory editor.

The Rust process is packaged and verified like a managed runtime. It never uses
`cargo install` on an end-user machine, never inherits Provider credentials,
and is stopped with Desktop rather than being presented as durable background
execution.
