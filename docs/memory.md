# Memory

SwarmX Memory uses actual bounded `USER.md` and `MEMORY.md` global files plus a
local collection of Markdown entity pages that an authorized host can create,
read, list, search, update, and delete. Double-bracket links between current
entity pages are projected into directed knowledge-graph edges. Global files and
entity pages share one Git authority. Neither surface turns Memory into Session
history, a graph database, or a workflow runtime.

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
`~/.swarmx/memory/`. The directory itself is a human-owned vault: open it
directly in Obsidian, VS Code, or any Markdown editor. `USER.md` and `MEMORY.md`
live at its root, while active entity pages use readable recursive paths under
`pages/`, for example `pages/Herdr.md` or `pages/Projects/SwarmX.md`. Each page
has a stable generated id in YAML frontmatter, a unique title and aliases,
Markdown content, optimistic revision, and creation/update timestamps. A page
may also use one of seven deliberately small kinds (`project`, `person`,
`organization`, `technology`, `decision`, `concept`, or `note`) plus a short
summary, human-checkable source links, and an optional Project/source scope.
These ordinary frontmatter fields are enough for useful templates without
creating an ontology framework. The id
is an internal permalink used by APIs and Git history; it is deliberately not
the filename and remains stable when a human moves or renames the file.

Markdown files are the source of truth. Before each operation, the runtime
reconciles bounded human changes under `pages/`: new notes are adopted, body or
supported frontmatter edits advance the revision, moves keep the same id and
revision, and deletions become hidden recoverable tombstones. Unknown YAML
frontmatter fields are preserved. Reconciliation is committed before search is
refreshed, so Obsidian and SwarmX see the same wiki links, content, and graph.
SwarmX does not write `.obsidian` settings or require an Obsidian plugin.

`INDEX.md` is a deterministic directory containing title-first page summaries
and backlinks. `DISAMBIGUATION.md` lists qualified pages that began with the
same human name. Both are rebuildable views of the page Markdown and may be
deleted safely; SwarmX recreates them without making either file an authority.
Search keeps its compatible full-page result and additionally projects
title-first results containing summary, kind, sources, and related page titles.

New Agent-created pages derive a portable, human-readable filename from the
title while preserving Unicode, spaces, and case; filesystem-reserved
characters become hyphens, and a numeric suffix resolves a path collision.
Existing `pages/mem_<id>.md` pages migrate once to title-derived paths without
changing their stable id, semantic revision, or recoverable Git history.
The migration is idempotent. A failed multi-file rename restores the page,
incoming links, and generated views to their pre-operation bytes and restages
that state, while a successful operation records all changes in one Git
version.
The Tantivy search index and TypeScript double-bracket link graph projection are
rebuildable and are not independent authorities. There is no JSON fallback or
second persistence authority.

### Crash recovery and publication

All Memory mutations that touch more than one path use one short-lived WAL in
`.runtime/transactions/<txid>`. The WAL contains the base HEAD, intended Git
commit, path manifest, before-images, and a content-free summary. The sidecar
holds a single-writer lock, prepares the commit without moving HEAD, durably
records the WAL, applies `write → fsync → rename → directory fsync` per path,
then advances the branch with a compare-and-swap. Index refresh and WAL
cleanup happen before success is reported.

On startup, recovery runs before migration, reconciliation, or search setup.
HEAD at the base commit restores the before-image; HEAD at the intended commit
rolls the working tree forward from Git. Any other HEAD or file/journal state
is retained and fails closed so an external editor's changes cannot be
overwritten. A recovery pass never creates a new commit, and a second restart
is therefore a no-op.

An API title change preserves the old title as an alias, moves the file within
its current human folder, rewrites exact incoming Wiki-link targets (including
headings and visible aliases), and refreshes the directory/backlink views in
the same version. When a second entity is created with the same title, SwarmX
qualifies both titles with their kind and optional scope, for example
`Mercury (organization, Project Atlas)`, and records the shared name in the
disambiguation view. It never overwrites the first entity or chooses one by
registration order.

Deletion removes the active page and leaves a content-free tombstone plus Git
history. Moving a page into `pages/Archive/` is the human archive operation: it
keeps identity, content, and revision history and is reconciled as a move.
Merging is intentionally not a hidden inference: update the chosen survivor,
redirect links through a rename when needed, then delete the duplicate. Every
step is visible and recoverable in Git; SwarmX never silently coalesces pages.

Authorized TypeScript hosts depend on the `MemoryBackend` contract and do
not open the repository directly. Desktop implements that contract through the
private `swarmx-mem` MCP server. The Agent-facing `Memory` tool accepts
the following strict operations:

```json
{ "operation": "create", "title": "SwarmX", "aliases": ["Swarm X"], "content": "SwarmX learns from [[Hermes Agent]]." }
{ "operation": "create", "title": "Mercury", "kind": "technology", "scope": "Project Orion", "summary": "A protocol.", "sources": ["https://example.test/protocol"], "content": "Verified facts and links." }
{ "operation": "search", "query": "Hermes", "limit": 10 }
{ "operation": "update", "id": "mem_...", "expectedRevision": 1, "content": "Updated Markdown" }
{ "operation": "history", "id": "mem_...", "limit": 20 }
{ "operation": "restore", "id": "mem_...", "expectedRevision": 2, "version": "<40-hex Git commit>" }
{ "operation": "delete", "id": "mem_...", "expectedRevision": 3 }
```

Updates, deletes, and restores require the currently observed revision. A stale
revision returns a conflict instead of overwriting newer knowledge. Each
mutation validates the bounded page, writes it, commits it, and refreshes the
engine index before reporting success. Delete removes the active note and keeps
a content-free tombstone outside the human-facing `pages/` tree;
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

The Core and Rust boundaries reject common credential-bearing content in global
files, page bodies, names, summaries, scopes, and source fields (private keys,
bearer authorization, embedded URL credentials, token-shaped values, and
populated secret/password/token fields).
Research pages distinguish `observed`, `derived`, `decision`, and `hypothesis`
claims and retain source locators, so facts, inferences, user decisions, and
external evidence remain visibly different without storing raw prompts,
responses, source code, or terminal transcripts.

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

The storage engine does not call an LLM, ingest sources, infer or admit claims,
mutate itself without explicit approval, provide vector retrieval, or render a
graph. It also does not convert knowledge edges into `SwarmConfig` edges.
`USER.md` owns personal preferences; `MEMORY.md` owns compact cross-Project
experience; detailed research belongs to entity pages. Hosts retain ownership
of reflection, candidate validation, Agent authorization, approvals, and UI.

The Rust process is packaged and verified like a managed runtime. It never uses
`cargo install` on an end-user machine, never inherits Provider credentials,
and is stopped with Desktop rather than being presented as durable background
execution.
