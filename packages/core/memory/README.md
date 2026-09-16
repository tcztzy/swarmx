# `@swarmx/memory`

Shared semantic memory for research Agents, persisted as private OKF Markdown concepts.
`MemoryService` validates requests, maintains the concept pool and index, and exposes explicit
retrieval, curation, and deterministic linting with source references and revision controls.
`CoreMemory` owns the bounded user note. Concept dependencies pin prerequisite revisions,
load in topological order, and expose stale knowledge without silently changing references.
The desktop Host adds frozen session context, journal-backed recall, background review and approval.

Concept files live directly under the Memory root beside one `index.md`; `README.md` and `USER.md`
are reserved. Reads and linting reject unsafe paths. Unknown frontmatter fields survive reads and
updates, while invalid UTF-8 is rejected.

The desktop Host exposes the service through its single `ProductServices` instance. Native Agent
carriers only forward calls to that owner.

See [Memory](../../../docs/memory.md) for the tool contract, validation, and layout.
