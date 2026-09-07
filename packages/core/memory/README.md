# `@swarmx/memory`

Shared semantic memory for research Agents, persisted as private OKF Markdown concepts.
`MemoryService` validates requests, maintains the vault and indexes, and exposes explicit
retrieval, curation, and deterministic linting with source references and revision controls.
`CoreMemory` owns bounded user/workspace notes. Vault dependencies pin prerequisite revisions,
load in topological order, and expose stale knowledge without silently changing references.
The desktop Host adds frozen session context, journal-backed recall, background review and approval.

The desktop Host exposes the service through its single `ProductServices` instance. Native Agent
carriers only forward calls to that owner.

See [Memory](../../../docs/memory.md) for the tool contract, validation, and storage upgrade.
