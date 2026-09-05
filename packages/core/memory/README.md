# `@swarmx/memory`

Shared semantic memory for research Agents, persisted as private OKF Markdown concepts.
`MemoryService` validates requests, maintains the vault and indexes, and exposes explicit
retrieval, curation, and deterministic linting with source references and revision controls.
Native Agents own their transcripts and context management.

The desktop Host exposes the service through its single `ProductServices` instance. Native Agent
carriers only forward calls to that owner.

See [Memory](../../../docs/memory.md) for the tool contract, validation, and storage upgrade.
