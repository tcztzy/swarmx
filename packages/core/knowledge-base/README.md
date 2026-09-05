# `@swarmx/knowledge-base`

Vendor-neutral private knowledge storage. `KnowledgeBaseService` validates bounded requests,
maintains Markdown concepts and search indexes under its configured vault, and records provenance
without owning Agent transcripts or transport state.

The desktop Host exposes the service through its single `ProductServices` instance. Native Agent
carriers only forward calls to that owner.
