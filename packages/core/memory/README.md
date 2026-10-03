# `@swarmx/memory`

Shared semantic memory for research Agents, persisted as private OKF Markdown concepts.
`MemoryService` validates requests, maintains the concept pool and index, and exposes explicit
retrieval, curation, and deterministic linting with source references and revision controls.
`CoreMemory` owns the bounded user note. Concept dependencies pin prerequisite revisions,
load in topological order, and expose stale knowledge without silently changing references.
The desktop Host adds frozen session context, journal-backed recall, background review and approval.

Assessment writes may supply `evaluation`: an observation, judgment or preference, its task,
criteria, execution evidence, counter-evidence and limitations. It persists as `swarmx_evaluation`;
an optional review reference identifies its review snapshot. Evidence uses exact
`urn:swarmx:execution:<UUID>` addresses. Writes merge those references into sources without losing
existing citations, within the 32-source limit. Updates preserve an assessment unless replaced.
Reads and lint validate assessment references even when hand edits omit them from sources. A missing
resolver is reported as a lint warning; resolver errors reject writes, while unresolved warnings
keep historical notes readable.

Concept files live directly under the Memory root beside one `index.md`; `README.md` and `USER.md`
are reserved. Reads and linting reject unsafe paths. Unknown frontmatter fields survive reads and
updates, while invalid UTF-8 is rejected.
`snapshotConcept(id, expectedRevision)` returns the parsed concept and its exact original
Markdown from the same read, rejecting stale revisions for reproducible evidence exports.

Concept updates accept an optional UUID `requestId` for crash recovery. Replaying the same parsed
request after its atomic write returns the saved concept and repairs its index without requiring a
new revision. Reusing that ID with changed request content is rejected. Any later successful update
replaces or clears this last-operation marker, so stale replay still fails after an intervening edit.
A saved content fingerprint also rejects replay after direct file edits that retain the marker.
Requests without an ID retain strict `expectedRevision` checks.

New request-ID creations also save `swarmx_create_revision`: create replay rejects changes to
content, metadata or original formatting before repairing the index. Legacy creations without this
fingerprint retain their earlier replay contract; reads and ordinary edits remain compatible.

The desktop Host exposes the service through its single `ProductServices` instance. Native Agent
carriers only forward calls to that owner.

`swarmx-memory` provides standalone read-only model-experience snapshots and owner-local external
observation imports against one explicit user-owned vault. See
[shared model experience](../../../docs/shared-model-experience.md) for schema, privacy and authority
boundaries. Ordinary read/search/snapshot operations no longer initialize or repair the vault.

See [Memory](../../../docs/memory.md) for the tool contract, validation, and layout.
