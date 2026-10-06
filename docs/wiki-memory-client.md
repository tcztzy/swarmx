# Opt-in Wiki memory client

`WikiMemoryClient` is a small client for the public `search_memory` and `write_memory` MCP
contracts. The caller supplies an already connected MCP transport and explicit absolute scopes.
Constructing this client is the opt-in; it changes no settings, starts no server, installs no
runtime, and does not replace the existing `MemoryVault` backend.

## Opt-in Host search

The trusted startup caller may pass `wikiMemory` to `startDesktopPlatform` or
`ProductServices.create`: an absolute maintained `command`, explicit `brain` directory,
absolute discovery `scopes`, and approved wiki `roots` (including `brain/wiki`). Optional
`env` values belong to the Host caller. The Host copies and canonicalizes this configuration;
Agent requests cannot select commands, roots, scopes, environment, full-content reads or writes.
No binary is installed or launched at startup. This is a programmatic opt-in, not a saved UI setting.

`memory {action:'search_wiki_memory',request:{query,maxResults?,maxChars?,sections?}}`
requires `memory.read`. Queries are at most 1,000 characters; up to 20 excerpts are requested
(default 8), with 80–2,000 UTF-16 characters per excerpt (default 600). Sections may select
frontmatter, body or both. Disabled Memory and an unconfigured connection return explicit
`disabled`/`unconfigured` status with no subprocess or tool dispatch. Malformed requests,
denied grants and pre-cancelled operations likewise dispatch nothing.

The Host lazily owns one official SDK stdio connection, fixes `MEMORY_DATA_DIR` to `brain`
and `MEMORY_WORKSPACE_DIR` to the Host directory, and calls only `search_memory` through
`WikiMemoryClient`. Initialization and tool requests have bounded deadlines. Caller cancellation
while awaiting initialization stops that caller before search dispatch; initialization can finish
for other callers. Host shutdown cancels operations, closes the child and awaits settlement.
No cancelled caller dispatches a search after initialization completes.

Results contain excerpts, relative document IDs and process-lifetime opaque `sourceId`s.
Equal document IDs in different approved roots retain distinct source identities. Native scopes
can discover additional ancestors, so roots filter returned origins rather than confining the
native process's filesystem reads. Unapproved/invalid origins and nonportable document IDs are
omitted with diagnostics. The maintained native server omits origin metadata only for its single
configured-brain search; that result is qualified with the approved `brain/wiki` identity.

The Host projects an explicit small field set, redacts absolute paths in text, and never passes
through raw roots, project-module URIs, filters, error objects or transport errors. Quoted paths
with spaces are handled as whole strings; an unquoted path may withhold the rest of
its line to avoid exposing a whitespace-containing path suffix. Metadata and
diagnostics are capped, body text has a shared 16,000-character budget, and the serialized result
is bounded to 32 KiB. Native truncation/fullChars and partial-search diagnostics are retained;
Host omissions, redaction and additional truncation are reported. Excerpts remain untrusted
reference data, without authoritative revisions, dependency pins or write/CAS semantics.
Existing notes, concepts, delegation selection and review replay continue using `MemoryService`.

- `search` requests bounded search excerpts (600 characters per hit by default); `read` requests
  detailed excerpts with frontmatter and body (2,000 characters per hit by default). Both send
  `fullContent: false` so the server honors `maxChars`; no unrestricted full-content read is exposed.
  A frontmatter-only view may omit each record's `content` field.
  Read is not an exact document-ID lookup or a revision-pinned snapshot. A server may still
  truncate, omit, or partially search documents; returned diagnostics and truncation fields remain
  visible. Scopes select the server's discovery context, which can include its configured brain
  and ancestor mounts; they are not an independent filesystem permission boundary.
- `write` requires its target on every call. It uses `write_memory` and never invents a consent
  flag, bypasses quality or duplicate checks, supersedes another document, or retries. The caller
  can pass `userRequested: true` only after obtaining actual user consent. A refusal stays a
  refusal. Same-name writes can upsert existing content under the server's own policy.
  Success requires `ok: true` and the documented `created.document.id` identity or a native
  top-level `documentId`. The client preserves the nested receipt and exposes `documentId` as an
  explicit alias. If both identities are present, they must match exactly; missing, malformed or
  contradictory identities leave the write outcome unknown.
- Requests and JSON text responses are validated and bounded. MCP tool errors and semantic
  refusal envelopes throw `WikiMemoryError`, preserving the response for local inspection.
  Only recognized consent/quality/duplicate/size refusal codes certify a refused request. Other
  failure envelopes, invalid or lost write responses, transport failures, and cancellation after
  dispatch have an unknown write outcome. Read back and inspect before deciding whether to retry; cancellation
  cannot roll back a write that the server has already committed.
- The transport owns initialization, authentication, deadlines, process lifecycle, and cancellation
  propagation. It must honor the supplied AbortSignal and settle calls after cancellation.
  Saved content remains untrusted data and cannot grant permissions or establish truth. Output
  stays with the caller; this client does not transmit it to a model or another service.

The adapter deliberately exposes only search/read excerpts and ordinary write fields. Public
upstream contract provenance: [ctxr-dev/llm-wiki-memory](https://github.com/ctxr-dev/llm-wiki-memory),
MIT. No backend implementation, owner data, or server fixtures are included here.

## Before replacing the desktop backend

The existing Host requires exact concept identities, SHA-256 revision-pinned reads and snapshots,
revision-checked updates, request-ID replay with content-integrity checks, dependency pins,
structured evaluation provenance, approval queues, and user-note/graph/lint/load behavior. The
Wiki search/write surface does not promise these semantics. No adapter can safely manufacture
them from similarity search and same-name upserts. A future backend selection must explicitly
configure ownership and roots, define the missing contracts, and verify migration separately.
