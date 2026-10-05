# Opt-in Wiki memory client

`WikiMemoryClient` is a small client for the public `search_memory` and `write_memory` MCP
contracts. The caller supplies an already connected MCP transport and explicit absolute scopes.
Constructing this client is the opt-in; it changes no settings, starts no server, installs no
runtime, and does not replace the existing `MemoryVault` or desktop Memory backend.

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
