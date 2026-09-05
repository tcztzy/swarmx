# Science resource addressing

SwarmX gives each local Science entity a typed, canonical address without changing its internal
identifier or the Science Journal schema:

```text
sx:<prefix>/<encoded-entity-id>
sx:<prefix>/<encoded-entity-id>@<revision>
```

The entity-id segment uses canonical `encodeURIComponent` encoding. The prefixes are `p` project,
`n` notebook, `a` artifact, `d` document, `f` figure, `r` research record, `e` experiment, and `x`
run. A bare UUID is not a resource address because it does not identify an entity type. Unknown
prefixes, empty or non-canonical encodings, addresses above 1024 characters, trailing input, and
non-positive or unsafe revisions are rejected. The formatter enforces the same encoded-address
limit, so every successful formatted value parses back to the original identity.

The address without `@revision` is the logical ID. The exact ID returned by every resource view adds
the current positive revision. In this phase an exact ID is a current-revision assertion, not a
historical read: a mismatch returns `RESOURCE_REVISION_MISMATCH` and never falls back to the latest
entity. Call `head` again and deliberately use its new `exactId` after deciding that the changed
entity is acceptable.

## Bounded views

`science_query` provides five ID-addressed read actions:

- `head` returns one resource ref, a deterministic short summary, and the capabilities currently
  supported for that entity.
- `batch_head` returns at most 50 heads in input order. Duplicate IDs remain duplicated in the
  result but are resolved once.
- `get` currently accepts only the `metadata` projection. It returns bounded decision-making fields,
  counts, and capped reference/key/tag lists, never full Notebook cells, document source, figure
  code, experiment protocol, environment, provenance payload, or a workspace snapshot.
- `select` works only for text and tabular Artifacts already accepted by the existing verified
  Artifact preview. A table window contains at most 100 rows and 32 requested columns. A text window
  contains at most 16 KiB of characters. Unknown or ambiguous table columns fail explicitly.
- `neighbors` returns at most 100 typed target refs for explicit Journal relations and structural
  project/member, source, hypothesis/run, and run/artifact links. It never infers a relation from
  titles or prose. Missing legacy targets are skipped; ambiguous untyped legacy targets fail closed.

Artifact `select` is a window over the existing bounded preview, not arbitrary file random access.
The source preview reads at most 64 KiB and materializes at most 500 rows. An Artifact beyond the
text preview limit or with an image, PDF, spreadsheet, or binary MIME returns an explicit
`too-large` or `unsupported` result. When rows or columns exist beyond the preview, `truncated` stays
true and no continuation is advertised outside the verified preview.

Every successful view includes the canonical logical ID, current exact ID, resource kind, title,
revision, and a trusted digest when one already exists. Artifact digests come from the immutable
Artifact Store; document and figure digests come from their latest stored source/code revision.
Other entity kinds report `null` rather than assigning a hash with unclear semantics.

## Authorization and graph boundaries

Each Host method validates the native conversation, derives its authorized workspace, loads only that
workspace's Science snapshot, and resolves the typed ID there. Knowing an ID from another workspace
does not make it discoverable. Model-visible resource data contains no host path, conversation identity,
unredacted environment, or raw Journal payload. Artifact bytes are read only through the existing
owner-only, digest-verifying Artifact Store.

The resource resolver is a narrow local read layer over Science Journal projections. RO-Crate
remains the project-level interchange graph and is not a command protocol or a replacement resource
address. This phase adds no identifiers.org registry/API, UniProt/PubMed/Crossref/PDB provider,
network access, session shorthand such as `@a1`, alias inheritance, database migration, historical
revision materialization, SQL/filter/sort/join language, UI, or RO-Crate identifier extension.

Future external Compact Identifiers such as `uniprot:P12345`, `taxonomy:9606`, and DOI addresses
will be routed through the same resolver boundary as additional providers; they will not replace
the local `sx:` namespace.
