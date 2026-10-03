# Evidence research objects

`@swarmx/evidence` owns the shared RO-Crate 1.3 constants, entity schema and metadata validation.
It depends only on Zod and has no Science, desktop, execution journal or provider dependency.
These existing validation rules are not a complete RO-Crate conformance checker. Domain
applications own scientific projections and artifact payloads; SwarmX owns its generic evaluation
exports. Packaging establishes content identity and provenance links, not judgment correctness.

## Evaluation research objects

H/M/P evaluations and agent/skill improvements are research outputs with an inspectable generation
process. The read-only Memory action `export_evaluation` accepts either `{id, expectedRevision}`
for one saved structured evaluation or `{source}` naming one immutable `review.started` execution
source. The latter includes no-change and failed reviews and resource-only improvement proposals.
It returns `{metadata, files: [{path, content}]}` for an Attached RO-Crate. The desktop downloads
these UTF-8 files as a ZIP with `ro-crate-metadata.json` at its root. This explicit local export
requires `memory.read`; it does not run a model, approve changes, or modify knowledge.

The root Dataset is `./`. Relative File entities name actual payload files with SHA-256 digests
and byte sizes. A saved concept includes its exact Markdown bytes and pinned revision; original
journal files preserve the stored `record_json` bytes. The bundle includes the cited execution
lifecycle, supporting and contrary evidence, and the selected review attempt's inputs, literal
model response when recorded, validated plan and application/approval records. Later attempts
are excluded. Older reviews without a literal response retain that absence explicitly.

Review entities describe assessments; CreateAction entities connect executions and review input
to outputs, and proposed resource changes describe their candidate and original revisions without
claiming that a proposal was applied or that passing a validator improved quality. Statistics
remain derived from the declared source records and recipe, with unknown values and cancelled
outcomes preserved. The exported record files allow independent recomputation. The same frozen
inputs produce identical metadata and payload bytes; export time is not substituted for observation
time. A changed concept revision, missing/foreign evidence, invalid review association, or an
8 MiB total UTF-8 payload/metadata limit rejects the export intact.

This is a private evidence package, distinct from a domain application's scientific project projection.
It can contain selected conversation text, tool results, local paths, session identifiers and
the complete review context. It is neither automatically published nor claimed to be redacted.
Packaging and hashes establish content identity and provenance links, not the truth of a judgment.
