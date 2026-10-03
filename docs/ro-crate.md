# Research Objects

SwarmX exposes each Science project as an [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/) Metadata Document. The append-only Science Journal remains the private operational store; it is not a second public Research Object vocabulary.

RO-Crate is the interchange/read model, not a command protocol. Create, modify, execution, and
annotation requests therefore keep strict task-specific schemas; replacing those RPC inputs with
JSON-LD would weaken validation without improving interoperability. The public project graph,
provenance read, and export are the surfaces standardized as RO-Crate.

Local `sx:` resource addresses are a separate, directory-authorized progressive-read layer. They
return bounded heads, metadata projections, Artifact preview windows, and relation refs without
changing the RO-Crate graph or its `urn:uuid:` entity identifiers. See
[`resource-addressing.md`](./resource-addressing.md).

## Boundary

The reusable `@swarmx/evidence` package owns the RO-Crate constants, entity schema and
metadata-document validation shared by evaluation exports and domain applications. It has no
Science, desktop, execution, Memory or provider dependency. Existing Science public exports
remain available, but generic desktop evaluation exports and graph rendering import this
contract directly. Scientific project IDs and projection rules stay in Science.

This extraction preserves the existing wire format and validation rules. It does not make the
validator a complete RO-Crate conformance checker or move scientific interpretation into SwarmX.

`ctx.science.getResearchObject(sessionId, { projectId })` and every new `science_export` result return the same deterministic, project-scoped `ro-crate-metadata.json` structure:

- `@context` is `https://w3id.org/ro/crate/1.3/context`.
- `@graph` is flat and contains one `ro-crate-metadata.json` descriptor plus one project `Dataset` root.
- Every entity has a unique `@id`, an `@type`, and a human-readable `name` when applicable.
- References to other entities use `{ "@id": "..." }`; nested entities are not emitted.
- Root `hasPart` makes every project entity reachable.
- Host paths, Session identifiers, unredacted environments, Journal payloads, and model-private reasoning are never included.

The live API document uses stable `urn:uuid:` entity identifiers and describes registered local artifacts as contextual `MediaObject` entities. It is not an Attached RO-Crate package containing payload files.

## Mapping

| Science projection | RO-Crate / Schema.org representation |
| --- | --- |
| Project | Root `Dataset` |
| Notebook | `SoftwareSourceCode` |
| Registered artifact | `MediaObject`, plus `ImageObject`, `DigitalDocument`, `Dataset`, or `SoftwareSourceCode` where applicable |
| Writing document | `DigitalDocument` |
| Figure source | `SoftwareSourceCode` |
| Research question | `Question` |
| Hypothesis | `CreativeWork` with textual `additionalType: "Hypothesis"` |
| Claim | `Claim` |
| Evidence supporting/refuting a claim | `Review` whose `itemReviewed` is the claim and whose separate `Rating` records support/refute direction |
| Experiment definition | `HowTo` |
| Run or Journal mutation | `CreateAction` or `UpdateAction`; inputs use `object`, outputs use `result`, and the experiment/software uses `instrument` |

Operational statuses use Schema.org `creativeWorkStatus` or `actionStatus`. Tags use `keywords`. Source links use `isBasedOn` or the Action input/output properties. `ScienceRelation { fromId, toId, type }` rows remain private Journal projections and are not a public graph format.

## Extensions

The projection uses standard terms and permitted textual `additionalType` values. It emits no SwarmX-specific Profile or ad-hoc compact JSON-LD keys.

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

This is a private evidence package, distinct from the default Science project projection above.
It can contain selected conversation text, tool results, local paths, session identifiers and
the complete review context. It is neither automatically published nor claimed to be redacted.
Packaging and hashes establish content identity and provenance links, not the truth of a judgment.
