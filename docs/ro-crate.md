# Research Objects

SwarmX exposes each Science project as an [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/) Metadata Document. The append-only Science Journal remains the private operational store; it is not a second public Research Object vocabulary.

RO-Crate is the interchange/read model, not a command protocol. Create, modify, execution, and
annotation requests therefore keep strict task-specific schemas; replacing those RPC inputs with
JSON-LD would weaken validation without improving interoperability. The public project graph,
provenance read, and export are the surfaces standardized as RO-Crate.

## Boundary

`ctx.science.getResearchObject(sessionId, { projectId })` and every new `science_export` result return the same deterministic, project-scoped `ro-crate-metadata.json` structure:

- `@context` is `https://w3id.org/ro/crate/1.3/context`.
- `@graph` is flat and contains one `ro-crate-metadata.json` descriptor plus one project `Dataset` root.
- Every entity has a unique `@id`, an `@type`, and a human-readable `name` when applicable.
- References to other entities use `{ "@id": "..." }`; nested entities are not emitted.
- Root `hasPart` makes every project entity reachable.
- Host paths, Session identifiers, unredacted environments, Journal payloads, and model-private reasoning are never included.

The live API document uses stable `urn:uuid:` entity identifiers and describes registered local artifacts as contextual `MediaObject` entities. It therefore does not pretend that an API response is an Attached RO-Crate package containing payload files. A future downloadable payload package must place `ro-crate-metadata.json` and every relative `File` data entity in one actual crate root.

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

Operational statuses use Schema.org `creativeWorkStatus` or `actionStatus`. Tags use `keywords`. Source links use `isBasedOn` or the Action input/output properties. The old `ScienceRelation { fromId, toId, type }` rows remain private Journal projections and are not a public graph format.

## Extensions

RO-Crate Profiles are the only supported route for a SwarmX-specific semantic extension. SwarmX must not declare profile conformance until the profile has a versioned persistent URI that resolves to human-readable documentation. Until that exists, the projection uses standard terms and permitted textual `additionalType` values; it does not emit ad-hoc compact JSON-LD keys.

Persisted `dsh-science-project@1` export events remain replayable as immutable history. They do not define the current export contract: all newly requested exports use RO-Crate 1.3.
