# Evidence contracts

`@swarmx/evidence` supplies the shared RO-Crate 1.3 constants, entity schema and metadata-document
validator used by SwarmX evaluation exports and scientific domain applications. It depends only
on Zod and runs without a desktop, Science workspace, journal or provider.

Import constants, `roCrateEntitySchema`, `roCrateMetadataDocumentSchema`, `RoCrateEntity` and
`RoCrateMetadataDocument` from `@swarmx/evidence`. The constants are `RO_CRATE_CONTEXT`,
`RO_CRATE_PROFILE`, `RO_CRATE_FILENAME`, `RO_CRATE_MEDIA_TYPE` and `RO_CRATE_FORMAT`.

These are the existing SwarmX evidence validation rules, not a complete RO-Crate conformance
implementation. Domain projection, artifact storage, execution and scientific interpretation
remain with their owning application. Parsing metadata does not validate payload bytes or prove
that an evaluation is correct.
