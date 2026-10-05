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

## Source dependencies

The package has its own build tools and a self-contained build configuration. `prepare` builds
JavaScript and declarations from source before Git dependency packaging; generated `lib/` stays
ignored and must not be committed. A package-local empty workspace isolates Git preparation from
the parent monorepo install. Its small local lockfile pins build inputs. Root SwarmX commands still
discover this package normally. No desktop or other workspace package is installed or built during
isolated source preparation.

With pnpm, pin an immutable upstream commit and select this package's subdirectory:

`"@swarmx/evidence": "git+https://github.com/tcztzy/swarmx.git#<full-commit-sha>&path:/packages/core/evidence"`

Commit the consumer's manifest and lockfile. Follow the consumer's build-approval policy for this
trusted source dependency; pnpm can require approval for the exact Git dependency identity,
including its commit and subdirectory. Disabling its prepare script leaves the source package
unbuilt. A registry release is not needed, and a compiled vendor tarball does not belong in the
consumer's Git history. The commit must contain the source build contract described here.

npm does not support pnpm's Git subdirectory selector. For an npm consumer, first check out and
verify the chosen upstream SHA, install/build this package independently with
`npm install --prefix /path/to/checkout/packages/core/evidence --workspaces=false`, then use a
`file:` directory dependency on that prepared package. Keep the source checkout pin/verification
alongside the consumer manifest and lock; the local file dependency alone does not pin a Git SHA.
