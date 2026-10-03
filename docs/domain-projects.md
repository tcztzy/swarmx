# Independent domain applications

SwarmX owns native Agent access, delegation, Host authority, Work, Memory and observed execution
evidence. Domain applications such as GEEPilot own scientific entities, methods, artifact storage,
notebooks, runtimes and interpretation. Their existing data, revisions and hashes remain theirs;
SwarmX does not delete or rewrite old scientific stores.

## Native execution and ACP

Set `SWARMX_CWD` to an existing project directory and optionally `SWARMX_HOME` to a dedicated
private product home, then run `pnpm start` from the SwarmX checkout. Independent programs and
native skills stay with that project. Native runtimes retain their own authentication, configuration
and transcripts. See [runtime setup](runtime-platform.md) and [ACP](acp.md).

External applications can launch the existing `pnpm --silent acp` entry point after building,
using official ACP initialization/session/prompt and native permission flows. ACP stdout is reserved
for protocol traffic. ACP execution access does not register a domain evidence resolver or grant
additional Host authority. Saving or resuming an ACP session is separate from scientific data access.

## Generic calls and evidence ownership

A coordinating Agent learns domain capabilities from the ordinary Agent/tool descriptions and
uses the existing authorized native or ACP call path. Domain-specific scientific validation stays
with that Agent/tool. SwarmX has no dedicated GEEPilot connector, resource resolver registration,
scientific URI parser or command protocol.

Work submissions contain opaque artifact identities plus observed execution sources:

```json
{"action":"submit","artifacts":[{"id":"domain-owned-id","revision":"domain-owned-revision","evidence":["urn:swarmx:execution:<record-UUID>"]}]}
```

`work.status` exposes `submissionEvidence`: up to the latest 100 source URNs, run IDs, event
types and observation times from this Work runtime, plus a truncation flag. It exposes no recorded
payloads and needs no broader Memory recall grant. The Agent can use these stable references
after ordinary producing/verification calls.

The Host requires an active managed Work execution. For every artifact, evidence must be nonempty
and refer to immutable records in this execution directory, Work item and runtime; delegated runs
in that runtime may supply producing or verification observations. Missing, foreign or mismatched
sources reject the whole submission. No `science.read` grant or domain lookup is involved.

The ID and revision are supplied domain claims. The Host does not check their syntax, existence,
currentness or scientific meaning. Even a valid source can contain a self-report or an incorrect
judgment: provenance proves what was observed, not that the artifact or conclusion is correct.
Independent user/trusted-validator acceptance pins the submitted identities and evidence, criteria
revision and evaluator report. Agent submission cannot accept its own work or establish domain truth.
A domain verification result must be obtained through an ordinary authorized Agent/tool call and
assessed against the acceptance criteria, not synthesized by this bookkeeping layer.

Existing Work records without evidence stay readable as unverified legacy records. New artifact
submissions lacking evidence are rejected; add observed sources rather than silently upgrading old
records. Existing scientific storage, revisions and hashes are not changed by this contract update.

Memory validates SwarmX execution URNs against its own directory-scoped journal. Other URI
references, including old `sx:` identifiers, are opaque external references and receive a
`source.unverified` warning. Local Memory paths retain their validation. This does not query external
services, infer scientific validity or promise that an unavailable link resolves.
