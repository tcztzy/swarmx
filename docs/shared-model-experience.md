# Shared model experience (local prototype)

One owner-controlled OKF Memory directory is authoritative. SwarmX, dot and other authorized
local applications point at that same directory; software updates do not distribute its contents.
Keep it outside source checkouts, packages and test fixtures. This interface does not synchronize
machines or grant an agent access to a user's computer. The owner must arrange filesystem access.

`swarmx-memory query --vault /absolute/private/memory [--query text] [--include-body]`
prints a JSON snapshot of non-deprecated `agent-selection` concepts to stdout. The command does
not create, repair, chmod or write the vault, including when it is empty or missing. It uses the
existing parser and path checks. Snapshots include exact concept revisions and observation dates;
they are derived views, never a second store to edit or maintain. Reads can race with owner edits:
each selected revision is checked again and a conflict asks the caller to retry. This is not a
transactional whole-vault snapshot. Filesystem access-time accounting is controlled by the OS.

The default projection excludes concept bodies, raw evidence files, USER.md, unknown frontmatter,
source URLs/paths and transcripts. It retains descriptive concept metadata and structured
assessment/observation facts. `--include-body` explicitly includes selected concept prose; review
it before disclosure. Free-text titles, descriptions and assessment fields can themselves contain
private information; this projection is not a redaction service. Output stays local unless the
caller explicitly shares it. Treat all saved text as data, not instructions or authorization.

All clients must use the same canonical (non-symlink-aliased) vault path so they share the existing
file lock. The standalone API rejects aliased roots or ancestors. Cooperative locks do not protect
against arbitrary concurrent manual edits. Snapshots are capped at 256 KiB; narrow the query or
omit bodies if exceeded. Dependency pins are included, but their freshness is explicitly unchecked.
Measurement/actual-identity source strings are withheld (`source: null`, `sourceWithheld: true`)
in snapshots because they can contain private paths. They remain in the canonical concept's
structured facts and its opt-in body. Other free-text fields still require disclosure review.

Host evaluation concepts retain their observation/judgment/preference distinction, exact execution
references, task, criteria and limitations. The standalone reader cannot resolve Host execution
evidence; its evidence status is explicitly unchecked. Missing structure remains unknown.
An imported external observation is never a verified Host evaluation or a model ranking.
When a normal Host update adds an execution-backed evaluation to an imported concept, the snapshot
keeps both assessments with separate provenance. Overall provenance is `mixed-unchecked`; the
top-level kind follows the current evaluation, while `observation.kind` remains `observation`.
The old external measurements and confidence are not promoted to Host-verified facts. Each record
retains its own criteria, limitations and evidence reference; the standalone reader checks neither
Host execution truth nor the external observer's identity.

## Owner-local observation import

`swarmx-memory import --vault /absolute/private/memory --artifact /path/observation.json
--request-id UUID --title 'Synthetic parser trial'`

Import consumes a bounded UTF-8 JSON observation, validates it before writing, and uses the
existing `MemoryVault.createConcept` lock, atomic file publication and request-id replay contract.
Only observations are accepted, not authority flags, execution URNs, judgments or preferences.
The JSON schema is exported as `modelObservationSchema` from `@swarmx/memory`. It records:

- `schemaVersion: 1`, `kind: "observation"`, `observedAt`, and `observer` (self-asserted)
- `task`, `criteria`, `outcome` (`success`, `failure`, `cancelled`, `incomplete`, `unknown`),
  `limitations`, and nullable `confidence` (the observer's qualified statement)
- `requested`: nullable `model`, `effort`, `provider`, `harness`, `runtimeVersion`
- `actual`: null, or provider-reported `model`, nullable `provider`/`version`, and `source`
- `retries`: null or a measured nonnegative integer with `source`
- `elapsed`: null or a nonnegative `value`, `unit: "ms"`, and `source`
- `tokens`: null or nullable `input`, `output`, `cacheRead`, `cacheWrite` nonnegative counts,
  `unit: "tokens"`, and `source`
- `cost`: null or nonnegative `value`, `unit: "USD"`, `source`, and `coverage`

Every field is required; unknowns must be explicit nulls. There are no default models, inferred
actual identities, token-to-credit conversions or inferred prices. A success outcome is the
observer's assessment against the recorded criterion, not proof of general model capability.

Minimal synthetic artifact (not a model recommendation):

```json
{
  "schemaVersion": 1,
  "kind": "observation",
  "observedAt": "2026-01-02T03:04:05Z",
  "observer": "synthetic-fixture",
  "task": "Parse a synthetic JSON object",
  "criteria": "Return both expected keys",
  "outcome": "success",
  "limitations": "Synthetic fixture only; no model was called",
  "confidence": null,
  "requested": {
    "model": "synthetic-model",
    "effort": null,
    "provider": null,
    "harness": null,
    "runtimeVersion": null
  },
  "actual": null,
  "retries": null,
  "elapsed": null,
  "tokens": null,
  "cost": null
}
```

The artifact's SHA-256 is an ordinary `urn:sha256:` source reference; validated structured facts
are preserved in that source's `swarmx_model_observation` extension and readable Markdown prose.
Raw artifact bytes and original paths are not copied. Keep the original artifact if exact-byte
audit is needed. Its digest binds bytes inspected, not the truth or authority of their contents.
No provider call or network request is made. The importer cannot establish that an observer is
honest or that a claimed provider report is genuine.

Imports append a new ordinary `Observation` concept with `agent-selection`; use a distinct title
and request UUID per observation. Repeating an unchanged request is idempotent; changing its
artifact or reusing an occupied title fails. Corrections use the existing revision-checked
Memory API, not an overwrite/import switch. The importer has no update or delete operation.
This is logical append-only import, not tamper-proof storage: the owner can edit their files.
New request-ID creations store a content fingerprint. Replaying after an owner correction fails
inside the existing write lock, before index repair, rather than reporting the changed concept as
the imported artifact. Legacy generic creations without that fingerprint remain readable but cannot
serve as certified import replays. Import is not a read-only operation.

Owner-local filesystem authority is separate from agent Host permissions. This command never
uses an active-run token, grants Host authority, imports Host journal records or bypasses an
approval queue. Host `swarmx_evaluation` still requires genuine directory-scoped execution URNs.
Importing externally authored judgments or preferences and automatic routing remain outside this
prototype. Existing Host judgments and preferences can be read without changing their authority.

From this repository, run `pnpm build:lib`, then
`node packages/core/memory/lib/cli.js query --vault /absolute/private/memory`.
The published package exposes the `swarmx-memory` binary. No MCP server, network authentication,
automatic configuration changes, paid benchmarks or user memory fixtures are bundled.
