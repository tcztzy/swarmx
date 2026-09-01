# WikiSkill evolution

SwarmX adopts WikiSkill as an explicit skill-evolution loop without replacing its conversation or
knowledge stores:

```text
Raw     DSH native Session log (`ctx.sessionQuery`)
Wiki    existing PKB OKF Vault (`AgentPattern` and `SkillImpact` are ordinary concept types)
Skills  `$SWARMX_HOME/skills` (`SKILL.md` + `PURPOSE.md`)
```

This follows the three-layer and validate-before-promotion method described in
[WikiSkill](https://arxiv.org/html/2608.27454). It is an implementation of the paper contract, not
an integration with an official WikiSkill library.

## Truth and ownership

- DSH remains the only Raw truth. `DshRawTraceReader` returns a bounded contiguous event window from
  one native Session and persists nothing. Proposals retain only `sessionId + startSeq + endSeq`.
- PKB remains the only Wiki truth. `AgentPattern` and `SkillImpact` use the current open OKF concept
  profile, workspace isolation, provenance, single-use approval, SHA revision, history, index, and
  atomic write behavior. WikiSkill code does not add a `wiki/` directory or write PKB implicitly.
- The WikiSkill store owns only operational proposals and active skills. It returns a bounded
  PKB-ready `SkillImpact` draft after resolution; a caller admits that draft through the existing PKB
  surface and approval policy.

Ordinary inference Agents do not receive PKB Wiki pages from this feature. Maintainer/Proposer code
may explicitly read selected DSH traces and PKB concepts outside the inference run.

## Layout

```text
$SWARMX_HOME/skills/
├── .swarmx/
│   └── write
├── staging/
│   └── <proposal-uuid>/
│       ├── proposal.json
│       ├── outcome.json          # after deterministic resolution
│       └── <skill>/
│           ├── SKILL.md
│           └── PURPOSE.md
└── active/
    └── <sha256(preset+model)>/
        ├── .target.json
        └── <skill>/
            ├── .wikiskill.json
            ├── PURPOSE.md
            └── SKILL.md
```

All directories are owner-only. `staging/` is never registered as a DSH skill root. A target key is
derived from the exact bounded `preset + model` tuple, so target strings never become host paths.

## Proposal contract

One proposal creates or patches exactly one kebab-case skill. It contains:

- a UUID proposal id and `create|patch` operation;
- one exact target preset and model;
- a bounded DSH-compatible `SKILL.md` with matching `name` and a description;
- a non-empty `PURPOSE.md` narrative;
- at least one DSH Raw locator and one exact PKB concept id + SHA revision;
- for a patch, the exact current active revision.

The store appends a deterministic provenance section to `PURPOSE.md`. Reusing a proposal id with
identical content is idempotent; reuse with changed content fails. An external active edit makes a
patch revision stale instead of being overwritten.

## Validation and resolution

Evaluation is supplied by an explicit benchmark runner; the store does not start a background model
or invent a score. Baseline and candidate must name the proposal's same target, benchmark, and task
set and provide finite aggregate scores and positive run counts.

The gate is deliberately strict:

```text
candidate score > baseline score  → accepted
candidate score ≤ baseline score  → rejected
```

Acceptance rechecks the active revision under the writer lock. `PURPOSE.md` and operational metadata
are written first; the existing DSH-visible `SKILL.md` is replaced atomically last. Rejection changes
no active file. Cancellation or conflict before that final instruction boundary changes neither the
active skill nor PKB. Both verdicts produce a `SkillImpact` draft containing target, revisions,
benchmark identity, scores, verdict, and source locators.

Resolved proposal metadata remains under the invisible staging root for idempotent recovery. It is
operational state, not a second Wiki; the admitted PKB `SkillImpact` remains the durable synthesis.

## Runtime loading

`registerWikiSkillProvider(agentCtx, store)` registers one provider in the exact DSH Agent scope. On
each pre-step it derives the live Agent's preset and model, switches only to that target's active root,
and invalidates the catalog when the tuple changes. Discovery and body loading delegate to
`@deepseek-ai/dsh-skill-filesystem`; `@deepseek-ai/dsh-tool-skill` continues to own the catalog,
progressive loading, explicit user invocation, and model-facing rendering.

Consequences:

- a candidate is never visible to a normal Session;
- a skill validated for another preset or model is absent rather than reused;
- `PURPOSE.md`, provider paths, and target metadata are not catalog entries;
- project/user skills keep the existing DSH precedence rules;
- a Session without an exact model selection receives no evolved skill (fail closed).

## Deliberate limits

This layer supplies the storage, provenance, gate, and runtime activation boundary. It does not yet
choose a benchmark domain, generate proposals, schedule repeated inference runs, or automatically
admit the returned impact draft. Those are explicit campaign/orchestration responsibilities and must
not become background PKB or active-skill writes.
