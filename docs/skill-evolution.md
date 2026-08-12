# Skill Self-Improvement (evolution) Loop

This document is the operator guide for SwarmX's first safe, auditable,
rollback-able Skill self-improvement vertical slice. It implements the loop:

```text
explicit train/dev data + immutable baseline Skill
  -> granted optimization WorkItem (durable task runtime)
  -> optimizer worker proposes an immutable candidate (deterministic fake or DSPy/GEPA)
  -> static checks in Core (digest, lineage, budget, secret scan)
  -> paired baseline/candidate evaluation on a hidden holdout through the
     same real SwarmX execution path (Core in-process or Inspect evidence)
  -> deterministic gate verdict in Core (quality up, safety/failure/context not down)
  -> human approval
  -> compare-and-swap promotion of the active revision pointer
  -> only future Task/Session executions see the new revision
  -> rollback restores any retained revision
```

Learning never happens in an active request, never mutates a running Session,
and never edits Skill files or the persisted `SwarmConfig`. DSPy proposes;
evaluation provides evidence; TypeScript Core decides promotion.

## Concepts

| Concept | Meaning |
| --- | --- |
| Baseline Skill | The immutable `prompt_fragment` variant whose content the loop starts from. |
| Optimization request | A strict, secret-free request naming skill, parent revision, baseline and train/dev dataset digests, optimizer fingerprint, budgets, and proposer mode. It can never contain a holdout ref. |
| Candidate | Immutable Skill Markdown produced by the optimizer, stored as a content-addressed artifact. Its revision id is derived from the content digest; it cannot be edited in place. |
| Evaluation | Paired execution of baseline and candidate over a hidden holdout with identical model, tool policy, and budget. |
| Active revision | The pointer that resolves which Skill content future executions receive. Promotion is compare-and-swap; rollback restores retained revisions. |

## CLI workflow

```shell
# 0. Compute the optimizer launch environment digest (worker, lockfile,
#    evolution sources, and Python version) for the request file.
swarmx evolution digest

# 1. Create a request file (see Request file below) and evolve.
swarmx evolution evolve request.json

# 2. Evaluate against a hidden holdout (Core paired path, requires a model
#    via the usual Provider environment) …
swarmx evolution evaluate skc_xxxx --holdout holdout.jsonl -c config.json

# … or record evidence produced by the independent Inspect adapter:
swarmx evolution evaluate skc_xxxx --evidence evidence.json

# 3. Inspect status.
swarmx evolution status

# 4. Approve and promote (human gate, compare-and-swap).
swarmx evolution promote skc_xxxx --actor <you> --reason "Eval passed" --yes

# 5. Roll back when needed.
swarmx evolution rollback math-coach --revision r_<parent> --actor <you> --reason "Restore baseline" --yes
```

`reject` and `quarantine` mark candidates terminal without promotion; a
candidate that is currently the active revision cannot be quarantined (roll
back first).

Promotion enters the real execution chain: new executions resolve the evolved
active revision from the ledger.

```shell
swarmx send "question" --resolve-skill math-coach:math-coach:default
swarmx eval-run --config eval.json --resolve-skill math-coach:math-coach:default
```

`--resolve-skill <skillId>:<variantId>` binds the promoted revision to every
direct native Agent node of the new execution (by each agent's real name, so
multi-agent configs do not leak one delivery everywhere); already-constructed
Swarms and frozen Sessions are untouched. Delivery always validates the
requested `variantId`, the node-derived `targetAgentId` (`<harness>:<model>`),
and (when supplied) the target model fingerprint against the promoted
candidate manifest — or the retained revision metadata for rolled-back
baselines — and refuses mismatches: a Skill evolved for Agent A can never be
injected into Agent B. External ACP
Harnesses and non-native backends reject Skill delivery at construction.
Desktop Main can use the same `resolveActiveSkillDeliveriesForAgent` entry when
composing direct native executions.

## Request file

`swarmx evolution evolve request.json` reads a JSON file:

```json
{
  "schemaVersion": 1,
  "skillId": "math-coach",
  "variantId": "math-coach:default",
  "parentRevisionId": "r_<baseline revision id>",
  "baselineContentPath": "./baseline.md",
  "trainDatasetPath": "./train.jsonl",
  "devDatasetPath": "./dev.jsonl",
  "targetAgentId": "swarmx:model-x",
  "targetModelFingerprint": "model-x@v1",
  "optimizer": {
    "optimizerId": "deterministic.v1",
    "optimizerVersion": "1",
    "environmentDigest": "sha256:<launch digest>",
    "seed": 7
  },
  "budget": {
    "maxWallTimeMs": 120000,
    "maxModelCalls": 0,
    "maxTokens": 0,
    "maxArtifactBytes": 262144
  },
  "proposer": "none"
}
```

The CLI digests the baseline and dataset files, derives the canonical optimizer
config digest, and verifies the optimizer `environmentDigest` against the
launch digest computed from the worker source, `pyproject.toml`, `uv.lock`,
the evolution sidecar sources, and the resolved Python version
(`swarmx evolution digest` prints it). Datasets are JSONL records with stable
`id`/`caseId` values plus `input`, `target`, and optionally `keyword` (the
deterministic fake optimizer appends a mandatory rule embedding the first
keyword it finds in train) or `safetyFlag` (scorer input).

Optimizer ids:

- `deterministic.v1` — dependency-free fake optimizer in the worker; the
  vertical slice runs entirely without model credentials.
- `dspy.gepa.v1` — locked DSPy/GEPA optimizer in the private `swarmx.rsi`
  MCP server from the standard root `swarmx` distribution. The durable worker obtains
  and verifies the three granted artifacts, then calls the server's only tool,
  `swarmx_rsi_optimize`. `proposer: gateway` requires `--model-command <cmd>` on
  `swarmx evolution evolve` (a local credential-free command that reads a JSON
  request on stdin and writes `{content, usage: {totalTokens}}` on stdout;
  Provider secrets never enter the worker or RSI process); `proposer:
  deterministic` is the offline test mode. A zero `maxModelCalls` budget is a
  hard error — GEPA
  never silently defaults. Tokens are a hard budget: a zero `maxTokens` grant
  denies every model call before dispatch, exhausted budgets are denied before
  the next paid call, and the host gateway re-checks remaining tokens from
  durable receipts before each dispatch. The RSI server requests model calls
  with MCP sampling; the worker's private MCP client maps each request onto the
  same grant-checked `skill_evolution:model.generate` gateway. Both processes
  use MCP Python SDK v2. Their private stdio client explicitly initializes a
  pre-2026 session so a single optimizer tool call can make multiple
  server-initiated sampling requests; this legacy protocol mode is confined to
  the private worker-to-RSI link. The CLI launch digest includes the
  `src/swarmx/rsi` server, MCP client, and optimizer sources, plus the
  interpreter's installed `dspy` and `mcp` versions verified against the pinned
  root project. The interpreter's environment must pass the strict locked sync
  (`uv sync --locked --check --no-default-groups`), so a launch can never
  silently run against an unverified installation. The CLI auto-discovers the
  standard locked `.venv/bin/python` and there is no silent optimizer fallback.

The worker capability budget is keyed exactly like the durable receipts
(`skill_evolution:read_artifact`, `skill_evolution:model.generate`) so the
granted limits are actually enforced end to end.

## Data and security

- Only explicitly provided synthetic/golden JSONL is accepted. No automatic
  scanning of `~/.swarmx/projects`, the prior `~/.swarmx/sessions` layout, or
  external Agent histories.
- Candidates are untrusted: digest re-verification, lineage checks, budget
  bounds, and secret scans run before anything may be evaluated; a secret scan
  failure quarantines the candidate.
- Eval is read-only with no network side effects beyond the model calls the
  operator configured; the holdout never reaches the optimizer worker.
- Provider credentials never enter either Python process, private MCP stdio,
  artifacts, traces, or logs: MCP sampling crosses the grant-checked capability
  gateway and credentials are resolved inside a host-owned handler.
- Audit events record intent before promotion/rollback/decide effects and only
  carry ids, digests, metric summaries, budget usage, and correlation ids.
  An unavailable audit authority fails the promotion closed.

## Gates

- Static gate: content digest verified, worker lineage matches the request
  exactly (a missing worker manifest fails closed), parent/lineage checks,
  instruction delta present, size within budget, `prompt_fragment` delivery
  supported (derived from the candidate media type), no secrets.
- Evaluation gate (deterministic scorer first; a blind LLM judge is a later
  phase): candidate quality strictly exceeds baseline by a minimum margin,
  safety does not drop, failure rate does not rise, context tokens do not
  grow, latency/cost stay under configured caps, the sample count meets a
  minimum, and the sample-level improvement ratio is at least the configured
  threshold — a single mean move never declares success. Baseline and
  candidate run through the same real SwarmX path in the seeded-randomized
  order that is recorded; evaluation swarms are restricted to direct native
  agents without queen agents, hooks, MCP servers, tool nodes, or external
  backends.
- Promotion gate: default `human`; `policy` auto-promotion fails closed until
  canary and drift monitoring exist. Promotion content coordinates must equal
  the candidate manifest exactly and the revision id must be derived from the
  content digest — a forged or replaced artifact is rejected at ledger replay.
  A candidate that is currently the active revision can be neither rejected
  nor quarantined (roll back first), and a CAS failure records a failed audit
  outcome alongside the durable intent.

## Persistence

- Evolution ledger: append-only JSONL under `~/.swarmx/skill-evolution/` with
  strict per-kind secret-free records, idempotency, torn-tail recovery,
  immutable candidates/evaluations, and request-anchored CAS on promotion
  receipts. Promotion records may only carry `promote` or `rollback`
  decisions — a `reject`/`quarantine` receipt is refused at replay and can
  never create an active pointer. Candidates must start `proposed` and move
  through `evaluating`; an evaluation record requires the evaluating state, a
  bound per-sample evidence artifact, and a verdict consistent with its own
  metrics and gate, so a self-reported `eligible` can never be forged.
  Retained baselines cannot be re-anchored with different content, a
  quarantined revision cannot be re-activated by rollback, and the active
  revision can be neither rejected nor quarantined while it keeps delivering.
- Content: content-addressed blobs in the same root; large artifacts, datasets,
  and running state never live in settings.
- Task runtime: optimization WorkItems and artifacts live in
  `~/.swarmx/task-runtime/` with the standard lease/heartbeat/cancel rules.
- Audit: standard `~/.swarmx/audit/` chain. Intent is durable before any
  effect and an unavailable intent fails closed; a terminal outcome write that
  fails after the CAS is reported as failed and left for operator
  verification — it is never claimed completed.

## Inspect adapter

`evals/inspect/skill_eval.py` exposes the `skill_paired_eval` task. It runs
each hidden holdout case through the real `swarmx eval-run` path twice — once
with the baseline `prompt_fragment` delivery, once with the candidate — in the
seeded-randomized order it records, using the same
`--skill-delivery`/`--skill-content-path` flags that production execution
uses. The adapter rejects side-effect surfaces in the eval config itself
(queen agents, hooks, MCP servers, tool nodes, non-native backends), kills the
child on timeout, and the deterministic scorer matches the Core gate (target
match, optional `safetyFlag`, real `contextTokens` from the eval-run metrics).
The evidence file binds the holdout digest and sample count, the target agent
and target model fingerprint, and a config-derived runtime fingerprint;
`swarmx evolution evaluate --evidence --holdout <jsonl> -c <config>` requires
the actual holdout file and config, re-verifies the holdout digest, verifies
the evaluator's runtime fingerprint against the host-computed config digest,
and Core compares the evidence case-id set against the parsed holdout case-id
set exactly (count, uniqueness, and equality), rejecting train/dev overlap and
target-agent/model mismatches before anything can reach the promotion gate.
Inspect binds deliveries to a single named agent
(`--skill-delivery-agent`, refused for ambiguous multi-agent configs), requires
target agent/model fingerprint parameters matching the optimization request,
scores `passed` only when safety also passed, and kills the whole child
process group (escalating to SIGKILL) on timeout so grandchildren cannot
leak. The adapter never writes the active revision and never decides
promotion. A no-credential smoke runs the echo fixture end to end.

## Current limits

- First generation supports `prompt_fragment` delivery in direct SwarmX
  runtimes only; external ACP Harnesses, native plugins, and rules files are
  rejected explicitly.
- The CLI launch digest covers worker, project, lockfile, RSI server/client and
  optimizer sources, Python version, and the verified locked DSPy/MCP versions;
  production Desktop wiring uses the full unified `@swarmx/runtime`
  environment digest.
- Not yet enabled (require separate authorization): opt-in Session datasets,
  canary and drift monitoring, scheduled optimization, policy auto-promotion,
  tool/system-prompt/code evolution, and blind LLM judges.
