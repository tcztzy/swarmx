# Roadmap

This file contains unfinished or partially verified work. It is not part of the
product contract in `SPEC.md`. Publication evidence is the ordering rule until
the research direction stops or the P5 artifact is complete; product breadth is
deliberately deferred. See the
[publication-first research strategy](docs/publication-research-strategy.md)
for the ontology, information bounds, literature record, experiments, and stop
conditions.

For code items, remove an item once its behavior is implemented, documented
where needed, and covered by focused tests. Research and submission items close
when their stated decision or artifact is recorded.

## P0: submit or archive the manuscripts already present

- [ ] Resolve the `SwarmX Project` author placeholder and choose a venue for
  `harness-context-engineering-review.tex` and
  `recursive-harness-self-improvement-review.tex`.
- [ ] Repeat each manuscript's closest-work search through the submission date,
  resolve or retain the documented author-confirmation gaps explicitly, and
  freeze source, bibliography, claim audit, and reproducible PDF build.
- [ ] Submit or archive each review before starting unrelated product features;
  this lane does not wait for the empirical paper.

## P1: freeze the formal claim before code

- [x] Define the agentic execution system without changing SwarmX Agent
  identity. Distinguish latent world, observable history, belief, task contract,
  Task instance/WorkItem, Run, residual task state, normative state, action
  specification, intention, request, authorization, occurrence, effect
  occurrence, observation, evidence, ledger claim, artifact, Session, and
  executor with explicit identity and disjointness rules.
- [ ] Define continuation equivalence, exact and approximate sufficiency,
  exact quotient size `K_Z`, declared incompatible response-set size `K_perp`,
  hard memory capacity `B_cap`, latent target support `K_star`, task distortion,
  and verification side information.
  Record all counting, Fano, and update-channel assumptions; do not present
  POMDP beliefs, causal states, rate-distortion, or requisite variety as new.
- [ ] Hand-check four compression twins where different observable histories
  require different responses but a named projection collides, and four
  observability twins where different latent worlds induce the same complete
  observable history and therefore require new verification.
- [ ] Freeze the hypotheses, outcome vector, independent
  `H_prefix x K_perp x B_cap x W_V` factors, where `W_V` is the declared
  verification-channel condition and `V` is its transcript random variable.
  Stop before implementation if a nontrivial finite task family, exact
  quotient, right-congruence check, and two isolated bottleneck families cannot
  be specified.

## P2: test the information bound without a language model

- [ ] Implement a dependency-free transactional simulator with only
  `inspect / apply / verify / compensate / finish`. Independently vary forced
  prefix depth `H_prefix`, incompatible response variety `K_perp`, hard memory
  capacity `B_cap`, latent target variety `K_star`, verification-channel
  condition `W_V`, distractor entropy, and uncertain or irreversible effects.
- [ ] Include `H_prefix=64, K_perp=2` versus
  `H_prefix=8, K_perp=256`; exhaustively verify quotient size and right
  congruence on small cases, then add per-step risk separately.
- [ ] Compare a latent-world oracle, full observable history, exact `Z` and `C*`,
  bounded suffix, a specified capacity-limited type-agnostic compressor,
  belief-only state, typed operational state, support-preserving typed-field
  interventions, reliance ablations, and read-only verification.
- [ ] Measure success, constraint violations, duplicate/omitted effects, false
  completion, deadlock, oracle regret, collisions, first divergence,
  response-decoding accuracy, `I(C*;M)`, and `I(C*;V|M)`. Report hard-state
  capacity, mutual information, and serialized bits/bytes/tokens separately.
- [ ] Keep this phase within three authored files under `evals/`, zero production
  modules, and zero model calls. Update `docs/codebase/` for every new source or
  test and run `pnpm docs:check`.
- [ ] Stop unless the exact capacity boundary, inability of memory to resolve
  observability twins, and type-specific failures appear as predicted and the
  information-margin model improves held-out predictive log loss by at least
  10% over prefix depth/transcript size/calls alone.

## P3: instantiate one SwarmX request-binding counterexample

- [ ] Add at most one focused experiment/test module around the existing
  capability gateway and `TaskSideEffectReceipt`, using a non-idempotent mock
  world and no model call.
- [ ] Test same-key/same-arguments replay, same-key/different-arguments semantic
  collision, mutation followed by lost response, explicit non-commit,
  corrupted committed detail, restart/replay, and Session link changes.
- [ ] Score the safe recovery decision `execute / replay / reconcile / reject`,
  world mutation count, duplicate effects, unresolved outcomes, and retained
  bytes. Use Session-only and audit-only projections as negative controls.
- [ ] Change zero production modules in the experiment and make no bug claim
  before it fails as predicted. If the candidate input-identity gap is
  confirmed and P2 survives, hash a fixed canonical, versioned serialization of
  `{capabilityId, operation, arguments}`, exclude retry-varying metadata, and
  add only the smallest replay check in existing Core modules. Call this a
  capability-call fingerprint, not an action or effect identity.
- [ ] Treat old receipts without the capability-call fingerprint as
  `unknown / reconcile`; never invent or backfill absent arguments. Account
  explicitly for the strict versioned event schema, add focused compatibility
  tests, update the codebase map, and run `pnpm docs:check` before any repair
  lands.

## P4: test model and executor external validity

- [ ] Only after P2 and P3 advance, compare transcript, natural-language
  summary, belief-only, typed, and oracle states once under matched hard-state
  capacity and separately under matched serialized bytes/tokens, using one
  fixed model, fixed tools, deterministic validators, and complete trajectories.
- [ ] Add a held-out task family and second model only after the first result is
  stable. Report paired uncertainty, bytes/tokens, verification cost, first
  divergence, and every preregistered failure class.
- [ ] Test executor replacement or cross-harness handoff only as a 2-by-2
  external-validity condition: executor same/different crossed with
  normative-causal state preserved/ablated. Do not create a handoff format or
  redesign `SwarmConfig`.
- [ ] Exclude private chain-of-thought and any hidden Harness state from transfer
  or scoring.

## P5: submission artifact

- [ ] Repeat the closest-work search through the submission date and have one
  control/information-theory reader and one formal-ontology reader attack the
  definitions, assumptions, and novelty boundary.
- [ ] Freeze task families, baselines, versions, checksums, exact commands,
  output schemas, exclusions, negative cases, and limitations. Add a manifest
  only for fields required to regenerate a declared result.
- [ ] Reproduce from a clean checkout and publish only synthetic, licensed,
  secret-scanned artifacts sufficient to regenerate every table and figure.
  Keep generated runs and local artifacts out of Git.
- [ ] Write around one result: safe continuation is jointly limited by capacity
  to preserve observed response-relevant distinctions and the informativeness
  of verification for resolving latent ambiguity, while forced prefix depth
  and accumulated risk are separate variables. Treat SwarmX as one empirical
  instantiation, not the theory.

## Research stop condition

- [ ] If the exact quotient, independent `H_prefix x K_perp` construction,
  isolated compression/observability twins, information-margin prediction,
  type-specific ablations, or real request-binding/replay case fails its frozen
  gate, preserve the negative result and select a new question from fresh
  literature. Do not add product features to rescue the thesis.

## Deferred product work

The remaining items are intentionally below the paper evidence gates. Resume
one only when an experiment identifies it as a required variable, baseline,
measurement, or failure repair.

### Carryover work to close

- [ ] Make registered Harness runtimes optional until a user selects them or a
  configured Agent requires them.
- [ ] Finish native Anthropic Messages and OpenAI Responses execution in the
  direct SwarmX Harness, using compatibility bridges only as fallback.
- [ ] Finish grouped Provider Model catalog persistence, safe Codex discovery,
  routing metadata, and effort metadata.

### Known native-tool parity gaps

- [ ] Add Claude-compatible `PowerShell` only with a Windows-native sandboxed
  process host.
- [ ] Add Claude-compatible `SendMessage` only with concurrently live teammate
  identities, mailboxes, and lifecycle control.
- [ ] Add Claude-compatible `Workflow` only with a deterministic, persisted,
  resumable workflow VM.

### Durable task service boundary

- [ ] Add optional login startup plus graceful upgrade/handoff for the on-demand
  local supervisor, without changing the canonical task authority or format.
- [ ] Add continuous lease-expiry/watchdog scheduling around the existing
  startup recovery path, including crash injection and concurrent-controller
  fencing tests. Route cancellation across controller processes so it does not
  depend on the process that originally launched the worker.
- [ ] Move synchronous event-store I/O off Electron Main's latency-sensitive
  path while preserving the single-writer lock, fsync boundary, and strict
  replay semantics.
- [ ] Materialize artifact-backed execution checkpoints into the immutable
  runtime store. The current app-attached slice resumes only inline checkpoint
  payloads and fails closed on artifact-backed checkpoint payloads.
- [ ] Implement production capability-gateway adapters with per-operation
  grants, Project/tool containment, Provider request brokering, audit receipts,
  and sandbox policy. Keep plaintext Provider credentials out of worker
  environments and protocol messages.
- [ ] Add a formal reconciliation API for `uncertain` external-effect receipts;
  current replay fails closed and requires manual intervention rather than
  inventing success or automatically repeating an unknown effect.
- [ ] Add production worker isolation and artifact-ingestion hardening,
  including OS sandbox profiles and defenses against filesystem races around
  worker-produced files.
- [ ] Add a real uv-managed, locked-environment-to-worker integration gate and
  a setup/run environment lock so concurrent repair cannot replace the selected
  interpreter after launch verification.

The pre-simplification status and its old `Txxx` links remain available with
`git show 780fb8e:SPEC.md`.
