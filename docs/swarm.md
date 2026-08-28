# DSH Swarm

`dsh-swarm` is SwarmX's opt-in scientific Team mode. It keeps the complete `dsh-science` Agent
composition and adds one aggregate `swarm` Tool. Standard and ordinary Science sessions do not see
that Tool.

## Ownership

- DeepSeek Harness owns Agents, continuable subagents, Sessions, model calls, Tool execution,
  approvals, persistence, and the browser conversation surface.
- `@swarmx/dsh-swarm` owns only Team roster, task, mailbox, scheduling, and orchestration
  provenance under `$DSH_HOME/swarm/swarm.sqlite`.
- Science Journal remains scientific truth. PKB remains personal synthesis. Swarm state is never
  written into the project checkout and private Team/Session ids never enter RO-Crate output.

## Identity and authority

One top-level Session is the Team lead and Team id. Every member is a continuable direct child and
uses its immutable child Session id as the authority credential; display names are immutable labels,
not credentials. Every Host mutation resolves the exact live Agent object. Lead-only actions are
Team creation, member admission, task reassignment, interruption, effect reconciliation, evidence
admission, and archival.

Every member also has one immutable orchestration role: `lead`, `legacy`, `researcher`,
`implementer`, `monitor`, or `verifier`. `legacy` is the explicit default for persisted members and
old `add_member` calls that predate role profiles; SwarmX does not guess their historical model.
Roles narrow actions but never grant authority: every mutation still requires the exact current
Agent. Researchers may create bounded tasks and messages, implementers may execute and submit their
assigned attempts, monitors may read safe projections and record only strict findings, and verifiers
may record verdicts only for the exact current submission assigned to them. Monitor/verifier roles
have no workspace-mutation, roster, budget-policy, nested-delegation, or PKB authority.

`add_member` may carry bounded `agentOptions.provider`, `agentOptions.model`, and
`agentOptions.maxTokens` plus an attempt-budget template. Provider/model labels are restricted
identifiers, not deployment objects, and no credential or environment configuration enters the
Swarm journal. The member policy is journaled before materialization, sent to the child's initial
`startContinuable`, and re-applied from that durable policy through the child-scoped DSH request seam
on cold resume. This second step matters because the current DSH continuable descriptor persists
provider/model but intentionally treats `maxTokens` as activation-local. Missing options preserve
the existing deployment defaults.

The lead is not created through `add_member`; Team creation snapshots the lead Agent's observable
provider/model/max-token options so lead work is not absent from attempt economics. A policy
snapshot records requested/observed/legacy-default provenance and is immutable after admission.

## Tasks and workspace safety

Tasks form a bounded immutable-dependency DAG and have one explicit work class:

- `read` (R) searches, analyzes, or produces candidates without changing the checkout, an
  experiment, an external system, Science Journal, or PKB. Independent R tasks may run in
  parallel.
- `write` (W) may change the checkout, an experiment, or an external system. One Team has at most
  one effective W attempt, even when declared write scopes do not overlap.
- `knowledge` (K) reviews a candidate for promotion. K analysis may overlap like R, but K cannot
  complete through ordinary task settlement: only the explicit evidence-admission operation below
  can commit and complete it.

Every transition uses an exact task revision. Executing ownership is one random attempt id; member
settlement must present both the current revision and attempt. A stale attempt cannot change the
board.

Non-knowledge work uses an explicit acceptance boundary. A task may declare a bounded acceptance
summary, required checks, expected artifact labels, and rubric plus one assigned verifier. The
owner of the exact current attempt may `submit_task` with a bounded summary, client-safe artifact
locators, and SHA-256 evidence digests. Submission changes the task to `submitted`, closes its write
authority immediately, and preserves the attempt for audit. It does not mean completion.

The assigned exact verifier or exact lead may claim `start_verification` and then
`record_verdict`. A `pass` alone completes the task. `fail` records a rejected terminal attempt that
the lead may explicitly reopen or reassign; `uncertain` and `escalate` enter lead review. Verdicts
carry bounded named check results and rationale. Task revision, attempt id, and submission id fence
every verification transition, so an old verdict cannot overwrite a new submission. A member cannot
verify its own submission. The exact lead may verify lead-owned work only as the explicit
single-Agent compatibility path, and the ledger/UI label that verdict `degraded` rather than
independent. Knowledge tasks keep their existing owner-preserving evidence-admission completion
path.

Team participants share one checkout. Read tasks may overlap. Agent-scoped Tool guards admit
`write`, `edit`, shell, Science mutation, and similar side-effecting tools only while that exact Agent
owns an active `write` attempt. The lead receives no ambient mutation exemption: lead integration
work must be an explicitly lead-assigned W task, while unowned tasks continue to schedule only to
members. At most one write attempt exists per Team. Declared write scopes are coordination hints,
not filesystem locks or rollback guarantees. Reassignment interrupts the old owner and waits for
quiescence before rotating the attempt.

Every admitted participant W Tool body is enclosed by a durable effect intent keyed by attempt plus DSH
call id. A successful Tool result settles the intent. A Tool error, cancellation, timeout, or host
crash after intent commit is conservatively `uncertain`: the effect may have happened. An unresolved
uncertain intent blocks later W dispatch for that task. The lead must record a typed postcondition or
operator observation as `observed` (do not retry) or `absent` (retry may use a new call id). Repeated
delivery of the same call id never re-enters the Tool body. This is verify-before-retry and duplicate
suppression at the Swarm boundary, not general exactly-once execution: an opaque Tool may perform
multiple internal effects, an external system may ignore idempotency, and effects outside registered
DSH Tools are outside the guarantee.

## Attempt budgets and economics

Each admitted attempt snapshots the member role, model policy, and budget template. The ledger
aggregates start/progress/submission/verification/end times, wall time, turn and Tool-call counts,
provider-reported input/output/cache tokens when available, client-safe artifacts/digests, verdict,
and terminal/escalation reason. It never copies prompts, assistant responses, message bodies, shell
output, file content, credentials, environment values, absolute paths, or raw provider payloads.

`maxWallMs` is a hard Host deadline: warning and exhaustion are deduplicated, exhaustion uses the
existing Agent cancel/interrupt path, revokes the attempt, and leaves the task `needs_attention`.
`maxTurns`, `maxInputTokens`, and `maxOutputTokens` are observed soft ceilings. Provider usage may
arrive only after one request, so these limits can overshoot that request; once observed exhausted,
the Host prevents further attempt work and escalates instead of claiming pre-request isolation.
When an adapter supplies no stable usage, token state is `unknown`; SwarmX never parses UI text or
invents precision. Local/open models still contribute wall-clock and later local-compute fields even
when API token cost is unavailable. Heterogeneous routing is therefore an experimentable capability,
not a claim that Teams are inherently cheaper or better.

## Event-driven monitoring

The deterministic Host monitor evaluates journal transitions, Agent lifecycle/status, task
progress, mailbox pressure, role-denied Tool calls, submissions, verification outcomes, and usage
events. It produces bounded stable findings with severity, code, safe subject, summary, and one of
`none|notify|interrupt|needs_attention|lead_review`. Findings cover approaching/exhausted wall
budgets, observed token/turn exhaustion, running-without-progress, long-held write attempts,
mailbox pressure, abnormal/provisioning exits, repeated verification failure, missing required
submission evidence, unattributed/unknown usage, and monitor/verifier policy violations. Dedupe keys
make evaluation idempotent; monitor findings do not recursively trigger more findings.

An optional semantic monitor is disabled by default. When enabled, only deduplicated submission,
repeated-failure, stall, budget, conflict, or lifecycle events wake a continuable `monitor` member.
It receives bounded path-free task/submission/check/usage summaries rather than transcript or
mailbox bodies, and may return only the strict finding vocabulary. It cannot run commands, write the
workspace, modify guards/budgets/verifier policy, or settle implementation tasks. There is no
standing model polling loop.

The Host service configuration exposes `monitorStallMs` (default five minutes) and
`semanticMonitor` (default `false`). Enabling the latter does nothing unless the Team has one active
`monitor` member; delivery failure is recorded as a bounded finding and never rolls back an already
committed task submission or verdict.

## Evidence admission

Task completion, peer messages, Tool results, and orchestration events are candidates, never Science
facts or PKB knowledge. `admit_knowledge` is a lead-only, exact-attempt operation for one active K
task. It requires a stable admission UUID, one or more typed source locators, and a verified record
containing method and time. Swarm stores only the request hash, source locators, verification state,
and owner receipt; it does not copy the proposed claim, evidence text, or PKB body into its journal.

The first executable boundary supports two owner-preserving commits:

- `science_evidence` calls the existing Science `linkEvidence` operation. Sources must all be
  workspace-local Science entity ids; Science validates the claim, project, and source ownership and
  remains the only scientific truth.
- `pkb_concept` asks for DSH `allowed-once` approval and calls the existing PKB Vault writer. The
  Vault remains the only PKB truth and records the admission UUID as its owner-side idempotency key.

The admission intent is durable before the owner call. If the owner commits but the Swarm receipt is
lost, recovery marks the admission uncertain and revokes the task attempt. A later current attempt
may retry the same admission UUID: Science and PKB verify the same idempotency key before returning
the existing owner receipt. A different payload under that UUID fails closed. Only a committed owner
receipt completes K. Unverified/rejected candidates, ordinary Swarm completion, and Swarm recovery
can never create Science or PKB state.

These rules define the system properties:

- **Authority safety:** only the exact current Agent/attempt can start effects; only the exact lead
  can admit knowledge or resolve uncertain effects.
- **Stale-attempt noninterference:** rotation revokes old-owner settlement, Tool, and
  evidence-admission authority; only the lead may later reconcile an old effect against its recorded
  attempt and the task's exact current revision, and quiescence precedes rotation.
- **Crash-closed recovery:** active attempts are revoked, started effects/admissions become
  uncertain, and neither is replayed automatically.
- **Epistemic noninterference:** R/W/K output remains orchestration data until a verified,
  owner-committed K admission exists.

## Mailbox, recovery, and retirement

Messages are durably queued before delivery with stable ids, bounded content, per-target ordering,
and pending-mail limits. An optional caller idempotency key makes repeated quiet/wakeup submission
return the original message status instead of delivering twice; reusing a key for different content
is rejected. Quiet messages inject context only into a resident member. Waking messages use DSH
continuable follow-up and may cold-resume the target. Delivery intent is journaled before either
call. If the process loses the outcome after that point, recovery leaves the message uncertain and
does not redeliver it automatically; a human must inspect the target before choosing a new message
id. Messages that were queued but never entered delivery remain eligible for recovery.

After process recovery, previously running tasks become `needs_attention` and lose their attempt;
SwarmX never silently replays potentially non-idempotent work. Plugin disposal closes admission,
releases long-poll waiters, aborts and settles admitted runtime work, revokes member-scoped
capabilities, then closes SQLite. `archive` is an honest durable retirement operation; it is not
physical deletion.

Submitted and verifying attempts are equally unsafe to continue after process recovery: they become
`needs_attention`, retain immutable submission/verdict history in the attempt ledger, and are never
automatically re-run or re-verified. The v3 migration adds replayable attempt materialization;
v1/v2 events rebuild with explicit legacy role/model/usage defaults. Migration is transactional and
idempotent, a database newer than the runtime fails loud, and owner-only directory/file permissions
remain `0700`/`0600`.

## Browser surface

`@swarmx/dsh-ui-swarm` reuses the generic per-Session Side View. Its strict Remote returns only
bounded member role/model/budget state, task status, recent verdict/check/economic/escalation
summaries, counts, and revision. It exposes no mutation method, path, workspace title, raw Session or
attempt id, artifact locator, message body, prompt, or credential. Long text/history is bounded,
English and Chinese share one complete locale contract, and one revision-based bounded long-poll is
active only for the rendered Session and is cancelled on switch, unmount, or HMR.

## Team sizing contract

SwarmX never creates or expands a Team automatically. The executable preflight evaluator reports
three measured inputs from a proposed R/W/K DAG: cognitive parallel width (R plus K candidates at
the same dependency depth), pairwise W scope-conflict rate, and effectful Tool density. It recommends
more than one member only when cognitive width is at least two. The recommendation is capped by that
width; W work contributes at most one serial lane, and high conflict/tool density is reported as
serialization pressure rather than hidden behind an adaptive policy. The lead or user remains the
decision maker.

## Verification and research boundary

The repository model checker exhaustively explores bounded attempt rotation, crashes, uncertain
Tool results, duplicate delivery, recovery, and knowledge admission. Its regression benchmark runs
the enforced transition system beside an intentionally prompt-only baseline and reports safety
violations, duplicate effects, recovery replays, knowledge pollution, and coordination writes.
This is an executable bounded model and fault benchmark, not a proof for arbitrary Tools or an
external environment. TLA+/Alloy remains a later route if the state space or cross-process protocol
outgrows the repository model.

The contribution claimed here is the composition of durable DSH Team attempts, Tool-effect fencing,
and explicit Science/PKB admission. It does not claim to originate shared-tree concurrency control,
policy-state serializability, recovery, verified Tool calls, or agent provenance; those are treated
as prior art in [CoAgent](https://arxiv.org/abs/2606.15376),
[Stateful Governance](https://arxiv.org/abs/2608.02764),
[AgentRewind](https://arxiv.org/abs/2608.14380),
[Verified Tool Calls](https://arxiv.org/abs/2608.02645),
[PROV-AGENT](https://arxiv.org/abs/2508.02866), and the 2026
[agent provenance survey](https://arxiv.org/abs/2606.04990).
