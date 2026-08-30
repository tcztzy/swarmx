# SwarmX Swarm

SwarmX Swarm is the opt-in scientific Team mode. DSH mounts it through the `dsh-swarm` preset;
Codex mounts the same coordinator through the owned SwarmX MCP server.

## Ownership

- The selected runtime owns Agents or Threads, model calls, Tool execution, approvals, and native
  conversation persistence.
- `SwarmCoordinator` owns only Team roster, task, mailbox, scheduling, and orchestration provenance
  under `$SWARMX_HOME/swarm/swarm.sqlite`; its DSH and Codex member handles are bottom adapters.
- Science Journal remains scientific truth. PKB remains personal synthesis. Swarm state is never
  written into the project checkout and private Team/Session ids never enter RO-Crate output.

## Identity and authority

On DSH, one top-level Session is the Team lead and Team id. Every member is a continuable direct
child and uses its immutable child Session id as the authority credential. On Codex, the owned MCP
server binds the lead to exact Codex-supplied native Thread metadata only after the bridge proves
that Thread belongs to the canonical workspace, then to an exact stable MCP session identity, and
otherwise to the current call-scoped request actor. It never invents a Thread identity. Native lead
authority and the private
reserved-member-to-native-child binding remain stable across MCP restarts; a child Thread's own MCP
call resolves to that reserved member, while the same Thread id on another workspace has no Team
authority. A stdio call without Thread or session metadata has no ambient continuation identity and
cannot inherit another call's Team. Codex member materialization uses native child Threads. Display names are
immutable labels, not credentials. Every Host mutation resolves the exact live adapter actor.
Lead-only actions are Team creation, member admission, task reassignment, interruption, effect
reconciliation, evidence admission, and archival.

Codex App Server may construct the required MCP process for a native child while its reserved
member is still provisioning. The one Electron primary owns a non-configurable platform recovery
service that starts before configurable Harness plugins and App Server and stops after both. An
auxiliary per-Thread MCP process opens only existing initialized storage, without migration or
projection rebuild, and never crash-recovers the journal on construction, first invocation, or
disposal. It only re-reads a committed exact native-Thread binding, so a pre-commit miss is retryable
and cannot fail the live provision. A present but inactive binding is denied rather than converted
into lead authority, and cached actors remain exact only while the same claim and active member phase
still exist. Bindings are immutable transactional claims per member and use exact conditional
release, so concurrent MCP processes cannot replace or delete one another's native handles. Native
child creation is a root-bridge operation: it is not cancelled by an MCP response disconnect, and
the bridge transactionally claims the returned Thread for the exact still-provisioning member before
it publishes the response. A lost response therefore leaves a durable handle for root recovery
rather than an invisible orphan. A bridge error or caller timeout is not proof that no native handle
exists, so the member remains provisioning until the root proves creation failed or cleanup finished.
Rollback uses a cancellation-independent cleanup operation: an exact source-tagged zero-turn Thread
is physically deleted because Codex cannot archive a Thread without a rollout, while a materialized
Thread is natively archived. Cleanup is single-flighted per exact member/Thread identity, and the
binding is released only after acknowledgement. When the returned handle is already claimed by
another member, the failed caller neither archives nor releases that Thread. Root
Codex startup alone reconciles interrupted provisioning and orphaned claims, and promotes only an
unarchived native child whose initial turn is observable. Cancellation, transient reads, and an
unsupported `thread/read` method fail loudly and retain the durable claim; they are not evidence that
the Thread is missing. The provisioning source tag is only a same-App-Server, zero-turn correlation:
blank Threads do not appear in `thread/list`, and Codex clears the tag after materialization. It is
never a persisted lookup index. Before the first turn the owning bridge retains the exact returned
Thread id; after the first turn the transactional member binding is the sole authority. Live synchronization also requires the same Thread id, workspace, and
unarchived state. A mismatch fails the member and revokes its attempts before further scheduling.

Every member also has one immutable orchestration role: `lead`, `legacy`, `researcher`,
`implementer`, `monitor`, or `verifier`. `legacy` is the explicit default for persisted members and
old `add_member` calls that predate role profiles; SwarmX does not guess their historical model.
Roles narrow Swarm actions and registered DSH/product MCP Tools but never grant authority: every such
mutation still requires the exact current Agent. Researchers may create bounded tasks and messages,
implementers may execute and submit their assigned attempts, monitors may read safe projections and
record only strict findings, and verifiers may record verdicts only for the exact current submission
assigned to them. Monitor/verifier roles have no guarded workspace-mutation, roster, budget-policy,
nested-delegation, or PKB authority. Codex-native Tools follow the separate boundary described below.

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
workspace-mutating DSH Tools and Codex product MCP calls, including Science and PKB mutations, only
while that exact Agent owns an active `write` attempt. The lead receives no ambient mutation
exemption: lead integration work must be an explicitly lead-assigned W task, while unowned tasks
continue to schedule only to members. At most one write attempt exists per Team. Declared write
scopes are coordination hints, not filesystem locks or rollback guarantees. Reassignment interrupts
the old owner and waits for quiescence before rotating the attempt.

Codex-native shell and file Tools remain owned and executed directly by Codex App Server; they do
not traverse the SwarmX product MCP or DSH Tool guard. Swarm role and attempt assignment therefore
coordinate those native actions but do not form a security boundary for them. Runtime sandbox and
approval policy remain the authority boundary for native Tool execution.

Every admitted participant W Tool body on a guarded carrier is enclosed by a durable effect intent
keyed by attempt plus the DSH Tool call id or derived Codex MCP request id. A successful Tool result
settles the intent. A Tool error, cancellation, timeout, or host crash after intent commit is
conservatively `uncertain`: the effect may have happened. An unresolved uncertain intent blocks later
W dispatch for that task. The lead must record a typed postcondition or operator observation as
`observed` (do not retry) or `absent` (retry may use a new call id). Repeated delivery of the same call
id never re-enters the Tool body. This is verify-before-retry and duplicate suppression at the Swarm
boundary, not general exactly-once execution: an opaque Tool may perform multiple internal effects,
an external system may ignore idempotency, and effects outside registered DSH or Codex product MCP
Tools are outside the guarantee.

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
and pending-mail limits. Queue insertion atomically owns idempotency comparison, pending-limit
enforcement, and target sequence. A separate delivery transaction atomically grants the first claim,
so concurrent processes cannot accept or enter delivery twice. An optional caller idempotency key makes repeated quiet/wakeup submission
return the original message status; reusing a key for different content is rejected. Quiet messages
inject context only into a resident member. Waking messages use native follow-up and may cold-resume
the target. A Codex lead may reload child handles for scheduling and lifecycle observation, but it
does not consume sibling mailboxes: queued mail is recovered only when the exact target Thread's MCP
carrier invokes Swarm. DSH wakeup remains queued until its parent handle is available. Delivery
intent is journaled before the native call. If the process loses the outcome after that point,
recovery leaves the message uncertain and does not redeliver it automatically; a human must inspect
the target before choosing a new message id. Messages that were queued but never entered delivery
remain eligible for exact-target recovery.

After platform-owner recovery, previously running tasks become `needs_attention` and lose their attempt;
started effects/admissions become `uncertain`, and SwarmX never silently replays potentially
non-idempotent work. The root carrier closes admission, aborts and settles admitted runtime work,
reconciles native member lifecycle, performs final crash-closed recovery, then closes SQLite. An
auxiliary Codex MCP process settles only its own operations/watchers and closes its client connection
without recovering or rebuilding shared state. DSH monitoring skips Teams with no runtime-owned
actor. A Codex native member watch/interrupt failure never becomes
an unhandled Promise or a fabricated idle acknowledgement. `archive` first commits one replayable
cross-process fence that closes admission, activation, scheduling, mailbox delivery, and ordinary
mutation. Provisioning already in flight must either publish and clean its exact native handle or
durably acknowledge that no handle was created; timeout, disconnect, and unclassified rejection keep
the fence open. Native archive/delete acknowledgement, member retirement,
and exact binding release cannot be split by a crash; the final archived edge is allowed only after
every child, non-terminal attempt, and started intent has drained. Archive is durable retirement, not
physical deletion of materialized conversation or domain state; deletion is limited to an exact
source-tagged provisioning Thread that has zero turns and was never published as a member.

Submitted and verifying attempts are equally unsafe to continue after process recovery: they become
`needs_attention`, retain immutable submission/verdict history in the attempt ledger, and are never
automatically re-run or re-verified. The v3 migration adds replayable attempt materialization, v4
adds exact transactional runtime-member claims, and v5 adds durable message idempotency and
target-sequence history that bounded Team projection trimming never forgets. Owner replay rebuilds
the message ledger deterministically with the earliest historical key as owner while leaving exact
runtime-member claims untouched. Legacy committed knowledge admissions that lack a valid matching
owner receipt migrate to `uncertain` without fabricating a receipt; v1/v2 events
rebuild with explicit legacy role/model/usage defaults. Migration acquires the write lock before
reading applied versions, is transactional and idempotent, rejects a database newer than the runtime,
and preserves owner-only `0700`/`0600` directory/file permissions.

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
