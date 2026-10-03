# Long-term work in the Host

See [product direction](product-direction.md) for the objective and [ROADMAP](../ROADMAP.md) for unfinished work.
The Host owns work cycles, items, admission budgets, decisions and acceptance. Native sessions,
Domain artifacts and execution records retain their existing owners; work records contain references.
Work state is stored in `$SWARMX_HOME/work/work.sqlite`, keyed by the canonical execution directory.
Separate directory keys share neither work items nor acceptance history, even in the same product home.

## Contract

- A cycle shares a USD budget and concurrency limit across work and configured background review.
  A root attempt reserves its runtime budget transactionally before dispatch; an omitted amount
  uses the available cycle budget. Descendants share that runtime identity and add no separate
  monetary reservation. Runtime `budgetUsd` and `timeoutMs` are independent of Agent configuration.
  Claude receives the available query budget through SDK `maxBudgetUsd`; other harnesses retain
  Host admission/cancellation without an equivalent native USD cap. This is not a universal hard
  spending limit, and actual observed spending can exceed admission estimates.
- Each item has an external goal, versioned acceptance criteria, task class, priority, value,
  deadline, risk, dependencies and attempt references. Splitting an execution creates no extra value.
- An Agent configuration contains harness, model, optional effort and profile; `id` identifies a
  saved preset and defaults to `temporary` when omitted. There is no configuration version or
  embedded budget. Native tools/skills belong to the harness; effort belongs to model selection.
  The user can choose a preset or supply a temporary configuration before execution. All choices
  remain subject to Host grants. Requested configuration is the shared identity; an explicit
  native conflict fails instead of silently accepting another configuration.
- Manual mode executes the selected Agent once. Managed mode uses the explicitly selected
  supervisor as the parent Agent. Before dispatch, the Host gives it the actual `swarm.prepare`
  result, goal, criteria, budget and candidate evidence. Its native tool loop delegates and decides
  subsequent messages from returned results; the Host does not create another judging loop.
  Prepared instructions are validated before reserving a runtime; an oversized input starts no
  Agent and holds no budget. The prepared task is not repeated outside the preparation result.
  Supervisor completion never accepts work. An untried configuration stays unknown.
- Stop and timeout propagate through the root runtime into every delegated call, including calls
  entering through the execution-bound MCP credential. Finishing asynchronous preparation after
  cancellation cannot dispatch a later model or tool request.
- Native completion, failure and cancellation remain execution outcomes. Only trusted Host code
  acting for a user or validator records acceptance. Agents may inspect work and submit pinned
  domain references; they cannot create budgets or approve themselves through product tools.
  Feedback may arrive after execution or a memory review and may correct earlier feedback.
  Each saved feedback fact is published once to the execution journal. Publication can be
  replayed after an interrupted write without duplicating the fact. Authorized automatic
  learning reviews newly arrived feedback even when the original run was already reviewed;
  the exact included feedback IDs are acknowledged only after a completed review.
- Each independent native query and tool charge counts once. Agent totals include all Host
  descendants and are used in configuration comparisons; cycle spending sums independent charges
  rather than overlapping Agent totals. Missing native cost leaves totals incomplete and retains
  the relevant reservation until trusted reconciliation. Unpriced tool calls initially cost zero
  and remain marked unpriced. All totals derive from existing logs; see [execution logs](execution-log.md).
  Cycle tool totals also include preparation calls before native execution starts. A reconciled
  invoice replaces that attempt's native estimate in work accounting and configuration comparisons.
  A finished child with unknown cost blocks further delegation until its bill is reconciled.
- Startup does not execute queued or interrupted work. Explicit continuation first reconciles
  recorded terminal events. An unknown execution outcome continues to occupy capacity, including
  review capacity, even after its invoice is reconciled. A trusted Host caller must confirm the
  external outcome before retrying; missing cost separately retains the monetary reservation.
  A reservation without a native start can still be preparing in another Host connection, so
  restart alone cannot establish that dispatch stopped. An observed cancellation before dispatch
  establishes termination but leaves unreported preparation costs unknown.
- Shared review cost is charged once to the earliest managed source item in its review batch.
  Other source items retain the review's source-run references; they do not duplicate the charge.
  A managed cycle must explicitly allocate a per-review reservation before background learning runs.
- Unobserved native preparation, catalog and internal calls remain coverage gaps. Trusted invoice
  reconciliation is separate from observed SDK estimates; no manual human-resource registry is used.

## Desktop controls

The conversation header opens a Long-term work side view without replacing the conversation or
its draft. A user creates a named cycle with a shared USD budget and optional Agent presets.
Each goal selects manual or managed mode, its execution Agent or supervisor, and optional runtime
budget and timeout. Temporary choices do not require a preset. Saving does not execute work.
Goals share the cycle budget; each has written acceptance criteria and a criteria version.

Start dispatches the selected goal through `ProductServices.runWork`. Run next item selects an
eligible goal by dependencies and priority through the desktop `startNext` command, using that goal's saved mode,
Agent and runtime limits. It does not bypass acceptance of completed work. Native confirmation
requests appear in the work view using the same schema fields as conversation confirmations.
Stop requests cancellation; the view continues to show execution as active until cleanup ends.
Closing the view does not cancel work. Closing the Host cancels owned execution; reopening reads
durable state and requires explicit continuation after unresolved effects and costs are reconciled.

The view distinguishes observed spending, held reservations and available budget, including
unknown costs. Budget edits require the displayed prior budget and cannot undercut already spent
or reserved money. Execution results and exact submitted artifact revisions remain inspectable
before a user records an acceptance verdict and report. Desktop acceptance is attributed to the
user, never to a validator. Corrections explicitly supersede the prior user verdict. Revising the
goal and criteria uses a new criteria version under the same work identity; it does not erase
old attempts, spending or feedback, and does not start another execution automatically.

Work commands use the explicit `work.read` and `work.command` preload methods. Only a trusted
top-level application window can invoke them. Agents retain the existing status/submission tool;
they cannot invoke desktop administration, answer native confirmations or accept their own work.

## Host APIs

Create and attach `ProductServices` as usual, then use its `work` manager. Administration methods
reject an active Agent caller. The desktop exposes these controls through authenticated window IPC.

| Operation | Contract |
| --- | --- |
| `work.createCycle({id, project, budgetUsd, configurations, concurrency?, reviewReserveUsd?})` | Save cycle limits and optional Agent presets. Configurations contain `id`, `harness`, `model`, optional `effort`/`profile`. Concurrency defaults to one; reviews require an explicit nonzero allowance. |
| `work.createItem({id, cycleId, goal, criteria, criteriaVersion, taskClass, mode, configuration?, supervisor?, runtime?, ...})` | Add work, its execution mode, Agent choice and runtime limits; retain priority, value, risk, deadline and same-cycle dependencies. Managed mode requires an explicit supervisor. |
| `work.setBudget({cycleId, expectedBudgetUsd, budgetUsd})` | Change a cycle budget after checking its displayed prior amount and existing commitments. Active or unresolved execution prevents a budget change. |
| `services.runWork(workId, signal, interact?, options?)` | Prepare context, check cancellation, then select the requested Agent and transactionally reserve its runtime budget with temporary start options applied. Dispatch recorded native execution with the preparation result in its input. A defer/stop decision starts no Agent. |
| `work.command({action: "startNext", cycleId})` | Inspect queued/blocked items in priority, earliest deadline, value and creation order. Skip unmet dependencies and unresolved attempts; start one eligible goal through the same lifecycle as `start`. |
| `work.accept(feedback)` | Save a versioned, artifact-pinned user/validator verdict and report for a finished primary attempt. Content, scientific-evidence and user feedback can accept work; structural validity alone cannot. |
| `work.revise({id, expectedCriteriaVersion, criteriaVersion, goal, criteria})` | Require the expected version and a new version while the item is not running. Queue the revised goal under the same work identity and budget; retain historical attempts and feedback. |
| `work.reconcileCharge({id, reservationId, costUsd, source: "invoice", reference})` | Replace an attempt's observed estimate or unknown cost with a trusted reconciled amount; an identical repeated ID is a no-op and a conflicting duplicate fails. Reconciliation records cost, not successful execution. |
| `work.reconcileOutcome({reservationId, outcome, reference})` | Record a trusted confirmation of completed, failed or cancelled external execution after an unknown outcome. Active or still-preparing reservations cannot be confirmed. Identical confirmations are idempotent and conflicting confirmations fail; the original journal, unknown costs and independent acceptance requirements remain intact. |
| `work.snapshot(cycleId)` | Reconcile journal records and return items, attempts, decisions, feedback, execution/tool costs, budget/coverage and accepted/partial configured value. User-configured value is not converted into cash or measured research benefit. |

Feedback includes `id`, `attemptId`, `criteriaVersion`, `verdict`, `accepted`, `fraction`, `layer`,
`source`, `evaluator`, `evaluatorVersion`, `report`, optional `artifacts`, `supersedes` and
`intervention`. Verdicts are passed, failed, partial or insufficient-evidence. Failed work has zero
accepted value; justified insufficient evidence can be accepted. Replacing feedback on the same
attempt/layer requires its previous ID in `supersedes`. Late feedback for historical attempts or
criteria versions remains visible but cannot accept a newer attempt or revised goal. The first
acceptance timestamp is retained across revisions. Native completion does not create feedback.

Selection evidence includes task class, risk, criteria revision, requested configuration, sample
counts, latest feedback, complete descendant cost, elapsed time and outcomes. Missing prices cannot
become evidence that a configuration is cheap. Decisions preserve the candidates, evidence and reason;
there is no versioned policy-injection interface.

Managed Agents call `work` with `{action: "status"}` or `{action: "submit", artifacts}`. Status and
`swarm.prepare` expose current work context, budget and evidence. Submission requires `science.read`
and a trusted configured domain provider. The provider must return the exact submitted ID and
matching revision; missing providers and mismatches fail closed. All provider grants are checked
before the callback. Submission creates no acceptance authority. See [domain references](domain-projects.md).
Delegated calls still require same-run preparation and a reason. They must match an admitted
configuration and share the root runtime's budget. Previously managed native sessions cannot escape
their original work identity and budget through another entry point.

## Initial scope

Additional native limits, fair scheduling, external-effect retry authorization and live-model
comparisons remain in [ROADMAP](../ROADMAP.md). Callers opt into managed work by creating a cycle
and item. Late acceptance updates selection evidence and enters the existing Memory review
backlog under the original execution's learning grants. Disabling learning or removing current
authority prevents review dispatch. Its extra execution consumes the same cycle's review budget.
