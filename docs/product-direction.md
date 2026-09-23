# Product direction

SwarmX is a long-term research work system. It combines native Agents, models, tools, skills
and execution environments to pursue research goals across tasks and sessions. The product
direction is to improve the value of accepted research outcomes relative to the full cost of
producing them, while respecting user authority, evidence requirements, deadlines and budgets.
This is an optimization objective to test against strong baselines, not a claim of global optimality.

This document defines the objective and how to evaluate progress. [SPEC](../SPEC.md) defines the
contract, [DESIGNS](../DESIGNS.md) assigns ownership, [readiness](product-readiness.md) lists checks
and limits, and [ROADMAP](../ROADMAP.md) tracks unfinished work.

## Work and Agent choices

A work item keeps its goal, versioned acceptance criteria and responsibility across
sessions, retries and delegation. A work cycle shares budget and context across related items.
Internal decomposition does not create additional business value or reset spending.

An Agent is a harness plus a model: model options include reasoning effort; native tools and skills
belong to the harness. Common choices are saved presets, while uncommon choices can be temporary.
These configurations need no version field. Runtime budgets and timeouts are separate from Agent
identity. The user chooses the Agent configuration before managed execution begins; full management
also requires an explicitly selected supervising Agent to decide what to ask next.

The intended work sequence is:

```text
Goal and acceptance criteria → Agent choice and runtime limits
→ native execution or delegation → measured resource use → independent acceptance
→ value and cost comparison → retained experience or evaluated behavior change
→ a later task that tests whether the change helped
```

Execution completion and acceptance are separate facts. Authorized users and trusted validators
supply acceptance; an Agent may submit work and request checks without certifying its own claims.
Evidence insufficient for a requested conclusion can be an accepted delivery when the criteria
require an honest assessment. Cancellation alone is not a quality failure.

## Resource and learning decisions

Selection uses the task, acceptance requirements, available budget and time, candidate capabilities,
explicit user choices, reusable resources and relevant historical feedback. Evidence retains its
requested configuration, native runtime version, sample size, timestamp and limitations. Untried candidates are unknown;
selected historical samples are not unbiased rankings. The decision record explains what was
visible and why an action was chosen; that explanation is not a reward signal.

Policies can choose direct work, a deterministic check, delegation, parallel work, review, reuse,
tool construction, clarification, postponement or stopping. Fixed routes and simple rules are useful
baselines. Explicit retry or escalation policies preserve the original failure, count all attempts
and protect non-repeatable side effects. An adapter must not silently turn failure into success on
a different route. Exploration spends only a user-authorized portion of the shared budget.

Account for recorded LLM and tool calls, including delegated work, review and validation, by deriving
totals from the original execution records. Each independent charge is counted once. An Agent's
total includes every descendant and is the cost used to compare configurations; cycle spending
does not add those overlapping totals. Unknown cost stays unknown and partial observations retain
their coverage. Initially unpriced tool calls can carry zero cost while preserving their records
for later pricing; human activity is not manually registered.

Saving an observation is different from validating a change to future behavior. Behavior changes
progress through candidate, evaluation, adoption, observation and withdrawal or replacement. Fixed
validators check structure and interfaces; paired behavioral evaluations check results, cost and
regressions on later tasks. User-authorized adoption can be automatic, but a learning Agent cannot
expand its own authority, change its judge or replace hidden tests. Late corrections remain eligible
feedback after an earlier execution has been reviewed.

## How progress is measured

Continuous research scenarios use the real Host/ProductServices entry points: data updates and
revised analyses; reusable knowledge invalidated by changed assumptions; competing work; correction
followed by unseen work; and cancellation or restart without duplicate side effects. Verifiers need
counterexamples for plausible but incorrect output, changed inputs, stale artifacts and unsupported
claims, while accepting different correct implementations.

Compare fixed strong and low-cost routes, simple static rules or verified escalation, dynamic
selection with frozen learning, and dynamic selection with evaluated learning. Paired repetitions
share data, permissions and budget; each group isolates workspace, product home, Memory, mutable
resources and native session/cache state. Allowed learning persists within a sequence. Analyze
sequences as samples, not every internal call as an independent experiment.

Reports retain source/data revisions, requested configuration, initial resources, runtime identity, policy,
budget, validator and random settings. Measure acceptance and valuable partial delivery, time to
first acceptance, queue/execution/human wait, attributable spending and its coverage, intervention,
rework and learning costs. Net learning benefit includes later gains and all retrieval, evaluation,
maintenance and regression costs. Paid model experiments require explicit opt-in and budget.
The deferred offline comparison suite is not part of this delivery. Publication protocols and
frozen scientific examples belong to `swarmx-paper`.
