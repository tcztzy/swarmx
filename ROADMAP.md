# Unfinished product work

This file lists unfinished milestones. See [product direction](docs/product-direction.md)
for the objective and [product readiness](docs/product-readiness.md) for verification commands.

| Milestone | Next implementation and acceptance |
| --- | --- |
| Recoverable long-term work | Extend explicit dependency/priority/deadline dispatch and journal reconciliation with fair automatic queues and native external-operation reconciliation. Authorize retries only after inspecting unresolved effects; verify non-repeatable operations are not duplicated. |
| Full resource accounting | Extend journal-derived LLM/tool costs with service billing and subscription/local metering, including local tool depreciation. Keep independent charges once and unknown coverage visible; human work is outside the current recording scope. |
| Shared foreground and learning capacity | Extend shared runtime budgets and review allowances to behavioral-evaluation scheduling and priority-aware waiting. Add native hard-limit integrations beyond Claude's query budget; report each enforcement boundary and potential overspend elsewhere. |
| Policy choices beyond route selection | Extend execute/defer/stop decisions with authorized deterministic checks, reuse, review, parallelism, clarification and explicit verified escalation. Compare against fixed and static-rule policies; preserve original failures, sample uncertainty and user choices. |
| Evaluated behavior changes | Extend registered old/candidate behavioral evaluation and guarded adoption with continued observation, withdrawal/replacement and unseen follow-up tasks. Schedule and account for evaluation in the shared cycle; isolate hidden fixtures from native tool access. |
| Feedback-driven review | Compare the implemented delayed-feedback and interval triggers against disabled learning and failure/value-based policies. Add priority-aware waiting so learning does not starve urgent delivery; include review and validation costs in comparisons. |
| Continuous research benefit experiments | Add the deferred offline comparison scenarios, then real data updates, changed assumptions, resource competition, learning regressions and restart. Run paired repeated opt-in model experiments using the comparison design in [product direction](docs/product-direction.md#how-progress-is-measured). |
| User-facing long-term controls | Extend desktop goals, budgets and acceptance controls with dependency/priority queue editing and shared scheduling visibility. Add explicit handover workflows while preserving work identity and existing evidence. |

No live-model quality, productivity or net-learning gain is established by this roadmap or by a
green engineering suite. Record an experiment as unrun until it has actually been executed.
