---
name: delegate
description: Select and call subagents using task-specific evidence about harness, model, effort, provider and execution settings, within the current work budget.
---

# Delegate work using observed combination performance

Use this skill when choosing or calling a subagent. A candidate is a combination of harness,
model, effort and provider, with profile and native runtime versions when available. Effort can change which
tasks a model can complete, its cost and latency; assess each level using its own evidence.
Provider price, latency and reliability can change the suitability of the same model. Use the requested
configuration as the Agent identity; an explicit native model/effort conflict must fail execution.
Original native reports remain in the raw journal; an unknown provider or version stays unknown.
Keep requested effort in the selection. An omitted effort is unknown, not an assumed default.
Native effort names are specific to a harness/model; the same
name does not establish equal computation across combinations. Do not pool different effort levels
when comparing task acceptance, cost or time.

`swarm.prepare {task, queries}` loads this skill for the exact task and appends relevant private
combination evaluations from Memory, with original sources and Host-computed statistics.
Read those evaluations before selecting a combination. Queries should name the task and useful
harness/model/effort/provider terms. The preparation ID belongs to this task and the current run.

## Choose a combination

Use the user's explicit selection first. Otherwise compare candidates against the actual goal,
acceptance criteria, budget, deadline, tool needs and risk. The work context gives registered
configurations, available budget and previous independent acceptance. `swarm.models` supplies
native model catalogs where supported; an admitted harness does not prove its runtime is available.
Choose from the selected model's supported effort levels and pass the selected effort explicitly.
If the runtime does not expose that choice, retain the unknown or observed native level.

For each relevant evaluated combination, read its task conditions and recommended scenarios,
sample count, acceptance results, cost coverage, time window, revisions and contrary evidence.
An observation, a user preference and an AI judgment have different evidential meaning.
Normal termination is not content acceptance; cancellation is not a quality failure. A provider
incident does not establish that every deployment of that model is unreliable. Selected samples
are not a randomized comparison, and an untried combination is not a demonstrated loser.

Use observed costs with their source: invoices, native estimates, price snapshots or unknown.
Partial token/cost coverage is not a complete price. Simulation and regression fixtures are
engineering evidence, not live provider prices or measured research benefit. Do not fill missing
evaluation rows with model-family reputations or invented scores. If evidence is insufficient,
say which comparison is unknown and use the applicable capability facts below or the user's choice.
An exploratory call needs an explicit allocation within the authorized budget.
For a routine task with a direct check, prefer the least costly effort with supporting acceptance
evidence. For demanding analysis or revision after a failed check, compare higher-effort combinations
against other models using the available budget and deadline. Higher effort is not automatically better
or cheaper overall; a failed low-effort attempt followed by escalation incurs both attempts' costs.

## Capability-based starting choices

These are integration-based scenario choices, not measured quality rankings. Resolve each model,
effort and provider to an admitted, explicit route; use the appended local evaluations when they exist.

| Scenario | Candidate combination and required check |
| --- | --- |
| A research lead needs SwarmX Science/Memory tools and configurable provider routes | Pi + an exact advertised `provider/model` route + that configured provider. Check its supported thinking levels and the local evaluation for this task class. |
| Repository work needs Codex's native session, tool and approval behavior | Codex + a model from its native catalog + the provider/deployment actually reported by that runtime. Compare relevant acceptance and cost coverage; the harness name alone is not a quality score. |
| A managed attempt needs a native USD query cap | Claude + an admitted catalog model + its configured provider. SwarmX passes the available runtime budget through the SDK's `maxBudgetUsd`; this query-scoped limit and its reported estimates do not establish complete external spending. |
| A standalone persistent-shell task needs a small tool composition | DSH + an explicit `provider/model` + profile `sdk-minimal`, with a chosen reasoning effort. It has no SDK model catalog and cannot resume or steer through this integration. Use `sdk` when its full tool composition is required. |
| Work depends on an existing native gateway/session environment | Hermes or OpenClaw + its discovered model and provider settings. Check catalog, configuration changes and task capabilities before dispatch; do not substitute another deployment's reliability record. |

These integration facts are checked by this release's native integration tests. Interpret them
against the same application revision as this skill; they do not describe another installation.

## Execute and retain responsibility

Call `swarm.send_message {agentId, text, model, effort?, profile?, preparationId, reason}` with
the prepared task. Explain the chosen combination and cite the applicable skill/Memory evidence.
The reason is an audit record, not proof that the choice was good. DSH model IDs include provider;
its profile changes tool composition, not Host permissions. Other harnesses keep their native
configuration semantics.

Retain the original work identity, budget and acceptance criteria across delegation, parallel
work and retry. Splitting work creates no extra business value or spending authority. Inspect
unresolved external effects before retrying. Return pinned outputs for independent acceptance;
an agent cannot certify its own answer by declaring success.

## Improve the evaluations from subsequent work

Save useful findings as Memory concepts tagged `agent-selection`, reusing an existing concept
for the same combination and task scope. Put the concrete evaluation and appropriate scenarios
in the concept body: what this combination did well or poorly, under which conditions, and what
should be selected differently next time. Include provider/deployment, model/harness versions,
profile/effort, measured acceptance, observed cost and coverage, latency, error/cancellation
distinctions, sample dates/counts and limitations when those facts are available. Avoid a
provider-wide score from a single incident. Preserve conflicting and corrected evidence.

Supply `evaluation {kind, task, criteria, evidence, counterEvidence, limitations}` with original
execution-event references. The Host resolves citations and computes scoped statistics; a
resolvable citation does not itself certify the written conclusion. Independent work acceptance
and later corrections are original evidence even if the execution was already reviewed.
The next preparation loads the updated concept into this skill's body. It also checks current Host
feedback for the cited executions: a correction marks an older assessment for reconsideration even
before its written body is updated. Work-attempt acceptance does not isolate a contributing run's quality.
Private incidents and
execution URNs stay in the user's Memory, outside the shipped skill.

Changes to project prompt/skill behavior use the existing registered learning-resource workflow:
candidate, fixed validation/evaluation, revision-checked adoption, and subsequent observation.
Do not change validators, expand permissions or claim that an available file revision was
actually loaded by a native runtime without evidence. A saved observation is not a demonstrated
behavior improvement; judge it on later accepted work and the additional learning cost.
