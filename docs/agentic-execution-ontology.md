# Agentic Execution Ontology Charter

Status: P1 definition baseline

This document freezes the operational meanings, identity criteria, and category
boundaries used by SwarmX's publication-first research program. Its purpose is
to prevent invalid substitutions between task, execution, authority, world,
and evidence concepts before experiments or product changes are designed.

This charter is a research and design contract. It does not add a product
entity, persistence format, runtime service, workflow language, or permission
mechanism. Current source mappings below are descriptive: an incomplete mapping
does not imply that the missing concept has already been implemented.

## Scope and commitments

The unit of study is an **agentic execution system**: a controller coupled to
memory, tools, and a verifier, acting under authority delegated by a principal
and interacting with an external environment. An Agent may be the controller,
but the environment, authority, memory, and verifier do not become parts of the
Agent merely because they participate in the same execution.

The charter makes five commitments:

1. Identity is stated explicitly and never inferred from similar content,
   correlation, or a shared execution context.
2. Ontic facts, epistemic representations, normative relations, and control
   state remain distinct.
3. A request, authorization, execution occurrence, world effect, observation,
   evidence, and ledger claim are not stages of one interchangeable object.
4. Completion is relative to a versioned task contract and its evidence rule;
   a status field alone is not proof of completion.
5. Ontology is used to reject category errors. It is not a reason to build a
   universal vocabulary or infer new authority.

The following are out of scope:

- RDF, OWL, SHACL, SPARQL, a triple store, or an ontology service;
- a universal world model, event schema, trace format, or task ontology;
- a second authority for Workflow, Memory, Session, task, or audit data;
- automatic permission grants, execution transitions, or truth claims derived
  from graph traversal;
- any change to the SwarmX Agent identity `harnessId:modelId`.

## Four-way task distinction

Every research fixture and interpretation of a durable task must distinguish
the following four kinds.

| Kind | Operational meaning | Identity criterion | Must not be identified with |
| --- | --- | --- | --- |
| Task contract `G@v` | Versioned information object declaring goals, constraints, admissible effect semantics, and evidence-based completion | Contract id plus immutable version or digest | Prompt wording, Plan, WorkItem, Run, or current progress |
| Task instance / WorkItem `W` | Persistent particular that instantiates a contract and carries its own lineage | Stable WorkItem id; lineage is explicit metadata, not content equality | Session, Run, transcript, contract, or residual state |
| Run `R_i` | One process occurrence attempting work for a Task instance | Stable Run id bound to exactly one WorkItem | Task identity, Agent identity, or successful Effect |
| Residual task state `S_t^W` | Time-indexed description of progress, remaining obligations, outcome claims, and relevant evidence for one WorkItem | WorkItem id plus observation boundary and revision/time | The WorkItem itself, World state, transcript, or workflow position |

These relations follow:

```text
W --instantiates--> G@v
R_i --attempts----> W
S_t^W --describes-> W at observation boundary t
Session --observes-> W
```

Consequences:

- Two WorkItems may have equal residual state without becoming the same Task.
- One WorkItem may pass through unequal residual states without changing
  identity.
- A retry normally creates a new Run for the same WorkItem; it does not create
  a new Task unless an explicit operation creates one.
- Replacing an executor changes an execution role or Run, not WorkItem identity.
- Linking, unlinking, switching, forking, or archiving a Session does not change
  WorkItem identity or terminate its execution.
- Revising a Task contract creates a new contract version. Continuing the same
  WorkItem under that version requires an explicit amendment relation that
  states how existing goals, constraints, authority, and commitments carry
  forward.

SwarmX currently has a first-class WorkItem and Run, but no first-class generic
Task contract. Experiments must therefore declare `G@v` in their fixture and
must not treat an input prompt as the complete contract.

## Upper categories and identity rules

The categories below are operational rather than metaphysical. A stored object
may play more than one declared role—for example, an immutable Artifact may
serve as Evidence—but the roles and the propositions they support remain
separately identified.

### Actors, components, and execution roles

| Kind | Meaning and identity rule |
| --- | --- |
| Principal | Actor on whose behalf authority is delegated. Identity comes from the declared governance boundary, not from the active Session. |
| Controller | Actor or policy component selecting actions from the information available to it. Identity is the declared controller id and revision for one role assignment; it does not absorb tools, Memory, verifier, or environment. |
| Model | Independent product entity with a stable Model id and capability metadata. |
| Harness | Reproducible runtime recipe with its own stable identity and revision. |
| Provider | Connection and credential source identified by its Provider profile id. It may supply Models but does not own Model identity. |
| ModelSupply | Route identified by its supply id and relating exactly one Model to one Provider profile under declared API compatibility. It is neither the Model nor the Provider. |
| Agent | Exactly one Harness paired with one Model, identified as `harnessId:modelId`. Provider route, effort, worker, Run, and Session do not change this identity. |
| Executor role | Relation identified by Run, assigned actor/component, operation, and validity interval. A change of executor creates a new role assignment or Run, not a new Agent or WorkItem by itself. |
| Worker | Concrete executor process or backend instance identified by its instance id within a verified environment. Worker identity and environment belong to a Run; they do not create a new Agent or WorkItem. |
| Verifier | Actor and rule applying an evidence criterion to a proposition, identified by verifier id plus rule version. Being the controller does not automatically make an actor an authoritative verifier. |
| Environment | Declared external-system/resource boundary in which relevant facts and effects exist; observations additionally require a time or version. Environment and Project containment are not Agent components or authorization. |

An **execution role** is the relation by which an Agent, worker, policy
component, or human performs part of a Run. It is not a new Agent identity and
does not transfer authority by itself.

### State and representation kinds

| Kind | Meaning | Identity boundary | Category boundary |
| --- | --- | --- | --- |
| World state `X_t` | What actually exists or has happened in the task-relevant external world at time `t`, including actual effects | Declared environment/resource boundary and world time/version; equality is relative to named task-relevant propositions | Not prompt context, Memory, a status, or model confidence |
| Observable history `H_t` | Ordered actions and observations available to the declared controller up to boundary `t` | Controller id, observation boundary, ordered source occurrences, and projection version | Not latent World state, a Session, or automatically the complete system history |
| Belief state `b_t` | Uncertain information held by a controller about relevant world propositions | Controller id, observation boundary, proposition vocabulary, and representation version | May be wrong or incomplete; not World state |
| Normative state `q_t` | Remaining goals, constraints, authority relations, and undischarged commitments | WorkItem id, Task contract version, observation boundary, and normative projection version | Not a workflow marking, Plan, or approval receipt |
| Control state `u_t` | Current scheduler/workflow marking, retry state, and open control subgoals | Workflow or scheduler instance, observation boundary, and control-schema version | Reaching a node does not prove an external effect or discharge a commitment |
| Operational memory `M_t` | Bounded representation retained for future control | Representation id/digest, producing boundary, and format version | Not canonical history, World state, or Task identity |
| Residual task state `S_t^W` | Typed task-relative projection containing distinct belief, normative, control, outcome-claim, and evidence-reference fields | WorkItem id, observation boundary, and projection version | An aggregate description; its fields do not lose their category boundaries |

`unknown` and `uncertain` are epistemic labels. They are not World states or
Effect occurrences. A true no-effect outcome is a proposition about the world;
it is not an Effect occurrence called `no_effect`.

### Specifications, occurrences, claims, and support

| Kind | Meaning | Identity criterion |
| --- | --- | --- |
| Action specification | Repeatable operation schema with parameters, preconditions, and possible effects | Capability/operation id plus schema version |
| Intention | State directed toward a possible future action | Intending actor, content, and decision boundary |
| Request | Information object asking that a bound action be performed | Request id plus versioned canonical semantics; transport correlation alone is insufficient |
| Authorization | Normative relation or gate permitting an actor to perform the bound request in a scope and validity interval | Principal/policy, actor, exact request semantics, scope, decision, and validity |
| Action occurrence | One concrete execution attempt by an actor in time | Occurrence id bound to actor, Run, request/specification, and time |
| Effect occurrence | Actual world transition causally produced by an Action occurrence | External transaction/resource identity plus causal binding; an idempotency key alone is insufficient |
| Observation event | Occurrence in which a signal about the world is received | Observation id, source, time, and observed signal |
| Proposition | Truth-evaluable statement about the world, task, or outcome | Canonical content and declared temporal/scope qualifiers |
| Evidence | Support role played by an object or record for a Proposition relative to a Verifier and rule | Evidence object/version plus supported proposition, provenance, verifier, and rule |
| Ledger claim | Durable assertion that an attempt or outcome is, for example, committed, not committed, or unresolved | Claim/receipt id, schema version, subject occurrence/request, and recorded status |
| Artifact | Persistent entity used or generated by an activity | Stable artifact id and immutable version/hash; a path or name is not identity |
| Plan | Information object describing intended future action | Plan id/version or immutable content digest |
| Session | Canonical persisted conversation and observation view | Session id within its persistence authority; it may observe but does not own a WorkItem |

Evidence is a relational role, not a synonym for an Artifact, AuditEvent,
receipt, or truth. The same immutable object can support different propositions
under different verifier rules. Provenance permits a verifier to evaluate
support; provenance does not make the proposition true automatically.

## Required distinctions

The following disjointness rules are mandatory for research models and design
reviews. “Distinct” means the kinds cannot be substituted merely because one
record, process, or file participates in multiple relations.

1. Task contract, Task instance, Run, Session, and residual task state are
   distinct.
2. Model, Harness, Provider, ModelSupply, Agent, Controller, Executor role, and
   Worker retain separate product, actor, and execution identities.
3. World state, Observable history, and Belief state are distinct; an observed
   signal can be incomplete or wrong, and confidence never promotes belief to
   fact.
4. Normative state and Control state are distinct; control progress never
   creates authority or discharges an obligation by itself.
5. Action specification, Intention, Request, Authorization, and Action
   occurrence are distinct.
6. Action occurrence and Effect occurrence are distinct. One attempt may cause
   zero, one, or multiple effects, and an effect may be observed later.
7. Effect occurrence and Observation event are distinct. Observation may be
   delayed, incomplete, incorrect, or absent.
8. Evidence, Ledger claim, and the Proposition supported or asserted are
   distinct. A claim remains a claim even when its status is `committed`.
9. Artifact identity is not path identity; version and digest differences are
   semantically relevant whenever a contract depends on content.
10. Plan and execution history are distinct; a Workflow edge records intended
    control flow, not proof that an external effect occurred.

## Typed action lifecycle

The action lifecycle is branching. It is not an assertion that every request
reaches every later kind.

```text
Action specification
      |
      +--> Intention
      |
      +--> Request --evaluate exact authorization--> Rejection
                         |
                         +--> Action occurrence
                                   |
                                   +--> zero or more Effect occurrences
                                   |
                                   +--> true no-effect Proposition
                                   |
                                   +--> outcome not yet known

World signal --> Observation event --> Evidence relation
                                      |
                                      +--> Ledger claim about an attempt/outcome
```

The minimum relation vocabulary is:

```text
instantiates(WorkItem, TaskContractVersion)
attempts(Run, WorkItem)
performs(ExecutionRole, ActionOccurrence)
conformsTo(ActionOccurrence, ActionSpecificationVersion)
requests(Request, ActionSpecificationVersion, Arguments)
authorizes(Authorization, Actor, Request, Scope, Validity)
causes(ActionOccurrence, EffectOccurrence)
observes(ObservationEvent, Signal, Source, Time)
supports(EvidenceObject, Proposition, Verifier, Rule)
asserts(LedgerClaim, Proposition, RecordedStatus)
describes(ResidualTaskState, WorkItem, ObservationBoundary)
observesWork(Session, WorkItem)
```

This vocabulary is notation for fixtures and reviews, not a persisted API.
Every relation instance must name its source authority. Absence of a relation
must not be treated as a negative fact unless the relevant authority declares
the dataset complete for that relation.

## Normative-causal continuity

**Normative-causal continuity** is the criterion for a valid continuation of
one WorkItem after interruption, retry, handoff, or executor replacement. It is
not the criterion of numerical WorkItem identity.

A continuation is valid only when:

1. the applicable Task contract version, or an explicit amendment lineage, is
   known;
2. remaining goals, constraints, authority, and commitments are preserved or
   revised by declared rules;
3. prior Action occurrences and known Effect occurrences retain their causal
   identities;
4. committed, not-committed, and unresolved outcome claims remain distinct;
5. every completion-relevant claim carries the evidence required by the
   contract's verifier rule;
6. unresolved outcomes lead to a declared reconcile, verify, compensate, or
   reject decision rather than an invented success or blind replay; and
7. the new executor possesses compatible capability and current authorization
   without inheriting hidden authority from the prior executor.

The same WorkItem may continue through a new Run or executor when these
conditions hold. Two different WorkItems do not become one merely because
their residual state and continuation policy are equal.

## Operational invariants

These stable identifiers are intended for fixtures, reviews, and later focused
tests.

| Id | Invariant |
| --- | --- |
| ONT-1 | An Agent is identified only by `harnessId:modelId`; Provider route, effort, Run, Worker, and Session never create another Agent identity. |
| ONT-2 | A WorkItem is not its contract, Run, Session, transcript, or residual state. |
| ONT-3 | Authorization must bind the actor, exact Request semantics, resource scope, decision, and validity before authority can expand or an effectful Action occurrence starts. |
| ONT-4 | Reusing an idempotency key is safe only when the replay mechanism proves that the versioned action specification and all effect-relevant arguments are the same. |
| ONT-5 | An Action occurrence is not an Effect occurrence; timeout or lost response preserves an unresolved outcome until evidence supports another claim. |
| ONT-6 | `uncertain` is an epistemic Ledger-claim status, `not_committed` asserts a no-effect proposition under a declared rule, and `committed` is still a claim rather than the Effect itself. |
| ONT-7 | Completion requires the Task contract's acceptance and evidence rules; Run success, workflow position, tool success, receipt status, or audit correlation alone is insufficient. |
| ONT-8 | Session links are observational. Session switching, unlinking, forking, or archiving does not mutate WorkItem authority or lifecycle. |
| ONT-9 | Executor replacement creates a new execution role or Run and must preserve fencing, environment, checkpoint, authority, and unresolved-outcome constraints. |
| ONT-10 | Memory links, catalog relations, audit correlations, and other knowledge projections never become `SwarmConfig` execution edges or permission grants. |
| ONT-11 | Audit events and receipts may support claims but do not independently reconstruct World state, WorkItem state, or causal Effect identity. |
| ONT-12 | Every derived ontology relation is deterministic, bounded, source-qualified, and rebuildable; it never becomes a new persistence authority. |

`ONT-4` is a research requirement, not a claim about the current implementation.
The planned request-binding counterexample must test it before any task-runtime
schema change is proposed.

## Category-error matrix

| Invalid inference | Why invalid | Required response |
| --- | --- | --- |
| “The Session ended, so the WorkItem ended.” | Session is only an observation view | Query task authority independently |
| “The Run succeeded, so the external effect committed.” | Run status and Effect occurrence are different kinds | Apply the contract's outcome verifier |
| “An approval record exists, so this call is authorized.” | Approval evidence may not bind the exact Request, scope, actor, or validity | Re-evaluate the full authorization relation |
| “The idempotency key matches, so the arguments match.” | A key is not canonical Request identity | Compare a versioned effect-relevant Request binding |
| “The receipt says committed, therefore it is the Effect.” | Receipt is a Ledger claim | Preserve its claim role and supporting evidence |
| “The model is confident, therefore the world fact is true.” | Confidence belongs to Belief state | Verify against an authorized world observation |
| “The workflow reached the next node, so the prior effect happened.” | Control state is not World state | Require outcome evidence before dependent effects |
| “Two Tasks have identical checkpoints, so they are one Task.” | State equality is not numerical identity | Retain both WorkItem ids and lineage |
| “The file path is unchanged, so the Artifact is unchanged.” | Artifact identity includes version/content | Compare immutable id and digest |
| “The Provider changed, so this is a different Agent.” | Provider is a supply route | Preserve `harnessId:modelId` identity |
| “Audit events correlate two operations, so one caused the other.” | Correlation and provenance do not establish causal identity | Require an explicit causal binding and verifier rule |
| “A Memory link names a dependency, so execution should follow it.” | Knowledge and workflow edges have separate semantics | Compile an explicit validated `SwarmConfig` edge if execution is intended |

## Formal-symbol boundary

The next information-theoretic task uses the following fixed meanings for one
declared Task contract `G` and WorkItem `W`:

| Symbol | Meaning |
| --- | --- |
| `X_t` | Latent task-relevant ontic World state; it may not be observable |
| `H_t` | Complete Agent-observable history of actions and observations up to `t` |
| `Z_t = f_G(H_t)` | Exact task-relative continuation state induced by observable history |
| `M_t = phi(H_t)` | Bounded representation retained by a checkpoint, summary, or Memory process |
| `C_t^* = g_G(X_t)` | Fixed task-scored response target at a declared bottleneck |

`Z_t`, `M_t`, and `S_t^W` are not WorkItem identity. Equal `H_t` may remain
compatible with multiple latent `X_t` values; no memory transformation can
resolve that ambiguity without a new informative observation. A verification
action adds an observation channel—it does not reveal facts merely by being
requested.

The exact continuation-equivalence relation, right-congruence assumptions,
capacity and decoding bounds, distortion, verification side information, and
twin fixtures remain the next P1 deliverable. Their working definitions and
stop conditions live in the
[publication-first research strategy](publication-research-strategy.md).

## Mapping to current SwarmX

| Ontology kind | Current realization | Limit |
| --- | --- | --- |
| Task contract | Executor operation plus fixture/input metadata | No first-class generic goal, constraint, effect, and acceptance contract |
| Task instance | `TaskWorkItem` and append-only `TaskRuntimeEvent` replay | Strong task authority; must not be reconstructed from Session or audit data |
| Run | `TaskRun`, lease, fencing token, retry, and cancellation | Process occurrence, not Task identity |
| Residual task state | Replayed task state, checkpoint, approvals, claims, and references | Distributed projection, not a unified typed state object |
| Normative state | Approval records, budgets, permission policy, cancellation, and capability grants | No unified residual norm automaton |
| Control state | WorkItem/Run statuses, leases, retries, and `SwarmConfig` markings | Does not prove World effects |
| Request | Worker capability call with call id, capability, operation, and arguments | No generic durable canonical Request identity yet |
| Authorization | Capability grants, permission policy, and approval decisions | Approval record is evidence, not the complete relation |
| Action occurrence | Capability/tool/process lifecycle records | No generic canonical occurrence identity across transports |
| Effect occurrence | External system transition | No generic first-class World or Effect identity |
| Ledger claim | `TaskSideEffectReceipt` with committed, not-committed, or uncertain status | Current request-binding strength is an experimental question |
| Evidence | Immutable artifacts, hashes, workspace observations, and hash-chained audit records | These can support claims but are not automatically World truth |
| Operational memory | Checkpoint plus immutable Artifact references | Context summary and Session history are separate projections |
| Observable history | Declared projection of actions and observations available to one controller | Session, task, worker, audit, and environment records are not one complete event stream |
| Session | Append-only per-Project JSONL history and explicit WorkItem observer links | Never participates in task replay |
| Agent | Harness plus Model | Identity remains `harnessId:modelId` |

Relevant implementation contracts are in
[`task-runtime.ts`](../packages/core/src/task-runtime.ts),
[`task-worker-protocol.ts`](../packages/core/src/task-worker-protocol.ts),
[`types.ts`](../packages/core/src/types.ts), and
[`audit.ts`](../packages/core/src/audit.ts).

## Authority separation

| Authority | Canonical subject | Ontology must not infer |
| --- | --- | --- |
| `SwarmConfig` | Executable workflow topology | Execution from Memory or catalog relations |
| Memory Markdown plus Git | User-owned subjective knowledge | World truth, permission, or task completion |
| Session JSONL | Conversation and observation history | WorkItem lifecycle or external effect |
| Task-runtime events | WorkItem, Run, lease, checkpoint, approval, and outcome-claim state | Complete World state from a receipt or status |
| Audit chain | Integrity-protected evidence about privileged decisions and effects | Causal Effect identity or canonical Task reconstruction |
| External environment and verifier | Actual world propositions under a declared observation/evidence rule | Truth from internal correlation alone |

Any later ontology projection must read these authorities without replacing or
merging them. It must carry source authority, source identity, schema/version,
and observation generation for every derived relation.

## Change and review policy

Before a production field or relation is added because of this charter, its
proposal must state:

1. the category and identity rule it implements;
2. its canonical source authority and integrity rule;
3. whether it records an ontic fact, belief, norm, control state, observation,
   evidence relation, or claim;
4. its persistence and compatibility consequences;
5. its bounded serialized byte/token cost;
6. the category error or paired experimental failure it prevents; and
7. the focused test and ablation that would detect its absence or corruption.

A read-only derived relation additionally requires deterministic ordering,
explicit bounds, provenance, ambiguity diagnostics, and proof that it cannot
grant authority or create an execution edge. If those requirements cannot be
met, the relation remains analysis notation rather than a product feature.
