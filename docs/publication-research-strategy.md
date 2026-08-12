# Publication-First Research Strategy

Status: working research plan
Literature cutoff: 2026-08-06
Decision owner: project maintainers

This document turns the repository into a falsifiable theory-and-systems
research program. It is a novelty sanity check, not a claim of exhaustive
coverage or priority over prior work.

## Decision

**PIVOT** from cross-harness handoff as the paper thesis to this question:

> Which already-observed distinctions must a tool-using Agent preserve in
> bounded memory, and which latent uncertainties must it resolve through new
> verification, to continue an operation safely after uncertain effects?

Working title:

> **State, Not Steps: Dual Information Bottlenecks in Verifiable Agent
> Operations**

The definition baseline is frozen in the
[agentic execution ontology charter](agentic-execution-ontology.md): a four-way
Task distinction, typed action lifecycle, explicit identity and disjointness
rules, and the `X / H / Z / M / C*` symbol boundary. The next action is to
freeze continuation equivalence and hand-check four compression twins and four
observability twins. Do not implement the simulator until those definitions and
fixtures are coherent.

The paper's single claim is:

> In a finite declared task family with uncertain external effects, safe
> continuation is jointly limited by memory capacity for preserving observed
> response-relevant distinctions and the informativeness of available
> verification for resolving latent world ambiguity; the bottlenecks can be
> isolated from each other and from forced prefix depth.

Handoff, checkpointing, context compaction, crash recovery, verification, and
executor replacement become experimental conditions of this general claim.
They are not separate contributions.

## The ontological thesis

The [ontology charter](agentic-execution-ontology.md) is authoritative for the
operational terms, identity criteria, category-error matrix, and stable
`ONT-*` invariants summarized in this section.

The basic unit of agentic computation is not a message or tool call. It is a
controlled transition in a temporally extended task, followed by evidence about
what actually happened.

SwarmX defines an `Agent` identity as `harnessId:modelId`; the paper must not
silently redefine it. The broader object of study is an **agentic execution
system**: an Agent or other controller coupled to memory, tools, and a verifier,
acting under authority delegated by a principal and interacting with an
external environment. Environment and authority are not components of the
Agent. Responsibility is assigned by governance relations, not produced merely
by system composition.

Ontology is useful here as a prohibition on invalid identities, not as a large
vocabulary. The paper must preserve these distinctions:

| Kind | Meaning | Category error that causes operational failure |
| --- | --- | --- |
| World state | What actually exists or has happened in files, services, institutions, and other external systems | Treating prompt context or workflow position as the world |
| Belief state | The Agent's uncertain information about the world after observations and actions | Treating belief or model confidence as fact |
| Task contract | An information object declaring goals, constraints, effect semantics, and evidence-based completion | Treating prompt wording as the complete contract |
| Task instance / WorkItem | A persistent individual identified by id, contract version, and lineage | Treating equal residual state as numerical identity, or a Session as the Task |
| Run | One process occurrence that attempts to realize a Task instance | Treating a retry or replacement Run as a new Task by default |
| Residual task state | The time-varying progress, obligations, and unresolved outcomes of one Task instance | Treating a state value as the identity of the Task |
| Normative state | Remaining goals, constraints, authority relations, and undischarged commitments | Treating a plan or summary as if it preserved permission and obligation |
| Control state | The current workflow marking, retry state, and open subgoals | Treating passage through a node as proof that an external effect occurred |
| Action type / specification | A repeatable operation schema with parameters, preconditions, and possible effects | Treating the tool schema as a concrete occurrence |
| Intention | A control or mental state directed toward a possible future action | Treating intent as a request, authorization, or occurrence |
| Request | An information object asking that an action be performed | Treating a request as execution |
| Authorization | A normative relation or gate permitting an actor to perform an action in a scope | Treating approval evidence as the authorization relation itself or as execution |
| Action occurrence | One concrete execution attempt by an actor in time | Treating the request, response, and occurrence as one event |
| Effect occurrence | An actual world transition causally produced by an occurrence; it may be absent | Reifying “no effect” or `unknown` as an Effect; the former is an outcome proposition and the latter is epistemic |
| Observation | An event in which the system receives a signal about the world | Treating an observation as automatically correct or complete |
| Evidence | An object or record playing a support role for a proposition relative to a verifier and rule | Treating a receipt or provenance chain as self-authenticating truth |
| Ledger claim / status | A persistent claim such as attempted, committed, not committed, or unresolved | Treating the claim as the world transition it describes |
| Artifact | A persistent entity used or generated by an activity; versions and hashes matter | Treating the same path or name as the same entity over time |
| Plan | An informational object describing intended future action | Treating an intention as an execution history |
| Session | A persisted conversation and observation view | Treating conversation history as task or world authority |
| Execution role | The current Agent, worker, or policy component performing part of a Run | Collapsing Model, Harness, Agent, worker, Run, and Task identity |

**Normative-causal continuity** is a criterion for a valid continuation of a
Task instance: the applicable contract persists, commitments are discharged or
revised by declared rules, effects retain causal identity, and completion is
supported by evidence. It is not a criterion of numerical Task identity. The
same Task may pass through different residual states, and two different Tasks
may temporarily have identical residual states.

The typed lifecycle is a branching relation, not an assertion that every
request reaches every later stage:

```text
action specification -> intention and/or request
request --authorization relation/gate--> execution attempt or rejection
execution attempt -> effect occurrence or a true no-effect proposition
observation event -> evidence relation + ledger claim about the outcome
```

This gives four compact principles:

1. a Task instance is not its transcript, Run, or residual state;
2. a requested action, an execution occurrence, an actual effect, and evidence
   about that effect are different kinds;
3. memory is not recollection but a bounded representation of observable
   history for future control;
4. verification is an information-acquisition action used when history does
   not determine the relevant world fact.

The implementation need not adopt OWL, RDF, or a new workflow language. In
SwarmX, the ontology should appear as a few disjoint types, identity rules, and
state-transition invariants.

## The information-theoretic thesis

### World, history, sufficient state, memory, and response

For a fixed Task contract `G`, distinguish five variables:

| Variable | Meaning |
| --- | --- |
| `X_t` | Latent ontic world-and-task state; it may not be observable |
| `H_t` | A reachable Agent-observable history of actions and observations |
| `Z_t = f_G(H_t)` | The exact continuation state induced by observable history |
| `M_t = phi(H_t)` | The bounded representation retained by a checkpoint, summary, or memory |
| `C_t^* = g_G(X_t)` | A fixed task-scored response target at a declared bottleneck; `Z_t` fixes its conditional law, but the realized target may remain latent |

Let `Y^G` contain task-relevant future outcomes: goal satisfaction, constraint
violations, duplicate or omitted effects, evidence completeness, cost, and
future observations. A declared response target `C*` is included when the
constructed family scores response decoding. In the compression family,
`C*=r_G(Z)` is deterministic: different observable histories already determine
different responses. In the observability family, different latent worlds may
share one `H` and `Z`; `Z` then determines `P(C*|H)`, while a later
verification transcript may reveal the realized target. Before that evidence
arrives, the safe control label can simply be `verify / reconcile` for every
twin.

Let `Adm_G(h)` be the currently admissible action labels. Compare future
interventions that are specified independently of the history prefix; an
infeasible action has a declared violation outcome. Two histories are
continuation-equivalent when they expose the same admissible labels and have
the same controlled future kernel for every such intervention:

```text
h \sim_G h'  iff
  Adm_G(h) = Adm_G(h') and
  for every prefix-independent continuation intervention u,
  P(Y^G, O_future | h, do(u)) = P(Y^G, O_future | h', do(u)).
```

For exact deterministic tasks, the definition can instead compare the language
of safe, goal-reaching continuation traces. For recursive online updates,
require right congruence: extending equivalent reachable histories by the same
admissible action-observation label must produce equivalent successors. In the
stochastic case, use a label-preserving probabilistic bisimulation. Equality of
these controlled kernels then implies equal outcomes for adaptive controllers
that start with the same internal state and use only future observations,
without inspecting the prior prefixes.

The task-relative operational causal state is the equivalence class

```text
Z_t=[H_t]_{\sim_G}.
```

A bounded representation `M = phi(H_t)` is exactly predictive/continuation
sufficient only if

```text
\phi(h)=\phi(h')\;\Longrightarrow\;h\sim_G h'.
```

For a narrower decision, `M` need preserve only the conditional law of the
required response `C*`, not every distinction in `Z`; in the deterministic
compression family this reduces to preserving `C*` itself. If a projection
merges histories with disjoint contract-satisfying responses, every controller
that sees only that projection must fail on at least one. A downstream
transformation of `M` cannot recover information deleted from it, but a new
world observation may reacquire information that memory lacks.

A practical, auditable approximation is

```text
\widehat Z_t=(b_t,q_t,c_t,e_t),
```

where `b_t` is a belief over relevant world facts, `q_t` is the residual goal,
constraint, authority, and commitment automaton, `c_t` is a ledger of action
occurrences and observed outcome claims, and `e_t` contains the minimal
artifact identity and evidence provenance needed by the contract. An
`unresolved` ledger status records ignorance; it is not an effect.
This factorization is not claimed to be more fundamental than a sufficiently
rich POMDP belief state; its value is that the distinctions are observable,
auditable, and separately ablatable.

### Predictive variety and requisite response variety

Let `K_Z` be the number of reachable continuation-equivalence classes. An exact
fixed-length code for the full `Z` needs at least `ceil(log2 K_Z)` bits. Under a
declared source distribution, `H(Z)` is an average information lower bound, not
the realized serialized length.

Full predictive state can be more detailed than successful control requires.
At the declared compression bottleneck, let `R_G(z)` be the nonempty set of
response labels accepted by the contract from feasible state `z`, with no new
observation before the response. The exact synthetic family preconstructs a
uniform set `{z_1,...,z_K}` whose accepted-label sets are pairwise disjoint; in
the simplest fixtures each `R_G(z_i)` is the singleton `{c_i}`. Define this
declared incompatible response-set size as `K_perp`. It is neither the total
number of causal states nor an unspecified maximal packing.

If `C*` is uniform over these `K_perp` mutually incompatible responses, there
is no new observation before the decision, and the memory alphabet has at most
`|M|` distinguishable values, then

```text
P_success <= min(1, |M| / K_perp).
```

For hard memory capacity `B_cap = log2 |M|`, hence

```text
P_success <= min(1, 2^B_cap / K_perp).
```

The bound is a one-shot counting argument, not a bound from the number of all
causal states. More generally, let `K_star` be the number of equiprobable target
labels in a declared response-decoding problem. In the compression construction,
`K_star=K_perp`; in an observability construction, several realized labels may
share the same pre-verification `Z`. For `K_star >= 2`, Fano's inequality gives

```text
P_dec_error >= 1 - (I(C*; M) + 1) / log2(K_star).
```

This becomes a task-failure bound only when success implies correct decoding of
`C*`. For non-uniform response distributions, use the form involving `H(C*)`.

`H_prefix` denotes the forced time depth to one common bottleneck decision;
`K_perp` denotes the response variety at that decision. In the first exact
experiment, the prefix is noiseless and semantically inert except for facts
that memory must carry. This makes `H_prefix=64, K_perp=2` comparable with
`H_prefix=8, K_perp=256`. It tests whether prefix depth and instantaneous
response variety are separable; it does not claim that horizon is irrelevant
to end-to-end reliability. Per-step failures and accumulated cost are added
later as separate variables.

### Memory, feedback, and semantic value

Let `V` be the complete read-only verification transcript available before the
response, including the verification query/action and result. It supplements
memory as side information:

```text
P_dec_error >=
  1 - (I(C*; M) + I(C*; V | M) + 1) / log2(K_star).
```

Zero conditional information cannot improve Bayes-optimal decoding in the
controlled setup. Positive information only relaxes the converse bound; it
does not guarantee practical benefit if evidence arrives late, costs too much,
or the controller cannot use it. Adaptive verification must count both query
and response in `V`.

For a fixed pre-verification response target `C*`, a read-only observation,
and an update satisfying the Markov condition
`C* -> (M_t, V_{t+1}) -> M_{t+1}`, data processing makes

```text
L_{t+1}
  := I(C*; M_t, V_{t+1}) - I(C*; M_{t+1})
   = I(C*; M_t, V_{t+1} | M_{t+1}) >= 0,
```

and one update has the accounting identity

```text
I(C*; M_{t+1})
  = I(C*; M_t) + I(C*; V_{t+1} | M_t) - L_{t+1}.
```

`L` is task information discarded by the update channel; it is not necessarily
caused only by compaction. If verification changes the world or the relevant
response variable, this identity applies only to the fixed pre-query `C*` and
must not be relabeled as a statement about the successor state.

For approximate control, declare a reachable-history distribution `p_G(h)`.
Let `J_G*(h)` be Bayes-optimal contract value from full observable history. A
decoder `delta` maps each codeword `m` to an initialized continuation controller
that subsequently sees only allowed future observations. Define
`d_{G,delta}(h,m) = J_G*(h) - J_G(h,delta(m))`, with explicit penalties for
goal, constraint, effect, evidence, and cost outcomes. Then

```text
R_G(D) =
  inf_{p(m | h), delta:
       E_{p_G(h)p(m | h)}[d_{G,delta}(h,m)] <= D} I(H; M).
```

This is the minimum retained information for a declared tolerance, rather than
the minimum text reconstruction error. A length- and marginal-preserving field
scramble measures fixed-policy reliance on that field; it does not prove causal
necessity and is not a general measurement of semantic information. Causal
necessity requires a support-preserving paired intervention or twin.

The connection to Ashby's requisite variety is concrete: memory, informative
feedback, and the available action repertoire must jointly cover the
task-relevant response variety that remains at the decision point.

Do not mix three budgets. `B_cap` is a hard number of memory states and belongs
in the counting margin. `I(C*;M,V)` belongs in decoding bounds. Serialized
bytes or tokens are engineering proxies reported separately.

The project's simplicity rule follows directly:

> Predictive simplification deletes only distinctions that cannot change the
> controlled future kernel. Decision-specific simplification may delete more,
> but only when the target-response distribution and tolerated task loss remain
> unchanged.

Fewer lines of code reduce audit surface, but the state channel cannot be
simplified below the task's requisite information.

## Falsifiable hypotheses

1. **Response variety over prefix depth.** At a common noiseless decision
   bottleneck, increasing `K_perp` at fixed `H_prefix` raises memory demand,
   while inert prefix padding at fixed `K_perp` does not change the exact bound.
2. **Counting and decoding margins.** In the constructed response family,
   `B_cap - log2(K_perp)` predicts the noiseless capacity transition, while
   `I(C*;M)/H(C*)` predicts noisy compression-family decoding better than
   prefix depth, raw transcript length, or call count.
3. **Typed inductive bias.** Against named suffix, type-agnostic compressor, and
   belief-only baselines, the typed approximation has lower regret and more
   interpretable failures. It is not claimed to beat the optimal arbitrary
   quotient. Hard-state capacity and serialized size are matched and reported
   as separate comparisons.
4. **Typed-field necessity.** A support-preserving paired intervention on a
   truly necessary belief, normative, outcome-claim, or evidence relation
   causes its preregistered failure class rather than a generic score decrease.
5. **Observability bottleneck.** No memory representation distinguishes latent
   worlds that induce the same `H`; a verification channel with zero
   `I(C*;V|M)` gives no Bayes decoding gain. The
   `I(C*;M,V)/H(C*)` margin is tested under controlled cost, timing, and
   controller competence without assuming proportional benefit.
6. **Continuation state over executor continuity.** After controlling primitive
   tool competence, deleting response-relevant state harms continuation more
   consistently than replacing the model or Harness while preserving it.
7. **Accumulated-risk qualification.** After adding per-step failures, horizon
   may reduce end-to-end reliability even at fixed bottleneck variety; this is
   modeled separately rather than attributed to memory demand.

## Decisive experiment

### Phase A: exact simulator, no language model

Build a dependency-free transactional-operation simulator only after the
definitions and twin fixtures agree. The environment exposes
`inspect / apply / verify / compensate / finish`. An application may commit,
fail without committing, time out after committing, or remain unresolved;
authority may expire, artifacts may change version, and receipts may be delayed
or noisy.

Construct the two bottlenecks separately before combining them. For the
**compression bottleneck**, generate different observable histories with
different required responses, then force a named bounded projection to collide:

```text
H != H', C*(H) != C*(H'), but phi(H) = phi(H').
```

The prefix is noiseless and semantically inert except for facts that memory
must carry. Every history reaches one common bottleneck after `H_prefix` steps,
where one of `K_perp` mutually incompatible responses is required and the
memory alphabet has hard capacity `B_cap = log2 |M|`. Include the paired
comparison `H_prefix=64, K_perp=2` versus `H_prefix=8, K_perp=256`.

Compression twins differ only in an already-observed fact:

1. authority was observed valid versus revoked or consumed;
2. a commitment was observed open versus discharged or released;
3. an outcome was observed committed versus explicitly not committed;
4. an artifact hash or accepted evidence lineage was observed to differ.

These facts are frozen after observation until the bottleneck. Any unannounced
drift would turn the fixture into an observability case.

For the **observability bottleneck**, generate different latent worlds that
induce the same complete observable history:

```text
X != X', C*(X) != C*(X'), but H(X) = H(X').
```

No alternative memory encoder can distinguish these worlds before new
evidence arrives. Observability twins differ only in a hidden fact:

1. a response was lost after an effect committed versus after no effect;
2. the world changed after checkpoint versus did not change;
3. an artifact changed behind the same path without an observation;
4. authority, capability, or a precondition changed without notification.

The correct policy for these cases is to verify or reconcile rather than
guess. Vary the complete verification transcript's information, timing, and
cost. Only after both isolated families pass should the simulator combine
uncertain or irreversible effects with per-step failure and accumulated cost.

Manipulate independently:

- forced prefix depth `H_prefix`;
- incompatible response variety `K_perp`;
- latent target-label variety `K_star` in the observability family;
- memory alphabet and hard capacity `B_cap`;
- verification-channel condition `W_V`, which fixes query options, noise,
  timing, and cost and produces transcript random variable `V`;
- distractor entropy; and
- effect uncertainty, irreversibility, per-step noise, and receipt noise.

Compare:

1. a privileged latent-world oracle, reported only as an observability upper
   reference;
2. full observable history with a Bayes-optimal controller;
3. the exact `Z` quotient and, at the constructed bottleneck, the exact `C*`
   label;
4. a bounded suffix or sliding window;
5. a specified capacity-limited, type-agnostic compressor;
6. belief-only structured state;
7. typed `belief + residual obligations + outcome-claim ledger + evidence`;
8. support-preserving typed-field twin interventions, one-field reliance
   ablations, and read-only verification channels.

Measure task success, violations, duplicate and omitted effects, false
completion, deadlock, oracle regret, representation collisions, first state
divergence, response-decoding accuracy, `I(C*;M)`, and `I(C*;V|M)`. Report hard
memory-state capacity, mutual information, and serialized bits, bytes, or
tokens as different quantities.

Advance only if all of these preregistered checks hold:

- the exact `Z` quotient has the declared count and satisfies right congruence
  on exhaustively enumerable small cases;
- in the noiseless response family, every deterministic encoder below the
  `K_perp` counting bound has a predicted collision, every randomized
  encoder-policy has nonzero decoding error, and the `C*` oracle is exact;
  noisy families are compared with their Bayes optimum rather than a universal
  100% target;
- every declared necessary type has at least one support-preserving twin and
  paired intervention whose failure is exposed by its ablation;
- no memory representation separates an observability twin before
  verification;
- a controlled symmetric verification channel with zero `I(C*;V|M)` gives no
  Bayes gain, while useful side information changes the frontier when timing
  and cost are fixed;
- on held-out configurations, the information-margin model improves predictive
  log loss by at least 10% over the best model using `H_prefix`, transcript
  size, and call count alone.

Stop the direction if the quotient is trivial, `H_prefix` and `K_perp` cannot
be varied independently, the compression and observability bottlenecks cannot
be isolated, named typed baselines have no auditable advantage, or the
information margins do not predict held-out outcomes. Do not repair a failed
theory by expanding the product.

### Phase B: one SwarmX request-binding counterexample, still no language model

Use the existing durable runtime as a concrete instance of idempotency-key
binding and effect-claim replay. Static inspection on 2026-08-06 found that
`TaskSideEffectReceipt` records WorkItem, Run, effect kind, idempotency key,
status, and result reference, but not a capability-call fingerprint. The
capability replay path appears to accept a committed receipt with the same key,
WorkItem, and effect kind without comparing current arguments. This is a
candidate request/replay binding collision, not yet an empirical bug claim.

Add at most one focused experiment/test module with a non-idempotent mock world:

- same key and same arguments must execute once and replay one result;
- same key and different arguments must be rejected as a semantic collision;
- mutation followed by lost response must remain unresolved and must not be
  repeated automatically;
- explicit non-commit, corrupted committed detail, restart/replay, and
  Session-link changes must produce their declared recovery decisions.

The primary outcome is agreement with the safe decision
`execute / replay / reconcile / reject`, plus world mutation count, duplicate
effects, ambiguous outcomes, and retained bytes. Session-only and audit-only
projections are negative controls; neither is durable task authority.

This phase changes zero production modules. If the counterexample is confirmed
and the general Phase A result survives, the smallest later repair is a
versioned capability-call fingerprint plus a replay check in existing Core
modules. The fingerprint is the SHA-256 digest of a fixed canonical
serialization of `{capabilityId, operation, arguments}`; the serialization rule
or its version is part of the persisted contract. It excludes retry-varying
Run, call, lease, sequence, and timestamp fields and remains invariant when a
receipt moves from `uncertain` to `committed`. This also avoids relying on the
current colon-joined `effectKind` as a unique encoding of capability and
operation.

This digest identifies parsed request input, not an action occurrence or actual
effect. Effect identity would additionally require an external transaction or
resource identity, a verified postcondition, and a causal binding to the
execution occurrence; Phase B does not claim to solve that larger problem.

Compatibility is part of the experiment, not cleanup. Existing event logs
cannot reconstruct arguments that were never stored. Prefer an optional parser
field whose new writers always populate it; a legacy receipt without the field
must fail closed as `unknown / reconcile`, never be silently backfilled. If the
strict versioned event schema instead requires a new event version and explicit
migration, expand the code budget openly. Add focused tests, update the codebase
map, and run the documentation check. This repair would be evidence for the
general theory, not the paper's contribution by itself.

### Phase C: small model study and external validity

Only after Phases A and B advance:

- compare transcript, natural-language summary, belief-only, typed, and oracle
  states once under matched hard-state capacity and separately under matched
  serialized bytes or tokens;
- hold model, tools, prompts, and generation settings fixed;
- use deterministic hidden validators and report every trajectory;
- test at least one held-out task family and one second model only after the
  fixed-model effect is stable;
- use executor replacement or cross-harness handoff as a 2-by-2 external
  validity test, not as the thesis.

No private chain-of-thought is transferred or scored. The study evaluates
observable state and action outcomes.

## Novelty audit

### Observed

- Ashby's law of requisite variety and the Conant-Ashby good-regulator theorem
  already connect regulation, internal models, and information
  ([Ashby](https://ashby.info/Ashby-Introduction-to-Cybernetics.pdf),
  [Conant and Ashby](https://doi.org/10.1080/00207727008920220)).
- POMDP belief state, predictive/causal states, and information bottlenecks
  already establish that histories may admit sufficient compressed states
  ([POMDP](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf),
  [causal states](https://arxiv.org/abs/cond-mat/9907176)).
- A 2026 bounded-interaction Myhill-Nerode result already gives canonical,
  minimal quotients for finite POMDP interaction. Historical quotienting is not
  itself a new theorem
  ([paper](https://arxiv.org/abs/2603.21399)).
- DeMem already frames Agent memory as decision-centric rate-distortion, while
  an information-theoretic Agent-system paper studies compressor-predictor
  mutual information. Generic rate-distortion or mutual information is not the
  gap
  ([DeMem](https://arxiv.org/abs/2605.10870),
  [compressor-predictor study](https://arxiv.org/abs/2512.21720)).
- Agent-BRACE separates belief and policy under partial observability; MAGE and
  LongHorizon-Harness manage execution state explicitly. “Explicit state beats
  a growing transcript” is occupied
  ([Agent-BRACE](https://arxiv.org/abs/2605.11436),
  [MAGE](https://arxiv.org/abs/2606.06090),
  [LongHorizon-Harness](https://arxiv.org/abs/2608.01964)).
- AgentO models agents, tasks, workflows, and resources in OWL/RDF, and Agentic
  Redux applies BFO plus typed semantics to auditable Agent domains. Applying
  an ontology to Agents is not new
  ([AgentO](https://eprints.cs.univie.ac.at/8749/),
  [Agentic Redux](https://arxiv.org/abs/2606.04903)).
- W3C PROV-O already separates Entity, Activity, Agent, Plan, use, generation,
  derivation, and delegation. Proof-Carrying Agent Actions already relates
  authority, action, approval, runtime receipt, and outcome evidence
  ([PROV-O](https://www.w3.org/TR/prov-o/),
  [PCAA](https://arxiv.org/abs/2606.04104)).
- CAVA already formalizes canonical action identity, approval binding, receipt
  integrity, and a runtime-portable projection. Always-OnAgents already treats
  permissions, commitments, provenance, and effects as persistent operational
  state. These works sharply limit any claim that typed action state or durable
  responsibility is itself new
  ([CAVA](https://arxiv.org/abs/2607.13716),
  [Always-OnAgents](https://arxiv.org/abs/2606.30306)).
- ACRFence already addresses semantic rollback, resurrected authority, and
  replay-or-fork decisions in Agent recovery. Receipt identity and safe replay
  are therefore an implementation surface and comparison point, not an
  independent novelty claim
  ([ACRFence](https://arxiv.org/abs/2603.20625)).
- Dynamic process migration has history-equivalence work, while recent Agent
  studies directly cover resume contracts and non-atomic tool effects
  ([process migration](https://arxiv.org/abs/2412.08314),
  [resume contracts](https://arxiv.org/abs/2608.03836),
  [verified tool calls](https://arxiv.org/abs/2608.02645)).
- Kolchinsky and Wolpert formalize semantic information through counterfactual
  causal value to viability; the paper may adapt that intuition but cannot
  claim to invent semantic information
  ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6227811/)).

### Inference

The candidate gap is the conjunction below, not any component in isolation:

1. separate loss of already-observed response-relevant information from latent
   world ambiguity that no memory could resolve;
2. define a task-relative continuation quotient whose labels include goals,
   safety and authority, effect identity, commitments, and evidence;
3. independently manipulate forced prefix depth `H_prefix` and incompatible
   response variety `K_perp` in a tool-Agent benchmark;
4. test hard-capacity and response-decoding bounds with verification as measured
   side information under uncertain and irreversible effects; and
5. operationalize the distinctions as a typed, auditable state and instantiate
   one request-binding collision surface in a small event-sourced runtime.

The searches did not locate a paper that performs all five. This supports a
candidate research gap; it does not establish priority or absence.

### Uncertainty

The strongest reviewer objection is that this is a renaming of POMDP belief
state, causal-state abstraction, Ashby, Fano, and DeMem. Only an independent
`H_prefix x K_perp` experiment, separate compression and observability twins,
preregistered information bounds, type-specific failures, and a real
request-binding/replay case can answer that objection. A broad ontology, more
runtime features, or philosophical language cannot.

Recent 2026 indexing is changing daily. Before submission, repeat the exact
queries, inspect cited-by links for every direct neighbor, and invite one
control/information-theory reader and one formal-ontology reader to attack the
claim.

## Theory-to-SwarmX mapping

SwarmX already contains useful but incomplete realizations of the theoretical
objects:

| Theory object | Current SwarmX realization | Research use or limitation |
| --- | --- | --- |
| Task instance | `TaskWorkItem` and append-only `TaskRuntimeEvent` replay | Strongest task authority, but no first-class goal/acceptance contract |
| Process occurrence | `TaskRun`, lease, progress, retry, and cancellation | Distinct from task and Session |
| Operational memory | Checkpoint plus immutable artifact references | Candidate projection, not automatically world state |
| Normative state | Approval records, budgets, cancellation, capability policy | Distributed; no unified residual norm automaton |
| Effect claim and replay binding | `TaskSideEffectReceipt` and gateway outcome | Strong low-cost test surface; the receipt is a claim and its request-input binding may be incomplete |
| Evidence | Content-addressed blobs, artifact hashes, and hash-chained `AuditEvent` records | Audit events support claims but are not receipts, world facts, or task authority |
| Observation view | Canonical Session JSONL and Session-to-WorkItem links | Session may observe work; it does not own work |
| Controller | Model, Harness, Agent, and worker kept as separate concepts | Enables executor-replacement ablation |
| World observation | Workspace hashes and diffs | No first-class World identity or snapshot exists; a mock capability ledger is a proposed fixture, not current product state |

Important limits from source inspection:

- `task` is used for requests, temporary todos, evaluation samples, and durable
  WorkItems; the experiment must use WorkItem ids explicitly.
- exported `ContextPacket` schemas are not yet used by `Agent.call`; they cannot
  be cited as an implemented context experiment.
- Session, task-runtime, audit, rendering, and evaluation events are distinct
  streams, not one canonical event ontology.
- audit correlation is useful evidence but is not sufficient to reconstruct a
  WorkItem.
- the repository has no first-class World or generic Task contract. The first
  experiment should use a mock world and declared validator, not redesign the
  product.

This is the desired theory-practice loop:

```text
ontology defines distinctions and identity criteria
                 |
                 v
continuation equivalence defines the minimal task state
                 |
                 v
information theory bounds memory and values verification
                 |
                 v
synthetic twins expose exact collisions and failure classes
                 |
                 v
SwarmX receipt replay provides one candidate request-binding collision surface
                 |
                 v
model and harness studies test external validity only after the bound survives
```

## Development order and code budget

### Submission lane already in the repository

The two existing review manuscripts remain the shortest path to an actual
submission. Resolve authorship placeholders, repeat their closest-work search,
freeze bibliography and reproducible builds, and submit or archive them in
parallel. They do not determine the new empirical paper's thesis.

### Research lane

1. Freeze definitions, assumptions, twin pairs, outcome vector, distortion,
   bounds, hypotheses, and stop rules. Zero code.
2. Implement the exact simulator and exhaustive checker in at most three
   authored evaluation files. Zero production modules and zero model calls.
3. Add one focused SwarmX request-binding experiment/test for same-key,
   different-argument replay. Zero production modules and no bug claim before
   the test fails as predicted.
4. Add a minimal repair only if the counterexample is confirmed and the general
   theory survives. Prefer a versioned capability-call fingerprint and
   fail-closed legacy handling in two existing Core modules plus focused tests
   over a new abstraction layer; account explicitly for versioned-log
   compatibility.
5. Run the small fixed-model study, then one executor-replacement study.
6. Freeze the artifact and write the paper around the information-variety
   result, not around SwarmX feature breadth.

Do not add a new service, store, ontology framework, workflow DSL, memory
subsystem, UI, or telemetry path for this paper. Every added field must name its
source, identity rule, integrity rule, bit/byte cost, paired twin, and ablation.

## Deliberately deferred

Until a surviving experiment names one as a required variable or repair, defer:

- general cross-harness live migration;
- daemonizing the task controller;
- Desktop surfaces for durable work;
- Provider and native-tool parity breadth;
- generic long-term memory, multi-agent collaboration, or skill evolution;
- a universal world model, event schema, trace format, or task ontology;
- a leaderboard or broad harness comparison.

## Literature search record

Searches were run on 2026-08-06, Asia/Shanghai, against arXiv, ACL Anthology,
conference or institutional repositories, standards sites, and primary papers.
Search ranking and same-day indexing remain sources of uncertainty.

Exact claim-selection queries included:

```text
control sufficient statistic POMDP history information bottleneck rate distortion state representation paper
information theoretic bounded rational control rate distortion sequential decision making paper
LLM agent complex multi-step tasks ontology state action event artifact provenance paper
constrained POMDP sufficient state representation safety constraints information bottleneck
LLM agents POMDP belief state context management multi-step tool use paper 2025 2026
LLM agent information bottleneck context compression task relevant state paper
agent memory control sufficient state long horizon tasks benchmark 2026
ontology agentic AI task action event artifact state workflow formal semantics paper 2025 2026
"An Information Theoretic Perspective on Agentic System Design"
site:arxiv.org "information-theoretic" agentic system design compressor predictor
"control-sufficient" representation information bottleneck sequential decision paper
"task information rate" control agent rate distortion
"control sufficient state" POMDP representation paper
rate distortion control state abstraction regret information bottleneck safe reinforcement learning
LLM agent explicit world state goal constraints provenance long-horizon execution state paper 2026
LLM agents ontic epistemic state distinction action effect intent evidence paper
agentic AI formal semantics action event task state ontology causal provenance paper
LLM agent world model state machine multi-step tools explicit state benchmark paper
LLM agent ontic epistemic deontic state provenance authority constraints long horizon paper
"normative state" LLM agent belief task execution
"causal sufficient state" agent sequential decision information theory
typed task state provenance constraints LLM agent long-horizon verification paper
site:arxiv.org agent history equivalence minimal sufficient state Myhill Nerode sequential decision process
site:arxiv.org bisimulation state abstraction constraints safety evidence provenance agent sequential decision
site:aclanthology.org LLM agent normative state authority provenance multi-step tool use
site:arxiv.org LLM agent ontic epistemic deontic state action effect verification
site:arxiv.org ontological typing state abstraction normative constraints provenance sequential decision agents
site:arxiv.org deontic action ontology agent authority permission commitment evidence provenance workflow
site:w3.org PROV-O activities entities agents provenance ontology recommendation
AgentO ontology agentic systems tasks workflows resources paper 2026
formal ontology LLM agents task workflow action event state provenance 2026 AgentO
"ontic" "epistemic" "agent" "provenance" artificial intelligence action
Ashby law requisite variety original book cybernetics pdf
semantic information autonomous agency viability Kolchinsky Wolpert paper
causal states computational mechanics minimal sufficient statistic future prediction paper
Conant Ashby every good regulator system must be model original paper pdf
site:arxiv.org LLM agent requisite variety long horizon task information rate
site:arxiv.org "rate-distortion" "constraint violation" state representation agents safety
site:arxiv.org "normative" "rate-distortion" agents memory
site:arxiv.org "operational state" LLM agent safety liveness provenance
site:arxiv.org agent memory constraints commitments provenance decision conflict long horizon
site:arxiv.org long-horizon agent benchmark independently vary horizon state complexity information budget
site:arxiv.org LLM agent operational variety horizon complexity benchmark memory bits
site:arxiv.org agent benchmark distinguish horizon length state-space complexity memory budget
site:aclanthology.org long-horizon agents state complexity horizon memory benchmark
"Myhill–Nerode theorem for bounded interaction" agent-bounded indistinguishability
"Remember the Decision, Not the Description" rate-distortion agent memory
2025 2026 LLM agent benchmark horizon versus state complexity entropy task success
LLM agent benchmark vary horizon and partial observability independently memory capacity
LLM agent non-atomic tool calls uncertain effects belief state idempotency formal verification 2026
semantic information control goals task success causal intervention information theory agents
task relevant information control rate distortion POMDP directed information
2025 2026 agent world model sufficient statistic history action value equivalence
2025 2026 LLM agent irreversible actions partial observability effect uncertainty commitment state
2025 2026 LLM agent "requisite variety" memory feedback information theory
LLM agent "operational causal state" long horizon
LLM agent "memory-feedback" information bound verification
LLM agent "task-relative causal state"
```

Direct-neighbor records additionally inspected by identifier on the same date
were CAVA (`arXiv:2607.13716`), Always-OnAgents (`arXiv:2606.30306`), and
ACRFence (`arXiv:2603.20625`). They were added as claim limiters, not counted as
evidence that the conjunction above is absent elsewhere.

Before submission, repeat these queries, search direct neighbors' references and
cited-by links, and record every included or excluded close work with a reason.
