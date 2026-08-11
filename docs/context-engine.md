# Coding-agent Context Engine

The Context Engine turns a fixed history snapshot into the bounded context a
coding Agent receives. Raw events and content-addressed artifacts are truth;
task state, evidence, visibility decisions, and manifests are rebuildable
projections.

## Authority and scope

Canonical Session and WorkItem logs keep their existing ownership. The engine
accepts events through an `EventStore` interface and never treats a summary,
index, or status projection as execution authority. Its SQLite WAL and JSONL
stores are standalone replay-harness implementations, not a third production
conversation or durable-task authority. Production hosts must adapt the
applicable canonical log into the same immutable event contract.

Version 1 includes:

- strict sourced event, artifact, task-state, evidence, configuration, and
  context-manifest contracts;
- an append-only SQLite WAL store, a JSONL replay adapter, and a local
  content-addressed artifact store;
- deterministic tool-call/result normalization and observation masking;
- rule-derived structured state whose fields cite source event ids;
- BM25 lexical evidence retrieval over a fixed snapshot;
- deterministic citation/hash verification; and
- priority-and-slot context assembly with explicit overflow.

The request budget is a property of the final Provider request, not only of the
historical projection. A compile accounts for Agent, Skill, and Memory
instructions, the current user input and attachments, discovered tool schemas,
provider-hosted tool descriptors and injected routing instructions, the
selected historical projection, and the reserved output. The selected
Model or Model Supply may declare the context window and output limit; an
explicit Agent client value overrides it. A compatibility fallback is allowed
only when the manifest identifies it as a fallback rather than a model fact.

Session compilation has two pure phases over the same immutable snapshot. The
preflight phase runs before MCP startup and can reject mandatory static input.
After tool discovery, finalization recompiles with the complete tool schemas.
Only the finalized manifest is admitted to audit history.

For paired low-level evaluation, `createContextEngineEvaluationConfig` exposes
stable projection variants over the same compiler and event snapshot:

- `full`: lossless history or explicit overflow;
- `mask_tail`: the former recent-unit/capsule baseline;
- `checkpoint_tail`: deterministic checkpoint plus verbatim recent tail;
- `checkpoint_tail_bm25`: checkpoint/tail plus verified BM25 evidence; and
- `auto`: full below pressure, then checkpoint/tail plus BM25.

The policy configuration stores projection and evidence choices separately.
`preserveRecentAtomicUnits`, fallback budgets, and the pressure ratio are
explicit inputs, so an evaluation does not inherit hidden per-variant values.

For harness and research comparisons,
`createContextEngineProfileConfig` exposes these named recipes:

| Profile | Fidelity recorded in the manifest | Pressured projection implemented here |
| --- | --- | --- |
| `opencode_v2` | public-source reimplementation | 20K trigger buffer, an up-to-8K serialized recent tail, 2K-character tool-result sketches, fixed anchored-summary headings, and a 4,096-token summary ceiling |
| `codex_cli` | public-source reimplementation | public local-compactor shape: canonical initial context, up-to-20K tokens of recent user messages, and the compaction summary last; this is not hosted `/responses/compact` parity |
| `claude_code` | public-behavior reimplementation | summary plus recent exchanges with trusted system/project observations re-injected; exact private prompt, retention heuristics, and service behavior are intentionally not claimed |
| `hermes` | public-source reimplementation | 50% trigger (75% floor below 512K), protected head, 20%-of-threshold/protected recent tail, clearing old tool results over 200 characters before a structured summary, and a dynamic summary ceiling up to 12K; iterative state and micro-compaction remain outside this snapshot recipe |
| `reasonix` | public-source reimplementation | 85% low-frequency trigger, pinnable first user turn, 10% bounded recent tail, protected errors/`[[keep]]` turns, one digest, and candidate-size validation |
| `lcm` | paper reimplementation | hierarchical source-linked summary tree plus exact read-only search/read tools over losslessly retained raw events |
| `parallel_compaction` | paper reimplementation | token-balanced old-history partitions summarized concurrently and merged in deterministic partition order |
| `resum` | paper reimplementation | original task turn plus a periodically regenerated compact reasoning-state summary |
| `swarmx_auto` | native | lossless below pressure; source-linked extractive checkpoint, recent atomic tail, and verified BM25 above pressure |
| `baseline_full` | native baseline | lossless full history or explicit overflow |

An evaluation arm changes only configuration and its injected summary provider:

```ts
const config = createContextEngineProfileConfig({
  profile: "reasonix",
  pressureThresholdRatio: 0.85,
  summaryFailureMode: "error",
});

const engine = createSessionContextEngine({
  sessionId,
  history,
  config,
  summaryProvider,
});
```

Use `summaryFailureMode: "error"` for fidelity comparisons so infrastructure
failure cannot silently improve apparent cost. Use `deterministic` for
availability experiments; the manifest then distinguishes the fallback arm.

All recipes deliberately keep three SwarmX invariants even where the original
harness differs: a tool call/result is never split, canonical history is never
rewritten, and every derived block remains source-addressable. Consequently,
the OpenCode adapter will not split inside a serialized atomic tool exchange;
the Codex adapter renders its replacement-history topology inside the compiled
instruction block rather than calling the opaque hosted endpoint; Hermes head
protection is evaluated per immutable snapshot rather than carrying private
cooldown state; and Reasonix uses the manifest's declared token estimator rather
than provider-calibrated tokens-per-character unless an evaluation adapter
supplies that calibration. These are visible adaptations, not silent parity
claims.

The current adapters compile one immutable request snapshot. They do not import
or mutate a harness's private rolling-session state, cooldown counters, cache
entries, or hosted compaction receipts; repeated evaluation boundaries rebuild
from canonical events. This avoids cascading summary loss and makes paired
replay deterministic, but it is a different experimental arm from the native
harness's full lifecycle. Native-lifecycle parity should be evaluated through
that harness itself, using the same TRACE-style continuation scorer.

The profile name selects a reproducible recipe, while the factory still accepts
explicit window, output-reserve, pressure, recent-tail, summary/evidence-budget,
one-to-four summary-partition, and summary-failure overrides. A model-backed
profile receives a `ContextSummaryProvider`. Preflight
only estimates and never spends a model call; finalization invokes the provider
with source event ids and a cancellation signal. The configured failure mode is
either explicit error or deterministic extractive fallback. Manifests record
profile, fidelity, provider/fallback mode, summary subcalls, and token usage.
Parallel and LCM leaf work is token-balanced into the arm's declared one-to-four
ordinary blocks; LCM may spend one additional parent-merge call. The schema cap
keeps a single oversized history from fanning out into unbounded Provider calls.

LCM installs `context_search` and `context_read` as ordinary read-only local
tools. Search returns verified event/hash/range citations; read returns an exact
bounded character range from the immutable compile snapshot. Their schemas are
therefore included in final Provider-request accounting.

The Core Session adapter converts an immutable `MessageChunk` history into the
same event contract for native Agent calls. It preserves each tool call/result
pair as one unit, keeps the current user turn as the Provider's native input,
and compiles prior history plus trusted host system observations before MCP
startup or a Provider request. Desktop direct-Agent, side-chat, child-Agent,
background-activation, and Workflow paths install this adapter; external ACP
Harnesses continue to own their native context and Session behavior. Successful
Desktop compiles append a content-free audit record containing only manifest
hashes, counts, model identity, and token totals.

Vector/hybrid retrieval, learned routing, WorkItem adapters, Hermes rolling
micro-compaction, and provider-hosted compaction remain replaceable later
implementations. Recursive Language Models are not exposed as a profile: the
RLM design treats the long input as an external environment and lets a model
write programs and recursively invoke models over selected slices. A faithful
implementation therefore needs a separately authorized sandboxed REPL and
recursive-call budget; calling retrieval plus a summary “RLM” would be a false
comparison. Changing an evidence provider does not change `EvidencePack` or the
assembler contract.

## Research basis and adaptation limits

The recipes were checked against the current public implementations and primary
papers, rather than inferred from product names:

- [OpenCode V2 compaction](https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/compaction.ts)
  supplies the serialized-tail, trigger-buffer, truncation, and anchored prompt
  constants.
- [Codex local compaction](https://github.com/openai/codex/blob/main/codex-rs/core/src/compact.rs)
  supplies the retained-user-message and replacement-history logic. Hosted
  compaction is provider-side and opaque, so it is excluded from parity claims.
- [Claude Code context-window documentation](https://code.claude.com/docs/en/context-window)
  documents automatic/manual compaction and reinjection behavior but not the
  private implementation.
- [Hermes context compression](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/context-compression-and-caching.md)
  and its public compressor supply the head/tail, pruning, summary, cache, and
  micro-compaction behavior; this profile implements the batch snapshot path.
- [Reasonix compaction](https://github.com/esengine/DeepSeek-Reasonix/blob/main-v2/internal/agent/compact.go)
  supplies the cache-first trigger, tail, pinning, keep, and candidate rules.
- [Recursive Language Models](https://arxiv.org/abs/2512.24601),
  [Lossless Context Management](https://arxiv.org/abs/2605.04050),
  [Parallel Context Compaction](https://arxiv.org/abs/2605.23296), and
  [ReSum](https://arxiv.org/abs/2509.13313) define the compared research
  mechanisms. LCM, parallel compaction, and ReSum fit the present projection
  authority; RLM does not.

The newest work also changes how these profiles should be evaluated. TRACE's
[paired closed-loop compaction-event evaluation](https://arxiv.org/abs/2608.06503)
reports that compression can weaken even recent interactions and increase
blocked or repeated actions; the RSI loop should therefore fork from the same
environment state at each compaction boundary and score task continuation,
repetition, blocked effects, reliability across seeds, tokens, latency, and
cost—not summary similarity alone. [SRLM](https://arxiv.org/abs/2603.15653)
finds that recursive program selection and uncertainty signals matter, and that
RLM recursion can hurt when the input already fits the model window. This
supports the engine's lossless-below-pressure rule and makes an unverified RLM
profile a poor first experiment.

Two adjacent 2026 directions remain outside this layer. [Recursive Agent
Harnesses](https://arxiv.org/abs/2606.13643) recurse over full tool-using agent
harnesses and therefore belong in the task/executor runtime, while [online KV
cache compaction](https://arxiv.org/abs/2608.00902) changes inference-server
token retention and needs Provider/runtime support. Both are worth later arms,
but neither can be reproduced honestly by rewriting application messages.

## Raw events and artifacts

Every event has one session, turn, monotonic sequence, stable id, content hash,
causal parents, labels, and optional task/tool/artifact metadata. The hash binds
all event content except the hash field itself. Appending a reused id with
different content, a non-increasing sequence, a missing causal parent, a torn
JSONL tail, or a complete malformed record fails closed.

A tool result must name an earlier tool call in the same snapshot. The
normalizer emits the call and result as one atomic unit; assemblers and maskers
cannot select only one half. An unmatched in-flight call may remain visible as
one pending unit, but an orphan result is rejected.

Large observations can be placed in `LocalContextArtifactStore`. References use
their SHA-256 digest, reads verify the digest, and callers can preview or read a
bounded byte range. The source event still retains the reference, hash, exit
code, affected paths, and a bounded payload capsule. The v1 externalizer applies
a configured byte threshold to tool results, patches, and test results before
append, preserving declared salient lines in the capsule.

## Projections

When the complete request remains below the configured pressure threshold, the
Session adapter uses a `full` projection: every prior atomic unit is rendered
verbatim and no age-based masker or slot quota may discard it. Recent-unit
counts affect only a pressured projection.

When the threshold is crossed, the adapter uses `checkpoint_tail`: an
extractive, source-linked `SummaryCheckpoint` covers older units, the newest
atomic units remain verbatim, and query-specific BM25 evidence is verified
against the same snapshot. Checkpoint generation is deterministic and
rebuildable; a failed or invalid checkpoint never becomes active. The manifest
records projection mode and requested/effective evidence mode separately, so a
full projection cannot claim that retrieval ran.
Any checkpoint that covers old units includes at least one source-id-qualified
excerpt, prioritizing user requests and failed effects; if even that minimum
cannot fit, compilation overflows instead of activating an empty checkpoint.

The deterministic masker returns exactly one visibility for every atomic unit:
`full`, `capsule`, `ref`, or `omit`. Current-turn units, the newest failure,
pinned constraints, blockers, decisions, task contracts, and uncommitted
changes remain full. Recent observations and retrieval hits remain at least
capsules. Older artifact-backed results may become refs, while superseded or
duplicate successful observations may be omitted. Mandatory semantic records
never disappear silently.

Structured state distinguishes observed facts, inferred hypotheses, and planned
work. Every sourced field records its source event ids and the sequence at which
it was valid. The projector only uses explicit structured payloads and does not
guess current Git, filesystem, branch, or test state from prose. Hosts inject
those current observations separately when assembling a request through the
strict `runtimeContext.contextObservations` boundary. Host observations are
trusted request state with explicit ids, slots, priorities, and mandatory
flags; arbitrary runtime objects are never serialized into model context.

BM25 retrieval searches normalized event text in one immutable snapshot. It
returns sources and coverage, not invented claims. Sources identify the event,
content hash, exact character range, excerpt, and current/superseded/conflicting
status. The deterministic verifier rejects a mismatched snapshot, nonexistent
event, bad hash, invalid range, or excerpt mismatch and removes every claim
without valid supporting sources.

## Assembly and observability

The priority assembler fills slots in this order: system/safety rules, current
task contract, structured state plus live observations, recent atomic units,
query-specific evidence, summaries, then other capsules. Each item declares its
slot, priority, token estimate, source event ids, and whether it is mandatory.
Optional items are selected within configured slot quotas and the global input
budget. If mandatory items alone exceed either boundary, compilation throws
`ContextOverflow`; no lower adapter may silently truncate the result.
Selection reserves the global tokens required by mandatory items in later
slots, so earlier optional evidence can never starve a checkpoint or another
mandatory item that already passed preflight.

Each successful compile records a manifest containing:

- the event snapshot id, configuration hash, and model version;
- requested and effective evidence mode plus any fallback chain;
- included item and event ids, omitted item ids and reasons;
- input/output token reservations and per-slot usage;
- context-window source, fixed-request tokens, total projected input, pressure
  threshold, and projection mode; and
- selected profile/fidelity, summary path, summary subcall/token counts, and
  any deterministic fallback; and
- compile latency and a deterministic rendered-context hash.

Agent instructions and the current user turn remain explicit Provider inputs.
The Session adapter budgets the selected historical projection and trusted host
system observations; it does not duplicate the current user text in the
compiled instruction block. A malformed Session history, including an orphan
tool result, fails before the Provider request.

Configuration selects stable component ids. Version 1 registers the SQLite WAL
event store, JSONL replay adapter, local CAS, deterministic
normalizer/masker/projector/verifier, BM25
retriever, and priority assembler. Unknown component ids fail validation instead
of falling back to a different semantic path.

## Replay acceptance contract

A replay is valid only when the same ordered events, snapshot, configuration,
model version, request, and live observations reproduce the same normalized
units, sourced state, evidence sources, included event ids, and context hash.
Focused tests cover atomic tools, append-only rejection, artifact range/hash
checks, supersession, unsupported claims, stale or forged citations, mandatory
overflow, prompt-injection text remaining untrusted data, and deterministic
manifest replay.

Focused Session-adapter tests additionally prove that more than eight old
atomic units remain verbatim below pressure, instructions/current input/tool
schemas participate in final request accounting, finalization uses the same
snapshot, pressured projections activate a source-linked checkpoint plus
verified BM25 evidence, and a manifest never reports retrieval when none ran.
