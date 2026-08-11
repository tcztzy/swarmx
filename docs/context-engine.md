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

The Core Session adapter converts an immutable `MessageChunk` history into the
same event contract for native Agent calls. It preserves each tool call/result
pair as one unit, keeps the current user turn as the Provider's native input,
and compiles prior history plus trusted host system observations before MCP
startup or a Provider request. Desktop direct-Agent, side-chat, child-Agent,
background-activation, and Workflow paths install this adapter; external ACP
Harnesses continue to own their native context and Session behavior. Successful
Desktop compiles append a content-free audit record containing only manifest
hashes, counts, model identity, and token totals.

Vector and hybrid retrieval, recursive summaries, static map-reduce, RLM depth
0/1, learned routing, and WorkItem adapters are later replaceable
implementations. Changing an evidence provider does not change `EvidencePack`
or the assembler contract.

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
those current observations separately when assembling a request.

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

Each successful compile records a manifest containing:

- the event snapshot id, configuration hash, and model version;
- requested and effective evidence mode plus any fallback chain;
- included item and event ids, omitted item ids and reasons;
- input/output token reservations and per-slot usage; and
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
