# Shared semantic memory

`@swarmx/memory` owns shared semantic memory: private, owner-readable research knowledge persisted
across sessions as OKF Markdown concepts directly under `$SWARMX_HOME/memory`. User notes,
searchable observed conversations, reusable procedures, and the structured concept pool have
separate roles. Native skills remain owned by their runtimes; concepts add explicit dependencies.

Memory supports the [long-term research product direction](product-direction.md). Saving an
observation or passing a structural validator does not establish that a policy, prompt or skill
improves later work. Host work management keeps independent acceptance and resource feedback;
behavioral comparison, adoption and withdrawal are tracked in [ROADMAP](../ROADMAP.md).

## Learning lifecycle

- `memory/USER.md` stores standing user preferences (1375 Unicode characters). Empty notes are
  valid. Updates require the last read revision, reject overflow, and use a private atomic file
  replacement.
- The Host freezes a short Memory tool entry point and the user note for each session. The full
  authoring guide and concepts are loaded on demand through `read_memory_guide` and the existing
  search/read/load actions. The navigation index is not injected into new session context.
  Existing session snapshots survive Host restarts; new sessions
  see new notes. The snapshot is installed when creating a task, or on the first observed turn
  of an existing native task. Disabling memory stops learning and gives new tasks empty snapshots;
  it does not erase context already given to an existing task. Retrieved knowledge is data and cannot grant tool permissions.
  Codex/Claude receive native developer/system context. Hermes/OpenClaw receive a marked context
  appendix on the first observed turn; it is removed from the displayed user message.
- Observed user/assistant messages are recalled from the execution directory's journal.
  Search returns original text, session/run identifiers and event references, not invented summaries.
- Eligible terminal executions (success, failure or cancellation) and trusted work-acceptance
  feedback form a durable backlog in the
  execution journal. The review interval counts across this execution directory, including short
  child sessions. Ten executions by default, ten tool calls in one execution, or an unsuccessful
  outcome starts a review. New acceptance or correction feedback starts a review without waiting
  for that interval. The feedback keeps its own source event ID, so acknowledging the earlier
  execution cannot consume a later assessment. A manual review is also available. Busy reviews do not drop new work.
  A directory-scoped SQLite execution lock admits one queue consumer across desktop and ACP Hosts.
  It is held through review and application, released on normal exit or by SQLite after process
  termination, and stores no separate review state. A competing consumer leaves the durable queue
  intact; later startup or resume rechecks it. Journal writes remain available while review runs.
  Startup resumes queued work after runtimes are attached; failures retain the batch for the next
  eligible completion or feedback, startup or manual retry, without a retry timer. Disabled learning pauses it.
  Source-run review eligibility and grants are preserved and intersected with current policy.
  Reviews produce validated candidate operations;
  they cannot approve themselves or execute product tools. Review failures are visible in the
  journal and UI; no claim is made that an LLM will extract every important fact correctly.
  The user chooses a configured Codex or Claude runtime for review (Codex by default); the
  foreground chat may use any supported runtime. Reviews time out after two minutes and are
  cancelled on Host shutdown. Reviews receive no Host product MCP credential and reject observed
  tool calls and interactions. The Host requests native restrictions; unmodified upstream adapters
  determine their effect, including internal title calls and session persistence. The review uses existing
  credentials and consumes model usage. Bounded snapshots include observed execution settings,
  outcomes, timestamps, tool evidence, delegation reasons, queued acceptance feedback and existing concepts. Omitted evidence
  is identified explicitly and is never treated as inspected. Unknown native versions or actual
  provider routes stay unknown. Cancellation is not a quality failure; normal completion is not
  proof of task correctness.
  One review considers user preferences, harness/model/effort/provider selection, versioned agent prompts
  and skill improvements. Selection experience uses the `agent-selection` tag. Memory creations
  and updates cite the journal snapshot through `urn:swarmx:execution:<event-id>`.
  Project-local agent/skill files can opt into validated evolution; see
  [learning resources](learning-resources.md). Native runtimes retain discovery and loading ownership.
  A review saves its validated operation plan before applying it. Restart replays that plan, not a
  new model response. Each applied or staged operation is recorded; idempotent writes close the
  write/receipt crash window. Only a completed batch acknowledges its exact source event IDs.
  The persisted `terminalIds` field now includes both terminal-execution and feedback event IDs.
  Batches contain at most 100 chronological sources. Before saving a job, the Host selects a
  complete prefix that fits the existing 40,000-character evidence limit. Queued sources have snapshot priority;
  feedback arriving during preparation or review stays in the next batch. If the bounded snapshot
  cannot fit one source, review fails without acknowledging it. Large collections of individually
  bounded feedback drain in separate batches. Concurrent executions stay pending.
  A no-change review records its reason. Writes are per-target,
  not a transaction spanning a whole review; errors leave completed writes visible and unfinished
  work pending. Conflicting external edits are never overwritten by replay. After a failed review,
  requesting a manual review explicitly replaces its failed plan with fresh analysis; already
  applied writes remain, and superseding a plan does not acknowledge its execution evidence.
- Automatic learning and write approval are Host settings. Approval is off by default,
  matching Hermes. When enabled, writes are staged durably for the user. Only authenticated
  desktop actions can approve/reject them; an Agent-supplied `approved` flag is rejected.
- Settings edit the shared user note and expose pending changes, review status and the concept
  graph in English and Chinese. Session recall retains the execution directory's boundary;
  the user note, review settings and concept pool are shared. Ordinary use stays conversational.

## Native knowledge dependencies

OKF remains portable Markdown with YAML frontmatter. `swarmx_dependencies` is a SwarmX extension:
an array of `{id, revision}` references to prerequisite concepts in this memory. It is separate
from ordinary Markdown links and evidence `sources`. Dependencies are acyclic.
Missing/deprecated targets and cycles are rejected before publication. Reads can load the complete
dependency closure in prerequisite order. Changed revisions and expired/deprecated prerequisites
are reported as stale; they are never silently refreshed or treated as verified evidence.
Graph and load operations are bounded and reject incomplete dependency closures.

## Layout and ownership

```text
memory/
├── index.md
├── README.md
├── USER.md
├── agent2agent.md
└── deepseek-harness.md
```

Concepts live directly at the memory root; there are no scope or category folders. `index.md`
is reserved for generated navigation entries, `README.md` for repository guidance and `USER.md`
for user notes; those names are not valid concept filenames. Use links and metadata for topical
organization. Memory has no separate Markdown change log and no private revision store: Git
history for this directory and the Host execution journal retain history. Domain applications remain
authoritative for scientific entities and evidence; native Agents remain authoritative for
their transcripts.

Concepts are bounded UTF-8 Markdown with YAML frontmatter: `type`, `title`, `description`,
`generated`, optional sources/tags/aliases, and a revision digest. Standard
Markdown links are allowed; Wikilinks, Obsidian block references, and executable embedded HTML are
rejected. Unknown frontmatter fields survive updates.

New concept filenames use the normalized title alone, such as `list-of-agent-protocols.md`, and
are unique across the pool. Creating an occupied filename returns
`REVISION_CONFLICT` with the existing concept ID; read it and update with its current revision.
The failed creation leaves the existing concept and index unchanged. Repeated calls with
the same `requestId` retain their existing idempotency contract.

## Authoring rules

Memory stores durable user-specific or research knowledge that adds value beyond public sources:
decisions and their reasons, observed constraints, verified findings and reusable experience.
Keep public facts only as necessary context; without durable added value, save nothing instead
of an encyclopedia summary. Memory content excludes migration notes, curation history,
source-scope bookkeeping and self-commentary.

Do not create standalone current concepts or first-level index, navigation or disambiguation
entries for merged or obsolete topics. Retain historical detail only when needed to understand
the current topic. These are Memory operating rules, not learned user preferences; do not copy
them into core notes or vault concepts.

Use concise, unambiguous concept titles. Reuse one page for the same entity; distinguish different
entities with meaningful names. Put detailed subtopics in the body, description and tags.
Use `List of ...` titles for pages that list related concepts, such as `List of Agent Protocols`.
Filenames derive from those titles without hashes, UUIDs or timestamps. Reserve `index.md`,
`README.md` and `USER.md`; do not add folder levels.

Write natural-language metadata in American English, including titles, descriptions, tags,
aliases and source titles. Preserve an original-language name when it identifies a concept or
entity specific to a language community, culture or institution and that name matters to its
identity. Keep surrounding explanatory metadata in American English. A non-English word alone
does not establish this exception; ordinary concepts use English metadata even with non-English bodies.

Examples include China's [法定节假日调休](https://www.gov.cn/gongbao/2025/issue_12406/202511/content_7048922.html)
holiday scheduling arrangement, Japan's [ふるさと納税](https://www.city.kyoto.lg.jp/gyozai/page/0000186773.html)
local government donation program, and Korea's [전세](https://english.seoul.go.kr/service/living/housing/1-wolse-jeonse/)
deposit-based housing lease. These can support reusable knowledge about a specific institution
or practice, rather than merely recording a word's spelling.

Body content and plain-text user notes may use any language or mix languages. Preserve
IDs, URLs, hashes, timestamps and other machine-readable values exactly. Changing a title does
not require changing an existing concept ID or its links.

The complete runtime authoring guide is bundled at
`packages/core/memory/skills/memory/SKILL.md` and exported as
`@swarmx/memory/skills/memory/SKILL.md`. Applications resolve this package resource without
depending on a Desktop directory. New session context and the Memory tool
description point to `read_memory_guide {}`; the Agent reads it when it needs to author durable
knowledge. Background reviews load the same full guide directly. Existing sessions retain their
frozen context, including snapshots created before this change. These are semantic authoring
instructions. The schema/linter validates structure; it does not
certify knowledge value, enforce editorial relevance, certify English usage or translate saved content.

## Agent selection experience

Other owner-authorized applications can consume the same user-owned vault through the
[standalone model-experience CLI](shared-model-experience.md). Its external observation imports
remain self-asserted ordinary concepts, separate from Host execution-backed evaluations.

Harness/model/effort/provider experience uses the existing private concept pool. Tag durable selection
concepts `agent-selection`, add route/task aliases or tags, and retain the observed conditions and
evidence. Explicit user preferences, observations and unverified opinions must remain distinguishable.
Assess acceptance, time and total descendant cost separately for each requested effort. Explicit native
configuration conflicts fail execution. Leave omitted levels unknown, and do not equate identically
named levels across different runtimes. Original native reports remain in the execution journal.
The shared delegation skill ships in `packages/core/swarm/skills/delegate/SKILL.md`, exported as
`@swarmx/swarm/skills/delegate/SKILL.md`.
At `swarm.prepare`, the Host appends relevant private concept bodies and their original evidence to the
skill text. Write the combination assessment and appropriate scenarios in the concept body; future
preparations read the latest saved revision. This reuses the existing durable review and concept pool;
application updates never overwrite private concepts. See [delegation preparation](swarm.md#delegation-preparation).

New selection evaluations carry an `evaluation` object: `kind` (`observation`, `judgment`,
or `preference`), `task`, `criteria`, `evidence` (execution-event URNs), `counterEvidence`,
and `limitations`. Memory persists it as `swarmx_evaluation`, with the Host's review snapshot
reference when generated by a review. Each reference must resolve in the current execution
directory. A new review may cite only original events included in its supplied snapshot;
omitted events and previous review conclusions cannot serve as fresh execution evidence.
New selection writes and prompt/skill improvement proposals require this structure. Existing
notes remain readable and are explicitly unverified when the structure or accessible evidence
is missing. Approval and `stable` status do not certify the truth of a judgment.

The Host computes statistics from the distinct cited executions, recording the recipe, exact
run IDs, observed routes/settings, task inputs and time window. Completion, errors, cancellation,
other stops and incomplete runs remain separate. Elapsed time is Host-observed task wall time,
including tools and waits, not model generation speed. Token counts and USD cost are reported
only when the native interface supplied unambiguous totals; unknown values remain null with
explicit coverage counts. A citation-selected sample is not a provider-wide error rate or a
controlled model comparison. Agent identity uses requested settings; native metadata remains in
the original events, and an explicit configuration conflict fails execution.

Reviews receive these computed facts and preserve their prompt identity and requested reviewer
model, with available provider and runtime version. They do not copy a second model identity into
the review response or plan. Qualitative conclusions remain judgments
against their recorded criteria, with supporting and contrary evidence and limitations. A
resource validator passing establishes only its stated check, not improved agent performance.
Selection preparation returns the same evidence summaries alongside the exact Memory revisions.
Execution-source buttons open the original cited records and computed summaries without starting
a Harness or sending private records to an external service. The same directory-scoped source
resolver governs new writes, lint diagnostics and the read-only evidence view.
An explicit `export_evaluation` request packages the selected evaluation or review attempt as an
Attached RO-Crate, including original evidence and its generation process. See
[evaluation research objects](ro-crate.md#evaluation-research-objects); domain scientific exports
continue to exclude private Memory and execution history.

Bundled selection guidance uses the same separation of observation, judgment and preference.
Any quality recommendation must identify its task/route/version, dated source or versioned
reproducible experiment, sample scope and limitations. Private execution URNs are not shipped
as public evidence. The bundled guide currently contains integration facts, not measured model
quality rankings.

`swarm.prepare` reads the current user note and searches `agent-selection` plus up to four caller
queries (five matches per query). It returns complete concept bodies, sources, revisions and dependency
graphs, limited to eight loaded groups and 48,000 serialized characters. Oversized or excess groups
are listed as omitted, never silently truncated; search diagnostics are retained. The lead can load
additional concepts with the ordinary Memory tool. This retrieval is fresh even when the parent
session retains an older frozen note. Disabled Memory or absent `memory.read` authority returns an
explicit status without reading private notes or concepts. See [delegation preparation](swarm.md#delegation-preparation).

## Product tool

The Host publishes one `memory` MCP tool:

- `read_memory_guide` — `{}`; read the complete bundled authoring guide on demand
- `search_memory`
- `search_wiki_memory` — opt-in Host-owned read-only wiki excerpts, requiring `memory.read`;
  disabled/unconfigured status dispatches nothing. See [Host wiki search](wiki-memory-client.md#opt-in-host-search)
- `read_memory`
- `create_memory`
- `update_memory`
- `deprecate_memory`
- `lint_memory`
- `graph_memory` — the authorized prerequisite graph, including transitive stale flags
- `load_memory` — a concept and its complete prerequisite closure (64 concepts / 128 KiB maximum)
- `read_core_memory` / `update_core_memory` — the shared user note; updates use full content
  and the last read `expectedRevision`
- `search_sessions` — optional `query`, `sessionId`, and `limit`, within the execution directory
- `export_evaluation` — `{id, expectedRevision}` or `{source}` for a review-start event;
  returns Attached RO-Crate metadata and exact text files, requires `memory.read`, and rejects
  packages above 8 MiB rather than truncating evidence

The library requires its caller to authorize writes. The Host supplies this authorization from
the user's persisted setting or a desktop approval, never from model arguments. Updates require the last read revision.
Writes occur under one file lock next to the memory directory (`memory.lock`), atomically replace
the concept, and then refresh `index.md`. There is no physical-delete action or second
authoritative transcript. Recall indexes only events already observed by this Host.

The desktop client calls the same `memory` tool through the Electron preload bridge:
`memory_status`, `memory_configure`, `update_core_memory`, `graph_memory`, `load_memory`,
`memory_review` and `memory_decide`. Those calls carry no execution identity. Native Agents reach it
through the stdio Host bridge, where every call must bind an active session/run. Pending writes are capped at
100 and retain their original expected revision. Conflicts remain pending until rejected or
replaced by a fresh proposal. Settings are saved in `$SWARMX_HOME/memory.json`.

## Deterministic validation

The shared validator reports `ruleId`, relative `path`, `line`, `column`, `severity`, `message`,
and the SHA-256 `revision` of the bytes inspected (`null` when an unsafe, missing, oversized,
or scan-limited path could not be read). Its clock is an explicit ISO datetime `now`;
the same authorized file and execution-journal snapshots and clock produce the same diagnostics. Unknown frontmatter
fields and concept types remain supported. These are SwarmX authoring rules, not a claim that
every warning violates OKF.

- Errors: invalid UTF-8/YAML, duplicate keys, empty required strings/body, size limits, invalid
  calendar timestamps, malformed known fields, duplicate source IDs or footnote definitions,
  undefined footnotes, nonportable active Markdown, and references escaping or hiding inside
  local paths.
- Warnings: unassociated or unused sources, a `Finding` without evidence, broken local links,
  missing/stale index entries, and expired concepts. Ordinary explanatory footnotes do not
  require a source. Markdown code blocks and inline code are literal examples, not references.
- Index syntax is checked separately from concept frontmatter. `README.md` and `USER.md` are
  notes, not concepts, and are excluded from concept checks.
- External URI sources, including legacy `sx:` identifiers, are opaque and receive `source.unverified` warnings. SwarmX does not validate their domain state; see [domain references](domain-projects.md).
  Invalid addresses are errors; unavailable resources or changed revisions require review.
  No network requests, automatic revision substitution, or claims of factual verification.

`lint_memory` accepts optional `id` and `now`: without `id` it checks the memory pool; with `id`
it returns that file's diagnostics against the same authorized
snapshot. It does not edit files. Successful MCP mutations await the same post-edit check and
return diagnostics alongside the edited concept. A post-edit error describes bytes already
written; it does not claim to undo the edit. Native editor hooks can invoke this read-only action;
this package does not install or alter provider hooks.

Reads reject structural errors. Default search also excludes deprecated concepts; explicit
`includeDeprecated: true` includes them. Search results expose `stale`, and explicit reads retain
the original lifecycle metadata. Warnings do not prevent saving drafts. Approval and concurrent
revision checks remain enforced by the owning write operation, independently of the linter.
An explicit file check still reports scan failures that prevent a complete snapshot.

## Acceptance

- New session context and the Memory tool manifest contain a short entry point without the
  complete guide or navigation index. `read_memory_guide` returns the full bundled instructions
  under `memory.read` permission; background reviews use the same guide directly. Local concept
  names and mixed-language bodies survive writes.
- OpenClaw sessions reach the same tool only through the bundled gateway plugin, while the Host
  holds a lease for that gateway session's active run. Calls without a lease, with another
  credential, or after the run ends are rejected; without the plugin, OpenClaw keeps context-only
  Memory and its native tools.
- New concepts have deterministic filenames. Duplicate names reject creation without overwriting.
- Concepts use the flat layout with one root index and do not generate Markdown change logs.
- Unsafe paths, malformed or oversized concepts, stale revisions, rejection, and cancellation do
  not publish a partial change.
- Owner edits remain visible and cause stale model updates to fail.
- Reads and tool results expose no absolute host path.
- The Host exposes note, recall and graph operations; data survive reopening.
- Notes reject overflow, stale revisions and redirected files. The same session keeps its original
  snapshot after edits/restart; new sessions receive new notes. Full streamed messages are recalled
  without mixing execution directories or inventing citations.
- Review thresholds, disabled learning, empty/failed/cancelled review, staged approval, duplicate
  decisions and conflicting revisions are checked independently of model correctness.
- Dependency cycles, missing targets and stale revision links are rejected on write;
  upstream updates propagate stale flags through the graph and preserve the original pins.

Configured Memory resource checkers receive references without a built-in scientific parser.
They return no diagnostic for schemes they do not own; local Memory path validation still runs.
The Host validates only its owned execution URNs and marks external URI sources unverified.
