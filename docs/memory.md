# Shared semantic memory

`@swarmx/memory` owns shared semantic memory: private, owner-readable research knowledge persisted
across sessions as OKF Markdown concepts directly under `$SWARMX_HOME/memory`. User notes,
searchable observed conversations, reusable procedures, and the structured concept pool have
separate roles. Native skills remain owned by their runtimes; concepts add explicit dependencies.

## Learning lifecycle

- `memory/USER.md` stores standing user preferences (1375 Unicode characters). Empty notes are
  valid. Updates require the last read revision, reject overflow, and use a private atomic file
  replacement.
- The Host freezes the user note and a bounded navigation index for each session. Full
  concepts are loaded on demand. Existing session snapshots survive Host restarts; new sessions
  see new notes. The snapshot is installed when creating a task, or on the first observed turn
  of an existing native task. Disabling memory stops learning and gives new tasks empty snapshots;
  it does not erase context already given to an existing task. Retrieved knowledge is data and cannot grant tool permissions.
  Codex/Claude receive native developer/system context. Hermes/OpenClaw receive a marked context
  appendix on the first observed turn; it is removed from the displayed user message.
- Observed user/assistant messages are recalled from the execution directory's journal.
  Search returns original text, session/run identifiers and event references, not invented summaries.
- Successful foreground turns trigger a background review after 10 prompts, or 10 tool calls in
  one turn. A manual review is also available. Reviews produce validated candidate operations;
  they cannot approve themselves or execute product tools. Review failures are visible in the
  journal and UI; no claim is made that an LLM will extract every important fact correctly.
  The user chooses a configured Codex or Claude runtime for review (Codex by default); the
  foreground chat may use any supported runtime. Reviews time out after two minutes and are
  cancelled on Host shutdown. Reviews receive no Host product MCP credential and reject observed
  tool calls and interactions. The Host requests native restrictions; unmodified upstream adapters
  determine their effect, including internal title calls and session persistence. The review uses existing
  credentials and consumes model usage. It sees bounded snapshots of up to 30 messages, 30 observed
  tool events and 20 concepts; oversized entries are omitted intact, not silently summarized.
  It may update the user note and supplied concepts or create draft concepts. New concepts
  cite the journal snapshot through `urn:swarmx:execution:<event-id>`. Writes are per-concept,
  not a transaction spanning a whole review; an error leaves prior successful writes visible.
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
history for this directory and the Host execution journal retain history. Science remains
authoritative for research entities and evidence; native Agents remain authoritative for
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

The Host shares one `MEMORY_AUTHORING_RULES` instruction across foreground session context,
the Memory tool description and background review prompts. Its content selection, naming, layout and language
rules are loaded automatically when Memory is enabled; they do not depend on skill discovery,
a skill invocation, or the Agent deciding to read an instruction file. Existing sessions retain their frozen
context; newly generated context and review prompts, and the tool manifest, carry the current rule.
These are semantic authoring instructions. The schema/linter validates structure; it does not
certify knowledge value, enforce editorial relevance, certify English usage or translate saved content.

## Product tool

The Host publishes one `memory` MCP tool:

- `search_memory`
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
the same authorized file and Science resource snapshots and clock produce the same diagnostics. Unknown frontmatter
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
- Science `sx:` sources use the directory-scoped Science resolver in the Host.
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

- Foreground Agent context, the Memory tool manifest and background reviews carry the same
  content selection and metadata language instructions, even with an empty user note. Local
  concept names and mixed-language bodies survive writes.
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
