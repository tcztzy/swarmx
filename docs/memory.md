# Shared semantic memory

`@swarmx/memory` owns shared semantic memory: private, owner-readable research knowledge persisted
across sessions as OKF Markdown concepts under `$SWARMX_HOME/memory/vault`. Short user/workspace
notes, searchable observed conversations, reusable procedures, and the structured vault have
separate roles. Native skills remain owned by their runtimes; the vault adds explicit dependencies.

## Learning lifecycle

- `memory/USER.md` stores standing user preferences (1375 Unicode characters);
  `memory/workspaces/<workspace-id>/MEMORY.md` stores stable workspace notes (2200 characters). Empty notes are valid. Updates
  require the last read revision, reject overflow, and use a private atomic file replacement.
- The Host freezes short notes and a bounded vault navigation index for each session. Full
  concepts are loaded on demand. Existing session snapshots survive Host restarts; new sessions
  see new notes. The snapshot is installed when creating a task, or on the first observed turn
  of an existing native task. Disabling memory stops learning and gives new tasks empty snapshots;
  it does not erase context already given to an existing task. Retrieved knowledge is data and cannot grant tool permissions.
  Codex/Claude receive native developer/system context. Hermes/OpenClaw receive a marked context
  appendix on the first observed turn; it is removed from the displayed user message.
- Observed user/assistant messages are recalled from the existing workspace execution journal.
  Search returns original text, session/run identifiers and event references, not invented summaries.
- Successful foreground turns trigger a background review after 10 prompts, or 10 tool calls in
  one turn. A manual review is also available. Reviews produce validated candidate operations;
  they cannot approve themselves or execute product tools. Review failures are visible in the
  journal and UI; no claim is made that an LLM will extract every important fact correctly.
  The user chooses a configured Codex or Claude runtime for review (Codex by default); the
  foreground chat may use any supported runtime. Reviews time out after two minutes and are
  cancelled on Host shutdown. Native tools/MCP/plugins are disabled for the review, with a
  read-only sandbox and denied interactions as additional boundaries. The review uses existing
  credentials and consumes model usage. It sees bounded snapshots of up to 30 messages, 30 observed
  tool events and 20 concepts; oversized entries are omitted intact, not silently summarized.
  It may update short notes and supplied workspace concepts or create draft concepts. New concepts
  cite the journal snapshot through `urn:swarmx:execution:<event-id>`. Vault writes are per-concept,
  not a transaction spanning a whole review; an error leaves prior successful writes visible.
- Automatic learning and write approval are workspace settings. Approval is off by default,
  matching Hermes. When enabled, writes are staged durably for the user. Only authenticated
  browser actions can approve/reject them; an Agent-supplied `approved` flag is rejected.
- General Settings edit shared user notes. Project Settings edit project notes and expose
  pending changes, review status and the vault graph in English and Chinese. Review policies,
  session recall and local concepts stay bound to the project URL across navigation; user
  preferences and explicitly global vault concepts remain shared. Ordinary use stays conversational.

## Native knowledge dependencies

OKF remains portable Markdown with YAML frontmatter. `swarmx_dependencies` is a SwarmX extension:
an array of `{id, revision}` references to prerequisite concepts in this vault. It is separate
from ordinary Markdown links and evidence `sources`. Dependencies are acyclic; local concepts
may depend on local or global concepts, while global concepts may only depend on global ones.
Missing/deprecated targets and cycles are rejected before publication. Reads can load the complete
dependency closure in prerequisite order. Changed revisions and expired/deprecated prerequisites
are reported as stale; they are never silently refreshed or treated as verified evidence.
Graph and load operations are bounded and reject incomplete dependency closures.

## Layout and ownership

```text
vault/
├── index.md
├── log.md
├── global/concepts/
└── workspaces/<safe-name>--<opaque-key>/concepts/
```

The workspace key is a salted digest of its canonical path. Absolute paths are not written into
concepts. Science remains authoritative for research entities and evidence; native Agents remain
authoritative for their transcripts.

Concepts are bounded UTF-8 Markdown with YAML frontmatter: `type`, `title`, `description`,
`generated`, `swarmx_scope`, optional sources/tags/aliases, and a revision digest. Standard
Markdown links are allowed; Wikilinks, Obsidian block references, and executable embedded HTML are
rejected. Unknown frontmatter fields survive updates.

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
- `read_core_memory` / `update_core_memory` — target `user` or `workspace`; updates use full content
  and the last read `expectedRevision`
- `search_sessions` — optional `query`, `sessionId`, and `limit`, within the current workspace

The library requires its caller to authorize writes. The Host supplies this authorization from
the user's persisted setting or a browser approval, never from model arguments. Updates require the last read revision.
Writes occur under one file lock, preserve the prior revision, atomically replace the concept, and
then refresh indexes and the append-only update log. There is no physical-delete action or second
authoritative transcript. Recall indexes only events already observed by this Host.

Browser routes under `/api/v1/memory` expose status, `/settings`, `/notes`, `/graph`,
`/concept?id=...`, `/pending/:id` and `/review`. These require the browser cookie and same-origin
checks; the MCP bearer token does not authorize browser approvals. Pending writes are capped at
100 and retain their original expected revision. Conflicts remain pending until rejected or
replaced by a fresh proposal. Settings are saved with the workspace under `workspaces/<id>/memory.json`.

## Deterministic validation

The shared validator reports `ruleId`, relative `path`, `line`, `column`, `severity`, `message`,
and the SHA-256 `revision` of the bytes inspected (`null` when an unsafe, missing, oversized,
or scan-limited path could not be read). Its clock is an explicit ISO datetime `now`;
the same authorized file and Science resource snapshots and clock produce the same diagnostics. Unknown frontmatter
fields and concept types remain supported. These are SwarmX authoring rules, not a claim that
every warning violates OKF.

- Errors: invalid UTF-8/YAML, duplicate keys, empty required strings/body, size limits, invalid
  calendar timestamps, malformed known fields, duplicate source IDs or footnote definitions,
  undefined footnotes, nonportable active Markdown, and scope/path mismatches.
- Warnings: unassociated or unused sources, a `Finding` without evidence, broken local links,
  missing/stale index entries, and expired concepts. Ordinary explanatory footnotes do not
  require a source. Markdown code blocks and inline code are literal examples, not references.
- Index/log syntax is checked separately from concept frontmatter. The private `.swarmx`
  history is not a second set of live concepts and is excluded from linting.
- Science `sx:` sources use the existing workspace-scoped Science resolver in the Host.
  Invalid addresses are errors; unavailable resources or changed revisions require review.
  No network requests, automatic revision substitution, or claims of factual verification.

`lint_memory` accepts optional `id` and `now`: without `id` it checks the current workspace
and global knowledge; with `id` it returns that file's diagnostics against the same authorized
snapshot. It does not edit files. Successful MCP mutations await the same post-edit check and
return diagnostics alongside the edited concept. A post-edit error describes bytes already
written; it does not claim to undo the edit. Native editor hooks can invoke this read-only action;
this package does not install or alter provider hooks.

Reads reject structural/scope errors. Default search also excludes deprecated concepts; explicit
`includeDeprecated: true` includes them. Search results expose `stale`, and explicit reads retain
the original lifecycle metadata. Warnings do not prevent saving drafts. Approval and concurrent
revision checks remain enforced by the owning write operation, independently of the linter.
An explicit file check still reports scan failures that prevent a complete snapshot.

## Storage upgrade

The Host moves an existing `$SWARMX_HOME/knowledge-base/vault` to `$SWARMX_HOME/memory/vault`
before opening Memory, preserving every concept byte, revision, workspace salt, and history file.
The empty previous directory is removed. Other files already under `$SWARMX_HOME/memory` are
untouched. If both vaults exist, startup fails instead of overwriting or choosing between them.
After the move, only the current vault is used. Package exports, the MCP tool, and its actions
use the Memory names without aliases.

## Acceptance

- Canonical and symlink aliases of one workspace resolve the same scope; unrelated paths do not.
- Unsafe paths, malformed or oversized concepts, stale revisions, rejection, and cancellation do
  not publish a partial change.
- Owner edits remain visible and cause stale model updates to fail.
- Reads and tool results expose no absolute host path or other workspace's concepts.
- The Host exposes note, recall and graph operations; data survive reopening and the one-time storage move.
- Notes reject overflow, stale revisions and redirected files. The same session keeps its original
  snapshot after edits/restart; new sessions receive new notes. Full streamed messages are recalled
  without mixing workspaces or inventing citations.
- Review thresholds, disabled learning, empty/failed/cancelled review, staged approval, duplicate
  decisions and conflicting revisions are checked independently of model correctness.
- Dependency cycles, missing/cross-scope targets and stale revision links are rejected on write;
  upstream updates propagate stale flags through the graph and preserve the original pins.
- Existing destination vaults are never overwritten during the storage upgrade.
