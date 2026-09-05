# Shared semantic memory

`@swarmx/memory` owns shared semantic memory: private, owner-readable research knowledge persisted
across sessions as OKF Markdown concepts under `$SWARMX_HOME/memory/vault`. Agents explicitly
search, read, curate, and lint this store through the Host. Native runtimes own conversation
history and their own context management; Memory does not automatically capture conversations
or inject knowledge into a new session.

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

The Host publishes one `memory` MCP tool with six actions:

- `search_memory`
- `read_memory`
- `create_memory`
- `update_memory`
- `deprecate_memory`
- `lint_memory`

Mutations require explicit single-call approval and an update requires the last read revision.
Writes occur under one file lock, preserve the prior revision, atomically replace the concept, and
then refresh indexes and the append-only update log. There is no physical-delete action,
conversation search, conversation capture, or transcript mirror.

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
- The Host exposes all six Memory actions; data survive reopening and the one-time storage move.
- Existing destination vaults are never overwritten during the storage upgrade.
