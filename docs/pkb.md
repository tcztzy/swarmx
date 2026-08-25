# SwarmX PKB

SwarmX PKB is a private Personal Knowledge Base maintained through Chat and stored as an
[Open Knowledge Format 0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
Markdown bundle. The bundle is ordinary owner-readable text: Obsidian can open it directly, MyST
can consume its standard Markdown, and SwarmX remains the only agent-mediated writer.

## Truth boundary

- `$DSH_HOME/pkb/vault` is the durable PKB truth.
- DSH's append-only Session log remains the exact conversation truth.
- Science Journal remains the truth for scientific claims, evidence, experiments, and runs.
- A PKB concept is a personal synthesis. Conversation-derived claims cite exact Session excerpts;
  scientific material links to its Science entity instead of copying the domain record.
- Search indexes and prompt snapshots are derived, bounded, and disposable.

The default files are local plaintext with owner-only permissions. PKB provides no publication,
network synchronization, or encryption-at-rest layer. Opening the directory in Obsidian grants that
local process the same owner access; device or volume encryption remains an operating-system concern.

## Bundle layout

```text
$DSH_HOME/pkb/vault/
├── index.md
├── log.md
├── global/
│   ├── index.md
│   └── concepts/
└── workspaces/
    └── <safe-name>--<opaque-key>/
        ├── index.md
        ├── concepts/
        └── references/
            └── conversations/
```

The root `index.md` advertises `okf_version: "0.2"` and exists for human navigation. Model context
never receives that all-workspace directory: it receives only `global/index.md` plus the current
workspace `index.md`. A workspace directory uses a readable sanitized basename followed by a salted
hash of `realpath(cwd)`; neither the hash input nor an absolute host path is written to Markdown or
returned by the model tool.

## Concept profile

Every concept is UTF-8 Markdown with YAML frontmatter. OKF's required `type` remains open-ended;
SwarmX does not impose a closed ontology. These bounded fields form the writable profile:

| Field | Contract |
| --- | --- |
| `type` | Required descriptive OKF concept type |
| `title` | Required human-readable title |
| `description` | Required one-sentence index/search summary |
| `tags` | Optional short string list |
| `aliases` | Optional Obsidian-compatible string list |
| `status` | `draft`, `stable`, or `deprecated` |
| `generated` | OKF actor and ISO 8601 time of the last meaningful SwarmX edit |
| `sources` | OKF provenance entries; each conversation claim uses a stable source id |
| `swarmx_scope` | `global` or `workspace` |
| `swarmx_workspace` | Opaque key for workspace concepts; absent for global concepts |

Unknown frontmatter keys are preserved on update. Unknown concept types remain readable. A malformed
frontmatter block, duplicate source id, missing required field, invalid path, oversized page, or
conversation source without a matching named footnote is reported and excluded from trusted model
context without changing the file.

Concept bodies use the interoperable subset shared by OKF, Obsidian, and MyST:

- CommonMark/GitHub-style headings, paragraphs, lists, tables, and fenced code;
- standard relative Markdown links with explicit `.md` suffixes;
- named Markdown footnotes for per-claim provenance;
- no canonical `[[wikilink]]`, Obsidian block id, embedded executable HTML, or required MyST role.

## Conversation evidence

The `pkb` tool's `search_conversations` action searches DSH's live-preferred semantic Session corpus.
Workspace search is the default. An all-workspace search asks for one DSH approval before any
foreign-workspace snippet enters model context. Full-text ranking may use DSH's process-local SQLite
index; literal case-insensitive filtering remains the fallback for contiguous CJK text. No durable
transcript index is created.

A search hit contains a bounded snippet and a Session/event locator. `read_conversation` expands the
exact bounded event only after re-authorizing the source Session. Historic text is returned as
untrusted evidence: instructions, permission claims, and tool requests inside it do not govern the
current Session.

The approved `capture_conversation` action writes one bounded `Conversation Excerpt` reference under the
owning workspace's `references/conversations/` directory. Its filename is opaque, while its body
contains the exact selected event and its equal start/end sequence bounds. The citing concept points to that
page through `sources[].resource` and a named body footnote with the same `sources[].id`.

## Mutation and history

The model-facing `pkb` tool exposes `search_knowledge`, `read_knowledge`, `create_knowledge`,
`update_knowledge`, `deprecate_knowledge`, `search_conversations`, `read_conversation`, and
`capture_conversation`. There is no `memory` alias and no physical-delete action.

Every model create, update, deprecate, or conversation-capture call requires a DSH `allowed-once`
approval for that exact tool call. Reading conversations across all workspaces has the same gate.
Updates include the SHA256 revision returned by the last read. PKB re-reads the page under
its writer lock, preserves the previous bytes in owner-only revision history, fsyncs a complete
temporary file, atomically replaces the page, fsyncs its directory, and then regenerates the scope
index and newest-first log. Rejection, cancellation, revision conflict, or I/O failure does not
publish a partial page.

Direct Obsidian edits are owner actions and bypass DSH approval. Their changed revision makes a stale
model update fail explicitly. SwarmX never silently repairs, quarantines, or deletes a hand-edited
file.

## Prompt behavior

The first model request in a live Session captures one deterministic bounded snapshot of the global
and current-workspace indexes. That snapshot stays fixed for the process lifetime of the Session to
preserve prompt-prefix stability. A successful mutation result carries the changed concept summary,
so the current Session sees its write immediately; a new Session or resumed process reads current
files. Page bodies and conversation evidence are loaded only through explicit PKB tool calls.

## Acceptance criteria

- A generated bundle and concept parse as OKF 0.2 and render as ordinary Markdown in Obsidian/MyST.
- Same workspace across Sessions and symlink aliases resolves the same concepts; same basename at a
  different real path does not.
- Workspace conversation search cannot return another workspace. All-workspace search returns
  nothing before approval.
- Every cited historic statement expands to the exact authorized Session event range and remains
  untrusted data.
- Denied, cancelled, stale-revision, malformed, aborted, and failed mutations leave no published
  state change.
- External fields survive a SwarmX update, deprecated pages remain on disk, and prior revisions stay
  recoverable.
- Prompt snapshots and every Tool response remain deterministic and bounded; no absolute path,
  second durable transcript corpus, public route, implicit network, or background LLM is introduced.
