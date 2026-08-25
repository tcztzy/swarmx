# @swarmx/dsh-pkb

Private, local-first Personal Knowledge Base for SwarmX. It stores durable knowledge as an
[Open Knowledge Format 0.2](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
Markdown bundle that Obsidian can open directly and MyST can consume without proprietary syntax.

The Cordis plugin mounts `ctx.pkb` and one aggregate `pkb` model tool. Knowledge is isolated by the
live Session workspace; cross-workspace conversation search, writes, deprecation, and evidence
capture use DSH's one-shot approval service. DSH Session logs remain conversation truth, and any
SQLite search state is process-local and disposable.

The complete storage, provenance, privacy, and mutation contract is documented in
[`docs/pkb.md`](../../../docs/pkb.md).
