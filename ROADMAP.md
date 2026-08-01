# Roadmap

This file contains only unfinished or partially verified product work. It is not
part of the product contract in `SPEC.md`. Remove an item once its behavior is
implemented, documented where needed, and covered by focused tests.

## Carryover work to close

- [ ] Make registered Harness runtimes optional until a user selects them or a
  configured Agent requires them.
- [ ] Finish removing ambient-environment Provider synthesis from Desktop so
  discovery starts only from explicit connections and Extension metadata.
- [ ] Verify the Model catalog persists across restarts without automatic
  Provider discovery.
- [ ] Finish native Anthropic Messages and OpenAI Responses execution in the
  direct SwarmX Harness, using compatibility bridges only as fallback.
- [ ] Finish grouped Provider Model catalog persistence, safe Codex discovery,
  routing metadata, and effort metadata.

## Known native-tool parity gaps

- [ ] Add Claude-compatible `PowerShell` only with a Windows-native sandboxed
  process host.
- [ ] Add Claude-compatible `SendMessage` only with concurrently live teammate
  identities, mailboxes, and lifecycle control.
- [ ] Add Claude-compatible `Workflow` only with a deterministic, persisted,
  resumable workflow VM.

The pre-simplification status and its old `Txxx` links remain available with
`git show 780fb8e:SPEC.md`.
