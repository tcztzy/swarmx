# Global Memory

SwarmX global Memory consists of two bounded Markdown files within the same
Git-backed authority as linked entity pages:

| File | Purpose | Injected |
| --- | --- | --- |
| `USER.md` | Stable user identity, preferences, communication style, and working habits. | Every Agent-bearing run |
| `MEMORY.md` | Cross-Project environment facts, conventions, decisions, and reusable experience. | Every Agent-bearing run |

Detailed research findings do not belong in these always-loaded files. They are
captured as linked entity pages and retrieved on demand through the `Memory`
tool.

## User controls

Open **Settings → Global Memory** to inspect or edit either global file. Saving rejects
blank text, unsupported control characters, unknown request fields, and content
over the displayed capacity. Forget is a separate confirmed action for one file.
Deleting, archiving, or forking a Session does not change global Memory.

Existing Settings-backed Personal Memory remains available as a legacy
`USER.md` source until the user saves `USER.md`. That successful save removes
the obsolete Settings value; SwarmX never silently discards it.

Do not store credentials or one-off task data in global Memory. The frozen
snapshot is sent to the configured model Provider on supported runs.

## Execution and transparency

At the start of each Agent-bearing run, Main reads one immutable snapshot and
Core appends the present files to Agent instructions as separately labelled
blocks. This applies to direct SwarmX Agents, external ACP prompt text, and every
Agent node in a `SwarmConfig` workflow. Edits made during a run affect only
later runs.

A direct SwarmX Agent can propose saving or forgetting either file through the
`Memory` tool. Main shows the proposed change and applies it only after the user
confirms. Read-only side chats and external ACP Harnesses receive the snapshot
but not SwarmX mutation tools.

Each run writes a visible Session receipt stating which files were used, their
sizes and update times, or why no global Memory was used. Receipts are not
replayed as model context. Memory plaintext is excluded from audit, Activity
Profile, trace, telemetry, hook input, and unrelated tool transport.
If the managed Memory runtime is unavailable, the receipt says so and the Agent
run continues without a global snapshot; Settings and mutation attempts still
surface the runtime error.

## Reflective review

An explicit request such as “remember this” prompts the active Agent to propose
the appropriate global-file or entity-page change immediately.

Otherwise review cadence is Session-scoped:

- each persisted Session has an independent review cursor;
- every ten completed foreground user-Agent turns adds a reflection reminder;
- switching Sessions or restarting Desktop preserves the cursor;
- different Sessions are never concatenated into one reflection window;
- an archived or sufficiently idle Session may review its remaining tail
  independently.

The reminder is added only to the active Session's normal bounded execution
context and names the unreviewed turn range. It does not copy dialogue from any
other Session into that context. The reminder asks the Agent to propose durable
changes; it does not itself write Memory or advance the active run's snapshot.

## Research capture

Competitor and product research should create or update entity pages only for
observations costly to reconstruct from ordinary public documentation. Each
candidate names the entity and aliases and carries typed observations:

- `observed` — directly established by code, runtime behavior, or an experiment;
- `derived` — synthesis from multiple cited sources or observations;
- `decision` — a durable product or engineering conclusion;
- `hypothesis` — useful but not yet verified.

Every observation includes a concise claim, why it is worth retaining, source
references, confidence, and Session provenance supplied by the host. Exact
normalized title or alias matching selects an existing page; otherwise the
confirmed mutation creates one. SwarmX appends a versioned research section,
preserves unrelated authored Markdown, and skips observations whose stable
fingerprint is already present.
