# Personal Memory

Personal Memory is the compact Settings snapshot within SwarmX Memory: a small,
durable set of user-managed facts and preferences injected at run start. It is
not a distinct Memory product, organization strategy, or renamed Activity
Profile.

## User controls

Open **Settings → Personal Memory** to inspect or edit the record. Saving
rejects blank text, unsupported control characters, unknown request fields, and
content over 4,000 characters. **Forget Personal Memory** is a separate
confirmed action that removes the record from `~/.swarmx/settings.json`.
Deleting, archiving, or forking a Session does not change Personal Memory.

Do not store credentials or one-off task data here. The current snapshot is
sent to the configured model Provider on supported runs.

## Execution and transparency

At the start of each Agent-bearing run, Main reads one read-only snapshot and
Core appends it to the Agent instructions as a dedicated context block. This
applies to direct SwarmX Agents, external ACP prompt text, and every Agent node
in a `SwarmConfig` workflow. Edits made while a run is active affect only later
runs.

A direct SwarmX Agent or a SwarmX-owned Agent node in a workflow can request
**save** or **forget** through the Personal Memory tool. Main shows the proposed
operation and applies it only after the user confirms. Denial leaves Memory
unchanged. Read-only side chats and external ACP Harnesses receive the snapshot
but not this SwarmX mutation tool; external Harnesses retain their native tool
surface.

Each run writes a visible Session receipt:

- **Used** includes the Settings source, snapshot size and update time, and a
  short preview.
- **Not used** explains that Memory is empty or that a workflow has no Agent
  nodes capable of consuming context.

Receipts are not replayed as model context. The Memory snapshot itself is not
copied into audit, Activity Profile, trace, telemetry, hook input, or tool
transport.

## Execution paths

Direct Agent Composition runs and side chats receive the snapshot. External ACP
Harnesses receive it in the explicit Agent-instructions block of each ACP prompt.
Workflows deliver the same frozen snapshot to all Agent nodes, including nested
swarms; tool-only workflows report that no Agent consumed Memory.
