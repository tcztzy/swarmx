# `@swarmx/dsh-swarm`

Host-owned orchestration for durable DSH-native agent teams. A Team is rooted at one exact live
DSH Session; every teammate is a continuable direct child whose immutable member identity is its
child Session id. SwarmX does not own a model client, Agent loop, Session log, or alternate
conversation store.

The authoritative Team journal is `$SWARMX_HOME/swarm/swarm.sqlite`. It stores bounded versioned
domain events and replayable projections in one SQLite WAL transaction. Workspace files are never
used as roster, task, mailbox, or authority state.

The `swarm` aggregate tool is mounted only by the `dsh-swarm` system preset. Administrative
actions are lead-only. Members may report, message peers, and settle only their exact active task
attempt. Workspace mutation tools are guarded in each member scope: an active `write` attempt is
required and only one such attempt may exist in a Team.

See [`docs/swarm.md`](../../../docs/swarm.md) for lifecycle, recovery, and UI contracts.
