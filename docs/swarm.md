# DSH Swarm

`dsh-swarm` is SwarmX's opt-in scientific Team mode. It keeps the complete `dsh-science` Agent
composition and adds one aggregate `swarm` Tool. Standard and ordinary Science sessions do not see
that Tool.

## Ownership

- DeepSeek Harness owns Agents, continuable subagents, Sessions, model calls, Tool execution,
  approvals, persistence, and the browser conversation surface.
- `@swarmx/dsh-swarm` owns only Team roster, task, mailbox, scheduling, and orchestration
  provenance under `$DSH_HOME/swarm/swarm.sqlite`.
- Science Journal remains scientific truth. PKB remains personal synthesis. Swarm state is never
  written into the project checkout and private Team/Session ids never enter RO-Crate output.

## Identity and authority

One top-level Session is the Team lead and Team id. Every member is a continuable direct child and
uses its immutable child Session id as the authority credential; display names are immutable labels,
not credentials. Every Host mutation resolves the exact live Agent object. Lead-only actions are
Team creation, member admission, task reassignment, interruption, and archival.

## Tasks and workspace safety

Tasks form a bounded immutable-dependency DAG. Every transition uses an exact task revision.
Executing ownership is one random attempt id; member settlement must present both the current
revision and attempt. A stale attempt cannot change the board.

Members share one checkout. Read tasks may overlap. A member-scoped Tool guard admits `write`,
`edit`, shell, Science mutation, and similar side-effecting tools only while that exact member owns
an active `write` attempt. At most one write attempt exists per Team. Declared write scopes are
coordination hints, not filesystem locks or rollback guarantees. Reassignment interrupts the old
owner and waits for quiescence before rotating the attempt.

## Mailbox, recovery, and retirement

Messages are durably queued before delivery with stable ids, bounded content, per-target ordering,
and pending-mail limits. Quiet messages inject context only into a resident member. Waking messages
use DSH continuable follow-up and may cold-resume the target.

After process recovery, previously running tasks become `needs_attention` and lose their attempt;
SwarmX never silently replays potentially non-idempotent work. Plugin disposal closes admission,
releases long-poll waiters, aborts and settles admitted runtime work, revokes member-scoped
capabilities, then closes SQLite. `archive` is an honest durable retirement operation; it is not
physical deletion.

## Browser surface

`@swarmx/dsh-ui-swarm` reuses the generic per-Session Side View. Its strict Remote returns only
bounded member status, task summaries, counts, and revision. It exposes no mutation method, path,
workspace title, raw Session id, or message body. One revision-based bounded long-poll is active
only for the rendered Session and is cancelled on unmount or HMR.
