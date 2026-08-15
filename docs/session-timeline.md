# Session causal timeline

The causal timeline is a rebuildable diagnostic view over canonical Session
JSONL. It is not another conversation log and it is not an execution state
machine.

For each safe projected event it can show:

- Project, Session, Turn, Step, correlation, and causation ids;
- deterministic sequence and a fixed content-free summary;
- tool lifecycle and permission outcomes;
- child/external work only when an existing structured identifier proves it;
- retry, exact duplicate, late-result, and unsettled-work diagnostics;
- why a Turn completed, failed, was cancelled, or remains conservatively
  unknown.

`messages_appended.requestId` or `messages_replaced.requestId` is the strongest
foreground Turn anchor. A `requestState: "started"` record opens the request's
Turn and a matching `requestState: "settled"` record closes that same Turn.
Older logs without these fields receive deterministic local ids and are
labelled inferred; the projector only associates an old lifecycle when one
unambiguous candidate exists.
Background Session observer records carry their durable `activationId`. The
projector uses that identifier to create a separate system-origin Turn and
keeps its messages, audit evidence, and terminal status on that Turn. It never
changes the foreground `currentTurn` or adds background work to the foreground
completion/debt set. If the identifier is absent, the projector does not guess
that a record belongs to the most recent activation.
`render.invocationId` binds a tool call, progress, and result to one Step. A
result that arrives in a later record remains attached to its original Turn and
is marked late. Exact repeated lifecycle observations are collapsed; the
projection never treats a repeated transport record as a second side effect.

Correlated compact audit events can enrich the view with approval decisions and
host effects. Audit remains the semantic authority for those privileged
decisions; the timeline references the evidence and does not copy it back into
Session history. Replaying the same ordered Session and audit events produces
the same timeline.

`unsettled` entries are diagnostic hints derived from unmatched structured tool
or child-task lifecycles. They are not pending obligations, do not open or close
the foreground completion barrier, and cannot authorize continuation. When the
execution kernel persists an authoritative settlement, the projector consumes
only the resulting canonical Session/audit evidence; it does not mirror that
state into a second machine.

Both Session and audit readers used by this command are no-write paths. A torn
Session tail is ignored with a diagnostic; a corrupt or unverifiable audit
chain fails closed instead of being repaired by inspection.

The timeline deliberately excludes prompts, message/response text, model
reasoning, source code, tool arguments or results, terminal streams,
credentials, URLs containing secrets, and environment snapshots. Summaries use
fixed phrases such as `Tool Read started` or `Permission denied`.

Use the CLI with a full Session id:

```shell
swarmx sessions timeline <session-id>
swarmx sessions timeline <session-id> --json
```

The human view is concise. JSON output uses the same strict content-free schema
for deterministic diagnostics and test replay.
