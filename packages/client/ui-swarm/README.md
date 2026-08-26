# `@swarmx/dsh-ui-swarm`

Read-only browser integration for `@swarmx/dsh-swarm`. It mounts the strict
`swarm/uiSnapshot|waitUi` Remote, one conversation header action, one keyed generic Side
View renderer, and a bounded historical card derived from ordinary `swarm` Tool results.

The browser receives no workspace path/title, raw Session id, mailbox body, model-private text, or
mutation endpoint. Long-poll requests are revision-based and are cancelled on Session switch,
unmount, and HMR disposal.
