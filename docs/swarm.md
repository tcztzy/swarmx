# Swarm

`createSwarm(name, lead)` returns the same `Agent<Observer>` interface as its lead. It delegates
list/create/read/start/steer/interrupt/dispose directly. A parent does not inspect provider identity;
a lead can itself be a Swarm, with no configured nesting limit.

The Host's `swarm` MCP tool creates named Swarms and exposes status/new_session/send_message/cancel.
The same ProductServices instance serves MCP and REST. Membership is in-memory; native Agents
own all transcripts. Delegation returns final text, not a second transcript.

ACP and A2A are external gateways into this composition, not internal transports. Interactive
delegation that has no connected user fails explicitly; it never auto-approves tools.

This is recursive composition and explicit delegation, not a durable scheduler, verification DAG,
automatic knowledge admission, or proof that every manuscript claim is implemented.
