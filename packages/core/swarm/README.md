# @swarmx/swarm

Recursive ACP composition through `createSwarm(name, connectLead)`.

A Swarm is an official SDK Agent app upstream and Client downstream. Its connector accepts a
Client app and returns an SDK connection, allowing another Swarm, a leaf app or an ACP stream.
Requests and cancellation travel down; updates, approvals and forms travel up. Create a fresh
Swarm app per connection. Closing either side closes the other. The Host owns permission policy,
native adapters and gateway projections. This package has no provider, transcript or UI code.
