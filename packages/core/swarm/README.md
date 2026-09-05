# @swarmx/swarm

Protocol-neutral `Agent<Observer>` and `createSwarm(name, lead)`.

A Swarm delegates native session operations to its lead through the same in-process interface.
Leads may themselves be Swarms. The package has no runtime dependencies, provider imports,
wire types, transcript storage, or UI code. External protocol gateways belong to the Host.
