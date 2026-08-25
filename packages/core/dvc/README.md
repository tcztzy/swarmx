# @swarmx/dsh-dvc

Host-only DVC capability for an existing Git-backed DVC project. It provides path-free status,
explicit live-workspace pull, and exact-HEAD reproduction inside a package-private Git worktree.
It does not register model tools, initialize projects, configure remotes, push data, or decide
scientific equivalence.

See [`docs/version-control.md`](../../../docs/version-control.md) for the public contract.
