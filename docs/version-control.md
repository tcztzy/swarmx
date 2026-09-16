# Git and DVC

`@swarmx/dvc` exposes `DvcService` for Git/DVC inspection, bounded pulls and reproduction in
disposable Git worktrees. Callers supply an execution directory and a process runner, and retain
the returned reproduction handle until they have finished inspecting its outputs.

The package is vendor-neutral and receives no Agent transcript, A2A state, AG-UI event, or renderer
type. Callers authorize the directory and operations; the service propagates cancellation to
its owned subprocesses and cleans up reproduction worktrees.
