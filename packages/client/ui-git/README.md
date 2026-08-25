# @swarmx/dsh-ui-git

Read-only version-control visibility for people: one per-Session header action opens a shared
`Version Control` Details view. Git `Changes` is always available in a repository; when the DVC
Remote detects a project, a second default-open `DVC` disclosure shows path-free pipeline and data
status. DVC absence never hides Git.

The panel deliberately provides no model tool and no stage, commit, branch, checkout, initialize,
sync, reproduce, fetch, pull, or push action; agents can use normal Git and DVC commands through
Bash.

See [`docs/version-control.md`](../../../docs/version-control.md) for the boundary.
