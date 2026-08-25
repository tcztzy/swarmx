# @swarmx/dsh-ui-dvc

Read-only per-Session DVC status Remote for SwarmX. The Host derives the workspace from the live
Session and delegates inspection to `@swarmx/dsh-dvc`; its client mounts only the strict Remote
contribution. `@swarmx/dsh-ui-git` consumes project results inside the shared Version Control view,
so DVC registers no second header action or Details view.

This package registers no model tool and exposes no DVC initialization, synchronization,
reproduction, experiment, or remote action.
