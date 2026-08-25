# Git and DVC UI

SwarmX delegates generic repository identity, data versioning, pipeline DAGs, cache management, and
remote synchronization to Git and DVC. The product-owned Science Journal continues to own scientific
facts, and the Artifact Registry continues to own only explicitly captured immutable deliverables.

`@swarmx/dsh-ui-git` and `@swarmx/dsh-ui-dvc` are read-only human interfaces.
`@swarmx/dsh-ui-git` owns the single browser Version Control surface; `@swarmx/dsh-ui-dvc`
contributes only its strict Remote status to that surface. `@swarmx/dsh-dvc` remains the Host-only
command capability. None registers a model tool, browser route, background task, or automatic
synchronization.

## Git UI boundary

`ctx.gitUi.snapshot(sessionId, signal?)` derives its workspace from the live Session, resolves the
configured Git executable only when called, and uses Git porcelain v2 with NUL-delimited paths to
return one typed repository, not-repository, or unavailable result. A repository result includes:

- Git version, full commit object id, and object format;
- branch, optional upstream, and ahead/behind counts;
- bounded repository-relative staged, unstaged, untracked, renamed, and conflicted entries;
- clean/dirty and truncation state.

The Remote result contains no canonical repository/cwd path. The browser mounts one compact Session
header action and one keyed `Version Control` Details view. Its Git `Changes` disclosure is open by
default and provides refresh, empty, error, and truncated states. It is strictly observational:
normal Git work remains available through Bash, so the UI does not add a second staging, commit,
branch, checkout, fetch, pull, or push path.

## DVC boundary

`ctx.dvc.inspect(cwd, signal?)` requires an existing DVC project contained by the addressed Git
repository. It resolves the configured DVC executable only when called, hashes bounded regular
`dvc.yaml` and `dvc.lock` files with SHA256, and returns deterministic path-free summaries of:

- DVC and Git versions plus the exact Git commit;
- the DVC root relative to the Git root;
- data status from `dvc data status --json --no-remote-refresh`;
- pipeline status from `dvc status --json`.

Status JSON is summarized as category counts plus a canonical SHA256 digest. Host paths, individual
changed paths, remote URLs, and credentials are not returned. Inspection performs no remote refresh,
network access, cache synchronization, or workspace mutation.

`ctx.dvcUi.snapshot(sessionId, signal?)` derives the workspace from the live Session and delegates
only to `ctx.dvc.inspect`. Its strict Remote result is one project, not-project, or unavailable
state. The project state contains the same path-free DVC inspection projection; expected missing
project and missing executable conditions become renderable states instead of desktop boot errors.

The DVC client mounts no second header action or Details view. The shared Version Control view
checks DVC independently while loading Git. Only a typed `project` result adds a default-open `DVC`
disclosure beneath Git `Changes`; `not-project`, `unavailable`, and failed DVC detection stay hidden
without hiding or degrading Git. The disclosure shows DVC/Git identity, the Git-relative DVC root,
manifest digests, and path-free data/pipeline category counts. It does not expose initialize, add,
pull, push, reproduce, or experiment controls. Those actions can mutate files or contact remotes and
therefore remain explicit Host or Bash operations until a separate approved workflow owns their
policy and result lifecycle.

`ctx.dvc.pull(cwd, request, signal?)` is an explicit low-level Host mutation. Targets and an optional
remote name are validated before argv construction. The package does not request approval itself
because it exposes no model surface; any future model-facing caller must obtain its own approval
before invoking it.

`ctx.dvc.reproduce(cwd, request, signal?)` rejects dirty, bare, or unborn Git state. Its
package-private Git runtime creates a detached worktree at the exact clean `HEAD`, privately reuses
the source DVC cache and local config, and runs `dvc repro` only in that disposable worktree. `pull`
is opt-in. The returned Host-only handle retains the isolated outputs for verification or Artifact
Registry capture until disposed. A nonzero stage exit is a failed reproduction result, not a
service success claim.

## Security and lifecycle

- Commands use `ctx.subprocess` with explicit argv/cwd/stdio/env and no shell concatenation.
- stdout and stderr are bounded; cancellation terminates the complete managed process tree.
- Targets reject option prefixes, NUL, absolute paths, and traversal segments.
- Mutating operations serialize per canonical repository.
- `.dvc/config.local` is copied only into the owner-only disposable tree as mode `0600`; its content
  and location never enter results or diagnostics.
- Diagnostics redact canonical workspace paths and credential-bearing remote URLs.
- Git and DVC UI return unavailable states when an optional CLI is missing; neither prevents desktop
  boot.

## Non-goals

These packages do not initialize Git/DVC, add or commit files, configure remotes, push data, run DVC
Experiments, compare scientific outputs, change Claim/Evidence status, or publish private execution
metadata through RO-Crate. Scientific replay plans and `VerificationReport` remain a separate
higher-level Science capability.
