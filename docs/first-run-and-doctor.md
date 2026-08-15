# First run and Doctor

SwarmX keeps the first useful path short while preserving the internal
separation between Harness, Provider, Model, Agent, and Project:

```text
Install
  -> inspect the local runtime
  -> discover usable Harnesses and Models
  -> configure one Provider when the selected Model needs it
  -> choose or create a writable Project when local tools are needed
  -> run a small read-only first task
```

The curated default is the direct SwarmX Harness with one explicitly configured
Model. External Harnesses are optional until selected by the user or required by
an Agent recipe. A local or already authenticated route does not require a
network check merely to pass Doctor.

## Reading a Doctor report

The report has an `ok` overall state when no actionable finding remains. Each
non-OK finding has one of four user-facing classifications:

- `warning`: the selected path can run, but a limitation is visible;
- `blocking`: the selected path cannot run safely;
- `repairable`: SwarmX can prepare an idempotent repair for explicit review;
- `decision`: a Provider login, Project choice, trust decision, or other human
  choice is required and cannot be automated.

Every non-OK check states the symptom, cause, impact, and next action. A
repairable check also points to a bounded repair action whose preview describes
what it will change. Inspection and `plan()` are read-only. `doctor --fix`
prints the plan; setup begins only after interactive confirmation or explicit
`--yes` in a non-interactive invocation.

## Common first-run blockers

| Check | Meaning | Next action |
| --- | --- | --- |
| Node.js missing | The CLI/runtime baseline is unavailable | Install an active LTS release, then run Doctor again |
| Selected external Harness missing | That Agent recipe cannot start | Review and confirm its setup action, or choose the direct Harness |
| Provider not authenticated | A Model exists but no usable credential route does | Open Provider setup and save the connection |
| Authentication reference invalid | Settings points at a missing/invalid Main-only credential entry | Reconfigure that Provider; no secret is displayed in Doctor |
| Project not writable | Local mutation tools cannot be contained safely | Choose a writable folder or continue with a read-only task |
| Several Harnesses available | More than one safe runtime can satisfy the task | Choose one; Doctor does not guess based on discovery order |
| Offline | Hosted discovery or a hosted Model is unavailable | Use a local/already cached route or reconnect; local checks still run |

Doctor also reports the OS-sandbox strategy separately from Harness readiness:
`native_allowed` means the selected boundary may use its native host, while
`protected_required` is blocking until a registered protected profile and its
runtime are ready. A protected check never silently falls back to native.

After a repair, running Doctor again is safe and should either return `ok` or
the same still-actionable finding. Repair planning never installs, changes a
credential, grants Extension permissions, or changes trust.

## CLI and Desktop

`swarmx doctor` prints the same classification and human fields used by the
Desktop Runtime surface. `swarmx doctor --fix` reviews the exact plan, and
`swarmx doctor --fix --yes` explicitly confirms it. Desktop additionally
supplies Main-owned Provider-auth and Project-writability observations without
exposing credential values or filesystem authority to the Renderer or Runtime
package.
