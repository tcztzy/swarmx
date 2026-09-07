# Current manuscript example

This guided integration example was run on 7 September 2026 against implementation
`9e1e818acd438dcbf30fb9dc46d758495d24d094`, without production-code changes.
The six CSV rows are synthetic. Their means are 79% and 87%, with three replicates per
group and an eight-percentage-point difference; they support no biological inference.

Verify the saved evidence from the repository root, without Docker or model access:

```sh
node examples/softwarex/current-release/verify.mjs
```

The verifier recomputes the arithmetic, checks eight file hashes, validates the two
successful notebook executions, the recorded parent/child link, the saved Memory
revision and the exported entities. It checks saved evidence, not model reliability.
`result.json` includes the actual notebook sources and the new session's retrieved
answer. `native-runs.json` projects run identities and completion from the Host journal;
private transcripts and credentials are excluded.

To run the native workflow again, first install the repository dependencies, configure
Codex authentication and start Docker, then run:

```sh
pnpm build
node examples/softwarex/current-release/run.mjs artifacts/paper-current-new
```

Use a fresh directory. The driver imports the input and supplies plotting code and exact
tool argument shapes to a lead Codex session. That session executes the analysis,
delegates a standard-library recomputation, records the claim and supporting entities,
and proposes a sourced Finding. Automatic memory review is off; write approval is on.
Native model and permission settings use the installed harness configuration.

Open the one-use `url` from the run directory's private `ui.json`. When the driver prints
that it is waiting for approval, inspect the pending Finding in project Settings and
choose **Approve and save**. A new native session then reads the saved result. The driver
exports evidence and leaves the browser Host running for screenshots; press Ctrl-C to
stop it. Do not publish `ui.json`, the private product home or native transcripts.

The retained run completed three native turns (analysis, delegated check and recall)
and two Docker computations. An earlier author attempt used incorrect tool arguments;
the retained driver supplies the corrected argument shapes. Native approval handling
also affected a child inspection attempt. This is a worked example, not a first-attempt
success-rate experiment. New runs may differ and generate new IDs and timestamps.

The reporting code was corrected after the retained run to count delegation using the
journal's `causedBy` link. The published count and run summaries were regenerated from
that journal; the computations were not rerun or changed.

Current browser captures are `docs/figures/swarmx-current-science.png` and
`docs/figures/swarmx-current-provenance.png`. They use this run's real application state.
