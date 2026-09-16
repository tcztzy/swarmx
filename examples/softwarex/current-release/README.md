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

The recorded workflow imported the input and supplied plotting code and exact tool
argument shapes to a lead Codex session. That session executed the analysis, delegated
a standard-library recomputation, recorded the claim and supporting entities, and
proposed a sourced Finding. Automatic memory review was off and write approval was on.
The author approved the Finding before a new native session retrieved it.

The files in this directory support offline verification of that recorded run. They do
not establish that the native workflow has been rerun against the current application.

The retained run completed three native turns (analysis, delegated check and recall)
and two Docker computations. An earlier author attempt used incorrect tool arguments;
the retained run used corrected argument shapes. Native approval handling
also affected a child inspection attempt. This is a worked example, not a first-attempt
success-rate experiment.

The reporting code was corrected after the retained run to count delegation using the
journal's `causedBy` link. The published count and run summaries were regenerated from
that journal; the computations were not rerun or changed.

The captures at `docs/figures/swarmx-current-science.png` and
`docs/figures/swarmx-current-provenance.png` show this recorded run's application state.
