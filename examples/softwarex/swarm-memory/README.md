# Swarm-to-Memory illustrative workflow

This directory contains the deterministic input and analysis used by the second SoftwareX
illustrative example. The data are synthetic: the example evaluates orchestration, verification,
and provenance behavior, not a biological hypothesis.

The workflow gives two independent read tasks to research agents. One checks the arithmetic in
`data/germination.csv`; the other checks completeness and records the input digest. A write task,
blocked by both reads, runs `scripts/summarize.mjs` to create the two files under `results/`. An
independent verifier checks the output and digest. Finally, the lead admits the verified summary
through a Swarm knowledge task into the workspace memory.

Memory is SwarmX's shared semantic memory store, implemented as OKF Markdown concepts. This
recorded workflow predates the naming update and documents the evaluated implementation, not
the current Swarm's feature set. The manifest and concept retain their original producer labels,
schema keys, identifiers, and bytes so the owner receipt remains independently checkable. The
surrounding documentation, reproduction paths, and RO-Crate use the current Memory name.

Run the deterministic analysis from any directory:

```sh
node examples/softwarex/swarm-memory/scripts/summarize.mjs
```

Run the complete fixture verification from the repository root:

```sh
pnpm paper:demo
```

This reruns the summarizer, checks that `results/` contains exactly the two declared outputs,
verifies every published digest, checks the numerical result and synthetic-data boundary, and
confirms that the exported memory page hashes to the recorded owner receipt.

Expected numerical result:

- control mean germination: 79%
- primed mean germination: 87%
- primed-minus-control difference: 8 percentage points

The exact task, attempt, submission, verdict, and admission identifiers from the evaluated Desktop
run are in `run/manifest.json`. It is a sanitized projection of the durable journal: native Thread
identifiers, host paths, credentials, and private model reasoning are excluded.
`run/memory-concept.md` is the committed concept returned by the memory owner. The
publication screenshots, manifest, concept, and file digests all describe the same run.

`ro-crate-metadata.json` describes the attached reproduction fixture as an RO-Crate 1.3 research
object. `MANIFEST.sha256` covers every file needed to inspect or rerun the example; the verification
command checks both records in addition to regenerating the result.
