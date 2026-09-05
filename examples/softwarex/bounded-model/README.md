# Bounded Swarm model result

This directory contains the deterministic depth-7 result reported in the SoftwareX manuscript.
The model exhaustively explores the finite transition system in
`packages/core/swarm/src/verification-model.ts`; it does not execute prompts or estimate model
quality.

From the repository root, reproduce the JSON report with:

```sh
pnpm install --frozen-lockfile
pnpm paper:model
```

The committed reference output is `results/depth-7.json`. The regression test in
`packages/core/swarm/tests/verification-model.test.ts` fixes the reported state counts, per-rule
violation counts, and earliest counterexample depths.
