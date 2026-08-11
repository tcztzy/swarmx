# Context strategy evaluation fixtures

`smoke-suite.json` is an editable, development-split suite for comparing the
SwarmX baseline, OpenCode, Codex CLI, Claude Code, Hermes, Reasonix, LCM,
Parallel Compaction, and ReSum context recipes. Its five cases cover durable
constraints, recovery, superseded state, duplicate-effect avoidance, and
safety under pressure.

Before running it, replace the two placeholder continuation/summary Model ids
with models available through the corresponding Provider environment. The
suite never stores credentials. Delete either Agent entry for a one-model run.

```shell
mkdir -p .local
swarmx eval-run \
  --context-suite evals/context/smoke-suite.json \
  --context-jsonl .local/context-eval-runs.jsonl \
  --pretty
```

The checked-in configuration is a bounded first pass: 10 arms × 2 Agents × 5
cases × 1 repetition = 100 runs. For a confirmation pilot, copy the file
outside the repository, use at least three `repetitionSeeds`, set
`search.rounds` to 2 or 3, and raise `maxRuns` to the exact preflight bound
reported by the CLI. Keep development and confirmation cases in separate
suite versions; adaptive candidates are review evidence, not production
defaults.

The simulator deliberately does not reproduce native Harness session state or
real repository side effects. Promote promising arms only after replaying a
sample through the corresponding native Harness lifecycle.
