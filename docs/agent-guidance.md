# Model, Harness, and Agent guidance

SwarmX distributes a conservative, source-dated task-guidance catalog from
`@swarmx/core/agent-guidance`. It describes where a Model, Harness, or exact
`harnessId:modelId` Agent has positive evidence. It is passive metadata: it does
not select a Provider, change permissions, make an incompatible route runnable,
or write user Memory.

## Boundaries

- Model capability metadata remains the authority for APIs, reasoning controls,
  context limits, and other executable facts.
- Harness metadata remains the authority for software, launch recipes, model
  control, supported APIs, and environment handling.
- Guidance records contain curated task-fit conclusions and their evidence.
- Missing guidance means **unrated**. It never implies weak or unsupported.
- `weak` is a comparative, evidence-backed assessment. `unsupported` is not a
  guidance verdict; it belongs to the capability and compatibility boundaries.
- User-specific experience belongs in confirmed Memory. Local evaluation runs
  remain structured evaluation evidence and are not copied into this catalog or
  Memory automatically.

The initial catalog intentionally has no automatic recommendation or promotion
side effect. Consumers can query the three layers and must keep the returned
conditions and limitations visible.

## Target and precedence

A record targets exactly one layer:

| Target | Meaning |
| --- | --- |
| `model` | Evidence about a Model under the recorded API/effort configuration |
| `harness` | Evidence about a Harness family; often indirect when the benchmark used the upstream native product |
| `agent` | Evidence about one exact `harnessId:modelId` composition |

`getAgentGuidance()` returns an exact Agent record before matching Model and
Harness records. It does not merge their verdicts into a synthetic score. This
keeps conflicting or differently scoped evidence inspectable.

## Task families

The version 1 taxonomy is deliberately small:

| Family | Scope |
| --- | --- |
| `general` | Broad objective multi-category quality |
| `reasoning` | Verifiable analytical and logical reasoning |
| `coding` | Code generation and completion without a full repository runtime |
| `agentic_coding` | Multi-step coding with an agent/tool loop |
| `repository_coding` | Issue resolution against a real repository and tests |
| `terminal_work` | End-to-end tasks completed in a terminal environment |
| `tool_use` | Function selection, arguments, multi-turn calls, and related agentic tool behavior |
| `mathematics` | Objective mathematical problem solving |
| `data_analysis` | Objective data interpretation and analysis |
| `language` | Language understanding and generation tasks |
| `instruction_following` | Compliance with explicit response constraints |

These labels describe benchmark constructs, not industries or safety domains.

## Initial evidence snapshot

All URLs below were checked on **2026-08-12**. Dynamic leaderboards are a
snapshot, not a promise that ranks or prices remain current.

| Source id | Version/snapshot | What it supports | Primary source |
| --- | --- | --- | --- |
| `livebench-2026-06-25` | LiveBench release `2026-06-25` | Model-level general, reasoning, coding, agentic coding, mathematics, data analysis, language, and instruction-following evidence | [LiveBench leaderboard](https://livebench.ai/) |
| `terminal-bench-2.1` | `terminal-bench@2.1`, leaderboard snapshot checked 2026-08-12 | Verified Agent×Model terminal-task results; upstream native Codex/Claude Code is indirect evidence for SwarmX ACP Harnesses | [Terminal-Bench 2.1 leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.1) |
| `swe-bench-verified` | Verified leaderboard snapshot checked 2026-08-12 | Agent×Model real-repository issue-resolution evidence | [SWE-bench Verified leaderboard](https://www.swebench.com/) |
| `bfcl-v4-2025.12.17` | BFCL V4 commit `f7cf735`, evaluator `2025.12.17`, leaderboard updated 2026-04-12 | Model/API function- and tool-calling evidence | [Berkeley Function Calling Leaderboard V4](https://gorilla.cs.berkeley.edu/leaderboard) |

Leaderboard measurements are not interchangeable. LiveBench category scores
do not establish a Harness result. Terminal-Bench and SWE-bench measure the
whole submitted system, so their records use `agent_model` evidence scope.
Terminal-Bench also demonstrates why dates matter: version 2.1 corrected 28 of
89 version 2.0 tasks, and the published comparison changed several Agent×Model
results materially. BFCL distinguishes native function calling from prompted
workarounds; the catalog records the evaluated mode.

## Verdicts and confidence

- `preferred`: leading positive evidence for the exact recorded scope and
  configuration. This is still advice, not a compatibility fact.
- `suitable`: positive evidence exists, but the target was not leading or the
  transfer to SwarmX is indirect.
- `weak`: comparative evidence supports a negative assessment. The initial
  catalog avoids this verdict where absence or configuration mismatch is the
  only signal.

Confidence describes evidence transfer, not model certainty. Exact model-level
benchmark results may be `high`; upstream Harness results projected onto an ACP
adapter are at most `medium` until SwarmX runs the same workload through that
adapter.

## Update procedure

1. Read the primary leaderboard and methodology, not a mirror or vendor claim.
2. Record the exact release/commit, evaluated Model and Harness labels, effort or
   API mode, metric, result date when available, URL, and `checkedAt` date.
3. Add a new source id when the benchmark release or scoring contract changes.
   Do not silently reinterpret existing measurements.
4. Map a benchmark label to a SwarmX id only when the identity is explicit.
5. Mark whole-system results as `agent_model`; do not attribute them to the bare
   Model. Mark upstream native Harness evidence as an explicit limitation when
   SwarmX runs it through an ACP adapter.
6. Keep claims narrow. Do not infer unsupported, security, domain expertise, or
   cost guarantees from an absent or unrelated score.
7. Update focused tests and this source table, then run the Core tests,
   `pnpm docs:check`, lint, and the relevant build gate.
