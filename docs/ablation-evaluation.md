# Built-in service ablation

SwarmX can run a single direct-backend `eval-run` sample with an explicit
`AblationProfile` to measure the contribution of three built-in Agent services:

| Seam | `production` | `baseline` |
| --- | --- | --- |
| `context_engine` | Uses the supplied Context Engine | Removes Context Engine compilation and tools |
| `memory` | Projects supplied global/personal Memory, reflection, and Memory-owned tools | Removes Memory prompt projection, reflection, and tools |
| `skill_evolution` | Delivers the supplied active or explicit Skill fragments | Removes evolved/experimental Skill delivery |

Profiles are evaluation inputs, not persisted workflows or production defaults.
All three selections are mandatory so an omitted seam cannot silently inherit a
different implementation.

```json
{
  "schemaVersion": 1,
  "profileId": "without_context_engine",
  "variants": {
    "context_engine": "baseline",
    "memory": "production",
    "skill_evolution": "production"
  }
}
```

Run the same immutable config and input once with an all-`production` profile
and once with a one-seam `baseline` profile:

```shell
swarmx eval-run \
  --config eval.json \
  --ablation-profile without-context.json \
  --memory-snapshot memory.json \
  "Continue the coding task"
```

An explicit ablation run supplies the built-in Context Engine to every profile;
the `context_engine` variant decides whether the Agent receives it. When Memory
is part of the experiment, pass the same strict global Memory snapshot to every
arm with `--memory-snapshot`; the `memory` variant decides whether it reaches
the Agent. Programmatic hosts may also supply Memory-owned tools through the
same seam. A Memory snapshot without `--ablation-profile` is rejected.

The result retains ordinary output and metrics and adds an `ablation` receipt
containing only the profile id, selected variant ids, and deterministic
Swarm/Agent topology. It contains no prompt, Memory, Skill content, model output,
tool arguments/results, credentials, or environment values. Compare the tagged
metrics and task scorer results across profiles; `eval-run` does not promote a
winner or reinterpret one sample as statistically conclusive.

Resolution is fail-closed. Duplicate providers or a profile naming any missing
variant stops Agent construction before MCP startup or a Provider request. A
custom registry without an explicit profile is rejected so activations cannot
lose their run identity. Explicit ablation is supported only by the direct
`swarmx` backend; echo and external ACP Harnesses do not execute these services
and therefore cannot emit an activation receipt.
