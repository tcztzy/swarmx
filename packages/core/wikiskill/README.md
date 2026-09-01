# @swarmx/dsh-wikiskill

Explicit WikiSkill evolution for SwarmX: DSH native Session logs are Raw, the existing PKB is Wiki,
and this package owns only staged and target-validated active `SKILL.md` + `PURPOSE.md` artifacts.

The package exposes a bounded DSH Raw reader, a revision-fenced proposal/validation store, a
PKB-ready `SkillImpact` draft, and an exact-Agent provider that reuses DSH's existing filesystem
discovery and model-facing skill loader. It registers no model Tool and performs no background LLM
or implicit PKB write.

See [`docs/wikiskill.md`](../../../docs/wikiskill.md) for the complete contract.
