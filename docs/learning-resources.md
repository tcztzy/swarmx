# Project learning resources

Projects can explicitly allow execution reviews to propose changes to existing Markdown agent
instructions and native skills. This is separate from saving a Memory Playbook: an applied
resource update replaces the registered project file.

Register resources in `.swarmx/learning.json`:

```json
{
  "resources": [
    {
      "id": "research-skill",
      "kind": "skill",
      "path": ".agents/skills/research/SKILL.md",
      "validate": ["node", "scripts/validate-research-skill.mjs"],
      "evaluate": ["node", "scripts/evaluate-research-skill.mjs"]
    }
  ]
}
```

`kind` is `agent` or `skill`. Paths name existing Markdown files within this project; absolute
paths, parent traversal, symbolic links and the registration file are rejected. An absent
registration file means no resources. The list is bounded to 10 distinct resources, with at
most 128 KiB per file and 32,000 serialized characters across all snapshots. Oversized snapshots
are rejected intact. Installed plugins and global resources are not registered implicitly.

The Host snapshots each resource's complete content, SHA-256 revision and registration revision.
Reviewers can propose only `{id, expectedRevision, content, evaluation}` for a supplied resource; they cannot
choose a different target or change its validator or evaluator. The durable learning plan retains this
snapshot and the candidate. Existing write-approval settings control when it can be applied.
The evaluation records its task, criteria, original execution evidence, counterevidence and
limitations using the same [Memory contract](memory.md#agent-selection-experience). References
must resolve to original events included in this review's snapshot before a proposal is staged
or a validator runs. The Host stamps the review source; a prior model conclusion cannot serve
as fresh evidence. Previously persisted plans keep their replay contract.

The registered validator runs in the project directory without a shell. Its fixed argument
list receives one final argument: the absolute candidate Markdown path. The candidate is a
private temporary file alongside the original, so ordinary relative Markdown links keep their
base directory. The validator must inspect that candidate; testing the unchanged original
does not validate an update. Registration authorizes this project-owned program to run, and
the Host does not sandbox its side effects. Use a validator appropriate to the resource's
structure, including native parsing and interface checks where relevant.

`evaluate` is optional and separate from the quick validator. When registered, this fixed
program receives two final arguments: an absolute private baseline Markdown path containing
the snapshotted original bytes, then the candidate path. Even when replaying an already applied
update, the baseline remains the original snapshot. The evaluator has five minutes and must
write one JSON object of at most 16 KiB to stdout:

```json
{
  "evaluatorVersion": "research-fixtures-v1",
  "passed": true,
  "baseline": {
    "revision": "sha256:<64 lowercase hexadecimal characters>",
    "passedCases": 3,
    "totalCases": 4,
    "cost": { "amount": 0.01, "unit": "USD" }
  },
  "candidate": {
    "revision": "sha256:<64 lowercase hexadecimal characters>",
    "passedCases": 4,
    "totalCases": 4,
    "cost": { "amount": 0.01, "unit": "USD" }
  },
  "summary": "Candidate passes the held-out scientific task fixtures."
}
```

`cost` is optional; omission means unknown, not zero. The trusted evaluator owns its task
fixtures, cost requirements and stricter acceptance criteria. The Host additionally requires
matching content revisions, a nonempty equal case count, no reduction in passed cases and
`passed: true`. A zero exit alone is insufficient. The evaluator must use the same cases for
both versions, include content-error counterexamples, redact its bounded summary and keep
hidden materials outside Agent access. Configuring an evaluator authorizes its execution,
not extra Agent access to its fixtures. Both temporary files must remain unchanged.

Application requires a successful validator exit within 30 seconds, a passing evaluation when
configured, an unchanged registration and the expected original revision. Failure, cancellation or revision conflicts prevent the
Host from replacing the resource. Cancellation, timeout and oversized reports stop the command
and its ordinary descendants: an owned process group on macOS/Linux, or Windows `taskkill /T /F`.
Registered commands must not detach children from that lifetime; this is not a sandbox. Windows
remains unvalidated. After successful validation the Host atomically replaces it
and returns the original/candidate/configuration digests, candidate/validation/evaluation/
adoption stages, validation/evaluation elapsed time, and the parsed evaluation report with its
SHA-256 digest. Automatic review and manual approval store this result in existing journal
receipts. They also retain rejected comparisons and their digests as
`swarmx.memory.resource.evaluation.rejected` events. Malformed or oversized reports fail without
retaining raw output. Replaying an operation whose desired bytes are already present
still reruns validation, but does not rewrite the resource. Temporary candidates are removed.

Without `evaluate`, existing registrations remain supported and the receipt explicitly says
`structural-only`; they provide no behavioral acceptance. Evaluated updates say
`behavior-tested`, scoped to the reported fixtures; they do not establish general model-quality
improvement. Continued observation, withdrawal and long-term net-benefit comparisons remain
the next resource lifecycle milestone. Native runtimes retain ownership of skill discovery and context
loading. Updated resources apply when a runtime next loads them for a new task; existing
conversations are not promised a live refresh.
