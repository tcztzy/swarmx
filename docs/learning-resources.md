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
      "validate": ["node", "scripts/validate-research-skill.mjs"]
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
Reviewers can propose only `{id, expectedRevision, content}` for a supplied resource; they cannot
choose a different target or change its validator. The durable learning plan retains this
snapshot and the candidate. Existing write-approval settings control when it can be applied.

The registered validator runs in the project directory without a shell. Its fixed argument
list receives one final argument: the absolute candidate Markdown path. The candidate is a
private temporary file alongside the original, so ordinary relative Markdown links keep their
base directory. The validator must inspect that candidate; testing the unchanged original
does not validate an update. Registration authorizes this project-owned program to run, and
the Host does not sandbox its side effects. Use a validator appropriate to the resource's
behavior, including native parsing or task fixtures where relevant.

Application requires a successful validator exit within 30 seconds, an unchanged registration,
and the expected original revision. Failure, cancellation or revision conflicts prevent the
Host from replacing the resource. After successful validation the Host atomically replaces it
and reports the new digest. Replaying an operation whose desired bytes are already present
still reruns validation, but does not rewrite the resource. Temporary candidates are removed.

A passed validator records only that the configured checks passed; it is not a general claim
of improved model quality. Native runtimes retain ownership of skill discovery and context
loading. Updated resources apply when a runtime next loads them for a new task; existing
conversations are not promised a live refresh.
