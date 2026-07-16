# Extensions, Harness Recipes, and Custom Agents

SwarmX separates distribution, execution, and model choice:

```text
Extension = distributable Software, Skills, MCP servers, and other components
Harness   = Software + Skills + MCP servers + project context + policies + ...
Agent     = Harness + Model
```

An Extension can contribute reusable pieces without forcing them into every
Harness. A Harness is a versioned, reproducible recipe assembled from those
pieces. An Agent pairs one Harness recipe with one Model. This keeps
model-specific behavioral guidance out of the ambient context unless that Agent
actually needs it.

## Settings surfaces

Desktop management lives under the account menu's Settings workspace:

- **Providers** manages Model supply connections and usage metadata.
- **Extensions** manages marketplace sources, plugin revision state, component
  inventory, update/rollback actions, and Skill evolution policy.
- **Custom Agents** composes a Harness recipe from Software, Skill bindings, MCP
  servers, project context, delivery/permission policy, and a Model. Harness
  Software health and setup actions live beside the Software selector.
- **Runtime** presents Node.js as the standalone baseline, then lists every
  Harness tool with its detected semver or Install action, plus Apple Container.
  Doctor diagnostics and confirmed repair are built into the same page.

The conversation list has no permanent Doctor status card. `/doctor`,
`/doctor --fix`, `/setup`, and failed-send recovery continue to use the same
diagnostic APIs, while Runtime owns the normal Doctor UI. Repair remains
explicitly confirmed.

## Harness recipe and revision identity

`HarnessRecipeSchema` records the Software id/version, `off | auto | required`
Skill bindings, optional pinned Skill variants, MCP ids, project paths and
instruction files, delivery policy, permission policy, and contributing plugin
ids. Saving a Custom Agent canonicalizes that content, computes a SHA-256
digest, and derives a stable recipe revision. Changed revisions are retained in
the Agent's recipe history. Unchanged content reuses the existing revision.

The desktop projects each saved recipe into the runnable Harness inventory.
The ordinary composer therefore continues to select a Harness, Model, and
Effort; it does not need a separate Custom Agent execution path.

## Codex and Claude Code Agent compatibility

SwarmX treats host-native Agent files as definition adapters around its
canonical Agent profile rather than as a second execution model:

| Host | User scope | Project scope | Native format |
| --- | --- | --- | --- |
| Codex | `~/.codex/agents/*.toml` | `.codex/agents/*.toml` | TOML with required `name`, `description`, and `developer_instructions` |
| Claude Code | `~/.claude/agents/*.md` | `.claude/agents/*.md` | YAML frontmatter plus a Markdown system prompt |

The desktop scans only those four directories, only accepts the expected file
extension, and limits each file to 1 MiB. Malformed or secret-bearing files
become inventory warnings instead of breaking the Settings workspace. Within a
host, a project definition overrides a same-name user definition. Profile ids
are host-namespaced, so a Codex `reviewer` and Claude Code `reviewer` remain two
separate definitions.

Codex `model`, `model_reasoning_effort`, `sandbox_mode`,
`nickname_candidates`, `mcp_servers`, and `skills.config` are normalized into
the common metadata where semantics are safe. The full safe TOML table remains
available for round-trip projection. Claude Code fields continue to map from
frontmatter and the Markdown body remains the Agent instructions. Unknown safe
fields are preserved; inline secret-looking fields are rejected.

Native definitions are displayed under **Native definitions · read-only**.
Their Harness is fixed to `codex` or `claude_code`, but an omitted Model or the
native value `inherit` does not become a guessed SwarmX default. Such a profile
stays unresolved until an explicit composition provides a Model and passes the
normal Harness x Model preflight. Listing a definition never runs its hooks,
connects its inline MCP servers, loads its Skills, starts a subprocess, or
writes either the native file or SwarmX settings.

The browser-safe core API provides both directions for hosts that want an
explicit import/export workflow:

- `parseNativeAgentDefinition()` dispatches by `claude_code` or `codex`.
- `parseAgentDefinitionMarkdown()` and `parseCodexAgentDefinitionToml()` expose
  the individual codecs.
- `projectAgentDefinitionForClaudeCode()` and
  `projectAgentDefinitionForCodex()` emit conservative native projections.

Cross-host projection carries only shared role/model/effort fields. It does not
translate Codex sandbox or MCP configuration into Claude Code policy, or vice
versa, because those settings do not have guaranteed equivalent semantics.

Native format references: [Codex custom agents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [Claude Code custom subagents](https://code.claude.com/docs/en/sub-agents).

## Logical Skills and Agent-specific variants

A logical Skill owns one or more immutable variants. Variants may target an
Agent profile, exact Model, Model family or pattern, Model capabilities,
Harness, underlying Software, or platform. They also declare how their content
is delivered:

- `prompt_fragment`
- `host_native_plugin`
- `rules_file`
- `unsupported`

Resolution is deterministic. An explicit variant pin wins, followed by exact
Agent, exact Model, Model family/pattern, capability, Harness/Software,
platform, and finally the logical default. Priority breaks ties only within the
same target class. An unresolved tie is blocked instead of being chosen by
array order. A `required` binding blocks when delivery is unsupported; `auto`
may omit it. The UI shows the declared context-token estimate before save.

This permits two variants with the same logical Skill name to be optimized for
different Agents or Models. A strong base Model can set the binding to `off`,
while a weaker or behaviorally different Model can require a constrained
variant without changing the underlying Software.

## Legacy manifest migration

Existing Extension manifests remain valid. A legacy Skill record without a
`variants` array is normalized to one `${skillId}:default` variant with
`source: legacy` and version `0.0.0-legacy`. Its canonical path becomes a
prompt-fragment reference; a plugin-only host exposure becomes native-plugin
delivery; otherwise delivery is marked unsupported. A later manifest may add
explicit variants without changing the logical Skill id referenced by Harness
recipes.

## Marketplace, update, and rollback

Marketplace sources accept credential-free HTTPS catalogs/registries or local
paths. Credentials are references owned by the host and are never valid inline
manifest or settings fields. Extension lifecycle state distinguishes available,
installed, enabled, disabled, update-available, blocked, diverged, conflict, and
pinned revisions.

Install, update, trust changes, uninstall, and rollback require explicit
confirmation. An update promotes the supplied upstream revision while retaining
the previous immutable revision. Rollback selects a retained revision; a pinned
Extension remains pinned to the rollback target and cannot update until it is
unpinned. Built-in/read-only components cannot be removed.

The desktop manager refreshes credential-free HTTPS catalogs with redirects
disabled, a bounded response size, and a timeout; local sources read a bounded
`catalog.json`. Valid candidates are rebound to the selected source id, assigned
trust no higher than that source, hashed, and atomically cached. Remote component
code is still loaded only through the existing trusted host inventory loader:
refreshing a catalog discovers immutable candidates but does not evaluate remote
scripts, mount remote UI, or expose credentials.

## Skill evolution governance

Skill evolution uses immutable candidates rather than mutating an active Skill.
Each candidate records its parent/upstream revisions, target Agent, target Model
fingerprint, optimizer/version/config digest, and status. Evaluation records the
dataset digest, baseline and candidate metrics, and seed. The default eligibility
rule requires higher quality, no lower safety, no higher failure rate, and no
larger context footprint.

Settings can disable candidate generation or choose a human or evaluation-policy
promotion gate. Candidate, evaluation, promotion/canary, quarantine, and
rollback contracts are deliberately optimizer-neutral so a Hermes-style
self-improvement loop, SkillOpt-style optimizer, or another runner can plug into
the same audit trail. Optimizers cannot silently overwrite the active revision.

## Persistence and trust boundaries

Providers, Models, Model supplies, Custom Agents, Extension sources, installed
revisions, and evolution policy share one queued atomic settings store at
`~/.swarmx/settings.json`. Section updates merge against the latest document so
concurrent Provider and Extension writes do not erase one another. Zod validates
every IPC boundary and recursively rejects inline secret-shaped fields.
