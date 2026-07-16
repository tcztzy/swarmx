import { describe, expect, it } from "vitest";
import {
  createAgentProfileFromDefinition,
  parseAgentDefinitionMarkdown,
  parseAgentProfileMetadata,
  parseCodexAgentDefinitionToml,
  parseNativeAgentDefinition,
  projectAgentDefinitionForClaudeCode,
  projectAgentDefinitionForCodex,
  serializeAgentDefinitionMarkdown,
} from "../src/agent-profiles.js";

const agentMarkdown = `---
name: geepilot-reviewer
description: Review GEEPilot changes for spec drift and missing tests.
model: inherit
tools: Read, Grep
disallowedTools:
  - Bash
permissionMode: plan
mcpServers:
  - project-fs
maxTurns: 6
skills:
  - geepilot
initialPrompt: Use the repository specs.
memory: readonly
effort: high
background: false
isolation: workspace
color: blue
customHostField: preserved
geepilot:
  harness: claude
  supply: deepseek-claude
  selector: "@reviewer"
  enabled: true
  source: plugin
---

Review changes against the relevant specs. Return concrete findings first.
`;

const codexAgentToml = `name = "reviewer"
description = "Review changes for correctness, security, and missing tests."
developer_instructions = """
Review code like an owner.
Return concrete findings before summaries.
"""
nickname_candidates = ["Atlas", "Delta"]
model = "gpt-5.6-sol"
model_reasoning_effort = "max"
sandbox_mode = "read-only"
custom_host_field = "preserved"

[mcp_servers.github]
command = "github-mcp"

[[skills.config]]
path = "skills/code-review/SKILL.md"
enabled = true

[[skills.config]]
path = "skills/disabled/SKILL.md"
enabled = false
`;

describe("agent profile definition primitives", () => {
  it("parses Claude-compatible Markdown frontmatter and preserves inert unknown fields", () => {
    const definition = parseAgentDefinitionMarkdown(agentMarkdown, {
      source: { kind: "plugin", pluginId: "geepilot", path: "agents/reviewer.md" },
    });

    expect(definition.frontmatter).toMatchObject({
      name: "geepilot-reviewer",
      description: "Review GEEPilot changes for spec drift and missing tests.",
      model: "inherit",
      tools: ["Read", "Grep"],
      disallowedTools: ["Bash"],
      permissionMode: "plan",
      maxTurns: 6,
      skills: ["geepilot"],
      memory: "readonly",
      effort: "high",
      background: false,
      isolation: "workspace",
      color: "blue",
      customHostField: "preserved",
      geepilot: {
        harness: "claude",
        supply: "deepseek-claude",
        selector: "@reviewer",
        enabled: true,
        source: "plugin",
      },
    });
    expect(definition.body).toContain("Review changes against the relevant specs.");
    expect(definition.source).toMatchObject({
      kind: "plugin",
      pluginId: "geepilot",
    });
  });

  it("converts definitions into Harness x Model identity with separate supply routing", () => {
    const definition = parseAgentDefinitionMarkdown(agentMarkdown, {
      source: { kind: "plugin", pluginId: "geepilot", path: "agents/reviewer.md" },
    });

    const profile = createAgentProfileFromDefinition(definition, {
      modelId: "claude-sonnet-4-20250514",
      aliases: ["@spec-review"],
    });

    expect(profile).toMatchObject({
      id: "geepilot-reviewer",
      name: "geepilot-reviewer",
      aliases: ["@spec-review", "@reviewer"],
      harnessId: "claude",
      modelSupplyId: "deepseek-claude",
      modelId: "claude-sonnet-4-20250514",
      tools: ["Read", "Grep"],
      disallowedTools: ["Bash"],
      skills: ["geepilot"],
      mcpServers: ["project-fs"],
      permissionMode: "plan",
      maxTurns: 6,
      memory: "readonly",
      effort: "high",
      background: false,
      isolation: "workspace",
      color: "blue",
      enabled: true,
      readOnly: true,
      pluginIds: ["geepilot"],
    });
    expect(profile.instructions).toBe(
      "Review changes against the relevant specs. Return concrete findings first.",
    );
    expect(JSON.stringify(profile)).not.toContain("apiKey");
  });

  it("parses Codex TOML into the same profile model while preserving native config", () => {
    const definition = parseCodexAgentDefinitionToml(codexAgentToml, {
      source: { kind: "project", path: ".codex/agents/reviewer.toml" },
    });

    expect(definition).toMatchObject({
      format: "codex",
      frontmatter: {
        name: "reviewer",
        description: "Review changes for correctness, security, and missing tests.",
        nicknameCandidates: ["Atlas", "Delta"],
        model: "gpt-5.6-sol",
        effort: "max",
        sandboxMode: "read-only",
        mcpServers: ["github"],
        skills: ["skills/code-review/SKILL.md"],
      },
      source: {
        kind: "project",
        host: "codex",
        format: "codex",
      },
    });
    expect(definition.body).toContain("Review code like an owner.");
    expect(definition.nativeConfig).toMatchObject({
      custom_host_field: "preserved",
      mcp_servers: { github: { command: "github-mcp" } },
    });

    expect(
      createAgentProfileFromDefinition(definition, {
        id: "codex:reviewer",
        harnessId: "codex",
        readOnly: true,
      }),
    ).toMatchObject({
      id: "codex:reviewer",
      harnessId: "codex",
      modelId: "gpt-5.6-sol",
      nativeModel: "gpt-5.6-sol",
      effort: "max",
      sandboxMode: "read-only",
      nicknameCandidates: ["Atlas", "Delta"],
      mcpServers: ["github"],
      skills: ["skills/code-review/SKILL.md"],
      readOnly: true,
    });
  });

  it("round-trips Codex TOML and keeps host-only policy out of Claude projections", () => {
    const definition = parseNativeAgentDefinition(codexAgentToml, { format: "codex" });
    const projected = projectAgentDefinitionForCodex(definition);
    const reparsed = parseCodexAgentDefinitionToml(projected);
    const claudeProjection = projectAgentDefinitionForClaudeCode(definition);

    expect(projected).toContain('custom_host_field = "preserved"');
    expect(projected).toContain("[mcp_servers.github]");
    expect(projected).toContain('model_reasoning_effort = "max"');
    expect(reparsed.frontmatter).toMatchObject(definition.frontmatter);
    expect(reparsed.body).toBe(definition.body);
    expect(claudeProjection).toContain("name: reviewer");
    expect(claudeProjection).toContain("effort: max");
    expect(claudeProjection).not.toContain("sandbox_mode");
    expect(claudeProjection).not.toContain("sandboxMode");
    expect(claudeProjection).not.toContain("mcp_servers");
    expect(claudeProjection).not.toContain("custom_host_field");
  });

  it("keeps native inherit explicit instead of fabricating a SwarmX Model", () => {
    const definition = parseCodexAgentDefinitionToml(`
name = "worker"
description = "Use the parent model."
developer_instructions = "Implement the assigned change."
model = "inherit"
`);
    const profile = createAgentProfileFromDefinition(definition, {
      harnessId: "codex",
    });

    expect(profile.nativeModel).toBe("inherit");
    expect(profile.modelId).toBeUndefined();
  });

  it("requires the documented fields for native host definitions", () => {
    expect(() =>
      parseCodexAgentDefinitionToml(`
name = "reviewer"
description = "Missing instructions"
`),
    ).toThrow(/developer_instructions/i);

    expect(() =>
      parseNativeAgentDefinition(
        `---
name: reviewer
---

Review code.
`,
        { format: "claude_code" },
      ),
    ).toThrow(/requires name and description/i);
  });

  it("persists a reproducible Harness recipe separately from the Model", () => {
    const profile = parseAgentProfileMetadata({
      id: "researcher",
      name: "Researcher",
      harnessId: "research-harness",
      harnessRecipe: {
        id: "research-harness",
        revisionId: "research-harness@1",
        softwareId: "codex",
        skillBindings: [{ skillId: "paper-review", mode: "auto" }],
        mcpServerIds: ["zotero"],
      },
      modelId: "gpt-5",
    });

    expect(profile).toMatchObject({
      harnessId: "research-harness",
      harnessRecipe: { softwareId: "codex", revisionId: "research-harness@1" },
      modelId: "gpt-5",
    });
    expect(() =>
      parseAgentProfileMetadata({
        ...profile,
        harnessId: "different-harness",
      }),
    ).toThrow(/harnessId must match/i);
  });

  it("rejects inline secrets in definitions and profile metadata while allowing secret refs", () => {
    expect(() =>
      parseAgentDefinitionMarkdown(`---
name: bad
description: Bad profile
apiKey: sk-test
---

Do work.
`),
    ).toThrow(/inline secret field.*apiKey/);

    expect(() =>
      parseAgentProfileMetadata({
        id: "bad",
        name: "bad",
        providerApiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*providerApiKey/);

    expect(() =>
      parseCodexAgentDefinitionToml(`
name = "bad"
description = "Bad Codex profile"
developer_instructions = "Do work."
api_key = "sk-test"
`),
    ).toThrow(/inline secret field.*api_key/);

    expect(
      parseAgentProfileMetadata({
        id: "ok",
        name: "ok",
        secretRef: { source: "env", key: "OPENAI_API_KEY" },
      }),
    ).toMatchObject({
      id: "ok",
    });
  });

  it("rejects malformed or non-object frontmatter", () => {
    expect(() =>
      parseAgentDefinitionMarkdown(`---
- name
---

Body
`),
    ).toThrow(/frontmatter must be a YAML object/);

    expect(() =>
      parseAgentDefinitionMarkdown(`---
name: [unterminated
---

Body
`),
    ).toThrow(/Invalid agent definition frontmatter/);
  });

  it("serializes definitions and projects Claude Code frontmatter without GEEPilot-only metadata", () => {
    const definition = parseAgentDefinitionMarkdown(agentMarkdown);
    const serialized = serializeAgentDefinitionMarkdown(definition);
    const claudeProjection = projectAgentDefinitionForClaudeCode(definition);

    expect(serialized).toContain("geepilot:");
    expect(serialized).toContain("customHostField: preserved");
    expect(claudeProjection).toContain("name: geepilot-reviewer");
    expect(claudeProjection).toContain("customHostField: preserved");
    expect(claudeProjection).toContain("Review changes against the relevant specs.");
    expect(claudeProjection).not.toContain("geepilot:");
    expect(claudeProjection).not.toContain("supply: deepseek-claude");
  });
});
