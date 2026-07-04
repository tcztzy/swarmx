import { describe, expect, it } from "vitest";
import {
  createAgentProfileFromDefinition,
  parseAgentDefinitionMarkdown,
  parseAgentProfileMetadata,
  projectAgentDefinitionForClaudeCode,
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
  provider: deepseek
  selector: "@reviewer"
  enabled: true
  source: plugin
---

Review changes against the relevant specs. Return concrete findings first.
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
        provider: "deepseek",
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

  it("converts definitions into separate profile metadata without collapsing harness or provider state", () => {
    const definition = parseAgentDefinitionMarkdown(agentMarkdown, {
      source: { kind: "plugin", pluginId: "geepilot", path: "agents/reviewer.md" },
    });

    const profile = createAgentProfileFromDefinition(definition, {
      model: "claude-sonnet-4-20250514",
      aliases: ["@spec-review"],
    });

    expect(profile).toMatchObject({
      id: "geepilot-reviewer",
      name: "geepilot-reviewer",
      aliases: ["@spec-review", "@reviewer"],
      harnessId: "claude",
      providerProfileId: "deepseek",
      model: "claude-sonnet-4-20250514",
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
    expect(claudeProjection).not.toContain("provider: deepseek");
  });
});
