import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createExtensionInventory,
  loadExtensionInventory,
  parseAgentCompositionPlan,
  parseExtensionBundle,
  resolveAgentComposition,
  resolveAgentCompositionPlan,
  resolveAgentCompositionRuntimeEnv,
  validateSkillHostCompatibility,
} from "../src/extensions.js";

const tempRoots: string[] = [];

afterEach(async () => {
  await Promise.all(tempRoots.map((root) => rm(root, { recursive: true, force: true })));
  tempRoots.length = 0;
});

describe("extension inventory", () => {
  it("loads built-in harnesses and path manifests into one inventory", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-extension-"));
    tempRoots.push(root);
    await writeFile(
      path.join(root, "swarmx.extension.json"),
      JSON.stringify({
        schemaVersion: 1,
        id: "geepilot",
        name: "GEEPilot",
        version: "0.1.0",
        capabilities: {
          skills: [
            {
              id: "geepilot.biosecurity",
              name: "Biosecurity",
              path: "skills/biosecurity/SKILL.md",
              canonicalPath: "skills/biosecurity/SKILL.md",
              governanceRef: "docs/skills-governance.md",
              hostExposures: [
                {
                  host: "codex",
                  status: "plugin",
                  manifestPath: "./.codex-plugin/plugin.json",
                  marketplaceSourceId: "geepilot-codex-local",
                },
                { host: "opencode", status: "rules_only", rulesPath: "./AGENTS.md" },
              ],
              readOnly: true,
            },
            {
              id: "geepilot.memory",
              name: "Memory",
              path: "skills/memory/SKILL.md",
              canonicalPath: "skills/memory/SKILL.md",
              governanceRef: "docs/skills-governance.md",
              hostExposures: [
                {
                  host: "codex",
                  status: "plugin",
                  manifestPath: "./.codex-plugin/plugin.json",
                  marketplaceSourceId: "geepilot-codex-local",
                },
                {
                  host: "claude",
                  status: "plugin",
                  manifestPath: "./.claude-plugin/plugin.json",
                },
              ],
            },
          ],
          mcpServers: [
            {
              id: "geepilot.filesystem",
              name: "Filesystem",
              server: { type: "stdio", command: "npx", args: ["-y", "server"] },
            },
          ],
          harnesses: [
            {
              id: "geepilot-codex",
              label: "GEEPilot Codex",
              compatibleProviders: ["openai_responses"],
              backend: {
                type: "custom",
                program: "bun",
                args: ["x", "--silent", "@agentclientprotocol/codex-acp"],
              },
              software: { name: "codex-acp", version: "0.22.0" },
              skills: ["geepilot.memory"],
              mcps: ["geepilot.filesystem"],
              projectFiles: ["AGENTS.md"],
            },
          ],
          agents: [
            {
              id: "analysis-lead",
              name: "analysis lead",
              harnessId: "geepilot-codex",
              model: "gpt-5",
              skills: ["geepilot.memory"],
              mcpServers: ["geepilot.filesystem"],
            },
          ],
          uiContributions: [
            {
              id: "geepilot.nav",
              kind: "navigation_item",
              name: "GEEPilot",
              placement: "sidebar",
              route: "/extensions/geepilot",
              componentRef: "geepilot.ui.shell",
              assetRef: "geepilot-icon",
              sourcePluginId: "geepilot",
              readOnly: true,
            },
            {
              id: "geepilot.dashboard",
              kind: "dashboard_widget",
              name: "Analysis dashboard",
              placement: "dashboard",
              componentRef: "geepilot.ui.dashboard",
              permissionIds: ["project-read"],
              authPolicyIds: ["project-fs-auth"],
            },
            {
              id: "geepilot.composer.review",
              kind: "composer_action",
              name: "Review dataset",
              placement: "composer",
              commandId: "geepilot.refresh-index",
              settingIds: ["geepilot.indexRoot"],
            },
          ],
          commands: [
            {
              id: "geepilot.refresh-index",
              name: "Refresh index",
              command: ["geepilot", "index", "refresh"],
              scope: "plugin",
            },
          ],
          lspServers: [
            {
              id: "pyright",
              name: "Pyright",
              languages: ["python"],
              command: ["pyright-langserver", "--stdio"],
            },
          ],
          hooks: [
            {
              id: "before-run",
              event: "before_agent_run",
              command: ["geepilot", "hooks", "before-run"],
            },
          ],
          monitors: [
            {
              id: "catalog-refresh",
              trigger: "schedule",
              schedule: "PT1H",
            },
          ],
          outputStyles: [
            {
              id: "compact-review",
              name: "Compact Review",
              path: "output-styles/compact-review.md",
            },
          ],
          settings: [
            {
              id: "geepilot.indexRoot",
              name: "Index root",
              valueType: "string",
              required: false,
            },
          ],
          assets: [
            {
              id: "geepilot-icon",
              kind: "icon",
              path: "assets/icon.png",
              sha256: "abc123",
            },
          ],
          permissions: [
            {
              id: "project-read",
              kind: "filesystem",
              access: "read",
              target: "workspace",
              required: true,
            },
          ],
          authPolicies: [
            {
              id: "project-fs-auth",
              kind: "env",
              required: false,
              secretRefs: [{ source: "env", key: "GEEPILOT_PROJECT_TOKEN" }],
            },
          ],
          marketplaceSources: [
            {
              id: "geepilot-codex-local",
              name: "GEEPilot Codex marketplace",
              host: "codex",
              kind: "local_path",
              path: "./.agents/plugins/marketplace.json",
              readOnly: true,
            },
          ],
          pluginCatalog: [
            {
              id: "geepilot",
              name: "GEEPilot",
              version: "0.1.0",
              marketplaceSourceId: "geepilot-codex-local",
              bundleId: "geepilot",
              hosts: ["codex", "claude"],
              installState: "installed",
              updateState: "current",
              providesHarness: true,
              componentCounts: {
                commands: 1,
                skills: 2,
                mcpServers: 1,
                lspServers: 1,
                agents: 1,
                hooks: 1,
                monitors: 1,
                outputStyles: 1,
                uiContributions: 3,
                assets: 1,
                settings: 1,
                permissions: 1,
                authPolicies: 1,
              },
            },
          ],
        },
      }),
      "utf8",
    );

    const inventory = await loadExtensionInventory({ roots: [root] });

    expect(inventory.warnings).toEqual([]);
    expect(inventory.bundles.map((bundle) => bundle.id)).toContain("swarmx.builtin");
    expect(inventory.harnesses.map((harness) => harness.id)).toEqual(
      expect.arrayContaining(["swarmx", "codex", "geepilot-codex"]),
    );
    expect(inventory.skills.map((skill) => skill.id)).toEqual(
      expect.arrayContaining(["geepilot.biosecurity", "geepilot.memory"]),
    );
    expect(inventory.skills).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "geepilot.memory",
          canonicalPath: "skills/memory/SKILL.md",
          hostExposures: expect.arrayContaining([
            expect.objectContaining({
              host: "codex",
              status: "plugin",
              manifestPath: "./.codex-plugin/plugin.json",
              marketplaceSourceId: "geepilot-codex-local",
            }),
          ]),
        }),
      ]),
    );
    expect(inventory.agents.map((agent) => agent.id)).toContain("analysis-lead");
    expect(inventory.uiContributions).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "geepilot.nav",
          kind: "navigation_item",
          placement: "sidebar",
          componentRef: "geepilot.ui.shell",
          sourcePluginId: "geepilot",
        }),
        expect.objectContaining({
          id: "geepilot.dashboard",
          kind: "dashboard_widget",
          permissionIds: ["project-read"],
          authPolicyIds: ["project-fs-auth"],
        }),
        expect.objectContaining({
          id: "geepilot.composer.review",
          kind: "composer_action",
          commandId: "geepilot.refresh-index",
          settingIds: ["geepilot.indexRoot"],
        }),
      ]),
    );
    expect(inventory.commands.map((command) => command.id)).toContain("geepilot.refresh-index");
    expect(inventory.lspServers.map((server) => server.id)).toContain("pyright");
    expect(inventory.hooks.map((hook) => hook.id)).toContain("before-run");
    expect(inventory.monitors.map((monitor) => monitor.id)).toContain("catalog-refresh");
    expect(inventory.outputStyles.map((style) => style.id)).toContain("compact-review");
    expect(inventory.settings.map((setting) => setting.id)).toContain("geepilot.indexRoot");
    expect(inventory.assets.map((asset) => asset.id)).toContain("geepilot-icon");
    expect(inventory.permissions).toEqual([
      expect.objectContaining({ id: "project-read", kind: "filesystem" }),
    ]);
    expect(inventory.authPolicies).toEqual([
      expect.objectContaining({
        id: "project-fs-auth",
        secretRefs: [{ source: "env", key: "GEEPILOT_PROJECT_TOKEN" }],
      }),
    ]);
    expect(inventory.marketplaceSources).toEqual([
      expect.objectContaining({
        id: "geepilot-codex-local",
        host: "codex",
        path: "./.agents/plugins/marketplace.json",
      }),
    ]);
    expect(inventory.pluginCatalog).toEqual([
      expect.objectContaining({
        id: "geepilot",
        hosts: ["codex", "claude"],
        installState: "installed",
        providesHarness: true,
        componentCounts: expect.objectContaining({
          commands: 1,
          skills: 2,
          mcpServers: 1,
          lspServers: 1,
          agents: 1,
          hooks: 1,
          monitors: 1,
          outputStyles: 1,
          uiContributions: 3,
          assets: 1,
          settings: 1,
          permissions: 1,
          authPolicies: 1,
        }),
      }),
    ]);
  });

  it("rejects inline secret fields in extension manifests", () => {
    expect(() =>
      parseExtensionBundle({
        schemaVersion: 1,
        id: "bad",
        name: "Bad",
        version: "1.0.0",
        capabilities: {
          providers: [
            {
              id: "openai",
              label: "OpenAI",
              kind: "openai_responses",
              apiKey: "sk-test",
            },
          ],
        },
      }),
    ).toThrow(/inline secret field "apiKey"/);

    expect(() =>
      parseExtensionBundle({
        schemaVersion: 1,
        id: "bad-component",
        name: "Bad Component",
        version: "1.0.0",
        capabilities: {
          authPolicies: [
            {
              id: "bad-auth",
              kind: "api_key",
              apiKey: "sk-test",
            },
          ],
        },
      }),
    ).toThrow(/inline secret field "apiKey"/);

    expect(() =>
      parseExtensionBundle({
        schemaVersion: 1,
        id: "bad-ui",
        name: "Bad UI",
        version: "1.0.0",
        capabilities: {
          uiContributions: [
            {
              id: "bad-ui.inline",
              kind: "view",
              name: "Bad UI",
              placement: "workspace",
              componentRef: "bad.ui",
              html: "<script>alert(1)</script>",
            },
          ],
        },
      }),
    ).toThrow(/inline executable\/render field/);
  });

  it("validates host skill exposure without owning downstream skill logic", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "geepilot",
      name: "GEEPilot",
      version: "0.1.0",
      capabilities: {
        skills: [
          {
            id: "geepilot.biosecurity",
            name: "Biosecurity",
            path: "skills/biosecurity/SKILL.md",
            canonicalPath: "skills/biosecurity/SKILL.md",
            hostExposures: [
              {
                host: "codex",
                status: "plugin",
                manifestPath: "./.codex-plugin/plugin.json",
                marketplaceSourceId: "codex-local",
              },
              {
                host: "opencode",
                status: "rules_only",
                rulesPath: "./AGENTS.md",
              },
            ],
          },
          {
            id: "geepilot.primer3",
            name: "Primer3",
            path: "skills/primer3/SKILL.md",
            requiresGateSkillIds: ["geepilot.biosecurity"],
            hostExposures: [
              {
                host: "claude",
                status: "plugin",
                manifestPath: "./.claude-plugin/plugin.json",
              },
            ],
          },
        ],
        marketplaceSources: [
          {
            id: "codex-local",
            name: "Codex local marketplace",
            host: "codex",
            kind: "local_path",
            path: "./.agents/plugins/marketplace.json",
          },
        ],
      },
    });

    expect(
      validateSkillHostCompatibility(bundle, {
        canonicalRoots: ["skills/"],
        requireDotSlashLocalPathsForHosts: ["codex"],
      }),
    ).toEqual([]);
  });

  it("reports skill host compatibility issues as passive review data", () => {
    const issues = validateSkillHostCompatibility(
      {
        schemaVersion: 1,
        id: "bad-hosting",
        name: "Bad Hosting",
        version: "1.0.0",
        capabilities: {
          skills: [
            {
              id: "wrapper.memory",
              path: ".codex-plugin/skills/memory/SKILL.md",
              requiresGateSkillIds: ["missing.biosecurity"],
              hostExposures: [
                {
                  host: "codex",
                  status: "plugin",
                  manifestPath: ".codex-plugin/plugin.json",
                  marketplaceSourceId: "missing-source",
                },
                {
                  host: "opencode",
                  status: "rules_only",
                  manifestPath: "./opencode/plugin.json",
                },
              ],
            },
            {
              id: "self-gated",
              path: "skills/self/SKILL.md",
              requiresGateSkillIds: ["self-gated"],
            },
          ],
          marketplaceSources: [
            {
              id: "codex-local",
              name: "Codex local marketplace",
              host: "codex",
              kind: "local_path",
              path: ".agents/plugins/marketplace.json",
            },
          ],
        },
      },
      {
        canonicalRoots: ["skills/"],
        requireDotSlashLocalPathsForHosts: ["codex"],
      },
    );

    expect(issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining([
        "skill_path_outside_canonical_roots",
        "unknown_gate_skill",
        "unknown_marketplace_source",
        "host_local_path_must_be_dot_slash",
        "rules_only_manifest_claim",
        "skill_gate_self_reference",
        "marketplace_local_path_must_be_dot_slash",
      ]),
    );
    expect(issues.every((issue) => issue.level === "error")).toBe(true);
  });

  it("resolves an agent composition without copying provider secrets", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "geepilot",
      name: "GEEPilot",
      version: "0.1.0",
      capabilities: {
        providers: [
          {
            id: "openai-prod",
            label: "OpenAI Prod",
            kind: "openai_responses",
            model: "gpt-5",
            secretRef: { source: "env", key: "OPENAI_API_KEY" },
          },
        ],
        skills: [
          { id: "memory", name: "Memory" },
          { id: "paper-review", name: "Paper Review" },
        ],
        mcpServers: [
          {
            id: "project-fs",
            server: { type: "stdio", command: "npx", args: ["-y", "server"] },
          },
        ],
        harnesses: [
          {
            id: "codex-acp",
            label: "Codex ACP",
            compatibleProviders: ["openai_responses"],
            backend: {
              type: "custom",
              program: "bun",
              args: ["x", "--silent", "@agentclientprotocol/codex-acp"],
            },
            software: { name: "codex-acp", version: "0.22.0", runner: "bun" },
          },
        ],
        agents: [
          {
            id: "analysis-lead",
            name: "analysis lead",
            instructions: "Plan analysis work and cite evidence.",
            harnessId: "codex-acp",
            providerProfileId: "openai-prod",
            skills: ["memory"],
            mcpServers: ["project-fs"],
            tools: ["Read", "Grep"],
            disallowedTools: ["Bash"],
            permissionMode: "plan",
            maxTurns: 6,
            memory: "readonly",
            effort: "high",
            background: false,
            isolation: "workspace",
            color: "blue",
            definition: {
              kind: "plugin",
              pluginId: "geepilot",
              path: "agents/analysis-lead.md",
              readOnly: true,
            },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    const plan = resolveAgentCompositionPlan(
      {
        id: "run-analysis",
        agentProfileId: "analysis-lead",
        skills: ["paper-review"],
      },
      inventory,
    );

    expect(plan).toMatchObject({
      id: "run-analysis",
      agentId: "analysis-lead",
      agentProfileId: "analysis-lead",
      displayName: "analysis lead",
      canonicalSelector: "@analysis-lead",
      host: "local",
      status: "ready",
      healthStatus: "ready",
      harnessId: "codex-acp",
      harnessLabel: "Codex ACP",
      providerProfileId: "openai-prod",
      providerKind: "openai_responses",
      model: "gpt-5",
      definition: {
        source: "plugin",
        pluginId: "geepilot",
        path: "agents/analysis-lead.md",
        readOnly: true,
      },
      pluginIds: ["geepilot"],
      skills: [
        { id: "memory", name: "Memory", sourcePluginId: "geepilot", status: "ok" },
        {
          id: "paper-review",
          name: "Paper Review",
          sourcePluginId: "geepilot",
          status: "ok",
        },
      ],
      mcpServers: [{ id: "project-fs", sourcePluginId: "geepilot", status: "ok" }],
      context: { mode: "thread_packet", strategy: "auto", memory: "readonly" },
      permissions: {
        tools: "2 allowed tools",
        mcp: "selected",
        shell: "plan",
        mode: "plan",
      },
      visual: { label: "analysis lead", color: "blue" },
    });
    expect(plan.requirements).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "harness", status: "ok", id: "codex-acp" }),
        expect.objectContaining({ kind: "model", status: "ok", id: "gpt-5" }),
        expect.objectContaining({ kind: "plugin", status: "ok", id: "geepilot" }),
        expect.objectContaining({
          kind: "secret",
          status: "unknown",
          id: "env:OPENAI_API_KEY",
        }),
      ]),
    );
    expect(JSON.stringify(plan)).not.toContain("sk-runtime");

    const agent = resolveAgentComposition(
      {
        id: "run-analysis",
        agentProfileId: "analysis-lead",
        skills: ["paper-review"],
      },
      inventory,
    );

    expect(agent).toMatchObject({
      name: "analysis_lead",
      model: "gpt-5",
      instructions: "Plan analysis work and cite evidence.",
      backend: {
        type: "custom",
        program: "bun",
        args: ["x", "--silent", "@agentclientprotocol/codex-acp"],
      },
      mcpServers: {
        "project-fs": { type: "stdio", command: "npx", args: ["-y", "server"] },
      },
      parameters: {
        extension: {
          compositionId: "run-analysis",
          agentProfileId: "analysis-lead",
          harnessId: "codex-acp",
          providerProfileId: "openai-prod",
          host: "local",
          skills: ["memory", "paper-review"],
          mcpServers: ["project-fs"],
          profile: {
            tools: ["Read", "Grep"],
            disallowedTools: ["Bash"],
            permissionMode: "plan",
            maxTurns: 6,
            memory: "readonly",
            effort: "high",
            background: false,
            isolation: "workspace",
            color: "blue",
            definition: {
              kind: "plugin",
              pluginId: "geepilot",
              path: "agents/analysis-lead.md",
              readOnly: true,
            },
          },
        },
      },
    });
    expect(JSON.stringify(agent)).not.toContain("OPENAI_API_KEY");

    const runtimeEnv = resolveAgentCompositionRuntimeEnv(
      {
        id: "run-analysis",
        agentProfileId: "analysis-lead",
      },
      inventory,
      { env: { OPENAI_API_KEY: "sk-runtime" } },
    );
    expect(runtimeEnv).toEqual({
      OPENAI_API_KEY: "sk-runtime",
      OPENAI_MODEL: "gpt-5",
    });
    expect(JSON.stringify(agent)).not.toContain("sk-runtime");
  });

  it("fails explicit compositions with unknown harnesses or missing models", () => {
    const inventory = createExtensionInventory([]);

    const missingHarnessPlan = resolveAgentCompositionPlan(
      { id: "broken", harnessId: "missing", model: "gpt-5" },
      inventory,
    );
    expect(missingHarnessPlan).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({ kind: "harness", status: "missing", id: "missing" }),
        expect.objectContaining({ kind: "model", status: "ok", id: "gpt-5" }),
      ]),
    });
    expect(() =>
      resolveAgentComposition({ id: "broken", harnessId: "missing", model: "gpt-5" }, inventory),
    ).toThrow(/Unknown harness id "missing"/);

    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "minimal",
      name: "Minimal",
      version: "1.0.0",
      capabilities: {
        harnesses: [{ id: "echo", label: "Echo", backend: { type: "echo" } }],
      },
    });

    expect(() =>
      resolveAgentComposition(
        { id: "missing-model", harnessId: "echo" },
        createExtensionInventory([bundle]),
      ),
    ).toThrow(/must resolve one model/);
  });

  it("blocks compositions with unknown skills, disabled profiles, or unsupported context", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "strict",
      name: "Strict",
      version: "1.0.0",
      capabilities: {
        harnesses: [{ id: "echo", label: "Echo", backend: { type: "echo" } }],
        skills: [{ id: "known-skill", name: "Known Skill" }],
        agents: [
          {
            id: "disabled-agent",
            name: "disabled agent",
            enabled: false,
            harnessId: "echo",
            model: "gpt-5",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    const missingSkillPlan = resolveAgentCompositionPlan(
      { id: "bad-skill", harnessId: "echo", model: "gpt-5", skills: ["missing-skill"] },
      inventory,
    );
    expect(missingSkillPlan).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      skills: [{ id: "missing-skill", status: "missing" }],
      requirements: expect.arrayContaining([
        expect.objectContaining({
          kind: "skill",
          status: "missing",
          id: "missing-skill",
        }),
      ]),
    });
    expect(() =>
      resolveAgentComposition(
        { id: "bad-skill", harnessId: "echo", model: "gpt-5", skills: ["missing-skill"] },
        inventory,
      ),
    ).toThrow(/Unknown skill id "missing-skill"/);

    const disabledPlan = resolveAgentCompositionPlan(
      { id: "disabled-run", agentProfileId: "disabled-agent" },
      inventory,
    );
    expect(disabledPlan.status).toBe("disabled");
    expect(disabledPlan.requirements).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "agent_profile",
          status: "disabled",
          id: "disabled-agent",
        }),
      ]),
    );

    const unsupportedContextPlan = resolveAgentCompositionPlan(
      {
        id: "unsupported-context",
        harnessId: "echo",
        model: "gpt-5",
        context: { mode: "thread_packet", strategy: "raw_full_history" },
      },
      inventory,
    );
    expect(unsupportedContextPlan).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({
          kind: "context",
          status: "unsupported",
          id: "raw_full_history",
        }),
      ]),
    });
  });

  it("rejects inline secrets in composition inputs and plan records", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "minimal",
      name: "Minimal",
      version: "1.0.0",
      capabilities: {
        harnesses: [{ id: "echo", label: "Echo", backend: { type: "echo" } }],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    expect(() =>
      resolveAgentCompositionPlan(
        { id: "bad-secret", harnessId: "echo", model: "gpt-5", apiKey: "sk-test" },
        inventory,
      ),
    ).toThrow(/inline secret field "apiKey"/);

    expect(() =>
      parseAgentCompositionPlan({
        id: "plan",
        agentId: "agent",
        displayName: "Agent",
        canonicalSelector: "@agent",
        host: "local",
        status: "ready",
        healthStatus: "ready",
        definition: { source: "none" },
        apiKey: "sk-test",
      }),
    ).toThrow(/Record must not contain inline secret field/);
  });

  it("fails runtime provider resolution without compatible harness or env secret", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "providers",
      name: "Providers",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "anthropic-prod",
            label: "Anthropic Prod",
            kind: "anthropic",
            model: "claude-sonnet",
            secretRef: { source: "env", key: "ANTHROPIC_API_KEY" },
          },
        ],
        harnesses: [
          {
            id: "codex-only",
            label: "Codex Only",
            compatibleProviders: ["openai_responses"],
            backend: { type: "custom", program: "codex", args: ["acp"] },
          },
          {
            id: "claude",
            label: "Claude",
            compatibleProviders: ["anthropic"],
            backend: { type: "custom", program: "claude", args: ["acp"] },
          },
        ],
        agents: [
          {
            id: "bad-provider",
            name: "bad provider",
            harnessId: "codex-only",
            providerProfileId: "anthropic-prod",
          },
          {
            id: "missing-secret",
            name: "missing secret",
            harnessId: "claude",
            providerProfileId: "anthropic-prod",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    expect(() =>
      resolveAgentComposition({ id: "bad", agentProfileId: "bad-provider" }, inventory),
    ).toThrow(/not compatible/);
    expect(() =>
      resolveAgentCompositionRuntimeEnv(
        { id: "missing-secret", agentProfileId: "missing-secret" },
        inventory,
        { env: {} },
      ),
    ).toThrow(/requires env secret "ANTHROPIC_API_KEY"/);
  });
});
