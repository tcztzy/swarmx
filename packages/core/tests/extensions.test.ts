import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, expectTypeOf, it } from "vitest";
import {
  SWARMX_LOCAL_FILES_LSP_ID,
  SWARMX_SKILLS_LSP_ID,
  builtInExtensionBundle,
  createExtensionInventory,
  executeAgentComposition,
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
  it("V250 preserves typed Provider runtime readiness metadata", () => {
    const inventory = createExtensionInventory([
      parseExtensionBundle({
        id: "typed-provider-readiness",
        name: "Typed Provider readiness",
        version: "1.0.0",
        capabilities: {
          providers: [
            {
              id: "typed-provider",
              label: "Typed Provider",
              kind: "anthropic",
              runtimeReady: false,
              runtimeNote: "Credential is not configured.",
            },
          ],
        },
      }),
    ]);
    const provider = inventory.providers[0];

    expectTypeOf(provider?.runtimeReady).toEqualTypeOf<boolean | undefined>();
    expectTypeOf(provider?.runtimeNote).toEqualTypeOf<string | undefined>();
    expect(provider).toMatchObject({
      runtimeReady: false,
      runtimeNote: "Credential is not configured.",
    });
  });

  it("normalizes codex_responses mode in extension Provider profiles", () => {
    const bundle = parseExtensionBundle({
      id: "codex-responses-provider",
      name: "Codex Responses Provider",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "codex-subscription",
            label: "Codex subscription",
            kind: "openai_responses",
            api_mode: "codex_responses",
          },
        ],
      },
    });

    expect(bundle.capabilities.providers[0]?.apiMode).toBe("codex_responses");
  });

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
              modelControl: "session",
              modelCompatibility: "any",
              supportedModelApis: ["openai_responses"],
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
              modelId: "gpt-5",
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

  it("accepts LSP command string, args, and language id metadata", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "geepilot",
      name: "GEEPilot",
      version: "0.1.0",
      capabilities: {
        lspServers: [
          {
            id: "reference-lsp",
            name: "Reference LSP",
            command: "reference-lsp",
            args: ["--stdio"],
            languages: ["markdown"],
            languageIds: ["markdown", "plaintext"],
            mentionPrefixes: ["@"],
            cwd: ".",
            readOnly: true,
          },
        ],
      },
    });

    expect(bundle.capabilities.lspServers).toEqual([
      expect.objectContaining({
        id: "reference-lsp",
        command: "reference-lsp",
        args: ["--stdio"],
        languages: ["markdown"],
        languageIds: ["markdown", "plaintext"],
        mentionPrefixes: ["@"],
        cwd: ".",
        readOnly: true,
      }),
    ]);
  });

  it("declares the built-in local file LSP capability as passive metadata", () => {
    const bundle = builtInExtensionBundle();

    expect(bundle.capabilities.lspServers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: SWARMX_LOCAL_FILES_LSP_ID,
          name: "SwarmX Local Files",
          languageIds: ["markdown", "plaintext"],
          mentionPrefixes: ["@"],
          scope: "project",
          readOnly: true,
        }),
      ]),
    );
    expect(
      bundle.capabilities.lspServers.find((server) => server.id === SWARMX_LOCAL_FILES_LSP_ID)
        ?.command,
    ).toBeUndefined();
  });

  it("declares the built-in skill LSP capability as passive metadata", () => {
    const bundle = builtInExtensionBundle();

    expect(bundle.capabilities.lspServers).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: SWARMX_SKILLS_LSP_ID,
          name: "SwarmX Skills",
          languageIds: ["markdown", "plaintext"],
          mentionPrefixes: ["$"],
          scope: "project",
          readOnly: true,
        }),
      ]),
    );
    expect(
      bundle.capabilities.lspServers.find((server) => server.id === SWARMX_SKILLS_LSP_ID)?.command,
    ).toBeUndefined();
  });

  it("declares built-in LSP capabilities without commands", () => {
    const bundle = builtInExtensionBundle();

    expect(bundle.capabilities.lspServers.map((server) => server.id)).toEqual([
      SWARMX_LOCAL_FILES_LSP_ID,
      SWARMX_SKILLS_LSP_ID,
    ]);
    expect(bundle.capabilities.lspServers.every((server) => server.command === undefined)).toBe(
      true,
    );
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
        models: [
          {
            id: "gpt-5",
            label: "GPT-5",
            runtimeModel: "gpt-5",
            apiProtocols: ["openai_responses"],
          },
        ],
        providers: [
          {
            id: "openai-prod",
            label: "OpenAI Prod",
            kind: "openai_responses",
            secretRef: { source: "env", key: "OPENAI_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "gpt-5-openai-prod",
            modelId: "gpt-5",
            providerProfileId: "openai-prod",
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
            modelControl: "session",
            modelCompatibility: "any",
            supportedModelApis: ["openai_responses"],
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
            modelId: "gpt-5",
            modelSupplyId: "gpt-5-openai-prod",
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
      agentId: "codex-acp:gpt-5",
      agentProfileId: "analysis-lead",
      displayName: "analysis lead",
      canonicalSelector: "@analysis-lead",
      host: "local",
      status: "ready",
      healthStatus: "ready",
      harnessId: "codex-acp",
      harnessLabel: "Codex ACP",
      modelId: "gpt-5",
      runtimeModel: "gpt-5",
      modelSupplyId: "gpt-5-openai-prod",
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
      name: "codex_acp_gpt_5",
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
          modelId: "gpt-5",
          modelSupplyId: "gpt-5-openai-prod",
          host: "local",
          skills: ["memory", "paper-review"],
          mcpServers: ["project-fs"],
          profile: {
            tools: ["Read", "Grep"],
            disallowedTools: ["Bash"],
            permissionMode: "plan",
            maxTurns: 6,
            memory: "readonly",
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
      { id: "broken", harnessId: "missing", modelId: "gpt-5" },
      inventory,
    );
    expect(missingHarnessPlan).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({ kind: "harness", status: "missing", id: "missing" }),
        expect.objectContaining({ kind: "model", status: "missing", id: "gpt-5" }),
      ]),
    });
    expect(() =>
      resolveAgentComposition({ id: "broken", harnessId: "missing", modelId: "gpt-5" }, inventory),
    ).toThrow(/Unknown harness id "missing"/);

    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "minimal",
      name: "Minimal",
      version: "1.0.0",
      capabilities: {
        harnesses: [
          {
            id: "echo",
            label: "Echo",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_chat"],
            backend: { type: "echo" },
          },
        ],
      },
    });

    expect(() =>
      resolveAgentComposition(
        { id: "missing-model", harnessId: "echo" },
        createExtensionInventory([bundle]),
      ),
    ).toThrow(/must resolve one Model/);
  });

  it("V332 requires an explicit Model route for a session-controlled Harness", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "gateway-harness",
      name: "Gateway Harness",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "gateway-model",
            runtimeModel: "gateway-model",
            apiProtocols: ["openai_chat"],
          },
        ],
        providers: [
          {
            id: "gateway-provider",
            label: "Gateway Provider",
            kind: "openai_chat",
          },
        ],
        modelSupplies: [
          {
            id: "gateway-route",
            modelId: "gateway-model",
            providerProfileId: "gateway-provider",
            runtimeModel: "gateway/runtime-model",
            harnessIds: ["gateway"],
          },
        ],
        harnesses: [
          {
            id: "gateway",
            label: "Gateway",
            modelControl: "session",
            modelCompatibility: "any",
            requiresExplicitModelRoute: true,
            supportedModelApis: [],
            backend: { type: "custom", program: "gateway", args: ["acp"] },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    expect(
      resolveAgentCompositionPlan(
        { id: "gateway-run", harnessId: "gateway", modelId: "gateway-model" },
        inventory,
      ),
    ).toMatchObject({
      status: "ready",
      modelId: "gateway-model",
      runtimeModel: "gateway/runtime-model",
      requirements: expect.arrayContaining([
        expect.objectContaining({ kind: "model", status: "ok", id: "gateway-model" }),
      ]),
    });
    expect(
      resolveAgentComposition(
        { id: "gateway-run", harnessId: "gateway", modelId: "gateway-model" },
        inventory,
      ),
    ).toMatchObject({
      model: "gateway/runtime-model",
      backend: { type: "custom", program: "gateway", args: ["acp"] },
    });
    expect(
      resolveAgentCompositionPlan(
        { id: "bad-model-run", harnessId: "gateway", modelId: "invented-model" },
        inventory,
      ),
    ).toMatchObject({
      status: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({ kind: "model", status: "missing", id: "invented-model" }),
      ]),
    });
  });

  it("resolves Claude Code x DeepSeek through the fixed internal route", () => {
    const inventory = createExtensionInventory([builtInExtensionBundle()]);
    const composition = {
      id: "claude-deepseek",
      harnessId: "claude_code",
      modelId: "deepseek-v4-pro",
      effort: "max",
    };

    expect(resolveAgentCompositionPlan(composition, inventory)).toMatchObject({
      status: "ready",
      agentId: "claude_code:deepseek-v4-pro",
      modelId: "deepseek-v4-pro",
      runtimeModel: "deepseek-v4-pro[1m]",
      effort: "max",
    });
    expect(resolveAgentComposition(composition, inventory)).toMatchObject({
      name: "claude_code_deepseek_v4_pro",
      model: "deepseek-v4-pro[1m]",
    });
    expect(
      resolveAgentCompositionRuntimeEnv(composition, inventory, {
        env: { DEEPSEEK_API_KEY: "sk-deepseek-runtime" },
      }),
    ).toEqual({
      ANTHROPIC_BASE_URL: "https://api.deepseek.com/anthropic",
      ANTHROPIC_AUTH_TOKEN: "sk-deepseek-runtime",
      ANTHROPIC_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_OPUS_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_SONNET_MODEL: "deepseek-v4-pro[1m]",
      ANTHROPIC_DEFAULT_HAIKU_MODEL: "deepseek-v4-flash",
      CLAUDE_CODE_SUBAGENT_MODEL: "deepseek-v4-flash",
      CLAUDE_CODE_EFFORT_LEVEL: "max",
      CLAUDE_MODEL_CONFIG: '{"availableModels":["deepseek-v4-pro[1m]"]}',
    });
    expect(() => resolveAgentCompositionRuntimeEnv(composition, inventory, { env: {} })).toThrow(
      /requires env secret "DEEPSEEK_API_KEY"/,
    );
  });

  it("V333 keeps custom Harness identity while reusing its Software adapter", () => {
    const customHarness = parseExtensionBundle({
      id: "custom-claude-harness",
      name: "Custom Claude Harness",
      version: "1.0.0",
      capabilities: {
        harnesses: [
          {
            id: "researcher-harness",
            runtimeHarnessId: "claude_code",
            label: "Researcher Harness",
            modelControl: "session",
            modelCompatibility: "any",
            requiresExplicitModelRoute: true,
            supportedModelApis: [],
            backend: { type: "custom", program: "npx", args: ["claude-acp"] },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), customHarness]);
    const composition = {
      id: "custom-claude-deepseek",
      harnessId: "researcher-harness",
      modelId: "deepseek-v4-pro",
      effort: "max",
    };

    expect(resolveAgentCompositionPlan(composition, inventory)).toMatchObject({
      status: "ready",
      agentId: "researcher-harness:deepseek-v4-pro",
      runtimeModel: "deepseek-v4-pro[1m]",
    });
    expect(resolveAgentComposition(composition, inventory)).toMatchObject({
      name: "researcher_harness_deepseek_v4_pro",
      model: "deepseek-v4-pro[1m]",
    });
    expect(
      resolveAgentCompositionRuntimeEnv(composition, inventory, {
        env: { DEEPSEEK_API_KEY: "sk-deepseek-runtime" },
      }),
    ).toMatchObject({
      ANTHROPIC_MODEL: "deepseek-v4-pro[1m]",
      CLAUDE_MODEL_CONFIG: '{"availableModels":["deepseek-v4-pro[1m]"]}',
    });
  });

  it("chooses a ready internal ModelSupply without changing Harness x Model identity", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "dynamic-catalog",
      name: "Dynamic Catalog",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "remote-model",
            runtimeModel: "remote-model",
            apiProtocols: ["openai_chat"],
          },
        ],
        providers: [
          {
            id: "provider-unavailable",
            label: "Unavailable",
            kind: "openai_chat",
            secretRef: { source: "env", key: "MISSING_REMOTE_KEY" },
            runtimeReady: false,
          },
          {
            id: "provider-ready",
            label: "Ready",
            kind: "openai_chat",
            secretRef: { source: "env", key: "READY_REMOTE_KEY" },
            runtimeReady: true,
          },
        ],
        modelSupplies: [
          {
            id: "remote-via-unavailable",
            modelId: "remote-model",
            providerProfileId: "provider-unavailable",
            runtimeModel: "unavailable/remote-model",
          },
          {
            id: "remote-via-ready",
            modelId: "remote-model",
            providerProfileId: "provider-ready",
            runtimeModel: "ready/remote-model",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);
    const composition = {
      id: "automatic-supply",
      harnessId: "swarmx",
      modelId: "remote-model",
    };

    expect(resolveAgentCompositionPlan(composition, inventory)).toMatchObject({
      status: "ready",
      agentId: "swarmx:remote-model",
      modelId: "remote-model",
      modelSupplyId: "remote-via-ready",
      runtimeModel: "ready/remote-model",
    });
    expect(resolveAgentComposition(composition, inventory)).toMatchObject({
      name: "swarmx_remote_model",
      model: "ready/remote-model",
      parameters: {
        extension: {
          harnessId: "swarmx",
          modelId: "remote-model",
          modelSupplyId: "remote-via-ready",
        },
      },
    });
    expect(
      resolveAgentCompositionRuntimeEnv(composition, inventory, {
        env: { READY_REMOTE_KEY: "sk-ready-runtime" },
      }),
    ).toMatchObject({
      OPENAI_MODEL: "ready/remote-model",
      OPENAI_API_KEY: "sk-ready-runtime",
    });
  });

  it("uses request-scoped Provider secret overrides for local keychain references", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "keychain-provider",
      name: "Keychain Provider",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "claude-private",
            runtimeModel: "claude-private",
            apiProtocols: ["anthropic"],
          },
        ],
        providers: [
          {
            id: "anthropic-private",
            label: "Anthropic Private",
            kind: "anthropic",
            baseUrl: "https://anthropic.internal",
            authMode: "auth_token",
            secretRef: { source: "local_keychain", key: "anthropic-private" },
            runtimeReady: true,
          },
        ],
        modelSupplies: [
          {
            id: "claude-private-route",
            modelId: "claude-private",
            providerProfileId: "anthropic-private",
            runtimeModel: "claude-private",
            harnessIds: ["claude_code"],
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);
    const composition = {
      id: "private-token",
      harnessId: "claude_code",
      modelId: "claude-private",
    };

    expect(() => resolveAgentCompositionRuntimeEnv(composition, inventory, { env: {} })).toThrow(
      /unsupported secret source "local_keychain"/,
    );
    expect(
      resolveAgentCompositionRuntimeEnv(composition, inventory, {
        env: {},
        providerSecrets: { "anthropic-private": "private-auth-token" },
      }),
    ).toMatchObject({
      ANTHROPIC_AUTH_TOKEN: "private-auth-token",
      ANTHROPIC_BASE_URL: "https://anthropic.internal",
      ANTHROPIC_MODEL: "claude-private",
    });
  });

  it("blocks compositions with unknown skills, disabled profiles, or unsupported context", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "strict",
      name: "Strict",
      version: "1.0.0",
      capabilities: {
        models: [{ id: "gpt-5", runtimeModel: "gpt-5", apiProtocols: ["openai_chat"] }],
        harnesses: [
          {
            id: "echo",
            label: "Echo",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_chat"],
            backend: { type: "echo" },
          },
        ],
        skills: [{ id: "known-skill", name: "Known Skill" }],
        agents: [
          {
            id: "disabled-agent",
            name: "disabled agent",
            enabled: false,
            harnessId: "echo",
            modelId: "gpt-5",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    const missingSkillPlan = resolveAgentCompositionPlan(
      { id: "bad-skill", harnessId: "echo", modelId: "gpt-5", skills: ["missing-skill"] },
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
        {
          id: "bad-skill",
          harnessId: "echo",
          modelId: "gpt-5",
          skills: ["missing-skill"],
        },
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
        modelId: "gpt-5",
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
        harnesses: [
          {
            id: "echo",
            label: "Echo",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_chat"],
            backend: { type: "echo" },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    expect(() =>
      resolveAgentCompositionPlan(
        { id: "bad-secret", harnessId: "echo", modelId: "gpt-5", apiKey: "sk-test" },
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

  it("keeps Provider out of compatibility and fails only when a selected supply secret is missing", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "providers",
      name: "Providers",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "claude-sonnet",
            runtimeModel: "claude-sonnet",
            apiProtocols: ["anthropic"],
          },
        ],
        providers: [
          {
            id: "anthropic-prod",
            label: "Anthropic Prod",
            kind: "anthropic",
            secretRef: { source: "env", key: "ANTHROPIC_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "claude-anthropic",
            modelId: "claude-sonnet",
            providerProfileId: "anthropic-prod",
            apiCompatibility: { mode: "native" },
          },
        ],
        harnesses: [
          {
            id: "codex-only",
            label: "Codex Only",
            modelControl: "session",
            modelCompatibility: "any",
            supportedModelApis: ["openai_responses"],
            backend: { type: "custom", program: "codex", args: ["acp"] },
          },
          {
            id: "claude",
            label: "Claude",
            modelControl: "session",
            modelCompatibility: "any",
            supportedModelApis: ["anthropic"],
            backend: { type: "custom", program: "claude", args: ["acp"] },
          },
        ],
        agents: [
          {
            id: "codex-claude",
            name: "codex claude",
            harnessId: "codex-only",
            modelId: "claude-sonnet",
            modelSupplyId: "claude-anthropic",
          },
          {
            id: "missing-secret",
            name: "missing secret",
            harnessId: "claude",
            modelId: "claude-sonnet",
            modelSupplyId: "claude-anthropic",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    expect(
      resolveAgentCompositionPlan({ id: "codex", agentProfileId: "codex-claude" }, inventory),
    ).toMatchObject({ status: "ready", agentId: "codex-only:claude-sonnet" });
    expect(() =>
      resolveAgentCompositionRuntimeEnv(
        { id: "missing-secret", agentProfileId: "missing-secret" },
        inventory,
        { env: {} },
      ),
    ).toThrow(/requires env secret "ANTHROPIC_API_KEY"/);
  });

  it("resolves yallm bridge routing from ModelSupply metadata", () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "bridged-providers",
      name: "Bridged Providers",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "claude-sonnet",
            runtimeModel: "claude-sonnet",
            apiProtocols: ["anthropic", "openai_responses"],
          },
        ],
        providers: [
          {
            id: "anthropic-prod",
            label: "Anthropic Prod",
            kind: "anthropic",
            baseUrl: "https://api.anthropic.com",
            secretRef: { source: "env", key: "ANTHROPIC_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "claude-via-yallm",
            modelId: "claude-sonnet",
            providerProfileId: "anthropic-prod",
            apiCompatibility: { mode: "bridge", targetApi: "openai_responses" },
          },
        ],
        harnesses: [
          {
            id: "codex-only",
            label: "Codex Only",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_responses"],
            backend: { type: "custom", program: "codex", args: ["acp"] },
          },
        ],
        agents: [
          {
            id: "codex-with-anthropic",
            name: "codex with anthropic",
            harnessId: "codex-only",
            modelId: "claude-sonnet",
            modelSupplyId: "claude-via-yallm",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    const plan = resolveAgentCompositionPlan(
      { id: "bridged", agentProfileId: "codex-with-anthropic" },
      inventory,
    );
    expect(plan.status).toBe("ready");

    expect(
      resolveAgentCompositionRuntimeEnv(
        { id: "bridged", agentProfileId: "codex-with-anthropic" },
        inventory,
        { env: { ANTHROPIC_API_KEY: "sk-ant-runtime" } },
      ),
    ).toMatchObject({
      YALLM_DEFAULT_PROVIDER: "anthropic",
      ANTHROPIC_API_KEY: "sk-ant-runtime",
      ANTHROPIC_BASE_URL: "https://api.anthropic.com",
      OPENAI_API_KEY: "sk-swarmx-bridge",
      OPENAI_BASE_URL: "http://127.0.0.1:4000/v1",
      OPENAI_MODEL: "anthropic:claude-sonnet",
    });
  });

  it("executes an agent composition through a single-agent Swarm without leaking runtime secrets", async () => {
    const bundle = parseExtensionBundle({
      schemaVersion: 1,
      id: "runtime",
      name: "Runtime",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "gpt-5",
            runtimeModel: "gpt-5",
            apiProtocols: ["openai_responses"],
          },
        ],
        providers: [
          {
            id: "openai-prod",
            label: "OpenAI Prod",
            kind: "openai_responses",
            secretRef: { source: "env", key: "OPENAI_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "gpt-5-openai",
            modelId: "gpt-5",
            providerProfileId: "openai-prod",
          },
        ],
        harnesses: [
          {
            id: "echo",
            label: "Echo",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_responses"],
            backend: { type: "echo" },
          },
        ],
        agents: [
          {
            id: "analysis-lead",
            name: "analysis lead",
            instructions: "Plan analysis work.",
            harnessId: "echo",
            modelId: "gpt-5",
            modelSupplyId: "gpt-5-openai",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([bundle]);

    const streamed: Array<{ kind: string; content: string }> = [];
    const messages = await executeAgentComposition(
      { id: "run-analysis", agentProfileId: "analysis-lead" },
      [{ role: "user", content: "Review dataset evidence." }],
      {
        inventory,
        env: { OPENAI_API_KEY: "sk-runtime" },
        onChunk: (chunk) => streamed.push(chunk),
      },
    );

    expect(messages).toEqual([
      {
        role: "assistant",
        content: "Review dataset evidence.",
        kind: "message",
        agent: "echo_gpt_5",
      },
    ]);
    expect(streamed).toEqual(messages);
    expect(JSON.stringify(messages)).not.toContain("sk-runtime");

    await expect(
      executeAgentComposition(
        { id: "bad-run", harnessId: "missing", modelId: "gpt-5" },
        [{ role: "user", content: "hello" }],
        { inventory },
      ),
    ).rejects.toThrow(/Agent composition "bad-run" is blocked/);
  });
});
