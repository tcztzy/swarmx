import { Context } from "@deepseek-ai/cordis";
import { CODEX_MODULE_COMMAND, codexAcpLauncher } from "@swarmx/codex";
import { describe, expect, it, vi } from "vitest";
import { AcpLauncherService, coreRuntimePlugin, createCoreRuntime } from "../src/core-runtime.js";
import * as publicCore from "../src/index.js";
import { ProviderRuntimeEnvSchema } from "../src/providers.js";

const { testHome } = vi.hoisted(() => ({
  testHome: `${process.env.TMPDIR ?? "/tmp"}/swarmx-core-runtime-${process.pid}-${Date.now()}`,
}));
vi.mock("node:os", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:os")>();
  return { ...actual, homedir: () => testHome };
});

describe("Core Cordis runtime", () => {
  it("composes Codex as a launcher plugin and leaves Custom Harness commands untouched", async () => {
    const runtime = await createCoreRuntime({
      codex: {
        nodeExecutable: "/runtime/node",
        envPath: "",
        resolveModule: () =>
          "file:///Applications/SwarmX.app/Contents/Resources/app.asar/node_modules/@swarmx/codex/bin/swarmx-codex.js",
      },
    });
    try {
      expect(Context.is(runtime.context)).toBe(true);
      expect(runtime.resolveAcpLaunch({ command: CODEX_MODULE_COMMAND, args: [] })).toEqual({
        command: "/runtime/node",
        args: [
          "/Applications/SwarmX.app/Contents/Resources/app.asar.unpacked/node_modules/@swarmx/codex/bin/swarmx-codex.js",
        ],
        env: {},
      });
      expect(runtime.resolveAcpLaunch({ command: "custom-acp", args: ["serve"] })).toEqual({
        command: "custom-acp",
        args: ["serve"],
        env: {},
      });
      expect(
        runtime.context.harnessTransports.createClient({
          command: CODEX_MODULE_COMMAND,
          args: [],
          transport: "codex_server",
        }),
      ).toHaveProperty("prompt");
      expect(runtime.context.acpRuntime.createClient().prompt).toBeTypeOf("function");
    } finally {
      await runtime.dispose();
    }
  });

  it("runs packaged Electron as Node and rejects arguments on the managed token", async () => {
    const runtime = await createCoreRuntime({
      codex: {
        nodeExecutable: "/Applications/SwarmX.app/Contents/MacOS/SwarmX",
        electron: true,
        envPath: "",
        resolveModule: () => "file:///module/swarmx-codex.js",
      },
    });
    try {
      expect(runtime.resolveAcpLaunch({ command: CODEX_MODULE_COMMAND, args: [] })).toEqual({
        command: "/Applications/SwarmX.app/Contents/MacOS/SwarmX",
        args: ["/module/swarmx-codex.js"],
        env: { ELECTRON_RUN_AS_NODE: "1" },
      });
      expect(() =>
        runtime.resolveAcpLaunch({ command: CODEX_MODULE_COMMAND, args: ["unexpected"] }),
      ).toThrow(/does not accept arguments/);
    } finally {
      await runtime.dispose();
    }
  });

  it("binds launcher registration to the provider Fiber and rejects duplicate ownership", async () => {
    const context = new Context();
    await context.plugin(AcpLauncherService);
    const codexFiber = await context.plugin(codexAcpLauncher, {
      nodeExecutable: "/runtime/node",
      envPath: "",
      resolveModule: () => "file:///module/swarmx-codex.js",
    });
    try {
      await expect(context.plugin(codexAcpLauncher, {})).rejects.toThrow(/already registered/);
      await codexFiber.dispose();
      expect(context.acpLaunchers.resolve({ command: CODEX_MODULE_COMMAND, args: [] })).toEqual({
        command: CODEX_MODULE_COMMAND,
        args: [],
        env: {},
      });
    } finally {
      await context.fiber.dispose();
    }
  });

  it("fails closed after Core disposal", async () => {
    const runtime = await createCoreRuntime();
    await runtime.dispose();
    await runtime.dispose();
    expect(() => runtime.resolveAcpLaunch({ command: "custom-acp", args: [] })).toThrow(/disposed/);
  });

  it("owns Swarm execution in request Fibers without leaking them into the root", async () => {
    const runtime = await createCoreRuntime();
    const registrySize = runtime.context.registry.size;
    const swarm = runtime.prepareSwarm({
      name: "cordis_echo",
      root: "echo",
      nodes: {
        echo: { kind: "agent", agent: { name: "echo", backend: { type: "echo" } } },
      },
      edges: [],
    });
    try {
      await expect(
        swarm.execute({ messages: [{ role: "user", content: "owned by Cordis" }] }),
      ).resolves.toMatchObject([{ role: "assistant", content: "owned by Cordis" }]);
      const invalid = runtime.prepareSwarm({
        name: "invalid",
        root: "missing",
        nodes: {},
        edges: [],
      });
      await expect(invalid.execute({ messages: [] })).rejects.toThrow(/Root node/);
      expect(() =>
        runtime.prepareSwarm({
          name: "unknown_strategy",
          root: "missing",
          nodes: {},
          edges: [],
          strategy: "missing-strategy",
        }),
      ).toThrow(/Unknown swarm strategy/);
      await expect(runtime.listGroupedSessions({ harnessIds: [] })).resolves.toMatchObject({
        errors: [],
      });
      expect(runtime.context.registry.size).toBe(registrySize);
    } finally {
      await runtime.dispose();
    }
  });

  it("loads Provider, Harness, and Swarm modules as built-in DSH plugins", async () => {
    const runtime = await createCoreRuntime();
    try {
      expect(runtime.listSwarmStrategies()).toContain("dag");
      expect(runtime.getHarnessConnector("codex")?.config.backend).toMatchObject({
        type: "custom",
        program: CODEX_MODULE_COMMAND,
        transport: "codex_server",
      });
      expect(
        runtime.resolveProviderRuntimeEnv(
          {
            id: "builtin-provider",
            displayName: "Built-in Provider",
            kind: "openai_chat",
            baseUrl: "https://example.invalid/v1",
            apiEntrypoints: {},
          },
          { modelId: "model-a" },
        ).env,
      ).toMatchObject({ OPENAI_MODEL: "model-a", OPENAI_BASE_URL: "https://example.invalid/v1" });
      expect(runtime.resolveHarnessModelRuntimeEnv("codex", { modelId: "model-b" })).toEqual({
        CODEX_CONFIG: JSON.stringify({ model: "model-b" }),
      });
    } finally {
      await runtime.dispose();
    }
  });

  it("honors DSH plugins loaded after the built-in plugins", async () => {
    const transportClient = {
      async prompt() {
        return { messages: [] };
      },
      stderrOutput: () => "",
    };
    const providerEnv = ProviderRuntimeEnvSchema.parse({
      profileId: "plugin-provider",
      kind: "openai_chat",
      apiMode: "standard",
      authMode: "api_key",
      modelId: "plugin-model",
      runtimeModel: "plugin-model",
      apiEntrypoints: {},
      apiCompatibility: {},
      env: { OPENAI_MODEL: "plugin-model", SWARMX_PLUGIN_PROVIDER: "1" },
      requiresSecret: false,
      secretInjected: false,
    });
    const runtime = await createCoreRuntime({
      plugins: [
        {
          name: "test-harness-transport",
          inject: ["harnessTransports"],
          apply(ctx) {
            ctx.harnessTransports.register("test-transport", () => transportClient);
          },
        },
        {
          name: "test-provider-connector",
          inject: ["providerConnectors"],
          apply(ctx) {
            ctx.providerConnectors.register({
              id: "test-provider",
              kinds: ["openai_chat"],
              priority: 10,
              buildRuntimeEnv: () => providerEnv,
            });
          },
        },
        {
          name: "test-harness-connector",
          inject: ["harnessConnectors"],
          apply(ctx) {
            ctx.harnessConnectors.register({
              id: "plugin-harness",
              config: {
                label: "Plugin Harness",
                icon: "plugin",
                software: { name: "plugin-harness" },
                modelControl: "direct",
                modelCompatibility: "declared_apis",
                supportedModelApis: ["openai_chat"],
                passthroughEnv: [],
                backend: { type: "echo" },
              },
              resolveModelRuntimeEnv: () => ({ PLUGIN_HARNESS: "1" }),
            });
            ctx.harnessConnectors.register({
              id: "plugin-disabled",
              config: {
                label: "Plugin Disabled",
                icon: "plugin",
                software: { name: "plugin-disabled" },
                modelControl: "direct",
                modelCompatibility: "declared_apis",
                supportedModelApis: ["openai_chat"],
                passthroughEnv: [],
                backend: { type: "echo" },
                enabled: false,
              },
            });
          },
        },
        {
          name: "test-harness-permission",
          inject: ["harnessPermissions"],
          apply(ctx) {
            ctx.harnessPermissions.register(
              "test-harness-permission",
              async () => ({ outcome: "cancelled" as const }),
              10,
            );
          },
        },
        {
          name: "test-task-guidance",
          inject: ["taskGuidance"],
          apply(ctx) {
            ctx.taskGuidance.register("test-task-guidance", {
              records: [
                {
                  id: "test-plugin-harness-terminal-work",
                  target: { kind: "harness", harnessId: "plugin-harness" },
                  taskFamilies: ["terminal_work"],
                  verdict: "suitable",
                  confidence: "medium",
                  summary: "Plugin-provided task guidance.",
                  reviewedAt: "2026-08-12",
                  conditions: { benchmarkConfiguration: "plugin smoke" },
                  evidence: [
                    {
                      sourceId: "terminal-bench-2.1",
                      scope: "agent_model",
                      benchmarkTask: "plugin-smoke",
                      metric: "pass",
                      value: 1,
                      unit: "percent",
                      evaluatedModel: "gpt-5.6-sol",
                      evaluatedHarness: "plugin-harness",
                    },
                  ],
                  limitations: ["Plugin-only guidance."],
                },
              ],
            });
          },
        },
        {
          name: "test-swarm-strategy",
          inject: ["swarmStrategies"],
          apply(ctx) {
            ctx.swarmStrategies.register("test-strategy", {
              id: "test-strategy",
              prepare(config) {
                return {
                  name: config.name,
                  root: config.root,
                  models: [],
                  async execute() {
                    return [{ role: "assistant", content: "test-strategy", kind: "message" }];
                  },
                  async executeForEval() {
                    return {
                      output: "test-strategy",
                      messages: [],
                      trace: [],
                      error: null,
                      metrics: { steps: 0, messages: 0, toolCalls: 0, toolResults: 0 },
                    };
                  },
                  async listAllSessions() {
                    return [];
                  },
                };
              },
            });
          },
        },
      ],
    });
    try {
      expect(
        runtime.context.harnessTransports.createClient({
          command: "unused",
          args: [],
          transport: "test-transport",
        }),
      ).toBe(transportClient);
      expect(
        runtime.resolveProviderRuntimeEnv(
          {
            id: "ignored",
            displayName: "Ignored",
            kind: "openai_chat",
            apiEntrypoints: {},
          },
          { modelId: "model-c" },
        ),
      ).toBe(providerEnv);
      expect(runtime.listSwarmStrategies()).toEqual(["dag", "test-strategy"]);
      expect(runtime.context.harnessPermissions.resolve()).toBeTypeOf("function");
      expect(
        runtime.getTaskGuidanceForHarness("plugin-harness").map((record) => record.id),
      ).toContain("test-plugin-harness-terminal-work");
      expect(runtime.harnessCatalog.getHarness("plugin-harness")?.label).toBe("Plugin Harness");
      expect(runtime.harnessCatalog.getHarness("plugin-disabled")).toBeUndefined();
      expect(runtime.harnessCatalog.listHarnesses().map((entry) => entry.id)).toContain(
        "plugin-harness",
      );
      expect(
        runtime.harnessCatalog.resolveModelRuntimeEnv("plugin-harness", {
          modelId: "plugin-model",
        }),
      ).toEqual({ PLUGIN_HARNESS: "1" });
      const strategy = runtime.prepareSwarm({
        name: "plugin_strategy",
        root: "unused",
        nodes: {},
        edges: [],
        strategy: "test-strategy",
      });
      await expect(strategy.execute({ messages: [] })).resolves.toEqual([
        { role: "assistant", content: "test-strategy", kind: "message" },
      ]);
    } finally {
      await runtime.dispose();
    }
  });

  it("installs every Core service into an existing DSH Context", async () => {
    const context = new Context();
    const fiber = await context.plugin(coreRuntimePlugin, {
      codex: {
        nodeExecutable: "/runtime/node",
        envPath: "",
        resolveModule: () => "file:///module/swarmx-codex.js",
      },
    });
    try {
      expect(fiber).toBeTruthy();
      expect(context.swarmRuntime).toBeTypeOf("object");
      expect(context.providerConnectors.list()).toContainEqual(publicCore.builtinProviderConnector);
      expect(context.harnessConnectors.get("codex")).toBeTruthy();
      expect(context.swarmStrategies.listIds()).toContain("dag");
      expect(context.harnessTransports).toBeTypeOf("object");
      expect(context.harnessPermissions.resolve()).toBeTypeOf("function");
      expect(context.taskGuidance).toBeTypeOf("object");
      const swarm = context.swarmRuntime.prepare({
        name: "embedded_dsh_host",
        root: "echo",
        nodes: {
          echo: { kind: "agent", agent: { name: "echo", backend: { type: "echo" } } },
        },
        edges: [],
      });
      await expect(
        swarm.execute({ messages: [{ role: "user", content: "embedded host" }] }),
      ).resolves.toMatchObject([{ role: "assistant", content: "embedded host" }]);
    } finally {
      await context.fiber.dispose();
    }
  });

  it("retires the old runtime constructors from the package root", () => {
    expect(publicCore).not.toHaveProperty("Swarm");
    expect(publicCore).not.toHaveProperty("SwarmNode");
    expect(publicCore).not.toHaveProperty("Agent");
    expect(publicCore).not.toHaveProperty("AcpClient");
    expect(publicCore).not.toHaveProperty("McpManager");
    expect(publicCore.createCoreRuntime).toBeTypeOf("function");
  });

  it("prepares Agent and eval execution without exposing their constructors", async () => {
    const runtime = await createCoreRuntime();
    try {
      await expect(
        runtime
          .prepareAgent({ name: "echo", backend: { type: "echo" } })
          .call({ messages: [{ role: "user", content: "agent call" }] }),
      ).resolves.toMatchObject({ messages: [{ content: "agent call" }] });

      const swarm = runtime.prepareSwarm({
        name: "eval_echo",
        root: "echo",
        queen: { name: "queen", backend: { type: "echo" } },
        nodes: {
          echo: { kind: "agent", agent: { name: "echo", backend: { type: "echo" } } },
        },
        edges: [],
      });
      expect(swarm.models).toEqual([
        { id: "echo", object: "model" },
        { id: "queen", object: "model" },
      ]);
      await expect(
        swarm.executeForEval({ messages: [{ role: "user", content: "eval call" }] }),
      ).resolves.toMatchObject({ output: "eval call", error: null });
      await expect(
        runtime.loadDiscoveredSession({
          id: "missing-local-session",
          title: "Missing",
          cwd: process.cwd(),
          harnessId: "swarmx",
          harnessLabel: "SwarmX",
          source: "local",
        }),
      ).resolves.toBeNull();
    } finally {
      await runtime.dispose();
    }
  });

  it("makes every execution entry fail closed after host disposal", async () => {
    const runtime = await createCoreRuntime();
    await runtime.dispose();
    expect(() =>
      runtime.prepareSwarm({ name: "closed", root: "missing", nodes: {}, edges: [] }),
    ).toThrow(/disposed/);
    expect(() => runtime.prepareAgent({ name: "closed", backend: { type: "echo" } })).toThrow(
      /disposed/,
    );
    expect(() => runtime.listGroupedSessions({ harnessIds: [] })).toThrow(/disposed/);
    expect(() =>
      runtime.loadDiscoveredSession({
        id: "closed",
        title: "Closed",
        cwd: process.cwd(),
        harnessId: "swarmx",
        harnessLabel: "SwarmX",
        source: "local",
      }),
    ).toThrow(/disposed/);
  });
});
