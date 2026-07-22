import { describe, expect, it, vi } from "vitest";
import { Agent } from "../src/agent.js";
import {
  builtInExtensionBundle,
  createExtensionInventory,
  parseExtensionBundle,
  resolveAgentComposition,
  resolveAgentCompositionPlan,
} from "../src/extensions.js";
import {
  MODELS,
  MODEL_CAPABILITIES,
  ModelCapabilitySchema,
  ModelSchema,
  ModelSupplySchema,
  findModelCapability,
  modelCapabilityRegistry,
  modelReasoningRequestParameters,
  normalizeModelReasoningEffort,
  resolveHarnessModelInventory,
  resolveModelReasoningCapability,
} from "../src/model-capabilities.js";

describe("standalone Model capabilities", () => {
  it("keeps source-backed reasoning records separate from provider identity", () => {
    expect(MODEL_CAPABILITIES.length).toBeGreaterThan(0);
    for (const capability of MODEL_CAPABILITIES) {
      expect(ModelCapabilitySchema.parse(capability)).toBeDefined();
      expect(capability.source.url).toMatch(/^https:\/\//);
      expect(capability.source.checkedAt).toBe("2026-07-11");
      expect(capability).not.toHaveProperty("providerKind");
      expect(capability).not.toHaveProperty("providerProduct");
    }
  });

  it("builds independent Model entities and merges their supported APIs", () => {
    const gpt5 = MODELS.find((model) => model.id === "gpt-5");
    expect(gpt5).toMatchObject({
      id: "gpt-5",
      runtimeModel: "gpt-5",
      apiProtocols: ["openai_chat", "openai_responses"],
    });
    expect(ModelSchema.parse(gpt5)).toBeDefined();
  });

  it("looks up reasoning by Model and API only", () => {
    expect(
      findModelCapability({ modelId: "gpt-5.6", apiProtocol: "openai_responses" }),
    ).toMatchObject({
      reasoningControl: "effort_enum",
      supportedEfforts: ["none", "low", "medium", "high", "xhigh", "max"],
      defaultEffort: "medium",
    });
    expect(() =>
      findModelCapability({
        modelId: "gpt-5",
        apiProtocol: "openai_chat",
        providerProduct: "openai",
      }),
    ).toThrow();
    expect(
      findModelCapability({ modelId: "claude-fable-5", apiProtocol: "anthropic" }),
    ).toMatchObject({
      supportedEfforts: ["low", "medium", "high", "xhigh", "max"],
      defaultEffort: "high",
    });
  });

  it("resolves effort for a Harness x Model pair without a Provider", () => {
    expect(
      resolveModelReasoningCapability({
        harnessId: "swarmx",
        modelId: "gpt-5",
        apiProtocol: "openai_chat",
      }),
    ).toMatchObject({
      capabilityId: "openai-gpt-5-chat-effort",
      modelId: "gpt-5",
      apiProtocol: "openai_chat",
      supportedEfforts: ["minimal", "low", "medium", "high"],
    });
    expect(
      normalizeModelReasoningEffort({
        harnessId: "codex",
        modelId: "gpt-5.5",
        apiProtocol: "openai_responses",
        effort: " XHIGH ",
      }),
    ).toBe("xhigh");
    expect(
      modelReasoningRequestParameters({
        harnessId: "swarmx",
        modelId: "gpt-5.5",
        apiProtocol: "openai_chat",
        effort: "high",
      }),
    ).toEqual({ reasoning_effort: "high" });
  });

  it("keeps unknown reasoning controls non-executable", () => {
    expect(
      resolveModelReasoningCapability({
        harnessId: "swarmx",
        modelId: "deepseek-reasoner",
        apiProtocol: "openai_chat",
      }),
    ).toBeUndefined();
  });

  it("lets advertised effort override one Model without erasing sibling built-ins", () => {
    const registry = modelCapabilityRegistry([
      {
        id: "claude-fable-5",
        runtimeModel: "claude-fable-5",
        apiProtocols: ["anthropic"],
        reasoningCapabilities: [
          {
            id: "advertised-fable-effort",
            apiProtocol: "anthropic",
            modelIds: ["claude-fable-5"],
            reasoningControl: "effort_enum",
            supportedEfforts: ["low", "high"],
            defaultEffort: "low",
            parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
            source: {
              url: "https://provider.example.test/models",
              checkedAt: "2026-07-13",
              applicability: "Provider-advertised Fable route",
              version: "test",
            },
          },
        ],
      },
      {
        id: "claude-mythos-5",
        runtimeModel: "claude-mythos-5",
        apiProtocols: ["anthropic"],
      },
    ]);

    expect(
      findModelCapability({ modelId: "claude-fable-5", apiProtocol: "anthropic" }, registry),
    ).toMatchObject({ supportedEfforts: ["low", "high"], defaultEffort: "low" });
    expect(
      findModelCapability({ modelId: "claude-mythos-5", apiProtocol: "anthropic" }, registry),
    ).toMatchObject({
      supportedEfforts: ["low", "medium", "high", "xhigh", "max"],
      defaultEffort: "high",
    });
  });
});

describe("Harness x Model matrix", () => {
  const models = [
    {
      id: "shared-model",
      label: "Shared Model",
      runtimeModel: "shared-runtime",
      apiProtocols: ["openai_chat"],
    },
    {
      id: "claude-model",
      label: "Claude Model",
      runtimeModel: "claude-runtime",
      apiProtocols: ["anthropic"],
    },
  ];
  const supplies = [
    {
      id: "shared-openai",
      modelId: "shared-model",
      providerProfileId: "openai",
      runtimeModel: "shared-openai-runtime",
      harnessIds: ["codex"],
    },
    {
      id: "shared-local",
      modelId: "shared-model",
      providerProfileId: "local",
      runtimeModel: "shared-local-runtime",
      harnessIds: ["codex"],
    },
    {
      id: "claude-openai",
      modelId: "claude-model",
      providerProfileId: "openai",
      runtimeModel: "claude-openai-runtime",
      apiCompatibility: { mode: "bridge", targetApi: "openai_chat" },
      harnessIds: ["codex"],
    },
  ];

  it("models many-to-many supplies without multiplying agent identity", () => {
    for (const supply of supplies) expect(ModelSupplySchema.parse(supply)).toBeDefined();
    const matrix = resolveHarnessModelInventory({
      harnessId: "codex",
      models,
      supplies,
      providers: [
        { id: "openai", label: "OpenAI" },
        { id: "local", label: "Local", runtimeReady: false, runtimeNote: "offline" },
      ],
    });

    expect(matrix).toHaveLength(2);
    expect(matrix[0]).toMatchObject({
      agentId: "codex:shared-model",
      modelId: "shared-model",
      modelControl: "session",
      supplies: [
        { id: "shared-openai", providerProfileId: "openai" },
        { id: "shared-local", providerProfileId: "local", runtimeReady: false },
      ],
    });
    expect(matrix.filter((entry) => entry.modelId === "shared-model")).toHaveLength(1);
  });

  it("never uses Provider records to decide Harness compatibility", () => {
    const withoutProviders = resolveHarnessModelInventory({
      harnessId: "swarmx",
      models,
      supplies,
      providers: [],
    });
    const withUnreadyProviders = resolveHarnessModelInventory({
      harnessId: "swarmx",
      models,
      supplies,
      providers: [
        { id: "openai", enabled: false },
        { id: "local", runtimeReady: false },
      ],
    });

    expect(withoutProviders.map((entry) => entry.modelId)).toEqual([
      "shared-model",
      "claude-model",
    ]);
    expect(withUnreadyProviders.map((entry) => entry.modelId)).toEqual([
      "shared-model",
      "claude-model",
    ]);
  });

  it("V284 prefers the selected supply's native API over protocol conversion", () => {
    const model = {
      id: "multi-api-model",
      runtimeModel: "multi-api-model",
      apiProtocols: ["anthropic", "openai_chat", "openai_responses"],
    };
    const responseRoute = resolveHarnessModelInventory({
      harnessId: "swarmx",
      models: [model],
      supplies: [
        {
          id: "native-responses",
          modelId: model.id,
          providerProfileId: "responses-provider",
        },
      ],
      providers: [
        {
          id: "responses-provider",
          kind: "openai_responses",
          apiEntrypoints: {},
        },
      ],
    });
    const chatRoute = resolveHarnessModelInventory({
      harnessId: "swarmx",
      models: [model],
      supplies: [
        {
          id: "native-chat",
          modelId: model.id,
          providerProfileId: "chat-provider",
        },
      ],
      providers: [{ id: "chat-provider", kind: "openai_chat", apiEntrypoints: {} }],
    });

    expect(responseRoute[0]?.apiProtocol).toBe("openai_responses");
    expect(chatRoute[0]?.apiProtocol).toBe("openai_chat");
  });

  it("V332 fails closed unless a session Harness has an explicit executable route", () => {
    for (const harnessId of ["claude_code", "codex", "pi", "opencode", "hermes"]) {
      expect(resolveHarnessModelInventory({ harnessId, models })).toEqual([]);
    }

    expect(
      resolveHarnessModelInventory({
        harnessId: "opencode",
        models,
        supplies: [
          {
            id: "shared-opencode",
            modelId: "shared-model",
            providerProfileId: "openai",
            runtimeModel: "openai/shared-runtime",
            harnessIds: ["opencode"],
          },
        ],
      }),
    ).toContainEqual(
      expect.objectContaining({
        harnessId: "opencode",
        modelId: "shared-model",
        supplies: [
          expect.objectContaining({
            id: "shared-opencode",
            runtimeModel: "openai/shared-runtime",
          }),
        ],
      }),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "pi",
        models,
        supplies: [
          {
            id: "shared-pi",
            modelId: "shared-model",
            providerProfileId: "pi-anthropic",
            runtimeModel: "anthropic/shared-runtime",
            harnessIds: ["pi"],
          },
        ],
      }),
    ).toContainEqual(
      expect.objectContaining({
        harnessId: "pi",
        modelId: "shared-model",
        supplies: [
          expect.objectContaining({
            id: "shared-pi",
            runtimeModel: "anthropic/shared-runtime",
          }),
        ],
      }),
    );
    expect(
      resolveHarnessModelInventory({
        harnessId: "hermes",
        models,
        supplies: [
          {
            id: "shared-opencode",
            modelId: "shared-model",
            providerProfileId: "openai",
            runtimeModel: "openai/shared-runtime",
            harnessIds: ["opencode"],
          },
        ],
      }),
    ).toEqual([]);
    expect(resolveHarnessModelInventory({ harnessId: "openclaw", models })).toEqual([]);
  });

  it("V332 resolves a fixed runtime Model id through a custom Harness adapter", () => {
    expect(
      resolveHarnessModelInventory({
        harnessId: "custom-claude",
        models: [
          {
            id: "deepseek-v4-pro",
            runtimeModel: "deepseek-v4-pro",
            apiProtocols: ["openai_chat"],
            harnessRuntimeModels: { claude_code: "deepseek-v4-pro[1m]" },
          },
        ],
        harnesses: [
          {
            id: "custom-claude",
            runtimeHarnessId: "claude_code",
            modelControl: "session",
            modelCompatibility: "any",
            requiresExplicitModelRoute: true,
            supportedModelApis: [],
          },
        ],
      }),
    ).toContainEqual(
      expect.objectContaining({
        agentId: "custom-claude:deepseek-v4-pro",
        runtimeModel: "deepseek-v4-pro[1m]",
      }),
    );
  });
});

describe("agent composition identity", () => {
  function inventory() {
    const bundle = parseExtensionBundle({
      id: "standalone-models",
      name: "Standalone models",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "openai",
            label: "OpenAI",
            kind: "openai_chat",
          },
        ],
        modelSupplies: [
          {
            id: "gpt-5-openai",
            modelId: "gpt-5",
            providerProfileId: "openai",
          },
        ],
      },
    });
    return createExtensionInventory([builtInExtensionBundle(), bundle]);
  }

  it("uses Harness x Model as identity and keeps supply as routing metadata", () => {
    const composition = {
      id: "desktop-selection",
      harnessId: "swarmx",
      modelId: "gpt-5",
      modelSupplyId: "gpt-5-openai",
      effort: "high",
    };
    const plan = resolveAgentCompositionPlan(composition, inventory());
    expect(plan).toMatchObject({
      agentId: "swarmx:gpt-5",
      modelId: "gpt-5",
      runtimeModel: "gpt-5",
      apiProtocol: "openai_chat",
      modelSupplyId: "gpt-5-openai",
      effort: "high",
    });
    expect(plan).not.toHaveProperty("providerProfileId");

    const agent = resolveAgentComposition(composition, inventory());
    expect(agent).toMatchObject({
      name: "swarmx_gpt_5",
      model: "gpt-5",
      client: { apiProtocol: "openai_chat" },
    });
    expect(agent.parameters).toMatchObject({
      extension: {
        harnessId: "swarmx",
        modelId: "gpt-5",
        modelSupplyId: "gpt-5-openai",
      },
      reasoning: { effort: "high" },
    });
  });

  it("sends normalized effort in a direct native call", async () => {
    const config = resolveAgentComposition(
      { id: "native", harnessId: "swarmx", modelId: "gpt-5", effort: "high" },
      inventory(),
    );
    const agent = new Agent(config);
    const create = vi.fn().mockResolvedValue({
      choices: [{ message: { content: "done" } }],
    });
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    await agent.call({ messages: [{ role: "user", content: "go" }] });

    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "gpt-5",
      reasoning_effort: "high",
    });
  });

  it("rejects provider-owned model declarations", () => {
    expect(() =>
      parseExtensionBundle({
        id: "legacy-provider-model",
        name: "Legacy",
        version: "1.0.0",
        capabilities: {
          providers: [
            { id: "legacy", label: "Legacy", kind: "openai_chat", defaultModel: "gpt-5" },
          ],
        },
      }),
    ).toThrow(/Models and route compatibility belong to Model\/ModelSupply/);
  });
});
