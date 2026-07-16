import { describe, expect, it } from "vitest";
import {
  createExtensionInventory,
  parseExtensionBundle,
  resolveAgentCompositionRuntimeEnv,
} from "../src/extensions.js";
import {
  buildProviderRuntimeEnv,
  parseProviderProfileMetadata,
  parseProviderPromptRequest,
  parseProviderSecretStatus,
  resolveProviderProfile,
} from "../src/providers.js";

describe("provider supply primitives", () => {
  it("parses connection metadata without Model ownership", () => {
    expect(
      parseProviderProfileMetadata({
        id: "deepseek",
        preset_id: "deepseek-api",
        display_name: "DeepSeek",
        kind: "openai_chat",
        base_url: "https://api.deepseek.example/v1",
        secret_ref: { source: "local_keychain", key: "providers/deepseek" },
      }),
    ).toMatchObject({
      id: "deepseek",
      presetId: "deepseek-api",
      displayName: "DeepSeek",
      kind: "openai_chat",
      baseUrl: "https://api.deepseek.example/v1",
    });
  });

  it.each([
    ["model", "gpt-5"],
    ["models", ["gpt-5"]],
    ["defaultModel", "gpt-5"],
    ["harnessModelOverrides", { codex: "gpt-5" }],
    ["providerProduct", "openai"],
    ["apiCompatibility", { mode: "bridge" }],
  ])("rejects provider-owned field %s", (key, value) => {
    expect(() =>
      parseProviderProfileMetadata({
        id: "legacy",
        displayName: "Legacy",
        kind: "openai_chat",
        [key]: value,
      }),
    ).toThrow(/Models and route compatibility belong to Model\/ModelSupply/);
  });

  it("rejects inline secrets", () => {
    expect(() =>
      parseProviderProfileMetadata({
        id: "bad",
        displayName: "Bad",
        kind: "openai_responses",
        apiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*apiKey/);
    expect(() =>
      parseProviderSecretStatus({
        profile_id: "openai",
        source: "local_keychain",
        configured: true,
        value: "sk-test",
      }),
    ).toThrow(/must not include secret values/);
  });

  it("requires explicit selection when more than one supply label exists", () => {
    const profiles = [
      { id: "openai", displayName: "OpenAI", kind: "openai_responses" },
      { id: "local", displayName: "Local", kind: "ollama" },
    ];
    expect(resolveProviderProfile(profiles, { profile_id: "local" }).id).toBe("local");
    expect(resolveProviderProfile(profiles, { kind: "ollama" }).id).toBe("local");
    expect(() => resolveProviderProfile(profiles)).toThrow(/must be explicit/);
  });

  it("builds runtime env from an explicit Model and supply route", () => {
    const profile = parseProviderProfileMetadata({
      id: "openai",
      label: "OpenAI",
      kind: "openai_responses",
      baseUrl: "https://api.openai.example/v1",
      secretRef: { source: "env", key: "OPENAI_API_KEY" },
    });
    expect(() => buildProviderRuntimeEnv(profile, { modelId: "gpt-5" })).toThrow(
      /requires a secret value/,
    );

    const runtime = buildProviderRuntimeEnv(profile, {
      modelId: "gpt-5",
      runtimeModel: "gpt-5-route",
      secretValue: "sk-runtime",
    });
    expect(runtime).toMatchObject({
      profileId: "openai",
      modelId: "gpt-5",
      runtimeModel: "gpt-5-route",
      env: {
        OPENAI_API_KEY: "sk-runtime",
        OPENAI_BASE_URL: "https://api.openai.example/v1",
        OPENAI_MODEL: "gpt-5-route",
      },
    });
    expect(JSON.stringify(profile)).not.toContain("sk-runtime");
  });

  it("maps codex_responses subscription credentials without exposing them as OpenAI API keys", () => {
    const profile = parseProviderProfileMetadata({
      id: "codex-subscription",
      displayName: "Codex subscription",
      kind: "openai_responses",
      api_mode: "codex_responses",
      baseUrl: "https://chatgpt.com/backend-api/codex",
      authMode: "auth_token",
      secretRef: { source: "env", key: "CODEX_ACCESS_TOKEN" },
    });

    const runtime = buildProviderRuntimeEnv(profile, {
      modelId: "gpt-5.4",
      runtimeModel: "gpt-5.4",
      secretValue: "subscription-access-token",
    });

    expect(runtime).toMatchObject({
      apiMode: "codex_responses",
      targetApi: "openai_responses",
      bridgeEnabled: false,
      env: {
        SWARMX_API_MODE: "codex_responses",
        CODEX_ACCESS_TOKEN: "subscription-access-token",
        CODEX_BASE_URL: "https://chatgpt.com/backend-api/codex",
        OPENAI_MODEL: "gpt-5.4",
      },
    });
    expect(runtime.env.OPENAI_API_KEY).toBeUndefined();
    expect(() =>
      buildProviderRuntimeEnv(profile, {
        modelId: "gpt-5.4",
        secretValue: "subscription-access-token",
        targetApi: "openai_chat",
      }),
    ).toThrow(/requires the openai_responses API protocol/);
  });

  it("maps Anthropic API keys and auth tokens to distinct runtime variables", () => {
    const apiKeyProfile = parseProviderProfileMetadata({
      id: "anthropic-key",
      displayName: "Anthropic API key",
      kind: "anthropic",
      baseUrl: "https://anthropic.example",
      authMode: "api_key",
      secretRef: { source: "local_keychain", key: "anthropic-key" },
    });
    const tokenProfile = parseProviderProfileMetadata({
      id: "anthropic-token",
      displayName: "Anthropic auth token",
      kind: "anthropic",
      baseUrl: "https://anthropic.example",
      auth_mode: "auth_token",
      secretRef: { source: "local_keychain", key: "anthropic-token" },
    });

    expect(
      buildProviderRuntimeEnv(apiKeyProfile, {
        modelId: "claude-model",
        secretValue: "sk-ant-api-key",
      }).env,
    ).toMatchObject({
      ANTHROPIC_API_KEY: "sk-ant-api-key",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
    });
    expect(
      buildProviderRuntimeEnv(tokenProfile, {
        modelId: "claude-model",
        secretValue: "token-runtime",
      }).env,
    ).toEqual({
      ANTHROPIC_AUTH_TOKEN: "token-runtime",
      ANTHROPIC_BASE_URL: "https://anthropic.example",
      ANTHROPIC_MODEL: "claude-model",
    });
  });

  it("applies yallm bridge metadata from ModelSupply, not Provider", () => {
    const profile = parseProviderProfileMetadata({
      id: "anthropic",
      label: "Anthropic",
      kind: "anthropic",
      baseUrl: "https://api.anthropic.com",
    });
    const runtime = buildProviderRuntimeEnv(profile, {
      modelId: "claude-sonnet",
      runtimeModel: "claude-sonnet",
      targetApi: "openai_responses",
      apiCompatibility: { mode: "bridge", baseUrl: "http://127.0.0.1:4100/v1" },
    });
    expect(runtime).toMatchObject({
      bridgeEnabled: true,
      targetApi: "openai_responses",
      env: {
        YALLM_DEFAULT_PROVIDER: "anthropic",
        OPENAI_BASE_URL: "http://127.0.0.1:4100/v1",
        OPENAI_MODEL: "anthropic:claude-sonnet",
      },
    });
  });

  it("uses a declared native entrypoint for an explicit target API without a bridge", () => {
    const profile = parseProviderProfileMetadata({
      id: "deepseek",
      displayName: "DeepSeek",
      kind: "anthropic",
      baseUrl: "https://api.deepseek.com/anthropic",
      apiEntrypoints: {
        anthropic: "https://api.deepseek.com/anthropic",
        openai_chat: "https://api.deepseek.com",
      },
      secretRef: { source: "local_keychain", key: "deepseek" },
    });

    const preferred = buildProviderRuntimeEnv(profile, {
      modelId: "deepseek-v4-pro",
      secretValue: "shared-key",
    });
    expect(preferred).toMatchObject({
      targetApi: "anthropic",
      baseUrl: "https://api.deepseek.com/anthropic",
      bridgeEnabled: false,
      env: {
        ANTHROPIC_API_KEY: "shared-key",
        ANTHROPIC_BASE_URL: "https://api.deepseek.com/anthropic",
      },
    });

    const chat = buildProviderRuntimeEnv(profile, {
      modelId: "deepseek-v4-pro",
      secretValue: "shared-key",
      targetApi: "openai_chat",
    });
    expect(chat).toMatchObject({
      targetApi: "openai_chat",
      baseUrl: "https://api.deepseek.com",
      bridgeEnabled: false,
      env: {
        OPENAI_API_KEY: "shared-key",
        OPENAI_BASE_URL: "https://api.deepseek.com",
      },
    });
    expect(chat.env).not.toHaveProperty("YALLM_DEFAULT_PROVIDER");

    const chatPreferredProfile = parseProviderProfileMetadata({
      ...profile,
      kind: "openai_chat",
      baseUrl: "https://api.deepseek.com",
    });
    const chatPreferred = buildProviderRuntimeEnv(chatPreferredProfile, {
      modelId: "deepseek-v4-pro",
      secretValue: "shared-key",
    });
    expect(chatPreferred).toMatchObject({
      targetApi: "openai_chat",
      baseUrl: "https://api.deepseek.com",
      bridgeEnabled: false,
      env: {
        OPENAI_API_KEY: "shared-key",
        OPENAI_BASE_URL: "https://api.deepseek.com",
      },
    });
    expect(chatPreferred.secretRef).toEqual(preferred.secretRef);
  });

  it("preserves native Provider entrypoints through ModelSupply runtime resolution", () => {
    const bundle = parseExtensionBundle({
      id: "deepseek-dual-entrypoints",
      name: "DeepSeek dual entrypoints",
      version: "1.0.0",
      capabilities: {
        models: [
          {
            id: "deepseek-chat",
            runtimeModel: "deepseek-chat",
            apiProtocols: ["anthropic", "openai_chat"],
          },
        ],
        providers: [
          {
            id: "deepseek",
            label: "DeepSeek",
            kind: "anthropic",
            baseUrl: "https://api.deepseek.com/anthropic",
            apiEntrypoints: {
              anthropic: "https://api.deepseek.com/anthropic",
              openai_chat: "https://api.deepseek.com",
            },
            secretRef: { source: "env", key: "DEEPSEEK_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "deepseek-chat-native",
            modelId: "deepseek-chat",
            providerProfileId: "deepseek",
            apiCompatibility: { mode: "native", targetApi: "openai_chat" },
          },
        ],
        harnesses: [
          {
            id: "chat-harness",
            label: "Chat Harness",
            modelControl: "direct",
            modelCompatibility: "declared_apis",
            supportedModelApis: ["openai_chat"],
            backend: { type: "echo" },
          },
        ],
        agents: [
          {
            id: "deepseek-chat-agent",
            name: "DeepSeek Chat Agent",
            harnessId: "chat-harness",
            modelId: "deepseek-chat",
            modelSupplyId: "deepseek-chat-native",
          },
        ],
      },
    });

    expect(
      resolveAgentCompositionRuntimeEnv(
        { id: "deepseek-chat-run", agentProfileId: "deepseek-chat-agent" },
        createExtensionInventory([bundle]),
        { env: { DEEPSEEK_API_KEY: "shared-key" } },
      ),
    ).toEqual({
      OPENAI_API_KEY: "shared-key",
      OPENAI_BASE_URL: "https://api.deepseek.com",
      OPENAI_MODEL: "deepseek-chat",
    });
  });

  it("keeps direct prompt Model identity explicit", () => {
    expect(
      parseProviderPromptRequest({
        request_id: "prompt_1",
        profile_id: "openai",
        user_text: "Explain this dataset.",
        model_id: "gpt-5",
        runtime_model: "gpt-5-route",
      }),
    ).toMatchObject({
      profileId: "openai",
      modelId: "gpt-5",
      runtimeModel: "gpt-5-route",
    });
  });
});
