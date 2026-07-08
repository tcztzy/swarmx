import { describe, expect, it } from "vitest";
import {
  buildProviderRuntimeEnv,
  parseProviderProfileMetadata,
  parseProviderPromptRequest,
  parseProviderSecretStatus,
  providerModelForSelection,
  resolveProviderProfile,
} from "../src/providers.js";

describe("provider profile primitives", () => {
  it("parses provider profile metadata without inline secrets", () => {
    const profile = parseProviderProfileMetadata({
      id: "deepseek",
      preset_id: "deepseek-chat",
      display_name: "DeepSeek",
      kind: "openai_chat",
      default_model: "deepseek-chat",
      base_url: "https://api.deepseek.example/v1",
      is_default: true,
      harness_model_overrides: {
        claude: "deepseek-reasoner",
      },
      secret_ref: {
        source: "local_keychain",
        key: "providers/deepseek",
      },
    });

    expect(profile).toMatchObject({
      id: "deepseek",
      presetId: "deepseek-chat",
      displayName: "DeepSeek",
      kind: "openai_chat",
      defaultModel: "deepseek-chat",
      baseUrl: "https://api.deepseek.example/v1",
      isDefault: true,
      harnessModelOverrides: {
        claude: "deepseek-reasoner",
      },
      secretRef: {
        source: "local_keychain",
        key: "providers/deepseek",
      },
    });
    expect(JSON.stringify(profile)).not.toContain("sk-");
  });

  it("rejects inline provider secrets in profiles, statuses, and prompt requests", () => {
    expect(() =>
      parseProviderProfileMetadata({
        id: "bad",
        displayName: "Bad",
        kind: "openai_responses",
        defaultModel: "gpt-5",
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

    expect(() =>
      parseProviderPromptRequest({
        profile_id: "openai",
        user_text: "Summarize this.",
        parameters: {
          bearerToken: "secret-token",
        },
      }),
    ).toThrow(/inline secret field.*bearerToken/);
  });

  it("resolves explicit, default, and kind-scoped provider selections", () => {
    const profiles = [
      {
        id: "openai",
        displayName: "OpenAI",
        kind: "openai_responses",
        defaultModel: "gpt-5",
        isDefault: true,
      },
      {
        id: "ollama-local",
        displayName: "Ollama",
        kind: "ollama",
        defaultModel: "qwen3",
        baseUrl: "http://localhost:11434",
      },
    ];

    expect(resolveProviderProfile(profiles, { profile_id: "ollama-local" }).id).toBe(
      "ollama-local",
    );
    expect(resolveProviderProfile(profiles).id).toBe("openai");
    expect(resolveProviderProfile(profiles, { kind: "ollama" }).id).toBe("ollama-local");
    expect(() => resolveProviderProfile(profiles, { profile_id: "missing" })).toThrow(
      /Unknown provider profile id/,
    );
    expect(() =>
      resolveProviderProfile([
        { id: "a", displayName: "A", kind: "openai_chat", defaultModel: "a" },
        { id: "b", displayName: "B", kind: "openai_chat", defaultModel: "b" },
      ]),
    ).toThrow(/must be explicit/);
  });

  it("builds runtime env from explicit secret values without copying them into metadata", () => {
    const profile = parseProviderProfileMetadata({
      id: "openai",
      label: "OpenAI",
      kind: "openai_responses",
      model: "gpt-5",
      baseUrl: "https://api.openai.example/v1",
      harnessModelOverrides: {
        codex: "gpt-5-codex",
      },
      secretRef: {
        source: "env",
        key: "OPENAI_API_KEY",
      },
    });

    expect(providerModelForSelection(profile, { harnessId: "codex" })).toBe("gpt-5-codex");
    expect(() => buildProviderRuntimeEnv(profile, { harnessId: "codex" })).toThrow(
      /requires a secret value/,
    );

    const runtime = buildProviderRuntimeEnv(profile, {
      harnessId: "codex",
      secretValue: "sk-runtime",
    });

    expect(runtime).toMatchObject({
      profileId: "openai",
      kind: "openai_responses",
      model: "gpt-5-codex",
      requiresSecret: true,
      secretInjected: true,
      env: {
        OPENAI_API_KEY: "sk-runtime",
        OPENAI_BASE_URL: "https://api.openai.example/v1",
        OPENAI_MODEL: "gpt-5-codex",
      },
    });
    expect(JSON.stringify(profile)).not.toContain("sk-runtime");
  });

  it("builds API bridge runtime env for cross-harness provider use", () => {
    const profile = parseProviderProfileMetadata({
      id: "anthropic-prod",
      label: "Anthropic Prod",
      kind: "anthropic",
      model: "claude-sonnet",
      baseUrl: "https://api.anthropic.com",
      secretRef: {
        source: "env",
        key: "ANTHROPIC_API_KEY",
      },
    });

    const runtime = buildProviderRuntimeEnv(profile, {
      compatibleProviderKinds: ["openai_responses"],
      bridgeBaseUrl: "http://127.0.0.1:4100/v1",
      secretValue: "sk-ant-runtime",
    });

    expect(runtime).toMatchObject({
      profileId: "anthropic-prod",
      kind: "anthropic",
      targetKind: "openai_responses",
      model: "claude-sonnet",
      apiCompatibility: {
        mode: "auto",
        baseUrl: "http://127.0.0.1:4100/v1",
      },
      bridgeEnabled: true,
      env: {
        YALLM_DEFAULT_PROVIDER: "anthropic",
        ANTHROPIC_API_KEY: "sk-ant-runtime",
        ANTHROPIC_BASE_URL: "https://api.anthropic.com",
        OPENAI_API_KEY: "sk-swarmx-bridge",
        OPENAI_BASE_URL: "http://127.0.0.1:4100/v1",
        OPENAI_MODEL: "anthropic:claude-sonnet",
      },
    });
    expect(JSON.stringify(profile)).not.toContain("sk-ant-runtime");
  });

  it("supports direct provider prompt metadata without requiring a harness", () => {
    expect(
      parseProviderPromptRequest({
        request_id: "prompt_1",
        profile_id: "openai",
        user_text: "Explain this dataset.",
        model: "gpt-5",
        context_packet_id: "ctx_1",
        parameters: {
          temperature: 0.2,
        },
      }),
    ).toMatchObject({
      requestId: "prompt_1",
      profileId: "openai",
      userText: "Explain this dataset.",
      model: "gpt-5",
      contextPacketId: "ctx_1",
      parameters: {
        temperature: 0.2,
      },
    });
  });
});
