import { describe, expect, it } from "vitest";
import {
  createDefaultDesktopSettings,
  createLocaleRegistry,
  parseDesktopSettingsDocument,
  parseDesktopUiState,
  resolveDesktopRoot,
  resolveLocaleSelection,
} from "../src/desktop-settings.js";

const localeRegistry = {
  defaultLocaleId: "en-US",
  locales: [
    {
      id: "en-US",
      label: "English",
      nativeLabel: "English",
      messages: { send: "Send" },
    },
    {
      id: "zh-CN",
      label: "Chinese",
      nativeLabel: "中文",
      messages: { send: "发送" },
    },
  ],
};

describe("desktop settings primitives", () => {
  it("parses settings documents with provider and agent metadata kept separate", () => {
    const settings = parseDesktopSettingsDocument({
      schema_version: 1,
      desktop: {
        root: "/Users/example/.swarmx",
      },
      server: {
        base_url: "http://127.0.0.1:8787",
        data_root: "/srv/swarmx",
      },
      ui: {
        locale: "zh-CN",
        theme: "dark",
      },
      providers: [
        {
          id: "deepseek",
          display_name: "DeepSeek",
          kind: "openai_responses",
          model: "deepseek-reasoner",
          secret_ref: { source: "env", key: "DEEPSEEK_API_KEY" },
        },
      ],
      agents: [
        {
          id: "reviewer",
          name: "Reviewer",
          aliases: ["@reviewer"],
          harnessId: "claude",
          providerProfileId: "deepseek",
          model: "inherit",
        },
      ],
      extensions: {
        enabledPluginIds: ["geepilot"],
      },
    });

    expect(settings).toMatchObject({
      schemaVersion: 1,
      desktop: { root: "/Users/example/.swarmx" },
      server: {
        baseUrl: "http://127.0.0.1:8787",
        dataRoot: "/srv/swarmx",
      },
      ui: {
        locale: "zh-CN",
        theme: "dark",
      },
      providers: [
        {
          id: "deepseek",
          displayName: "DeepSeek",
          defaultModel: "deepseek-reasoner",
          secretRef: { source: "env", key: "DEEPSEEK_API_KEY" },
        },
      ],
      agents: [
        {
          id: "reviewer",
          name: "Reviewer",
          aliases: ["@reviewer"],
          harnessId: "claude",
          providerProfileId: "deepseek",
        },
      ],
      extensions: {
        enabledPluginIds: ["geepilot"],
      },
    });
  });

  it("resolves desktop roots by env, settings, legacy fallback, and default precedence", () => {
    const settings = {
      desktop: {
        root: "/settings/desktop",
        legacy_app_root: "/settings/legacy-desktop",
      },
      server: {
        data_root: "/settings/server-data",
        app_root: "/settings/server-app",
      },
    };

    expect(
      resolveDesktopRoot({
        settings,
        env: {
          GEEPILOT_DESKTOP_ROOT: "/env/desktop",
          GEEPILOT_APP_ROOT: "/env/legacy",
          GEEPILOT_SERVER_DATA_ROOT: "/env/server-data",
        },
      }),
    ).toMatchObject({
      desktopRoot: "/env/desktop",
      source: "env_desktop_root",
      legacyFallback: false,
      serverDataRoot: "/env/server-data",
      serverDataRootSource: "env_server_data_root",
    });

    expect(
      resolveDesktopRoot({
        settings,
        env: { GEEPILOT_APP_ROOT: "/env/legacy" },
      }),
    ).toMatchObject({
      desktopRoot: "/settings/desktop",
      source: "settings_desktop_root",
      legacyFallback: false,
      serverDataRoot: "/settings/server-data",
      serverDataRootSource: "settings_server_data_root",
    });

    expect(
      resolveDesktopRoot({
        settings: { server: { app_root: "/settings/server-app" } },
        env: { GEEPILOT_APP_ROOT: "/env/legacy" },
      }),
    ).toMatchObject({
      desktopRoot: "/env/legacy",
      source: "env_legacy_app_root",
      legacyFallback: true,
    });

    expect(
      resolveDesktopRoot({
        settings: { server: { app_root: "/settings/server-app" } },
        env: {},
      }),
    ).toMatchObject({
      desktopRoot: "/settings/server-app",
      source: "settings_legacy_app_root",
      legacyFallback: true,
    });

    expect(
      resolveDesktopRoot({
        settings: { server: { data_root: "/settings/server-data" } },
        defaultDesktopRoot: "/default/desktop",
      }),
    ).toMatchObject({
      desktopRoot: "/default/desktop",
      source: "default",
      legacyFallback: false,
      serverDataRoot: "/settings/server-data",
      serverDataRootSource: "settings_server_data_root",
    });

    expect(
      resolveDesktopRoot({
        settings: { server: { data_root: "/settings/server-data" } },
      }),
    ).toMatchObject({
      source: "unresolved",
      serverDataRoot: "/settings/server-data",
      serverDataRootSource: "settings_server_data_root",
    });
  });

  it("centralizes locale registry validation and locale selection fallback", () => {
    const registry = createLocaleRegistry(localeRegistry);

    expect(resolveLocaleSelection({ registry, requestedLocaleId: "zh-CN" })).toMatchObject({
      locale: { id: "zh-CN", messages: { send: "发送" } },
      source: "requested",
      fallback: false,
    });
    expect(
      resolveLocaleSelection({
        registry,
        settings: { ui: { locale: "zh-CN" } },
      }),
    ).toMatchObject({
      locale: { id: "zh-CN" },
      source: "settings",
      fallback: false,
    });
    expect(
      resolveLocaleSelection({
        registry,
        env: { LANG: "zh_CN.UTF-8" },
      }),
    ).toMatchObject({
      locale: { id: "zh-CN" },
      source: "env",
      fallback: false,
    });
    expect(
      resolveLocaleSelection({
        registry,
        requestedLocaleId: "fr-FR",
      }),
    ).toMatchObject({
      locale: { id: "en-US" },
      source: "default",
      fallback: true,
    });

    expect(() =>
      createLocaleRegistry({
        defaultLocaleId: "en-US",
        locales: [
          { id: "en-US", label: "English" },
          { id: "en-US", label: "Duplicate" },
        ],
      }),
    ).toThrow(/Duplicate locale id/);
    expect(() =>
      createLocaleRegistry({
        defaultLocaleId: "missing",
        locales: [{ id: "en-US", label: "English" }],
      }),
    ).toThrow(/Default locale.*missing.*is not registered/);
  });

  it("rejects inline secrets while allowing secret references", () => {
    expect(() =>
      parseDesktopSettingsDocument({
        server: { apiToken: "secret-token" },
      }),
    ).toThrow(/inline secret field.*apiToken/);
    expect(() =>
      parseDesktopUiState({
        locale: "en-US",
        telemetryToken: "secret-token",
      }),
    ).toThrow(/inline secret field.*telemetryToken/);
    expect(() =>
      parseDesktopSettingsDocument({
        providers: [
          {
            id: "openai",
            displayName: "OpenAI",
            kind: "openai_responses",
            apiKey: "sk-test",
          },
        ],
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(
      parseDesktopSettingsDocument({
        providers: [
          {
            id: "openai",
            displayName: "OpenAI",
            kind: "openai_responses",
            secretRef: { source: "env", key: "OPENAI_API_KEY" },
          },
        ],
      }),
    ).toMatchObject({
      providers: [{ secretRef: { source: "env", key: "OPENAI_API_KEY" } }],
    });
  });

  it("creates default settings without choosing product-owned roots", () => {
    expect(createDefaultDesktopSettings()).toMatchObject({
      schemaVersion: 1,
      desktop: {},
      server: {},
      ui: { theme: "system" },
      providers: [],
      agents: [],
      extensions: {
        enabledPluginIds: [],
        disabledPluginIds: [],
        trustedSourceIds: [],
      },
    });
  });
});
