import { type ProviderProfile, createExtensionInventory } from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";
import {
  ProviderUsageService,
  queryCodexAppServer,
  queryCodexAppServerRequest,
} from "./provider-usage.js";

describe("ProviderUsageService", () => {
  it("queries DeepSeek and regional Kimi balances without exposing credentials", async () => {
    const fetch = vi.fn(async (url: string) => {
      if (url.includes("deepseek")) {
        return response({
          is_available: true,
          balance_infos: [
            {
              currency: "CNY",
              total_balance: "110.00",
              granted_balance: "10.00",
              topped_up_balance: "100.00",
            },
            {
              currency: "USD",
              total_balance: "2.50",
              granted_balance: "0.50",
              topped_up_balance: "2.00",
            },
          ],
        });
      }
      return response({
        code: 0,
        status: true,
        data: { available_balance: 49.5, voucher_balance: 46.5, cash_balance: 3 },
      });
    });
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ deepseek: "deep-secret", kimi: "kimi-secret" }),
      env: { DEEPSEEK_API_KEY: "ambient-must-not-use" },
      fetch,
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory(
        userProvider("deepseek", "DeepSeek", "https://api.deepseek.com/v1"),
        userProvider("kimi", "Kimi CN", "https://api.moonshot.cn/v1"),
      ),
    );

    expect(fetch).toHaveBeenCalledWith(
      "https://api.deepseek.com/user/balance",
      expect.objectContaining({
        redirect: "manual",
        headers: expect.objectContaining({ Authorization: "Bearer deep-secret" }),
      }),
    );
    expect(fetch).toHaveBeenCalledWith(
      "https://api.moonshot.cn/v1/users/me/balance",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer kimi-secret" }),
      }),
    );
    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        adapterId: "deepseek",
        status: "ready",
        meters: [
          expect.objectContaining({ currency: "CNY", total: "110.00" }),
          expect.objectContaining({ currency: "USD", total: "2.50" }),
        ],
      }),
    );
    expect(snapshot.providers[1]?.meters[0]).toEqual(
      expect.objectContaining({ currency: "CNY", total: "49.5" }),
    );
    expect(JSON.stringify(snapshot)).not.toMatch(/deep-secret|kimi-secret/);
  });

  it("queries an explicitly selected New API endpoint on the configured HTTPS origin", async () => {
    const fetch = vi.fn(async () =>
      response({
        code: true,
        message: "ok",
        data: {
          object: "token_usage",
          name: "Packy",
          total_granted: 1_000_000,
          total_used: "250000",
          total_available: 750_000,
          unlimited_quota: false,
          expires_at: 1_784_354_741,
        },
      }),
    );
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ packy: "packy-secret" }),
      fetch,
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory({
        ...userProvider("packy", "Packy", "https://www.packyapi.com/v1"),
        usageAdapter: "new_api",
      }),
    );

    expect(fetch).toHaveBeenCalledWith(
      "https://www.packyapi.com/api/usage/token/",
      expect.objectContaining({
        redirect: "manual",
        headers: expect.objectContaining({ Authorization: "Bearer packy-secret" }),
      }),
    );
    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        adapterId: "new_api",
        status: "ready",
        meters: [
          {
            kind: "credit",
            label: "API key quota",
            remaining: "750000",
            unit: "quota points",
          },
        ],
        detail: expect.stringMatching(/Used 250000 of 1000000.*Expires/),
      }),
    );
    expect(JSON.stringify(snapshot)).not.toContain("packy-secret");
  });

  it("preserves unlimited New API quota and does not auto-probe custom hosts", async () => {
    const fetch = vi.fn(async () =>
      response({
        code: true,
        data: {
          object: "token_usage",
          total_granted: 0,
          total_used: 0,
          total_available: -8_562_615,
          unlimited_quota: true,
          expires_at: 0,
        },
      }),
    );
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ automatic: "automatic-secret", explicit: "explicit-secret" }),
      fetch,
      includeCodex: false,
    });

    const snapshot = await service.refresh(
      usageInventory(userProvider("automatic", "Automatic", "https://gateway.example.test"), {
        ...userProvider("explicit", "Explicit", "https://gateway.example.test"),
        usageAdapter: "new_api",
      }),
    );

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      "https://gateway.example.test/api/usage/token/",
      expect.anything(),
    );
    expect(snapshot.providers[0]).toEqual(expect.objectContaining({ status: "unsupported" }));
    expect(snapshot.providers[1]?.meters).toEqual([
      expect.objectContaining({ remaining: "Unlimited", unit: "quota" }),
    ]);
    expect(JSON.stringify(snapshot)).not.toMatch(/automatic-secret|explicit-secret/);
  });

  it("refreshes exactly one selected Provider or the Codex account", async () => {
    const fetch = vi.fn(async (url: string) =>
      url.includes("deepseek")
        ? response({
            is_available: true,
            balance_infos: [
              {
                currency: "CNY",
                total_balance: "10.00",
                granted_balance: "0.00",
                topped_up_balance: "10.00",
              },
            ],
          })
        : response({
            status: true,
            data: { available_balance: 5, voucher_balance: 0, cash_balance: 5 },
          }),
    );
    const codexReader = vi.fn(async () => ({
      rateLimits: {
        primary: { usedPercent: 10, windowDurationMins: 300, resetsAt: 1_783_865_430 },
        secondary: null,
        credits: null,
      },
    }));
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ deepseek: "deep-secret", kimi: "kimi-secret" }),
      fetch,
      codexReader,
      now: fixedClock(),
    });
    const inventory = usageInventory(
      userProvider("deepseek", "DeepSeek", "https://api.deepseek.com"),
      userProvider("kimi", "Kimi", "https://api.moonshot.ai/v1"),
    );

    const providerSnapshot = await service.refresh(inventory, {
      source: "provider",
      sourceId: "deepseek",
    });

    expect(providerSnapshot.providers.map((entry) => entry.sourceId)).toEqual(["deepseek"]);
    expect(providerSnapshot.toolAccounts).toEqual([]);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(codexReader).not.toHaveBeenCalled();

    const codexSnapshot = await service.refresh(inventory, {
      source: "tool_account",
      sourceId: "codex",
    });

    expect(codexSnapshot.providers).toEqual([]);
    expect(codexSnapshot.toolAccounts.map((entry) => entry.sourceId)).toEqual(["codex"]);
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(codexReader).toHaveBeenCalledTimes(1);
  });

  it("rejects malformed and unknown refresh targets before querying", async () => {
    const fetch = vi.fn();
    const codexReader = vi.fn();
    const service = new ProviderUsageService({ fetch, codexReader });
    const inventory = usageInventory(
      userProvider("deepseek", "DeepSeek", "https://api.deepseek.com"),
    );

    await expect(
      service.refresh(inventory, {
        source: "provider",
        sourceId: "",
      }),
    ).rejects.toThrow("Invalid Provider usage refresh target");
    await expect(
      service.refresh(inventory, {
        source: "provider",
        sourceId: "missing",
      }),
    ).rejects.toThrow('Provider "missing" was not found');
    await expect(
      service.refresh(inventory, {
        source: "tool_account",
        sourceId: "claude",
      }),
    ).rejects.toThrow('Tool account "claude" was not found');
    await expect(
      service.refresh(inventory, {
        source: "invalid",
        sourceId: "deepseek",
      } as never),
    ).rejects.toThrow("Invalid Provider usage refresh target");

    expect(fetch).not.toHaveBeenCalled();
    expect(codexReader).not.toHaveBeenCalled();
  });

  it("timestamps every queried row, including unsupported, unavailable, and error results", async () => {
    let tick = 0;
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ failing: "failing-secret" }),
      fetch: vi.fn(async () => {
        throw new Error("network failure");
      }),
      codexReader: vi.fn(async () => {
        throw new Error("not signed in");
      }),
      now: () => new Date(Date.UTC(2026, 6, 12, 12, 0, tick++)),
    });

    const snapshot = await service.refresh(
      usageInventory(
        userProvider("unsupported", "Anthropic", "https://api.anthropic.com"),
        userProvider("missing", "DeepSeek", "https://api.deepseek.com"),
        userProvider("failing", "Kimi", "https://api.moonshot.ai/v1"),
      ),
    );

    expect(snapshot.providers.map((entry) => entry.status)).toEqual([
      "unsupported",
      "unavailable",
      "error",
    ]);
    expect(snapshot.toolAccounts[0]?.status).toBe("unavailable");
    const rowTimestamps = [
      ...snapshot.providers.map((entry) => entry.fetchedAt),
      snapshot.toolAccounts[0]?.fetchedAt,
    ];
    expect(
      rowTimestamps.every((timestamp) => /^2026-07-12T12:00:\d{2}\.000Z$/.test(timestamp ?? "")),
    ).toBe(true);
    expect(new Set(rowTimestamps).size).toBe(rowTimestamps.length);
  });

  it("queries a New API account with separate credentials and paginates safe token summaries", async () => {
    const firstPage = Array.from({ length: 100 }, (_, index) => ({
      id: index + 1,
      name: `Token ${index + 1}`,
      key: `masked-credential-${index + 1}`,
      status: index === 0 ? 4 : 1,
      accessed_time: index === 0 ? 1_783_865_430 : 0,
      expired_time: -1,
      remain_quota: 500_000,
      unlimited_quota: false,
      used_quota: 100_000,
    }));
    const fetch = vi.fn(async (url: string, init: { headers: Record<string, string> }) => {
      if (url.endsWith("/api/usage/token/")) {
        return response({
          code: true,
          data: {
            object: "token_usage",
            total_granted: 1_000_000,
            total_used: 250_000,
            total_available: 750_000,
            unlimited_quota: false,
            expires_at: 0,
          },
        });
      }
      if (url.endsWith("/api/status")) {
        expect(init.headers).toEqual({ Accept: "application/json" });
        return response({
          success: true,
          data: {
            quota_per_unit: 500_000,
            quota_display_type: "CNY",
            usd_exchange_rate: 7.2,
            custom_currency_symbol: "¤",
            custom_currency_exchange_rate: 1,
          },
        });
      }
      expect(init.headers).toEqual(
        expect.objectContaining({
          Authorization: "Bearer account-access-secret",
          "New-Api-User": "42",
        }),
      );
      if (url.endsWith("/api/user/self")) {
        return response({
          success: true,
          data: {
            display_name: "Packy account",
            username: "packy-user",
            group: "vip",
            quota: 1_000_000,
            used_quota: 500_000,
            access_token: "must-not-leak",
          },
        });
      }
      if (url.endsWith("/api/token/?p=1&size=100")) {
        return response({
          success: true,
          data: { page: 1, page_size: 100, total: 101, items: firstPage },
        });
      }
      if (url.endsWith("/api/token/?p=2&size=100")) {
        return response({
          success: true,
          data: {
            page: 2,
            page_size: 100,
            total: 101,
            items: [
              {
                id: 101,
                name: "Disabled token",
                key: "masked-credential-101",
                status: 2,
                accessed_time: 0,
                expired_time: 1_784_354_741,
                remain_quota: 0,
                unlimited_quota: true,
                used_quota: 50_000,
              },
            ],
          },
        });
      }
      throw new Error(`Unexpected URL ${url}`);
    });
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({
        packy: "primary-api-secret",
        "packy:new-api-account": "account-access-secret",
      }),
      fetch,
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory({
        ...userProvider("packy", "Packy", "https://www.packyapi.com/v1"),
        usageAdapter: "new_api",
        newApiAccountUserId: "42",
      }),
    );

    expect(fetch).toHaveBeenCalledWith(
      "https://www.packyapi.com/api/usage/token/",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer primary-api-secret" }),
      }),
    );
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(
      expect.arrayContaining([
        "https://www.packyapi.com/api/status",
        "https://www.packyapi.com/api/user/self",
        "https://www.packyapi.com/api/token/?p=1&size=100",
        "https://www.packyapi.com/api/token/?p=2&size=100",
      ]),
    );
    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        status: "ready",
        meters: [
          {
            kind: "balance",
            label: "Account balance",
            currency: "CNY",
            total: "14.40",
          },
          expect.objectContaining({
            kind: "credit",
            label: "API key quota",
            remaining: "750000",
          }),
        ],
        account: expect.objectContaining({
          status: "ready",
          displayName: "Packy account",
          group: "vip",
          balance: { remaining: "14.40", used: "7.20", total: "21.60", unit: "CNY" },
          totalTokens: 101,
          tokens: expect.arrayContaining([
            expect.objectContaining({
              id: "1",
              name: "Token 1",
              status: "exhausted",
              remaining: "CN¥7.20",
              used: "CN¥1.44",
              total: "CN¥8.64",
              lastUsedAt: "2026-07-12T14:10:30.000Z",
            }),
            expect.objectContaining({
              id: "101",
              status: "disabled",
              remaining: "Unlimited",
              total: "Unlimited",
              expiresAt: "2026-07-18T06:05:41.000Z",
            }),
          ]),
        }),
      }),
    );
    expect(snapshot.providers[0]?.account?.tokens).toHaveLength(101);
    expect(JSON.stringify(snapshot)).not.toMatch(
      /primary-api-secret|account-access-secret|masked-credential|must-not-leak/,
    );
  });

  it("keeps primary New API usage ready when the optional account query fails", async () => {
    const fetch = vi.fn(async (url: string) => {
      if (url.endsWith("/api/usage/token/")) {
        return response({
          code: true,
          data: {
            object: "token_usage",
            total_granted: 1_000,
            total_used: 250,
            total_available: 750,
            unlimited_quota: false,
            expires_at: 0,
          },
        });
      }
      if (url.endsWith("/api/status")) {
        return response({
          success: true,
          data: { quota_per_unit: 500_000, quota_display_type: "USD" },
        });
      }
      if (url.endsWith("/api/user/self")) return response({}, 503);
      return response({
        success: true,
        data: { page: 1, page_size: 100, total: 0, items: [] },
      });
    });
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({
        packy: "primary-api-secret",
        "packy:new-api-account": "account-access-secret",
      }),
      fetch,
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory({
        ...userProvider("packy", "Packy", "https://www.packyapi.com"),
        usageAdapter: "new_api",
        newApiAccountUserId: "42",
      }),
    );

    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        status: "ready",
        meters: [expect.objectContaining({ label: "API key quota", remaining: "750" })],
        account: {
          kind: "new_api",
          status: "error",
          tokens: [],
          totalTokens: 0,
          detail: "Usage query failed with HTTP 503.",
        },
      }),
    );
    expect(JSON.stringify(snapshot)).not.toMatch(/primary-api-secret|account-access-secret/);
  });

  it("normalizes Z.AI and MiniMax 5-hour and weekly windows", async () => {
    const fetch = vi.fn(async (url: string) => {
      if (url.includes("bigmodel") || url.includes("z.ai")) {
        return response({
          success: true,
          data: {
            level: "Pro",
            limits: [
              {
                type: "TOKENS_LIMIT",
                unit: 6,
                number: 7,
                percentage: 42,
                nextResetTime: 1_780_848_000_000,
              },
              {
                type: "TOKENS_LIMIT",
                unit: 3,
                number: 5,
                percentage: 8,
                nextResetTime: 1_780_329_600_000,
              },
            ],
          },
        });
      }
      return response({
        model_remains: [
          {
            model_name: "MiniMax-M2.7",
            current_interval_total_count: 1_000,
            current_interval_usage_count: 980,
            current_interval_status: 1,
            current_weekly_total_count: 10_000,
            current_weekly_usage_count: 9_500,
            current_weekly_status: 1,
            end_time: 1_780_329_600_000,
            weekly_end_time: 1_780_848_000_000,
          },
        ],
        base_resp: { status_code: 0, status_msg: "success" },
      });
    });
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ zai: "zai-secret", minimax: "minimax-secret" }),
      env: {},
      fetch,
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory(
        userProvider("zai", "Z.AI", "https://dev.bigmodel.cn/api/anthropic"),
        userProvider("minimax", "MiniMax", "https://api.minimax.io/anthropic"),
      ),
    );

    expect(fetch).toHaveBeenCalledWith(
      "https://dev.bigmodel.cn/api/monitor/usage/quota/limit",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "zai-secret" }),
      }),
    );
    expect(fetch).toHaveBeenCalledWith(
      "https://api.minimax.io/v1/token_plan/remains",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer minimax-secret" }),
      }),
    );
    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        plan: "Pro",
        meters: [
          expect.objectContaining({ id: "five_hour", usedPercent: 8, remainingPercent: 92 }),
          expect.objectContaining({ id: "weekly", usedPercent: 42, remainingPercent: 58 }),
        ],
      }),
    );
    expect(snapshot.providers[1]?.meters).toEqual([
      expect.objectContaining({
        id: "five_hour:MiniMax-M2.7",
        usedPercent: 2,
        remainingPercent: 98,
      }),
      expect.objectContaining({
        id: "weekly:MiniMax-M2.7",
        usedPercent: 5,
        remainingPercent: 95,
      }),
    ]);
  });

  it("parses Kimi Code quota shapes used by the official CLI", async () => {
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ "kimi-code": "kimi-code-secret" }),
      env: {},
      fetch: vi.fn().mockResolvedValue(
        response({
          usage: {
            name: "Weekly limit",
            limit: 1000,
            remaining: 750,
            reset_at: "2026-07-19T00:00:00Z",
          },
          limits: [
            {
              window: { duration: 300, timeUnit: "MINUTE" },
              detail: { limit: "100", used: "10", resetIn: 3_600 },
            },
          ],
        }),
      ),
      includeCodex: false,
      now: fixedClock(),
    });

    const snapshot = await service.refresh(
      usageInventory(userProvider("kimi-code", "Kimi Code", "https://api.kimi.com/coding/v1")),
    );

    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({
        adapterId: "kimi_code",
        status: "ready",
        meters: [
          expect.objectContaining({ id: "weekly", usedPercent: 25, remainingPercent: 75 }),
          expect.objectContaining({
            id: "five_hour",
            usedPercent: 10,
            remainingPercent: 90,
            resetsAt: "2026-07-12T13:00:00.000Z",
          }),
        ],
      }),
    );
  });

  it("reads Codex subscription windows through the app-server response", async () => {
    const service = new ProviderUsageService({
      env: {},
      codexReader: async () => ({
        rateLimits: {
          limitId: "codex",
          primary: { usedPercent: 7, windowDurationMins: 300, resetsAt: 1_783_865_430 },
          secondary: { usedPercent: 47, windowDurationMins: 10_080, resetsAt: 1_784_354_741 },
          credits: { hasCredits: true, unlimited: true, balance: null },
          planType: "pro",
        },
        rateLimitsByLimitId: null,
        rateLimitResetCredits: { availableCount: 2 },
      }),
      now: fixedClock(),
    });

    const snapshot = await service.refresh(usageInventory());

    expect(snapshot.toolAccounts).toEqual([
      expect.objectContaining({
        sourceId: "codex",
        adapterId: "codex_app_server",
        status: "ready",
        plan: "pro",
        meters: [
          expect.objectContaining({ id: "five_hour", remainingPercent: 93 }),
          expect.objectContaining({ id: "weekly", remainingPercent: 53 }),
          expect.objectContaining({ kind: "credit", label: "Credits", remaining: "Unlimited" }),
          expect.objectContaining({ kind: "credit", label: "Full resets", remaining: "2" }),
        ],
      }),
    ]);
  });

  it("rejects lookalike hosts and reports unsupported official account APIs honestly", async () => {
    const fetch = vi.fn();
    const service = new ProviderUsageService({ fetch, includeCodex: false, env: {} });

    const snapshot = await service.refresh(
      usageInventory(
        provider("spoof", "Spoof", "https://api.deepseek.com.evil.test", "SPOOF_KEY"),
        provider("anthropic", "Anthropic", "https://api.anthropic.com", "ANTHROPIC_KEY"),
        provider("opencode", "OpenCode Go", "https://opencode.ai/zen/go", "OPENCODE_KEY"),
      ),
    );

    expect(fetch).not.toHaveBeenCalled();
    expect(snapshot.providers.map((entry) => entry.status)).toEqual([
      "unsupported",
      "unsupported",
      "unsupported",
    ]);
    expect(snapshot.providers[1]?.detail).toContain("Claude Code");
    expect(snapshot.providers[2]?.detail).toContain("no API-key quota endpoint");
  });

  it("isolates Provider failures and never returns secret-bearing errors", async () => {
    const fetch = vi.fn(async (url: string) => {
      if (url.includes("deepseek")) throw new Error("network failed with leaky-secret");
      return response({
        status: true,
        data: { available_balance: 5, voucher_balance: 0, cash_balance: 5 },
      });
    });
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ bad: "leaky-secret", good: "good-secret" }),
      env: {},
      fetch,
      includeCodex: false,
    });

    const snapshot = await service.refresh(
      usageInventory(
        userProvider("bad", "Bad", "https://api.deepseek.com"),
        userProvider("good", "Good", "https://api.moonshot.ai/v1"),
      ),
    );

    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({ status: "error", detail: expect.stringContaining("could not") }),
    );
    expect(snapshot.providers[1]).toEqual(expect.objectContaining({ status: "ready" }));
    expect(JSON.stringify(snapshot)).not.toMatch(/leaky-secret|good-secret/);
  });

  it("stops reading streamed usage responses at the byte limit", async () => {
    const text = vi.fn(async () => "should not be read");
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ deepseek: "deep-secret" }),
      env: { DEEPSEEK_API_KEY: "ambient-must-not-use" },
      fetch: vi.fn(async () => ({
        body: new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(new Uint8Array(256 * 1024 + 1));
            controller.close();
          },
        }),
        ok: true,
        status: 200,
        text,
      })),
      includeCodex: false,
    });

    const snapshot = await service.refresh(
      usageInventory(userProvider("deepseek", "DeepSeek", "https://api.deepseek.com")),
    );

    expect(snapshot.providers[0]).toEqual(
      expect.objectContaining({ status: "error", detail: expect.stringContaining("too large") }),
    );
    expect(text).not.toHaveBeenCalled();
    expect(JSON.stringify(snapshot)).not.toContain("deep-secret");
  });

  it("rejects ambient env and unbound keychain Provider usage credentials", async () => {
    const fetch = vi.fn();
    const authStore = memoryAuthStore({ victim: "victim-secret" });
    const service = new ProviderUsageService({
      authStore,
      env: { DEEPSEEK_API_KEY: "deep-secret" },
      fetch,
      includeCodex: false,
    });
    const inventory = usageInventory(
      provider("extension.deepseek", "Extension", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
      {
        ...userProvider("extension.local", "Alias", "https://api.deepseek.com"),
        secretRef: { source: "local_keychain" as const, key: "victim" },
      },
    );
    inventory.modelCatalog.userProviderIds = ["extension.local"];

    const snapshot = await service.refresh(inventory);

    expect(fetch).not.toHaveBeenCalled();
    expect(authStore.get).not.toHaveBeenCalled();
    expect(snapshot.providers.map((entry) => entry.status)).toEqual(["unsupported", "unsupported"]);
    expect(JSON.stringify(snapshot)).not.toMatch(/deep-secret|victim-secret/);
  });

  it("rejects redirects and body-level authorization failures per Provider", async () => {
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ zai: "zai-secret", deepseek: "deep-secret" }),
      env: {},
      fetch: vi.fn(async (url: string) =>
        url.includes("bigmodel") ? response({ code: 401, success: false }, 200) : response({}, 302),
      ),
      includeCodex: false,
    });

    const snapshot = await service.refresh(
      usageInventory(
        userProvider("zai", "Z.AI", "https://open.bigmodel.cn/api/anthropic"),
        userProvider("deepseek", "DeepSeek", "https://api.deepseek.com"),
      ),
    );

    expect(snapshot.providers).toEqual([
      expect.objectContaining({
        status: "unavailable",
        detail: expect.stringContaining("authorized"),
      }),
      expect.objectContaining({
        status: "unsupported",
        detail: expect.stringContaining("redirect"),
      }),
    ]);
    expect(JSON.stringify(snapshot)).not.toMatch(/zai-secret|deep-secret/);
  });

  it("preserves MiniMax boosted and unlimited quota semantics", async () => {
    const service = new ProviderUsageService({
      authStore: memoryAuthStore({ minimax: "minimax-secret" }),
      fetch: vi.fn(async () =>
        response({
          model_remains: [
            {
              model_name: "MiniMax-M2.7",
              current_interval_remaining_percent: 80,
              current_interval_status: 1,
              current_weekly_remaining_percent: 100,
              current_weekly_status: 3,
              weekly_boost_permille: 1_500,
            },
            {
              model_name: "MiniMax-M2.7-fast",
              current_interval_remaining_percent: 100,
              current_interval_status: 1,
              current_weekly_remaining_percent: 100,
              current_weekly_status: 1,
              weekly_boost_permille: 1_500,
            },
          ],
        }),
      ),
      includeCodex: false,
    });

    const snapshot = await service.refresh(
      usageInventory(userProvider("minimax", "MiniMax", "https://api.minimax.io/anthropic")),
    );

    expect(snapshot.providers[0]?.meters).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "credit",
          label: "MiniMax-M2.7 Weekly",
          remaining: "Unlimited",
        }),
        expect.objectContaining({
          id: "weekly:MiniMax-M2.7-fast",
          remainingPercent: 150,
          usedPercent: 0,
        }),
      ]),
    );
  });

  it("frames the Codex app-server protocol and filters unrelated Provider secrets", async () => {
    const previousSecret = process.env.DEEPSEEK_API_KEY;
    process.env.DEEPSEEK_API_KEY = "must-not-reach-codex";
    const fixture = String.raw`
      let input = "";
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (chunk) => {
        input += chunk;
        const lines = input.split("\n");
        input = lines.pop() || "";
        for (const line of lines) {
          if (!line) continue;
          const request = JSON.parse(line);
          if (request.id === 1) {
            process.stdout.write(JSON.stringify({ id: 1, result: {} }) + "\n");
          }
          if (request.id === 2) {
            process.stdout.write(JSON.stringify({
              id: 2,
              result: {
                inheritedSecret: process.env.DEEPSEEK_API_KEY || null,
                method: request.method,
                params: request.params,
              },
            }) + "\n");
          }
        }
      });
    `;

    try {
      await expect(queryCodexAppServer(process.execPath, 2_000, ["-e", fixture])).resolves.toEqual({
        inheritedSecret: null,
        method: "account/rateLimits/read",
      });
      await expect(
        queryCodexAppServerRequest(
          "model/list",
          { cursor: null, limit: 100 },
          process.execPath,
          2_000,
          ["-e", fixture],
        ),
      ).resolves.toEqual({
        inheritedSecret: null,
        method: "model/list",
        params: { cursor: null, limit: 100 },
      });
      await expect(
        queryCodexAppServer(process.execPath, 2_000, [
          "-e",
          `process.stdout.write("x".repeat(${512 * 1024 + 1}))`,
        ]),
      ).rejects.toThrow("too much output");
    } finally {
      if (previousSecret === undefined) process.env.DEEPSEEK_API_KEY = undefined;
      else process.env.DEEPSEEK_API_KEY = previousSecret;
    }
  });
});

function usageInventory(...providers: ProviderProfile[]) {
  return {
    ...createExtensionInventory([]),
    providers,
    modelCatalog: {
      manualModelIds: [],
      userProviderIds: providers
        .filter(
          (candidate) =>
            candidate.secretRef?.source === "local_keychain" &&
            candidate.secretRef.key === candidate.id,
        )
        .map((candidate) => candidate.id),
      providers: [],
    },
  };
}

function provider(id: string, label: string, baseUrl: string, secretKey: string) {
  return {
    id,
    label,
    kind: "openai_chat" as const,
    baseUrl,
    authMode: "api_key" as const,
    secretRef: { source: "env" as const, key: secretKey },
  };
}

function userProvider(id: string, label: string, baseUrl: string) {
  return {
    id,
    label,
    kind: "openai_chat" as const,
    baseUrl,
    authMode: "api_key" as const,
    secretRef: { source: "local_keychain" as const, key: id },
  };
}

function memoryAuthStore(secrets: Record<string, string>) {
  return {
    get: vi.fn(async (key: string) => secrets[key]),
    set: vi.fn(async (key: string, value: string) => {
      secrets[key] = value;
    }),
    delete: vi.fn(async (key: string) => {
      delete secrets[key];
    }),
  };
}

function response(payload: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(payload),
  };
}

function fixedClock(value = "2026-07-12T12:00:00.000Z") {
  return () => new Date(value);
}
