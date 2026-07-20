import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import { type ExtensionInventory, type ProviderProfile, SWARMX_VERSION } from "@swarmx/core";
import { z } from "zod";
import type { ProviderAuthStore } from "./provider-auth.js";
import { newApiAccountCredentialKey } from "./provider-auth.js";
import { type ProviderKeyUsageSummary, isOpenCodeGoBaseUrl } from "./provider-key-pool.js";

const DEFAULT_USAGE_TIMEOUT_MS = 8_000;
const MAX_USAGE_RESPONSE_BYTES = 256 * 1024;
const MAX_CODEX_OUTPUT_BYTES = 512 * 1024;
const MAX_NEW_API_TOKEN_PAGES = 20;
const NEW_API_TOKEN_PAGE_SIZE = 100;

const ProviderKeyUsageSummarySchema = z.object({
  id: z.string().min(1),
  label: z.string().min(1),
  enabled: z.boolean(),
  status: z.enum(["ready", "cooling", "disabled"]),
  requestCount: z.number().int().nonnegative(),
  inputTokens: z.number().int().nonnegative(),
  outputTokens: z.number().int().nonnegative(),
  reasoningTokens: z.number().int().nonnegative(),
  cachedInputTokens: z.number().int().nonnegative(),
  totalTokens: z.number().int().nonnegative(),
  lastUsedAt: z.string().datetime().optional(),
  cooldownUntil: z.string().datetime().optional(),
});

export type ProviderUsageStatus = "ready" | "unsupported" | "unavailable" | "error";

export interface ProviderBalanceUsageMeter {
  kind: "balance";
  label: string;
  currency: string;
  total: string;
  granted?: string;
  toppedUp?: string;
}

export interface ProviderWindowUsageMeter {
  kind: "window";
  id: string;
  label: string;
  usedPercent: number;
  remainingPercent: number;
  resetsAt?: string;
}

export interface ProviderCreditUsageMeter {
  kind: "credit";
  label: string;
  remaining: string;
  unit: string;
}

export type ProviderUsageMeter =
  | ProviderBalanceUsageMeter
  | ProviderWindowUsageMeter
  | ProviderCreditUsageMeter;

export interface ProviderApiTokenSummary {
  id: string;
  name: string;
  status: "active" | "disabled" | "exhausted" | "expired" | "unknown";
  remaining: string;
  used?: string;
  total?: string;
  lastUsedAt?: string;
  expiresAt?: string;
}

export interface ProviderAccountSummary {
  kind: "new_api";
  status: "ready" | "unavailable" | "error";
  displayName?: string;
  group?: string;
  balance?: {
    remaining: string;
    used: string;
    total: string;
    unit: string;
  };
  tokens: ProviderApiTokenSummary[];
  totalTokens: number;
  detail?: string;
}

export interface ProviderUsageEntry {
  source: "provider" | "tool_account";
  sourceId: string;
  providerProfileId?: string;
  label: string;
  adapterId: string;
  status: ProviderUsageStatus;
  meters: ProviderUsageMeter[];
  fetchedAt?: string;
  detail?: string;
  plan?: string;
  account?: ProviderAccountSummary;
  keys?: ProviderKeyUsageSummary[];
}

export interface ProviderUsageSnapshot {
  fetchedAt: string;
  providers: ProviderUsageEntry[];
  toolAccounts: ProviderUsageEntry[];
}

export interface ProviderUsageRefreshTarget {
  source: "provider" | "tool_account";
  sourceId: string;
}

const ProviderUsageRefreshTargetSchema = z
  .object({
    source: z.enum(["provider", "tool_account"]),
    sourceId: z.string().trim().min(1).max(256),
  })
  .strict();

export interface ProviderUsageServiceOptions {
  authStore?: ProviderAuthStore;
  env?: NodeJS.ProcessEnv;
  fetch?: ProviderUsageFetch;
  now?: () => Date;
  timeoutMs?: number;
  codexCommand?: string;
  codexReader?: () => Promise<unknown>;
  includeCodex?: boolean;
}

interface ProviderUsageHttpResponse {
  body?: ReadableStream<Uint8Array> | null;
  ok: boolean;
  status: number;
  text(): Promise<string>;
}

type ProviderUsageFetch = (
  url: string,
  init: {
    method: "GET";
    headers: Record<string, string>;
    redirect: "manual";
    signal: AbortSignal;
  },
) => Promise<ProviderUsageHttpResponse>;

type ProviderUsageAdapterId = "deepseek" | "kimi" | "kimi_code" | "zai" | "minimax" | "new_api";

interface ProviderUsageAdapter {
  id: ProviderUsageAdapterId;
  url: string;
  headers(secret: string): Record<string, string>;
  parse(
    payload: unknown,
    now: Date,
  ): { meters: ProviderUsageMeter[]; detail?: string; plan?: string };
}

const DeepSeekBalanceSchema = z.object({
  is_available: z.boolean(),
  balance_infos: z.array(
    z.object({
      currency: z.string().min(1),
      total_balance: z.string(),
      granted_balance: z.string(),
      topped_up_balance: z.string(),
    }),
  ),
});

const DecimalValueSchema = z.union([z.string(), z.number().finite()]).transform(String);
const KimiBalanceSchema = z.object({
  status: z.boolean().optional(),
  data: z.object({
    available_balance: DecimalValueSchema,
    voucher_balance: DecimalValueSchema,
    cash_balance: DecimalValueSchema,
  }),
});

const KimiCodeUsageSchema = z
  .object({
    usage: z.record(z.string(), z.unknown()).optional(),
    limits: z.array(z.unknown()).optional(),
  })
  .passthrough();

const ZaiUsageSchema = z.object({
  code: z.union([z.string(), z.number()]).optional(),
  success: z.boolean().optional(),
  data: z
    .object({
      level: z.string().optional(),
      limits: z.array(
        z
          .object({
            type: z.string(),
            unit: z.number().int().optional(),
            percentage: z.number().finite(),
            nextResetTime: z.number().int().optional(),
          })
          .passthrough(),
      ),
    })
    .optional(),
});

const MiniMaxUsageSchema = z.object({
  base_resp: z
    .object({
      status_code: z.number().int(),
      status_msg: z.string().optional(),
    })
    .optional(),
  model_remains: z.array(
    z
      .object({
        model_name: z.string(),
        current_interval_total_count: z.number().finite().optional(),
        current_interval_usage_count: z.number().finite().optional(),
        current_interval_remaining_percent: z.number().finite().optional(),
        current_interval_status: z.number().int().optional(),
        current_weekly_total_count: z.number().finite().optional(),
        current_weekly_usage_count: z.number().finite().optional(),
        current_weekly_remaining_percent: z.number().finite().optional(),
        current_weekly_status: z.number().int().optional(),
        weekly_boost_permille: z.number().finite().optional(),
        end_time: z.number().int().optional(),
        weekly_end_time: z.number().int().optional(),
      })
      .passthrough(),
  ),
});

const NewApiQuotaValueSchema = z
  .union([z.number().finite(), z.string().regex(/^-?\d+(?:\.\d+)?$/)])
  .transform(String);
const NewApiTimestampSchema = z
  .union([z.number().int().nonnegative(), z.string().regex(/^\d+$/)])
  .transform(Number);
const NewApiTokenUsageSchema = z.object({
  code: z.boolean(),
  data: z
    .object({
      object: z.literal("token_usage"),
      total_granted: NewApiQuotaValueSchema,
      total_used: NewApiQuotaValueSchema,
      total_available: NewApiQuotaValueSchema,
      unlimited_quota: z.boolean(),
      expires_at: NewApiTimestampSchema,
    })
    .optional(),
});

const NewApiNumericValueSchema = z
  .union([z.number().finite(), z.string().regex(/^-?\d+(?:\.\d+)?$/)])
  .transform(Number);
const NewApiStatusResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    quota_per_unit: NewApiNumericValueSchema.pipe(z.number().positive()).default(500_000),
    quota_display_type: z.enum(["USD", "CNY", "TOKENS", "CUSTOM"]).default("USD"),
    usd_exchange_rate: NewApiNumericValueSchema.pipe(z.number().positive()).default(1),
    custom_currency_symbol: z.string().max(16).default("¤"),
    custom_currency_exchange_rate: NewApiNumericValueSchema.pipe(z.number().positive()).default(1),
  }),
});
const NewApiUserSelfResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    display_name: z.string().optional(),
    username: z.string().optional(),
    group: z.string().optional(),
    quota: NewApiNumericValueSchema,
    used_quota: NewApiNumericValueSchema,
  }),
});
const NewApiAccountTokenSchema = z.object({
  id: z.union([z.number().int(), z.string().min(1)]).transform(String),
  name: z.string(),
  status: z.number().int(),
  accessed_time: NewApiTimestampSchema,
  expired_time: z.union([z.number().int(), z.string().regex(/^-?\d+$/)]).transform(Number),
  remain_quota: NewApiNumericValueSchema,
  unlimited_quota: z.boolean(),
  used_quota: NewApiNumericValueSchema,
});
const NewApiTokenListResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    page: z.number().int().positive(),
    page_size: z.number().int().positive(),
    total: z.number().int().nonnegative(),
    items: z.array(NewApiAccountTokenSchema),
  }),
});

type NewApiStatus = z.infer<typeof NewApiStatusResponseSchema>["data"];
type NewApiAccountToken = z.infer<typeof NewApiAccountTokenSchema>;

const CodexRateLimitWindowSchema = z.object({
  usedPercent: z.number().finite(),
  windowDurationMins: z.number().int().positive().nullable().optional(),
  resetsAt: z.number().int().nullable().optional(),
});

const CodexRateLimitSchema = z.object({
  limitId: z.string().nullable().optional(),
  primary: CodexRateLimitWindowSchema.nullable().optional(),
  secondary: CodexRateLimitWindowSchema.nullable().optional(),
  planType: z.string().nullable().optional(),
  credits: z
    .object({
      hasCredits: z.boolean(),
      unlimited: z.boolean(),
      balance: z.string().nullable().optional(),
    })
    .nullable()
    .optional(),
});

const CodexRateLimitsResponseSchema = z.object({
  rateLimits: CodexRateLimitSchema,
  rateLimitsByLimitId: z.record(CodexRateLimitSchema).nullable().optional(),
  rateLimitResetCredits: z
    .object({
      availableCount: z.number().int().nonnegative(),
    })
    .nullable()
    .optional(),
});

export class ProviderUsageService {
  private readonly authStore?: ProviderAuthStore;
  private readonly env: NodeJS.ProcessEnv;
  private readonly fetch: ProviderUsageFetch;
  private readonly now: () => Date;
  private readonly timeoutMs: number;
  private readonly codexReader: () => Promise<unknown>;
  private readonly includeCodex: boolean;

  constructor(options: ProviderUsageServiceOptions = {}) {
    this.authStore = options.authStore;
    this.env = options.env ?? process.env;
    this.fetch = options.fetch ?? ((url, init) => fetch(url, init));
    this.now = options.now ?? (() => new Date());
    this.timeoutMs = options.timeoutMs ?? DEFAULT_USAGE_TIMEOUT_MS;
    this.codexReader =
      options.codexReader ??
      (() => queryCodexAppServer(options.codexCommand ?? "codex", this.timeoutMs));
    this.includeCodex = options.includeCodex ?? true;
  }

  async refresh(
    inventory: ExtensionInventory,
    target?: ProviderUsageRefreshTarget,
  ): Promise<ProviderUsageSnapshot> {
    let refreshTarget: ProviderUsageRefreshTarget | undefined;
    if (target !== undefined) {
      const parsed = ProviderUsageRefreshTargetSchema.safeParse(target);
      if (!parsed.success) throw new Error("Invalid Provider usage refresh target.");
      refreshTarget = parsed.data;
    }
    const userProviderIds = providerUsageUserProviderIds(inventory);
    const providersToQuery = refreshTarget
      ? refreshTarget.source === "provider"
        ? inventory.providers.filter((provider) => provider.id === refreshTarget.sourceId)
        : []
      : inventory.providers;
    if (refreshTarget?.source === "provider" && providersToQuery.length === 0) {
      throw new Error(`Provider "${refreshTarget.sourceId}" was not found.`);
    }
    const queryCodex = refreshTarget
      ? refreshTarget.source === "tool_account" && refreshTarget.sourceId === "codex"
      : this.includeCodex;
    if (refreshTarget?.source === "tool_account" && (!queryCodex || !this.includeCodex)) {
      throw new Error(`Tool account "${refreshTarget.sourceId}" was not found.`);
    }
    const [providers, codex] = await Promise.all([
      Promise.all(
        providersToQuery.map((provider) => this.queryProvider(provider, userProviderIds)),
      ),
      queryCodex ? this.queryCodex() : Promise.resolve(undefined),
    ]);
    return {
      fetchedAt: this.now().toISOString(),
      providers,
      toolAccounts: codex ? [codex] : [],
    };
  }

  private async queryProvider(
    provider: ProviderProfile,
    userProviderIds: ReadonlySet<string>,
  ): Promise<ProviderUsageEntry> {
    if (isOpenCodeGoBaseUrl(provider.baseUrl)) {
      const keys = providerKeyUsageSummaries(provider);
      return providerEntry(provider, "opencode_go_local", "ready", [], {
        detail:
          "OpenCode Go has no official usage endpoint. Counts are observed locally; quota errors temporarily cool only the affected key.",
        fetchedAt: this.now().toISOString(),
        keys,
      });
    }
    const detected = providerUsageAdapter(provider);
    if ("unsupported" in detected) {
      return providerEntry(provider, detected.adapterId, "unsupported", [], {
        detail: detected.unsupported,
        fetchedAt: this.now().toISOString(),
      });
    }

    if (!providerUsageCredentialAllowed(provider, userProviderIds)) {
      return providerEntry(provider, detected.id, "unsupported", [], {
        detail: "Automatic usage is disabled for extension-owned or unbound credential references.",
        fetchedAt: this.now().toISOString(),
      });
    }

    const secret = await this.resolveSecret(provider);
    if (!secret) {
      return providerEntry(provider, detected.id, "unavailable", [], {
        detail: "A configured Provider credential is required to query usage.",
        fetchedAt: this.now().toISOString(),
      });
    }

    try {
      const payload = await this.fetchJson(detected.url, detected.headers(secret));
      const parsed = detected.parse(payload, this.now());
      const account =
        detected.id === "new_api" ? await this.queryNewApiAccount(provider) : undefined;
      const accountMeter = newApiAccountBalanceMeter(account);
      return providerEntry(
        provider,
        detected.id,
        "ready",
        [...(accountMeter ? [accountMeter] : []), ...parsed.meters],
        {
          fetchedAt: this.now().toISOString(),
          detail: parsed.detail,
          plan: parsed.plan,
          account,
        },
      );
    } catch (error) {
      const failure = publicUsageFailure(error);
      return providerEntry(provider, detected.id, failure.status, [], {
        detail: failure.detail,
        fetchedAt: this.now().toISOString(),
      });
    }
  }

  private async queryCodex(): Promise<ProviderUsageEntry> {
    try {
      const response = CodexRateLimitsResponseSchema.parse(await this.codexReader());
      const limits = response.rateLimitsByLimitId?.codex ?? response.rateLimits;
      const meters: ProviderUsageMeter[] = [limits.primary, limits.secondary]
        .filter((window): window is z.infer<typeof CodexRateLimitWindowSchema> => !!window)
        .sort(
          (left, right) =>
            (left.windowDurationMins ?? Number.MAX_SAFE_INTEGER) -
            (right.windowDurationMins ?? Number.MAX_SAFE_INTEGER),
        )
        .map((window, index) => codexWindowMeter(window, index));
      if (
        limits.credits &&
        (limits.credits.unlimited ||
          (limits.credits.balance !== undefined && limits.credits.balance !== null))
      ) {
        meters.push({
          kind: "credit",
          label: "Credits",
          remaining: limits.credits.unlimited ? "Unlimited" : (limits.credits.balance ?? "0"),
          unit: "credit",
        });
      }
      const resetCredits = response.rateLimitResetCredits?.availableCount ?? 0;
      if (resetCredits > 0) {
        meters.push({
          kind: "credit",
          label: "Full resets",
          remaining: String(resetCredits),
          unit: resetCredits === 1 ? "reset" : "resets",
        });
      }
      if (meters.length === 0) {
        return toolAccountEntry("codex", "Codex", "codex_app_server", "unavailable", [], {
          detail: "Codex returned no rate-limit windows for the current account.",
          fetchedAt: this.now().toISOString(),
        });
      }
      return toolAccountEntry("codex", "Codex", "codex_app_server", "ready", meters, {
        fetchedAt: this.now().toISOString(),
        plan: limits.planType ?? undefined,
      });
    } catch {
      return toolAccountEntry("codex", "Codex", "codex_app_server", "unavailable", [], {
        detail: "Sign in with Codex and ensure the codex app-server is available.",
        fetchedAt: this.now().toISOString(),
      });
    }
  }

  private async resolveSecret(provider: ProviderProfile): Promise<string | undefined> {
    const ref = provider.secretRef;
    if (!ref) return undefined;
    if (ref.source === "env") return this.env[ref.key]?.trim() || undefined;
    if (ref.source === "local_keychain") {
      if (!this.authStore) return undefined;
      try {
        return (await this.authStore.get(ref.key))?.trim() || undefined;
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  private async queryNewApiAccount(
    provider: ProviderProfile,
  ): Promise<ProviderAccountSummary | undefined> {
    const userId = stringProperty(provider, "newApiAccountUserId");
    if (!userId) return undefined;
    if (!/^[1-9]\d*$/.test(userId)) {
      return {
        kind: "new_api",
        status: "error",
        tokens: [],
        totalTokens: 0,
        detail: "The configured New API User ID is invalid.",
      };
    }
    if (!this.authStore) {
      return unavailableNewApiAccount(
        "A New API account access token is required to query the account wallet.",
      );
    }

    let accessToken: string | undefined;
    try {
      accessToken = (await this.authStore.get(newApiAccountCredentialKey(provider.id)))?.trim();
    } catch {
      return unavailableNewApiAccount(
        "The New API account access token is unavailable in secure storage.",
      );
    }
    if (!accessToken) {
      return unavailableNewApiAccount(
        "A New API account access token is required to query the account wallet.",
      );
    }

    let origin: string;
    try {
      const baseUrl = new URL(provider.baseUrl ?? "");
      if (baseUrl.protocol !== "https:" || baseUrl.username || baseUrl.password) {
        throw new Error("unsafe origin");
      }
      origin = baseUrl.origin;
    } catch {
      return {
        kind: "new_api",
        status: "error",
        tokens: [],
        totalTokens: 0,
        detail: "New API account queries require a credential-free HTTPS origin.",
      };
    }

    const accountHeaders = newApiAccountHeaders(accessToken, userId);
    try {
      const [statusPayload, userPayload, tokenResult] = await Promise.all([
        this.fetchJson(new URL("/api/status", origin).toString(), {
          Accept: "application/json",
        }),
        this.fetchJson(new URL("/api/user/self", origin).toString(), accountHeaders),
        this.queryNewApiTokens(origin, accountHeaders),
      ]);
      const statusResponse = NewApiStatusResponseSchema.parse(statusPayload);
      const userResponse = NewApiUserSelfResponseSchema.parse(userPayload);
      if (!statusResponse.success || !userResponse.success) {
        throw new UsageQueryFailure("error", "The New API account query was rejected.");
      }

      const remaining = newApiQuotaDisplay(userResponse.data.quota, statusResponse.data);
      const used = newApiQuotaDisplay(userResponse.data.used_quota, statusResponse.data);
      const total = newApiQuotaDisplay(
        userResponse.data.quota + userResponse.data.used_quota,
        statusResponse.data,
      );
      const tokens = tokenResult.tokens.map((token) =>
        newApiTokenSummary(token, statusResponse.data),
      );
      return {
        kind: "new_api",
        status: "ready",
        displayName: userResponse.data.display_name || userResponse.data.username,
        group: userResponse.data.group,
        balance: {
          remaining: remaining.amount,
          used: used.amount,
          total: total.amount,
          unit: remaining.unit,
        },
        tokens,
        totalTokens: tokenResult.total,
        ...(tokens.length < tokenResult.total
          ? { detail: `Showing the first ${tokens.length} of ${tokenResult.total} API tokens.` }
          : {}),
      };
    } catch (error) {
      const failure = publicUsageFailure(error);
      return {
        kind: "new_api",
        status: failure.status === "error" ? "error" : "unavailable",
        tokens: [],
        totalTokens: 0,
        detail: failure.detail,
      };
    }
  }

  private async queryNewApiTokens(
    origin: string,
    headers: Record<string, string>,
  ): Promise<{ tokens: NewApiAccountToken[]; total: number }> {
    const tokens = new Map<string, NewApiAccountToken>();
    let total = 0;
    for (let page = 1; page <= MAX_NEW_API_TOKEN_PAGES; page += 1) {
      const payload = await this.fetchJson(
        new URL(`/api/token/?p=${page}&size=${NEW_API_TOKEN_PAGE_SIZE}`, origin).toString(),
        headers,
      );
      const response = NewApiTokenListResponseSchema.parse(payload);
      if (!response.success) {
        throw new UsageQueryFailure("error", "The New API token list query was rejected.");
      }
      total = Math.max(total, response.data.total);
      for (const token of response.data.items) tokens.set(token.id, token);
      if (
        tokens.size >= total ||
        response.data.items.length < NEW_API_TOKEN_PAGE_SIZE ||
        response.data.page * response.data.page_size >= total
      ) {
        break;
      }
    }
    return { tokens: [...tokens.values()], total };
  }

  private async fetchJson(url: string, headers: Record<string, string>): Promise<unknown> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetch(url, {
        method: "GET",
        headers,
        redirect: "manual",
        signal: controller.signal,
      });
      if (response.status >= 300 && response.status < 400) {
        throw new UsageQueryFailure("unsupported", "Usage endpoints may not redirect credentials.");
      }
      if (response.status === 401 || response.status === 403) {
        throw new UsageQueryFailure(
          "unavailable",
          "The Provider credential is not authorized for usage queries.",
        );
      }
      if (response.status === 404) {
        throw new UsageQueryFailure(
          "unsupported",
          "This Provider account does not expose the supported usage endpoint.",
        );
      }
      if (response.status === 429) {
        throw new UsageQueryFailure("error", "The Provider rate-limited the usage query.");
      }
      if (!response.ok) {
        throw new UsageQueryFailure("error", `Usage query failed with HTTP ${response.status}.`);
      }
      const body = await readBoundedResponseBody(response);
      try {
        return JSON.parse(body) as unknown;
      } catch {
        throw new UsageQueryFailure("error", "The Provider returned invalid usage data.");
      }
    } catch (error) {
      if (controller.signal.aborted) {
        throw new UsageQueryFailure("error", `Usage query timed out after ${this.timeoutMs}ms.`);
      }
      if (error instanceof UsageQueryFailure) throw error;
      throw new UsageQueryFailure("error", "The Provider usage query could not be completed.");
    } finally {
      clearTimeout(timeout);
    }
  }
}

async function readBoundedResponseBody(response: ProviderUsageHttpResponse): Promise<string> {
  if (!response.body) {
    const body = await response.text();
    if (Buffer.byteLength(body, "utf8") > MAX_USAGE_RESPONSE_BYTES) {
      throw new UsageQueryFailure("error", "The Provider usage response was too large.");
    }
    return body;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let byteCount = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      byteCount += value.byteLength;
      if (byteCount > MAX_USAGE_RESPONSE_BYTES) {
        await reader.cancel().catch(() => undefined);
        throw new UsageQueryFailure("error", "The Provider usage response was too large.");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  return Buffer.concat(
    chunks.map((chunk) => Buffer.from(chunk)),
    byteCount,
  ).toString("utf8");
}

function providerUsageUserProviderIds(inventory: ExtensionInventory): ReadonlySet<string> {
  const catalog = (inventory as ExtensionInventory & { modelCatalog?: unknown }).modelCatalog;
  if (!isRecord(catalog) || !Array.isArray(catalog.userProviderIds)) return new Set();
  return new Set(
    catalog.userProviderIds.filter((id): id is string => typeof id === "string" && id.length > 0),
  );
}

function providerUsageCredentialAllowed(
  provider: ProviderProfile,
  userProviderIds: ReadonlySet<string>,
): boolean {
  const ref = provider.secretRef;
  if (!ref) return false;
  return (
    ref.source === "local_keychain" && userProviderIds.has(provider.id) && ref.key === provider.id
  );
}

function providerUsageAdapter(
  provider: ProviderProfile,
): ProviderUsageAdapter | { adapterId: string; unsupported: string } {
  let baseUrl: URL;
  try {
    baseUrl = new URL(provider.baseUrl ?? "");
  } catch {
    return {
      adapterId: "unsupported",
      unsupported: "No safe usage adapter is available for this Provider.",
    };
  }
  if (baseUrl.protocol !== "https:" || baseUrl.username || baseUrl.password) {
    return {
      adapterId: "unsupported",
      unsupported: "Usage queries require a credential-free official HTTPS origin.",
    };
  }

  const host = baseUrl.hostname.toLowerCase();
  if (stringProperty(provider, "usageAdapter") === "new_api") {
    return {
      id: "new_api",
      url: new URL("/api/usage/token/", baseUrl.origin).toString(),
      headers: bearerHeaders,
      parse: parseNewApiTokenUsage,
    };
  }
  if (host === "api.deepseek.com") {
    return {
      id: "deepseek",
      url: "https://api.deepseek.com/user/balance",
      headers: bearerHeaders,
      parse: parseDeepSeekBalance,
    };
  }
  if (host === "api.moonshot.cn" || host === "api.moonshot.ai") {
    return {
      id: "kimi",
      url: `https://${host}/v1/users/me/balance`,
      headers: bearerHeaders,
      parse: (payload) => parseKimiBalance(payload, host.endsWith(".cn") ? "CNY" : "USD"),
    };
  }
  if (host === "api.kimi.com") {
    return {
      id: "kimi_code",
      url: "https://api.kimi.com/coding/v1/usages",
      headers: bearerHeaders,
      parse: parseKimiCodeUsage,
    };
  }
  if (host === "api.z.ai" || host === "open.bigmodel.cn" || host === "dev.bigmodel.cn") {
    return {
      id: "zai",
      url: `https://${host}/api/monitor/usage/quota/limit`,
      headers: (secret) => ({
        Authorization: secret,
        Accept: "application/json",
        "Accept-Language": "en-US,en",
        "Content-Type": "application/json",
      }),
      parse: parseZaiUsage,
    };
  }
  if (host === "api.minimaxi.com" || host === "www.minimaxi.com") {
    return {
      id: "minimax",
      url: "https://api.minimaxi.com/v1/token_plan/remains",
      headers: bearerHeaders,
      parse: parseMiniMaxUsage,
    };
  }
  if (host === "api.minimax.io" || host === "www.minimax.io") {
    return {
      id: "minimax",
      url: "https://api.minimax.io/v1/token_plan/remains",
      headers: bearerHeaders,
      parse: parseMiniMaxUsage,
    };
  }
  if (host === "api.openai.com") {
    return {
      adapterId: "openai_api",
      unsupported:
        "OpenAI API keys do not expose Codex subscription quota; Codex login usage is shown separately.",
    };
  }
  if (host === "api.anthropic.com") {
    return {
      adapterId: "anthropic_api",
      unsupported:
        "Standard Anthropic API keys do not expose Claude Code 5-hour or weekly subscription quota.",
    };
  }
  if (host === "generativelanguage.googleapis.com" || host.endsWith(".aiplatform.googleapis.com")) {
    return {
      adapterId: "google_gemini",
      unsupported: "Gemini API keys have no public endpoint for current remaining account quota.",
    };
  }
  if (host === "opencode.ai") {
    return {
      adapterId: "opencode",
      unsupported:
        "OpenCode Go and Zen publish usage in their console but no API-key quota endpoint.",
    };
  }
  return {
    adapterId: "unsupported",
    unsupported: "No safe usage adapter is available for this Provider.",
  };
}

function bearerHeaders(secret: string): Record<string, string> {
  return {
    Authorization: `Bearer ${secret}`,
    Accept: "application/json",
    "Content-Type": "application/json",
  };
}

function newApiAccountHeaders(accessToken: string, userId: string): Record<string, string> {
  return {
    Authorization: `Bearer ${accessToken}`,
    "New-Api-User": userId,
    Accept: "application/json",
    "Content-Type": "application/json",
  };
}

function unavailableNewApiAccount(detail: string): ProviderAccountSummary {
  return {
    kind: "new_api",
    status: "unavailable",
    tokens: [],
    totalTokens: 0,
    detail,
  };
}

function newApiAccountBalanceMeter(
  account: ProviderAccountSummary | undefined,
): ProviderBalanceUsageMeter | undefined {
  if (account?.status !== "ready" || !account.balance) return undefined;
  return {
    kind: "balance",
    label: "Account balance",
    currency: account.balance.unit,
    total: account.balance.remaining,
  };
}

function newApiTokenSummary(
  token: NewApiAccountToken,
  status: NewApiStatus,
): ProviderApiTokenSummary {
  const remaining = newApiQuotaDisplay(token.remain_quota, status);
  const used = newApiQuotaDisplay(token.used_quota, status);
  const total = newApiQuotaDisplay(token.remain_quota + token.used_quota, status);
  return {
    id: token.id,
    name: token.name,
    status: newApiTokenStatus(token.status),
    remaining: token.unlimited_quota ? "Unlimited" : remaining.inline,
    used: used.inline,
    total: token.unlimited_quota ? "Unlimited" : total.inline,
    ...(token.accessed_time > 0 ? { lastUsedAt: isoFromSeconds(token.accessed_time) } : {}),
    ...(token.expired_time > 0 ? { expiresAt: isoFromSeconds(token.expired_time) } : {}),
  };
}

function newApiTokenStatus(status: number): ProviderApiTokenSummary["status"] {
  if (status === 1) return "active";
  if (status === 2) return "disabled";
  if (status === 3) return "expired";
  if (status === 4) return "exhausted";
  return "unknown";
}

function newApiQuotaDisplay(
  quota: number,
  status: NewApiStatus,
): { amount: string; inline: string; unit: string } {
  const type = status.quota_display_type;
  if (type === "TOKENS") {
    const amount = formatNewApiAmount(quota, false);
    return { amount, inline: `${amount} tokens`, unit: "tokens" };
  }

  const rate =
    type === "CNY"
      ? status.usd_exchange_rate
      : type === "CUSTOM"
        ? status.custom_currency_exchange_rate
        : 1;
  const amount = formatNewApiAmount((quota / status.quota_per_unit) * rate, true);
  const symbol = type === "USD" ? "$" : type === "CNY" ? "CN¥" : status.custom_currency_symbol;
  const unit = type === "CUSTOM" ? status.custom_currency_symbol : type;
  return { amount, inline: `${symbol}${amount}`, unit };
}

function formatNewApiAmount(value: number, currency: boolean): string {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: currency ? 2 : 0,
    maximumFractionDigits: currency && Math.abs(value) < 0.01 ? 4 : currency ? 2 : 0,
    useGrouping: true,
  }).format(value);
}

function parseDeepSeekBalance(payload: unknown) {
  const parsed = DeepSeekBalanceSchema.parse(payload);
  return {
    meters: parsed.balance_infos.map(
      (balance): ProviderBalanceUsageMeter => ({
        kind: "balance",
        label: `${balance.currency} balance`,
        currency: balance.currency,
        total: balance.total_balance,
        granted: balance.granted_balance,
        toppedUp: balance.topped_up_balance,
      }),
    ),
    ...(!parsed.is_available
      ? { detail: "The balance is currently unavailable for API calls." }
      : {}),
  };
}

function parseKimiBalance(payload: unknown, currency: string) {
  const parsed = KimiBalanceSchema.parse(payload);
  if (parsed.status === false) throw new Error("Kimi balance query failed.");
  return {
    meters: [
      {
        kind: "balance" as const,
        label: "Available balance",
        currency,
        total: parsed.data.available_balance,
        granted: parsed.data.voucher_balance,
        toppedUp: parsed.data.cash_balance,
      },
    ],
  };
}

function parseKimiCodeUsage(payload: unknown, now: Date) {
  const parsed = KimiCodeUsageSchema.parse(payload);
  const meters: ProviderUsageMeter[] = [];
  if (parsed.usage) {
    const meter = kimiCodeWindowMeter(parsed.usage, "Weekly", "weekly", now);
    if (meter) meters.push(meter);
  }
  for (const [index, item] of (parsed.limits ?? []).entries()) {
    if (!isRecord(item)) continue;
    const detail = isRecord(item.detail) ? item.detail : item;
    const window = isRecord(item.window) ? item.window : {};
    const duration = numericValue(window.duration ?? item.duration ?? detail.duration);
    const timeUnit = String(
      window.timeUnit ?? item.timeUnit ?? detail.timeUnit ?? "",
    ).toUpperCase();
    const defaultLabel =
      duration === 300 && timeUnit.includes("MINUTE")
        ? "5-hour"
        : duration === 7 && timeUnit.includes("DAY")
          ? "Weekly"
          : `Limit ${index + 1}`;
    const id =
      defaultLabel === "5-hour"
        ? "five_hour"
        : defaultLabel === "Weekly"
          ? "weekly"
          : `limit_${index + 1}`;
    const meter = kimiCodeWindowMeter(detail, defaultLabel, id, now);
    if (meter) meters.push(meter);
  }
  if (meters.length === 0) throw new Error("Kimi Code usage response has no limits.");
  return { meters };
}

function kimiCodeWindowMeter(
  data: Record<string, unknown>,
  defaultLabel: string,
  id: string,
  now: Date,
): ProviderWindowUsageMeter | undefined {
  const limit = numericValue(data.limit);
  const remaining = numericValue(data.remaining);
  const used =
    numericValue(data.used) ??
    (limit !== undefined && remaining !== undefined ? limit - remaining : undefined);
  if (limit === undefined || used === undefined || limit <= 0) return undefined;
  const label = String(data.name ?? data.title ?? defaultLabel);
  const reset = data.reset_at ?? data.resetAt ?? data.reset_time ?? data.resetTime;
  const relativeReset = numericValue(data.reset_in ?? data.resetIn ?? data.ttl ?? data.window);
  const resetsAt =
    absoluteResetTime(reset) ??
    (relativeReset !== undefined
      ? new Date(now.getTime() + relativeReset * 1_000).toISOString()
      : undefined);
  return windowMeter(id, label, (used / limit) * 100, resetsAt);
}

function parseZaiUsage(payload: unknown) {
  const parsed = ZaiUsageSchema.parse(payload);
  const code = String(parsed.code ?? "");
  if (parsed.success === false || (code && code !== "0" && code !== "200")) {
    throw new UsageQueryFailure(
      code === "401" || code === "403" ? "unavailable" : "error",
      code === "401" || code === "403"
        ? "The Provider credential is not authorized for usage queries."
        : "The Provider rejected the usage query.",
    );
  }
  if (!parsed.data) throw new Error("Z.AI usage response has no limits.");
  const windows = parsed.data.limits.filter((limit) => limit.type.toUpperCase() === "TOKENS_LIMIT");
  const classified = new Map<string, (typeof windows)[number]>();
  const unclassified: typeof windows = [];
  for (const window of windows) {
    if (window.unit === 3 && !classified.has("five_hour")) classified.set("five_hour", window);
    else if (window.unit === 6 && !classified.has("weekly")) classified.set("weekly", window);
    else unclassified.push(window);
  }
  unclassified.sort(
    (left, right) =>
      (left.nextResetTime ?? Number.MAX_SAFE_INTEGER) -
      (right.nextResetTime ?? Number.MAX_SAFE_INTEGER),
  );
  for (const window of unclassified) {
    if (!classified.has("five_hour")) classified.set("five_hour", window);
    else if (!classified.has("weekly")) classified.set("weekly", window);
  }
  const meters: ProviderUsageMeter[] = [];
  for (const [id, label] of [
    ["five_hour", "5-hour"],
    ["weekly", "Weekly"],
  ] as const) {
    const window = classified.get(id);
    if (!window) continue;
    meters.push(
      windowMeter(id, label, window.percentage, isoFromMilliseconds(window.nextResetTime)),
    );
  }
  return { meters, plan: parsed.data.level };
}

function parseMiniMaxUsage(payload: unknown) {
  const parsed = MiniMaxUsageSchema.parse(payload);
  if (parsed.base_resp && parsed.base_resp.status_code !== 0) {
    throw new Error("MiniMax usage query failed.");
  }
  const meters: ProviderUsageMeter[] = [];
  for (const quota of parsed.model_remains) {
    const prefix = quota.model_name === "general" ? "" : `${quota.model_name} `;
    const currentRemaining = miniMaxRemainingPercent(
      quota.current_interval_remaining_percent,
      quota.current_interval_usage_count,
      quota.current_interval_total_count,
    );
    if (quota.current_interval_status === 3) {
      meters.push({
        kind: "credit",
        label: `${prefix}5-hour`,
        remaining: "Unlimited",
        unit: "quota",
      });
    } else if (currentRemaining !== undefined) {
      meters.push(
        remainingWindowMeter(
          `five_hour:${quota.model_name}`,
          `${prefix}5-hour`,
          currentRemaining,
          isoFromMilliseconds(quota.end_time),
        ),
      );
    }

    const weeklyRemaining = miniMaxRemainingPercent(
      quota.current_weekly_remaining_percent,
      quota.current_weekly_usage_count,
      quota.current_weekly_total_count,
      quota.weekly_boost_permille,
    );
    if (quota.current_weekly_status === 3) {
      meters.push({
        kind: "credit",
        label: `${prefix}Weekly`,
        remaining: "Unlimited",
        unit: "quota",
      });
    } else if (weeklyRemaining !== undefined) {
      meters.push(
        remainingWindowMeter(
          `weekly:${quota.model_name}`,
          `${prefix}Weekly`,
          weeklyRemaining,
          isoFromMilliseconds(quota.weekly_end_time),
        ),
      );
    }
  }
  if (meters.length === 0) throw new Error("MiniMax usage response has no quota meters.");
  return { meters };
}

function parseNewApiTokenUsage(payload: unknown) {
  const parsed = NewApiTokenUsageSchema.parse(payload);
  if (!parsed.code) {
    throw new UsageQueryFailure("error", "The New API usage query was rejected.");
  }
  if (!parsed.data) throw new Error("New API usage response has no quota data.");
  const quota = parsed.data;
  const expiry = quota.expires_at > 0 ? isoFromSeconds(quota.expires_at) : undefined;
  return {
    meters: [
      {
        kind: "credit" as const,
        label: "API key quota",
        remaining: quota.unlimited_quota ? "Unlimited" : quota.total_available,
        unit: quota.unlimited_quota ? "quota" : "quota points",
      },
    ],
    detail: [
      `Used ${quota.total_used} of ${quota.total_granted} quota points.`,
      expiry ? `Expires ${expiry}.` : undefined,
    ]
      .filter((value): value is string => !!value)
      .join(" "),
  };
}

function miniMaxRemainingPercent(
  explicitPercent: number | undefined,
  remainingCount: number | undefined,
  totalCount: number | undefined,
  boostPermille = 1_000,
): number | undefined {
  const base =
    explicitPercent ??
    (remainingCount !== undefined && totalCount !== undefined && totalCount > 0
      ? (remainingCount / totalCount) * 100
      : undefined);
  if (base === undefined) return undefined;
  return Math.max(0, Math.min(200, base * (Math.max(0, boostPermille) / 1_000)));
}

function codexWindowMeter(
  window: z.infer<typeof CodexRateLimitWindowSchema>,
  index: number,
): ProviderWindowUsageMeter {
  const minutes = window.windowDurationMins;
  const id = minutes === 300 ? "five_hour" : minutes === 10_080 ? "weekly" : `window_${index}`;
  const label =
    minutes === 300 ? "5-hour" : minutes === 10_080 ? "Weekly" : `${minutes ?? "Rate"} min`;
  return windowMeter(
    id,
    label,
    window.usedPercent,
    window.resetsAt ? new Date(window.resetsAt * 1_000).toISOString() : undefined,
  );
}

function windowMeter(
  id: string,
  label: string,
  usedPercent: number,
  resetsAt?: string,
): ProviderWindowUsageMeter {
  const normalizedUsedPercent = Math.max(0, Math.min(100, usedPercent));
  return {
    kind: "window",
    id,
    label,
    usedPercent: normalizedUsedPercent,
    remainingPercent: 100 - normalizedUsedPercent,
    ...(resetsAt ? { resetsAt } : {}),
  };
}

function remainingWindowMeter(
  id: string,
  label: string,
  remainingPercent: number,
  resetsAt?: string,
): ProviderWindowUsageMeter {
  const normalizedRemainingPercent = Math.max(0, Math.min(200, remainingPercent));
  return {
    kind: "window",
    id,
    label,
    usedPercent: Math.max(0, Math.min(100, 100 - normalizedRemainingPercent)),
    remainingPercent: normalizedRemainingPercent,
    ...(resetsAt ? { resetsAt } : {}),
  };
}

function isoFromMilliseconds(value: number | undefined): string | undefined {
  if (value === undefined) return undefined;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString();
}

function isoFromSeconds(value: number): string | undefined {
  return isoFromMilliseconds(value * 1_000);
}

function stringProperty(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) return undefined;
  const property = value[key];
  return typeof property === "string" && property.length > 0 ? property : undefined;
}

function numericValue(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string" || !value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function absoluteResetTime(value: unknown): string | undefined {
  if (typeof value === "string" && value.trim()) {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) return date.toISOString();
  }
  const numeric = numericValue(value);
  if (numeric === undefined) return undefined;
  const milliseconds = numeric > 10_000_000_000 ? numeric : numeric * 1_000;
  return isoFromMilliseconds(milliseconds);
}

function providerEntry(
  provider: ProviderProfile,
  adapterId: string,
  status: ProviderUsageStatus,
  meters: ProviderUsageMeter[],
  extra: Pick<ProviderUsageEntry, "account" | "detail" | "fetchedAt" | "keys" | "plan"> = {},
): ProviderUsageEntry {
  return {
    source: "provider",
    sourceId: provider.id,
    providerProfileId: provider.id,
    label: provider.label,
    adapterId,
    status,
    meters,
    ...definedEntryExtra(extra),
  };
}

function toolAccountEntry(
  sourceId: string,
  label: string,
  adapterId: string,
  status: ProviderUsageStatus,
  meters: ProviderUsageMeter[],
  extra: Pick<ProviderUsageEntry, "detail" | "fetchedAt" | "plan"> = {},
): ProviderUsageEntry {
  return {
    source: "tool_account",
    sourceId,
    label,
    adapterId,
    status,
    meters,
    ...definedEntryExtra(extra),
  };
}

function definedEntryExtra(
  extra: Pick<ProviderUsageEntry, "account" | "detail" | "fetchedAt" | "keys" | "plan">,
): Pick<ProviderUsageEntry, "account" | "detail" | "fetchedAt" | "keys" | "plan"> {
  return {
    ...(extra.account ? { account: extra.account } : {}),
    ...(extra.detail ? { detail: extra.detail } : {}),
    ...(extra.fetchedAt ? { fetchedAt: extra.fetchedAt } : {}),
    ...(extra.keys ? { keys: extra.keys } : {}),
    ...(extra.plan ? { plan: extra.plan } : {}),
  };
}

function providerKeyUsageSummaries(provider: ProviderProfile): ProviderKeyUsageSummary[] {
  const input = (provider as Record<string, unknown>).runtimeKeyUsage;
  if (!Array.isArray(input)) return [];
  return input.flatMap((entry) => {
    const parsed = ProviderKeyUsageSummarySchema.safeParse(entry);
    return parsed.success ? [parsed.data] : [];
  });
}

class UsageQueryFailure extends Error {
  constructor(
    readonly status: Exclude<ProviderUsageStatus, "ready">,
    readonly detail: string,
  ) {
    super(detail);
  }
}

function publicUsageFailure(error: unknown): {
  status: Exclude<ProviderUsageStatus, "ready">;
  detail: string;
} {
  if (error instanceof UsageQueryFailure) return error;
  if (error instanceof z.ZodError) {
    return { status: "error", detail: "The Provider returned unsupported usage data." };
  }
  return { status: "error", detail: "The Provider usage query could not be completed." };
}

export async function queryCodexAppServer(
  command = "codex",
  timeoutMs = DEFAULT_USAGE_TIMEOUT_MS,
  args: readonly string[] = ["app-server", "--stdio"],
): Promise<unknown> {
  return queryCodexAppServerRequest("account/rateLimits/read", undefined, command, timeoutMs, args);
}

export async function queryCodexAppServerRequest(
  method: string,
  params: unknown,
  command = "codex",
  timeoutMs = DEFAULT_USAGE_TIMEOUT_MS,
  args: readonly string[] = ["app-server", "--stdio"],
): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, [...args], {
      env: codexSubprocessEnvironment(process.env),
      shell: false,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const decoder = new StringDecoder("utf8");
    let settled = false;
    let outputBytes = 0;
    let stdoutBuffer = "";
    const timeout = setTimeout(
      () => finish(new Error("Codex app-server request timed out.")),
      timeoutMs,
    );

    const finish = (error?: Error, value?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      child.stdin.end();
      if (!child.killed) child.kill();
      if (error) reject(error);
      else resolve(value);
    };

    const processLine = (line: string) => {
      if (settled || !line.trim()) return;
      let message: unknown;
      try {
        message = JSON.parse(line) as unknown;
      } catch {
        return;
      }
      if (!isRecord(message)) return;
      if (message.id === 1) {
        if ("error" in message) {
          finish(new Error("Codex app-server initialization failed."));
          return;
        }
        if ("result" in message) {
          child.stdin.write(`${JSON.stringify({ method: "initialized", params: {} })}\n`);
          child.stdin.write(`${JSON.stringify({ id: 2, method, params })}\n`);
        }
        return;
      }
      if (message.id === 2) {
        if ("error" in message) finish(new Error(`Codex app-server ${method} request failed.`));
        else finish(undefined, message.result);
      }
    };

    const processBufferedLines = () => {
      while (!settled) {
        const newline = stdoutBuffer.indexOf("\n");
        if (newline < 0) return;
        const line = stdoutBuffer.slice(0, newline).replace(/\r$/, "");
        stdoutBuffer = stdoutBuffer.slice(newline + 1);
        processLine(line);
      }
    };

    child.on("error", () => finish(new Error("Codex app-server is unavailable.")));
    child.on("close", () => {
      if (!settled) finish(new Error("Codex app-server closed before returning usage."));
    });
    child.stdin.on("error", () => finish(new Error("Codex app-server input closed.")));
    child.stderr.resume();
    child.stdout.on("data", (chunk: Buffer) => {
      outputBytes += chunk.byteLength;
      if (outputBytes > MAX_CODEX_OUTPUT_BYTES) {
        finish(new Error("Codex app-server returned too much output."));
        return;
      }
      stdoutBuffer += decoder.write(chunk);
      processBufferedLines();
    });
    child.stdout.on("end", () => {
      stdoutBuffer += decoder.end();
      processBufferedLines();
      if (!settled && stdoutBuffer.trim()) processLine(stdoutBuffer);
    });

    child.stdin.write(
      `${JSON.stringify({
        id: 1,
        method: "initialize",
        params: {
          clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
          capabilities: null,
        },
      })}\n`,
    );
  });
}

function codexSubprocessEnvironment(source: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const key of [
    "APPDATA",
    "CODEX_HOME",
    "ComSpec",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LOCALAPPDATA",
    "LOGNAME",
    "NODE_EXTRA_CA_CERTS",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SystemRoot",
    "TERM",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "WINDIR",
    "XDG_CONFIG_HOME",
  ]) {
    if (source[key] !== undefined) environment[key] = source[key];
  }
  return environment;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
