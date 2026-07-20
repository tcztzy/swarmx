import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { ModelTokenUsage } from "@swarmx/core";
import { z } from "zod";

const DEFAULT_COOLDOWN_MS = 5 * 60 * 60 * 1_000;
const MAX_PROVIDER_COOLDOWN_MS = 30 * 24 * 60 * 60 * 1_000;

const StoredProviderKeyStateSchema = z.object({
  requestCount: z.number().int().nonnegative().default(0),
  inputTokens: z.number().int().nonnegative().default(0),
  outputTokens: z.number().int().nonnegative().default(0),
  reasoningTokens: z.number().int().nonnegative().default(0),
  cachedInputTokens: z.number().int().nonnegative().default(0),
  totalTokens: z.number().int().nonnegative().default(0),
  lastUsedAt: z.string().datetime().optional(),
  cooldownUntil: z.string().datetime().optional(),
});

const ProviderKeyUsageDocumentSchema = z.object({
  schemaVersion: z.literal(1),
  keys: z.record(StoredProviderKeyStateSchema).default({}),
});

type ProviderKeyUsageDocument = z.infer<typeof ProviderKeyUsageDocumentSchema>;

export interface ProviderKeySlot {
  id: string;
  label: string;
  enabled: boolean;
}

export interface ProviderKeyCandidate {
  id: string;
  value: string;
}

export interface ProviderKeyUsageSummary extends ProviderKeySlot {
  status: "ready" | "cooling" | "disabled";
  requestCount: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cachedInputTokens: number;
  totalTokens: number;
  lastUsedAt?: string;
  cooldownUntil?: string;
}

export interface ProviderKeyUsageStoreOptions {
  path?: string;
  now?: () => Date;
}

export function isOpenCodeGoBaseUrl(value: string | undefined): boolean {
  if (!value) return false;
  try {
    const url = new URL(value);
    const pathname = url.pathname.replace(/\/+$/, "");
    return (
      url.protocol === "https:" &&
      url.hostname.toLowerCase() === "opencode.ai" &&
      !url.port &&
      (pathname === "/zen/go" || pathname === "/zen/go/v1")
    );
  } catch {
    return false;
  }
}

export class ProviderKeyUsageStore {
  readonly path: string;
  private readonly clock: () => Date;
  private mutations: Promise<void> = Promise.resolve();

  constructor(options: ProviderKeyUsageStoreOptions = {}) {
    this.path = options.path ?? join(homedir(), ".swarmx", "provider-key-usage.json");
    this.clock = options.now ?? (() => new Date());
  }

  currentTime(): Date {
    return this.clock();
  }

  async recordSuccess(providerId: string, keyId: string, usage?: ModelTokenUsage): Promise<void> {
    await this.update((document) => {
      const storageKey = providerKeyStateId(providerId, keyId);
      const state = StoredProviderKeyStateSchema.parse(document.keys[storageKey] ?? {});
      document.keys[storageKey] = {
        ...state,
        requestCount: state.requestCount + 1,
        inputTokens: state.inputTokens + (usage?.inputTokens ?? 0),
        outputTokens: state.outputTokens + (usage?.outputTokens ?? 0),
        reasoningTokens: state.reasoningTokens + (usage?.reasoningTokens ?? 0),
        cachedInputTokens: state.cachedInputTokens + (usage?.cachedInputTokens ?? 0),
        totalTokens: state.totalTokens + (usage?.totalTokens ?? 0),
        lastUsedAt: this.clock().toISOString(),
        cooldownUntil: undefined,
      };
    });
  }

  async recordQuotaExhausted(
    providerId: string,
    keyId: string,
    cooldownUntil: string,
  ): Promise<void> {
    const parsedCooldown = z.string().datetime().parse(cooldownUntil);
    await this.update((document) => {
      const storageKey = providerKeyStateId(providerId, keyId);
      const state = StoredProviderKeyStateSchema.parse(document.keys[storageKey] ?? {});
      document.keys[storageKey] = { ...state, cooldownUntil: parsedCooldown };
    });
  }

  async reset(providerId: string, keyId: string): Promise<void> {
    await this.update((document) => {
      const storageKey = providerKeyStateId(providerId, keyId);
      const state = document.keys[storageKey];
      if (state) document.keys[storageKey] = { ...state, cooldownUntil: undefined };
    });
  }

  async remove(providerId: string, keyId: string): Promise<void> {
    await this.update((document) => {
      delete document.keys[providerKeyStateId(providerId, keyId)];
    });
  }

  async removeProvider(providerId: string): Promise<void> {
    await this.update((document) => {
      const prefix = `${encodeURIComponent(providerId)}/`;
      for (const key of Object.keys(document.keys)) {
        if (key.startsWith(prefix)) delete document.keys[key];
      }
    });
  }

  async summaries(
    providerId: string,
    slots: readonly ProviderKeySlot[],
  ): Promise<ProviderKeyUsageSummary[]> {
    await this.mutations;
    const document = await this.read();
    const now = this.clock().getTime();
    return slots.map((slot) => {
      const state = StoredProviderKeyStateSchema.parse(
        document.keys[providerKeyStateId(providerId, slot.id)] ?? {},
      );
      const cooling = state.cooldownUntil ? new Date(state.cooldownUntil).getTime() > now : false;
      return {
        ...slot,
        status: slot.enabled ? (cooling ? "cooling" : "ready") : "disabled",
        requestCount: state.requestCount,
        inputTokens: state.inputTokens,
        outputTokens: state.outputTokens,
        reasoningTokens: state.reasoningTokens,
        cachedInputTokens: state.cachedInputTokens,
        totalTokens: state.totalTokens,
        ...(state.lastUsedAt ? { lastUsedAt: state.lastUsedAt } : {}),
        ...(cooling && state.cooldownUntil ? { cooldownUntil: state.cooldownUntil } : {}),
      };
    });
  }

  private async update(mutator: (document: ProviderKeyUsageDocument) => void): Promise<void> {
    const operation = this.mutations.then(async () => {
      const document = await this.read();
      mutator(document);
      await writeJsonAtomic(this.path, ProviderKeyUsageDocumentSchema.parse(document));
    });
    this.mutations = operation.catch(() => undefined);
    return operation;
  }

  private async read(): Promise<ProviderKeyUsageDocument> {
    try {
      return ProviderKeyUsageDocumentSchema.parse(JSON.parse(await readFile(this.path, "utf8")));
    } catch (error) {
      if (isNodeError(error, "ENOENT")) return { schemaVersion: 1, keys: {} };
      throw error;
    }
  }
}

export interface ProviderKeyAttemptObservation {
  markOutput(): void;
  recordUsage(usage: ModelTokenUsage): void;
}

export interface ProviderKeyPoolExecutionOptions<T> {
  providerId: string;
  routingKey: string;
  candidates: readonly ProviderKeyCandidate[];
  run(candidate: ProviderKeyCandidate, observation: ProviderKeyAttemptObservation): Promise<T>;
}

export class ProviderKeyPoolRuntime {
  constructor(private readonly usageStore: ProviderKeyUsageStore) {}

  async execute<T>(options: ProviderKeyPoolExecutionOptions<T>): Promise<T> {
    if (options.candidates.length === 0) {
      throw new Error(`Provider "${options.providerId}" has no configured API keys.`);
    }
    const summaries = await this.usageStore.summaries(
      options.providerId,
      options.candidates.map((candidate, index) => ({
        id: candidate.id,
        label: `Key ${index + 1}`,
        enabled: true,
      })),
    );
    const readyIds = new Set(
      summaries.filter((summary) => summary.status === "ready").map((summary) => summary.id),
    );
    const candidates = rotateCandidates(
      options.candidates.filter((candidate) => readyIds.has(candidate.id)),
      options.routingKey,
    );
    if (candidates.length === 0) throw allKeysCoolingError(options.providerId, summaries);

    for (const candidate of candidates) {
      let outputObserved = false;
      const usages: ModelTokenUsage[] = [];
      try {
        const result = await options.run(candidate, {
          markOutput: () => {
            outputObserved = true;
          },
          recordUsage: (usage) => usages.push(usage),
        });
        await this.usageStore.recordSuccess(
          options.providerId,
          candidate.id,
          usages.length > 0 ? mergeUsage(usages) : undefined,
        );
        return result;
      } catch (error) {
        const exhaustion = providerQuotaExhaustion(error, this.usageStore.currentTime());
        if (!exhaustion || outputObserved) throw error;
        await this.usageStore.recordQuotaExhausted(
          options.providerId,
          candidate.id,
          exhaustion.cooldownUntil,
        );
      }
    }

    throw allKeysCoolingError(
      options.providerId,
      await this.usageStore.summaries(
        options.providerId,
        options.candidates.map((candidate, index) => ({
          id: candidate.id,
          label: `Key ${index + 1}`,
          enabled: true,
        })),
      ),
    );
  }
}

export function providerQuotaExhaustion(
  error: unknown,
  now = new Date(),
): { cooldownUntil: string } | undefined {
  const records = errorRecords(error);
  const statuses = records.flatMap((record) => numberValue(record.status));
  const codes = records.flatMap((record) => stringValue(record.code)).map(normalizeErrorCode);
  const messages = records.flatMap((record) => stringValue(record.message));
  const quotaCode = codes.some((code) =>
    [
      "billing_hard_limit_reached",
      "credit_balance_exhausted",
      "balance_not_enough",
      "insufficient_credits",
      "insufficient_quota",
      "payment_required",
      "quota_exceeded",
      "usage_limit_exceeded",
    ].includes(code),
  );
  const quotaMessage = messages.some((message) =>
    /(insufficient\s+(?:balance|credits?|quota)|balance\s+is\s+insufficient|(?:credit|quota)\s+(?:balance\s+)?(?:exhausted|exceeded)|(?:no|not\s+enough|out\s+of)\s+credits?|usage\s+limit\s+(?:has\s+been\s+)?(?:exceeded|reached)|billing\s+(?:hard\s+)?limit)/iu.test(
      message,
    ),
  );
  if (!statuses.includes(402) && !quotaCode && !quotaMessage) return undefined;

  const resetAt = records.flatMap((record) => headerCooldown(record.headers, now))[0];
  const fallback = new Date(now.getTime() + DEFAULT_COOLDOWN_MS);
  return { cooldownUntil: (resetAt ?? fallback).toISOString() };
}

function mergeUsage(usages: readonly ModelTokenUsage[]): ModelTokenUsage {
  return {
    inputTokens: usages.reduce((sum, usage) => sum + usage.inputTokens, 0),
    outputTokens: usages.reduce((sum, usage) => sum + usage.outputTokens, 0),
    reasoningTokens: usages.reduce((sum, usage) => sum + usage.reasoningTokens, 0),
    cachedInputTokens: usages.reduce((sum, usage) => sum + usage.cachedInputTokens, 0),
    totalTokens: usages.reduce((sum, usage) => sum + usage.totalTokens, 0),
    estimated: usages.some((usage) => usage.estimated),
  };
}

function rotateCandidates(
  candidates: readonly ProviderKeyCandidate[],
  routingKey: string,
): ProviderKeyCandidate[] {
  if (candidates.length < 2) return [...candidates];
  const start = stableHash(routingKey) % candidates.length;
  return [...candidates.slice(start), ...candidates.slice(0, start)];
}

function stableHash(value: string): number {
  let hash = 2_166_136_261;
  for (const byte of Buffer.from(value, "utf8")) {
    hash ^= byte;
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

function allKeysCoolingError(
  providerId: string,
  summaries: readonly ProviderKeyUsageSummary[],
): Error {
  const retryAt = summaries
    .flatMap((summary) => (summary.cooldownUntil ? [summary.cooldownUntil] : []))
    .sort()[0];
  return new Error(
    retryAt
      ? `All API keys are cooling for Provider "${providerId}" until ${retryAt}.`
      : `Provider "${providerId}" has no available API keys.`,
  );
}

function errorRecords(error: unknown): Record<string, unknown>[] {
  const records: Record<string, unknown>[] = [];
  const pending: unknown[] = [error];
  const seen = new Set<unknown>();
  while (pending.length > 0 && records.length < 8) {
    const value = pending.shift();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    if (value instanceof Error) {
      const record = value as Error & Record<string, unknown>;
      records.push({ ...record, message: value.message });
      pending.push(record.cause, record.error, record.response);
    } else if (typeof value === "object" && !Array.isArray(value)) {
      const record = value as Record<string, unknown>;
      records.push(record);
      pending.push(record.cause, record.error, record.response);
    }
  }
  return records;
}

function headerCooldown(value: unknown, now: Date): Date[] {
  const retryAfter = headerValue(value, "retry-after");
  if (!retryAfter) return [];
  const seconds = Number(retryAfter);
  const candidate = Number.isFinite(seconds)
    ? new Date(now.getTime() + seconds * 1_000)
    : new Date(retryAfter);
  const duration = candidate.getTime() - now.getTime();
  return Number.isFinite(candidate.getTime()) &&
    duration > 0 &&
    duration <= MAX_PROVIDER_COOLDOWN_MS
    ? [candidate]
    : [];
}

function headerValue(value: unknown, name: string): string | undefined {
  if (value instanceof Headers) return value.get(name) ?? undefined;
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  const direct = record[name] ?? record[name.toLowerCase()] ?? record[name.toUpperCase()];
  return typeof direct === "string" && direct.trim() ? direct.trim() : undefined;
}

function normalizeErrorCode(code: string): string {
  return code
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/gu, "_");
}

function stringValue(value: unknown): string[] {
  return typeof value === "string" && value.trim() ? [value] : [];
}

function numberValue(value: unknown): number[] {
  return typeof value === "number" && Number.isFinite(value) ? [value] : [];
}

function providerKeyStateId(providerId: string, keyId: string): string {
  return `${encodeURIComponent(providerId)}/${encodeURIComponent(keyId)}`;
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.tmp-${process.pid}-${Date.now()}`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
  await rename(temporaryPath, path);
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}
