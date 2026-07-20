import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ProviderKeyPoolRuntime,
  ProviderKeyUsageStore,
  providerQuotaExhaustion,
} from "./provider-key-pool.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("ProviderKeyUsageStore", () => {
  it("V484 persists only normalized local per-key usage and cooldown state", async () => {
    const path = await usagePath();
    const store = new ProviderKeyUsageStore({
      path,
      now: () => new Date("2026-07-20T08:00:00.000Z"),
    });

    await store.recordSuccess("opencode-go", "primary", {
      inputTokens: 100,
      outputTokens: 25,
      reasoningTokens: 5,
      cachedInputTokens: 40,
      totalTokens: 130,
      estimated: false,
    });
    await store.recordQuotaExhausted("opencode-go", "secondary", "2026-07-20T13:00:00.000Z");

    expect(
      await store.summaries("opencode-go", [
        { id: "primary", label: "Key 1", enabled: true },
        { id: "secondary", label: "Key 2", enabled: true },
      ]),
    ).toEqual([
      expect.objectContaining({
        id: "primary",
        label: "Key 1",
        status: "ready",
        requestCount: 1,
        totalTokens: 130,
      }),
      expect.objectContaining({
        id: "secondary",
        status: "cooling",
        cooldownUntil: "2026-07-20T13:00:00.000Z",
      }),
    ]);
    const persisted = await readFile(path, "utf8");
    expect(persisted).not.toMatch(/sk-|quota exceeded|Authorization/i);
  });
});

describe("providerQuotaExhaustion", () => {
  it("V485 classifies explicit quota failures but not a generic rate limit", () => {
    const now = new Date("2026-07-20T08:00:00.000Z");

    expect(
      providerQuotaExhaustion({ status: 429, error: { code: "insufficient_quota" } }, now),
    ).toEqual({ cooldownUntil: "2026-07-20T13:00:00.000Z" });
    expect(providerQuotaExhaustion({ status: 402 }, now)).toEqual({
      cooldownUntil: "2026-07-20T13:00:00.000Z",
    });
    expect(
      providerQuotaExhaustion({ status: 429, message: "Too many requests" }, now),
    ).toBeUndefined();
    expect(
      providerQuotaExhaustion({ status: 429, message: "Account balance is insufficient" }, now),
    ).toEqual({ cooldownUntil: "2026-07-20T13:00:00.000Z" });
  });

  it("uses bounded Provider reset metadata when present", () => {
    const headers = new Headers({ "retry-after": "120" });
    expect(
      providerQuotaExhaustion(
        { status: 429, code: "usage_limit_exceeded", headers },
        new Date("2026-07-20T08:00:00.000Z"),
      ),
    ).toEqual({ cooldownUntil: "2026-07-20T08:02:00.000Z" });
  });
});

describe("ProviderKeyPoolRuntime", () => {
  it("V485 switches keys after a pre-output quota error and records the successful key", async () => {
    const store = new ProviderKeyUsageStore({
      path: await usagePath(),
      now: () => new Date("2026-07-20T08:00:00.000Z"),
    });
    const runtime = new ProviderKeyPoolRuntime(store);
    const attempts: string[] = [];

    const result = await runtime.execute({
      providerId: "opencode-go",
      routingKey: "session-one",
      candidates: [
        { id: "primary", value: "sk-primary" },
        { id: "secondary", value: "sk-secondary" },
      ],
      run: async (candidate, observation) => {
        attempts.push(candidate.id);
        if (attempts.length === 1) {
          throw Object.assign(new Error("usage limit exceeded"), { status: 429 });
        }
        observation.recordUsage({
          inputTokens: 10,
          outputTokens: 4,
          reasoningTokens: 0,
          cachedInputTokens: 0,
          totalTokens: 14,
          estimated: false,
        });
        return "completed";
      },
    });

    expect(result).toBe("completed");
    expect(attempts).toHaveLength(2);
    const summaries = await store.summaries("opencode-go", [
      { id: "primary", label: "Key 1", enabled: true },
      { id: "secondary", label: "Key 2", enabled: true },
    ]);
    expect(summaries.find((entry) => entry.id === attempts[0])?.status).toBe("cooling");
    expect(summaries.find((entry) => entry.id === attempts[1])?.requestCount).toBe(1);
  });

  it("V485 does not replay after observable output", async () => {
    const runtime = new ProviderKeyPoolRuntime(
      new ProviderKeyUsageStore({ path: await usagePath() }),
    );
    const run = vi.fn(async (_candidate, observation) => {
      observation.markOutput();
      throw Object.assign(new Error("quota exceeded"), { status: 429 });
    });

    await expect(
      runtime.execute({
        providerId: "opencode-go",
        routingKey: "session-two",
        candidates: [
          { id: "primary", value: "sk-primary" },
          { id: "secondary", value: "sk-secondary" },
        ],
        run,
      }),
    ).rejects.toThrow("quota exceeded");
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("V486 reports an actionable error when every key is cooling", async () => {
    const store = new ProviderKeyUsageStore({
      path: await usagePath(),
      now: () => new Date("2026-07-20T08:00:00.000Z"),
    });
    await store.recordQuotaExhausted("opencode-go", "primary", "2026-07-20T13:00:00.000Z");
    const runtime = new ProviderKeyPoolRuntime(store);

    await expect(
      runtime.execute({
        providerId: "opencode-go",
        routingKey: "session-three",
        candidates: [{ id: "primary", value: "sk-primary" }],
        run: vi.fn(),
      }),
    ).rejects.toThrow(/all API keys are cooling.*2026-07-20T13:00:00.000Z/i);
  });
});

async function usagePath(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-provider-key-usage-"));
  temporaryRoots.push(root);
  return join(root, "provider-key-usage.json");
}
