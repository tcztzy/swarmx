import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { AGENT_IDS, loadAgent, selectedAgent } from "../src/agent.js";
import { HarnessAccessSchema, policyPermissions } from "../src/permissions.js";

afterEach(() => vi.unstubAllEnvs());

it("defaults to Codex and selects a configured external endpoint without granting it authority", () => {
  vi.stubEnv("SWARMX_AGENT", undefined);
  vi.stubEnv("SWARMX_ACP_AGENT", undefined);
  expect(selectedAgent()).toBe("codex");
  vi.stubEnv("SWARMX_ACP_AGENT", "/configured/agent.json");
  expect(selectedAgent()).toBe("acp");
  expect(policyPermissions({}).harnesses.acp).toBeUndefined();
  expect(selectedAgent("claude")).toBe("claude");
});

it("retains old Pi policy values but never advertises or loads the retired runtime", async () => {
  expect(HarnessAccessSchema.parse({ pi: ["provider/model"] })).toEqual({ pi: ["provider/model"] });
  expect(AGENT_IDS).not.toContain("pi");
  expect(policyPermissions({}).harnesses.pi).toBeUndefined();
  expect(() => selectedAgent("pi")).toThrow("built-in Pi agent has been retired");
  await expect(
    loadAgent("pi" as Parameters<typeof loadAgent>[0], {
      cwd: "/unused",
      mcp: { command: "node", args: [], env: {} },
    }),
  ).rejects.toThrow("built-in Pi agent has been retired");
  vi.stubEnv("SWARMX_AGENT", "pi");
  expect(() => selectedAgent()).toThrow(
    "Existing Pi authentication and session files are unchanged",
  );
});

it("keeps the embedded implementation as a byte-preserved historical reference outside core", () => {
  const root = process.cwd();
  expect(existsSync(join(root, "apps/desktop/src/agents/pi.ts"))).toBe(false);
  const manifest = JSON.parse(readFileSync(join(root, "apps/desktop/package.json"), "utf8"));
  expect(manifest.dependencies["@earendil-works/pi-ai"]).toBeUndefined();
  expect(manifest.dependencies["@earendil-works/pi-coding-agent"]).toBeUndefined();
  expect(existsSync(join(root, "examples/legacy-pi/manifest.json"))).toBe(true);
});
