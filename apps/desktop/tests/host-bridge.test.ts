import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import type { NativeAgent } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { BootstrapSchema, SettingsResponseSchema } from "../src/bridge-contract.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";

const native = {
  name: "native",
  capabilities: HARNESS_CAPABILITIES.codex,
  models: async () => ({ models: [], current: {} }),
  list: async () => [],
  create: async () => "codex:new",
  read: async () => {},
  start: async () => ({ stopReason: "end_turn" as const }),
  steer: async () => {},
  interrupt: async () => {},
  dispose: async () => {},
} satisfies NativeAgent;

it("routes renderer operations and tool calls through one Host", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-host-bridge-"));
  try {
    const products = await ProductServices.create({ productHome: join(root, "home"), cwd: root });
    const host = await startHost({ products, agent: native, agentId: "codex" });
    try {
      const operations = new HostOperations(host);
      expect(BootstrapSchema.parse(await operations.bootstrap())).toMatchObject({
        cwd: products.options.cwd,
      });
      expect(SettingsResponseSchema.parse(await operations.settings())).toMatchObject({
        cwd: products.options.cwd,
      });
      const status = await operations.callTool(
        "memory",
        {
          action: "memory_status",
          request: {},
        },
        randomUUID(),
        host.signal,
      );
      expect(status).toMatchObject({ action: "memory_status", data: { note: { content: "" } } });
    } finally {
      await host.dispose();
    }
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
