import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { expect, it } from "vitest";
import { HARNESS_CAPABILITIES, type NativeAgent } from "../src/agents/types.js";
import { AgentRegistry } from "../src/host/agent-registry.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { bridgeClient } from "./mcp-bridge-support.js";

it("keeps concurrent Hosts' native MCP sockets and credentials independent", async () => {
  const root = await mkdtemp("/tmp/sx-socket-");
  const cleanup: Array<() => Promise<void>> = [];
  async function start(name: string) {
    const cwd = join(root, name);
    await mkdir(cwd);
    const token = randomUUID();
    let socket = "";
    const native = {
      name: "native",
      capabilities: HARNESS_CAPABILITIES.codex,
      models: async () => ({ models: [], current: {} }),
      list: async () => [],
      create: async () => `codex:${name}`,
      read: async () => {},
      start: async () => ({ stopReason: "end_turn" as const }),
      steer: async () => {},
      interrupt: async () => {},
      dispose: async () => {},
    } satisfies NativeAgent;
    const agents = new AgentRegistry(async (_id, options) => {
      socket = options.mcp.env.SWARMX_MCP_SOCKET ?? "";
      options.registerMcp?.(token);
      return native;
    });
    cleanup.push(() => agents.dispose());
    const products = await ProductServices.create({ productHome: join(root, "home"), cwd, agents });
    cleanup.push(() => products.dispose());
    const host = await startHost({ products, agentId: "codex" });
    cleanup.push(() => host.dispose());
    expect(socket).not.toBe("");
    return { host, socket, token };
  }
  async function tools(socket: string, token: string) {
    const client = await bridgeClient(socket, token);
    try {
      return (await client.listTools()).tools.map((tool) => tool.name);
    } finally {
      await client.close();
    }
  }
  try {
    const first = await start("first");
    await expect(tools(first.socket, first.token)).resolves.toContain("memory");
    const second = await start("second");
    await expect(tools(first.socket, first.token)).resolves.toContain("memory");
    await expect(tools(second.socket, second.token)).resolves.toContain("memory");
    expect(first.socket).not.toBe(second.socket);
    await expect(tools(first.socket, second.token)).rejects.toThrow();
    await expect(tools(second.socket, first.token)).rejects.toThrow();
    await second.host.dispose();
    await expect(tools(first.socket, first.token)).resolves.toContain("memory");
  } finally {
    for (const dispose of cleanup.reverse()) await dispose();
    await rm(root, { recursive: true, force: true });
  }
}, 15_000);
