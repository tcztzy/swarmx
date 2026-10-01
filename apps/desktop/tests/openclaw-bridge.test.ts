import { mkdtemp, readdir, readFile, rm, stat } from "node:fs/promises";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { AgentMemory } from "../src/host/memory.js";
import { ProductServices } from "../src/host/product-services.js";
import { recordedAgent } from "../src/host/recorded-agent.js";
import { startHost } from "../src/host/server.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const sink: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };

function bridgeCall(
  socket: string,
  payload: Record<string, unknown>,
): Promise<{ ok: boolean; value?: unknown; error?: string }> {
  return new Promise((resolve, reject) => {
    const connection = connect(socket);
    let buffer = "";
    connection.once("error", reject);
    connection.once("connect", () =>
      connection.write(`${JSON.stringify({ id: 1, ...payload })}\n`),
    );
    connection.on("data", (chunk: Buffer) => {
      buffer += chunk.toString("utf8");
      const index = buffer.indexOf("\n");
      if (index < 0) return;
      connection.end();
      resolve(JSON.parse(buffer.slice(0, index)));
    });
  });
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-openclaw-"));
  const productHome = join(root, "product");
  const products = await ProductServices.create({ productHome, cwd: root });
  const memory = new AgentMemory(
    { productHome, cwd: root },
    products.memory,
    products.journal,
    products.settings,
    async () => '{"summary":"No new evidence.","operations":[]}',
  );
  cleanups.push(async () => {
    await memory.close();
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  const observed: Array<Record<string, unknown>> = [];
  const native: NativeAgent = {
    name: "Fixture",
    capabilities: HARNESS_CAPABILITIES.openclaw,
    models: async () => ({ models: [], current: {} }),
    list: async () => [],
    create: async () => "agent:main:probe",
    read: async () => {},
    start: vi.fn(async (sessionId, _text, observer) => {
      const executionId = observer.executionId;
      if (executionId === undefined) throw new Error("The fixture needs a recorded execution.");
      observed.push(
        (await bridgeCall(products.mcpSocket, {
          token: products.openclawToken,
          tool: "memory",
          args: { action: "read_core_memory", request: {} },
          sessionKey: "agent:main:probe",
          toolCallId: "call-before-bind",
        })) as unknown as Record<string, unknown>,
      );
      const lease = products.registerOpenClaw("agent:main:probe");
      lease.bind(sessionId, executionId);
      observed.push(
        (await bridgeCall(products.mcpSocket, {
          token: products.openclawToken,
          tool: "memory",
          args: { action: "read_core_memory", request: {} },
          sessionKey: "agent:main:probe",
          toolCallId: "call-bound",
        })) as unknown as Record<string, unknown>,
      );
      lease.release();
      observed.push(
        (await bridgeCall(products.mcpSocket, {
          token: products.openclawToken,
          tool: "memory",
          args: { action: "read_core_memory", request: {} },
          sessionKey: "agent:main:probe",
          toolCallId: "call-released",
        })) as unknown as Record<string, unknown>,
      );
      return { stopReason: "end_turn" };
    }),
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  const host = await startHost({ products, agent: native });
  cleanups.push(() => host.dispose());
  return {
    productHome,
    products,
    observed,
    agent: recordedAgent(products.journal, "openclaw", native, memory),
  };
}

it("publishes the gateway bridge descriptor with owner-only permissions", async () => {
  const { productHome, products } = await fixture();
  const directory = join(productHome, "openclaw", "bridges");
  const [file] = await readdir(directory);
  const path = join(directory, file);
  const descriptor = JSON.parse(await readFile(path, "utf8"));
  expect(descriptor).toMatchObject({
    version: 1,
    socket: products.mcpSocket,
    token: products.openclawToken,
  });
  expect(descriptor.tools.map((tool: { name: string }) => tool.name)).toContain("memory");
  expect((await stat(path)).mode & 0o777).toBe(0o600);
});

it("leases Host tools to the OpenClaw session run and revokes them when it ends", async () => {
  const { products, observed, agent } = await fixture();
  const session = await agent.create();
  expect(session).toBe("agent:main:probe");
  await agent.start(session, "Run the probe", sink);

  expect(observed[0]).toMatchObject({ ok: false });
  expect(String(observed[0]?.error)).toContain("no active Host execution");
  expect(observed[1]).toMatchObject({ ok: true });
  expect(observed[2]).toMatchObject({ ok: false });
  expect(String(observed[2]?.error)).toContain("no active Host execution");

  const after = await bridgeCall(products.mcpSocket, {
    token: products.openclawToken,
    tool: "memory",
    args: { action: "read_core_memory", request: {} },
    sessionKey: "agent:main:probe",
    toolCallId: "call-after-run",
  });
  expect(after).toMatchObject({ ok: false });
  expect(products.journal.activeRuns()).toHaveLength(0);
});

it("rejects OpenClaw tool calls without a session claim or with a foreign credential", async () => {
  const { products } = await fixture();
  const missing = await bridgeCall(products.mcpSocket, {
    token: products.openclawToken,
    tool: "memory",
    args: { action: "read_core_memory", request: {} },
  });
  expect(missing).toMatchObject({ ok: false });
  expect(String(missing.error)).toContain("session key and tool call id");
  const foreign = await bridgeCall(products.mcpSocket, {
    token: "not-a-host-credential",
    tool: "memory",
    args: { action: "read_core_memory", request: {} },
    sessionKey: "agent:main:probe",
    toolCallId: "call-foreign",
  });
  expect(foreign).toMatchObject({ ok: false });
  expect(String(foreign.error)).toContain("Unknown MCP execution credential");
});
