import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as acp from "@agentclientprotocol/sdk";
import { afterEach, expect, it, vi } from "vitest";
import { HARNESS_CAPABILITIES, type Interaction, type NativeAgent } from "../src/agents/types.js";
import { acpAgent } from "../src/host/acp.js";
import { acpClient } from "../src/host/acp-client.js";
import { acknowledgedPermissions } from "../src/host/acp-extension.js";
import { ProductServices } from "../src/host/product-services.js";
import { projectPermissions } from "../src/permissions.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const negotiated = { _meta: { swarmx: { version: 2, permissions: true } } };
const requestMeta = {
  swarmx: {
    version: 2,
    permissions: { tools: ["memory.read", "science.read"], delegation: false },
  },
};
const question: Interaction = {
  id: "tool",
  title: "Write?",
  schema: {},
  permission: {
    toolCall: { toolCallId: "tool", title: "Write?" },
    options: [
      { optionId: "yes", name: "Allow once", kind: "allow_once" },
      { optionId: "no", name: "Reject", kind: "reject_once" },
    ],
    answers: { yes: { allow: true }, no: { allow: false } },
  },
};

it("relays native modes through nested Swarms and preserves the choice when selecting a model", async () => {
  const { products, native, root, start } = await fixture();
  native.models = async () => ({
    models: [{ id: "small", name: "Small", efforts: [] }],
    modes: [
      { id: "plan", name: "Plan" },
      { id: "full", name: "Full access" },
    ],
    current: { mode: "plan" },
  });
  await products.callTool(
    "swarm",
    { action: "create", id: "nested", leadAgentId: "swarm" },
    { actorId: "test", callId: "create", signal: new AbortController().signal },
  );
  const connection = acp.client().connect(acpAgent(await products.agent("nested"), root));
  try {
    await connection.agent.request(acp.methods.agent.initialize, {
      protocolVersion: acp.PROTOCOL_VERSION,
    });
    const session = await connection.agent.request(acp.methods.agent.session.new, {
      cwd: root,
      mcpServers: [],
    });
    expect(session.configOptions).toContainEqual(
      expect.objectContaining({
        id: "mode",
        category: "mode",
        currentValue: "plan",
        options: [
          { value: "plan", name: "Plan" },
          { value: "full", name: "Full access" },
        ],
      }),
    );
    await connection.agent.request(acp.methods.agent.session.setMode, {
      sessionId: session.sessionId,
      modeId: "full",
    });
    await connection.agent.request(acp.methods.agent.session.setConfigOption, {
      sessionId: session.sessionId,
      configId: "model",
      value: "small",
    });
    await connection.agent.request(acp.methods.agent.session.prompt, {
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "work" }],
    });
    expect(start.mock.lastCall?.[3]).toMatchObject({ mode: "full", model: "small" });
    expect(products.journal.sessionPermissions(session.sessionId)).toEqual(projectPermissions({}));
    await expect(
      connection.agent.request(acp.methods.agent.session.setMode, {
        sessionId: session.sessionId,
        modeId: "invented",
      }),
    ).rejects.toThrow("advertised");
  } finally {
    connection.close();
  }
});

it("opens every recursive edge through ACP and isolates concurrent connection authorities", async () => {
  const { products, start } = await fixture();
  const context = { actorId: "test", callId: "create", signal: new AbortController().signal };
  for (const [id, leadAgentId] of [
    ["inner", "codex"],
    ["middle", "inner"],
    ["outer", "middle"],
  ])
    await products.callTool(
      "swarm",
      {
        action: "create",
        id,
        leadAgentId,
        ...(id === "middle" ? { permissions: { tools: ["memory.read", "science.read"] } } : {}),
      },
      context,
    );
  const outer = await products.agent("outer");
  const restricted = await outer.create();
  const writable = await products.rootAgent.create();
  const seen = new Map<string, string[]>();
  const both = Promise.withResolvers<void>();
  start.mockImplementation(async (sessionId, _text, output) => {
    output.raw({ sessionId }, { "rpc.method": "native", "gen_ai.response.id": undefined });
    const before = products.journal.scope.getStore();
    seen.set(sessionId, before?.permissions?.tools ?? []);
    if (seen.size === 2) both.resolve();
    await both.promise;
    expect(products.journal.scope.getStore()).toBe(before);
    return { stopReason: "end_turn" };
  });
  const observer = { text() {}, tool() {}, raw: vi.fn(), interact: async () => undefined };
  const connect = vi.spyOn(acp.ClientApp.prototype, "connect");
  try {
    await Promise.all([
      outer.start(restricted, "read", observer),
      products.rootAgent.start(writable, "write", observer),
    ]);
    expect(connect).toHaveBeenCalledTimes(6); // outer → middle → inner → leaf, root → leaf
    expect(seen.get(restricted)).toEqual(["memory.read", "science.read"]);
    expect(seen.get(writable)).toEqual(projectPermissions({}).tools);
    expect(observer.raw).toHaveBeenCalledWith(
      { sessionId: restricted },
      { "rpc.method": "native", "gen_ai.response.id": undefined },
    );
    await expect(
      outer.start(restricted, "escalate", observer, {
        permissions: { tools: ["memory.read", "memory.write", "science.read", "science.write"] },
      }),
    ).rejects.toThrow(/product tool permission/);
    expect(start).toHaveBeenCalledTimes(2);
  } finally {
    connect.mockRestore();
  }
});

it.each(["unsupported", "missing", "wider"])(
  "internal ACP clients reject %s permission acknowledgements",
  async (failure) => {
    const create = vi.fn(() => ({
      sessionId: "native",
      ...(failure === "missing"
        ? {}
        : { _meta: { swarmx: { version: 2, permissions: projectPermissions({}) } } }),
    }));
    const client = acpClient(
      "test",
      HARNESS_CAPABILITIES.codex,
      "/workspace",
      (app) =>
        app.connect(
          acp
            .agent()
            .onRequest(acp.methods.agent.initialize, () => ({
              protocolVersion: acp.PROTOCOL_VERSION,
              agentCapabilities:
                failure === "unsupported"
                  ? {}
                  : { _meta: { swarmx: { version: 2, permissions: true } } },
            }))
            .onRequest(acp.methods.agent.session.new, create),
        ),
      () => projectPermissions({ tools: ["memory.read", "science.read"] }),
    );
    await expect(client.create()).rejects.toThrow();
    expect(create).toHaveBeenCalledTimes(failure === "unsupported" ? 0 : 1);
    await client.dispose();
  },
);

it("closing the upstream ACP connection cancels its native prompt", async () => {
  const { products, native, root, start } = await fixture();
  const entered = Promise.withResolvers<void>();
  const stopped = Promise.withResolvers<acp.PromptResponse>();
  start.mockImplementation(async () => {
    entered.resolve();
    return stopped.promise;
  });
  native.interrupt = vi.fn(async () => {
    stopped.resolve({ stopReason: "cancelled" });
  });
  const connection = acp.client().connect(acpAgent(products.rootAgent, root));
  await connection.agent.request(acp.methods.agent.initialize, {
    protocolVersion: acp.PROTOCOL_VERSION,
  });
  const session = await connection.agent.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
  });
  const pending = connection.agent.request(acp.methods.agent.session.prompt, {
    sessionId: session.sessionId,
    prompt: [],
  });
  const rejected = expect(pending).rejects.toThrow(/closed/);
  await entered.promise;
  connection.close();
  await rejected;
  await vi.waitFor(() => expect(products.journal.activeRuns()).toHaveLength(0));
  expect(native.interrupt).toHaveBeenCalled();
});

it("cancels during ACP model configuration before native dispatch", async () => {
  const { products, native, start } = await fixture();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  native.models = async () => {
    entered.resolve();
    await release.promise;
    return { models: [{ id: "small", name: "Small", efforts: [] }], current: {} };
  };
  const observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
  const running = products.rootAgent.start("codex:queued", "work", observer, { model: "small" });
  await entered.promise;
  await products.rootAgent.interrupt("codex:queued");
  release.resolve();
  expect(await running).toEqual({ stopReason: "cancelled" });
  expect(start).not.toHaveBeenCalled();
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-acp-contract-"));
  const products = await ProductServices.create({
    productHome: join(root, "home"),
    workspace: { id: "acp", label: "ACP", root },
  });
  let counter = 0;
  const start = vi.fn<NativeAgent["start"]>(async () => ({ stopReason: "end_turn" }));
  const native: NativeAgent = {
    name: "Fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    create: vi.fn(async () => `codex:${++counter}`),
    list: async () => [],
    models: async () => ({ models: [], current: {} }),
    read: async (_id, output) => output.text("history", "Native history"),
    start,
    steer: vi.fn(async () => {}),
    interrupt: vi.fn(async () => {}),
    dispose: async () => {},
  };
  await products.attachAgents("http://localhost", "test", native);
  cleanups.push(async () => {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  const connect = (client = acp.client({ name: "contract-test" })) => {
    const connection = client.connect(acpAgent(products.rootAgent, root));
    cleanups.push(async () => {
      await connection.close();
    });
    return connection.agent;
  };
  return { root, products, native, start, connect };
}

it("negotiates and acknowledges persistent permissions with the official ACP client", async () => {
  const { root, products, start, connect } = await fixture();
  const updates: acp.SessionNotification[] = [];
  const client = connect(
    acp
      .client({ name: "permissions" })
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        updates.push(params);
      }),
  );
  const init = await client.request(acp.methods.agent.initialize, {
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: negotiated,
  });
  expect(init.agentCapabilities?._meta?.swarmx).toMatchObject({ version: 2, permissions: true });
  const session = await client.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
    _meta: requestMeta,
  });
  expect(
    acknowledgedPermissions(session._meta, {
      tools: ["memory.read", "science.read"],
      delegation: false,
    }),
  ).toMatchObject({ tools: ["memory.read", "science.read"], delegation: false });
  const result = await client.request(acp.methods.agent.session.prompt, {
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "read" }],
  });
  expect(result.stopReason).toBe("end_turn");
  expect(
    acknowledgedPermissions(result._meta, { tools: ["memory.read", "science.read"] }).tools,
  ).toEqual(["memory.read", "science.read"]);
  expect(updates.find((update) => update._meta?.swarmx)?._meta?.swarmx).toMatchObject({
    execution: {
      runId: expect.any(String),
      parentRunId: null,
      permissions: { tools: ["memory.read", "science.read"] },
    },
  });
  await expect(
    client.request(acp.methods.agent.session.prompt, {
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "write" }],
      _meta: {
        swarmx: {
          version: 2,
          permissions: { tools: ["memory.read", "memory.write", "science.read", "science.write"] },
        },
      },
    }),
  ).rejects.toThrow(/product tool permission/);
  expect(start).toHaveBeenCalledTimes(1);
  expect(products.journal.sessionPermissions(session.sessionId)?.tools).toEqual([
    "memory.read",
    "science.read",
  ]);

  const standard = connect();
  await standard.request(acp.methods.agent.initialize, { protocolVersion: acp.PROTOCOL_VERSION });
  await standard.request(acp.methods.agent.session.prompt, {
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "continue" }],
  });
  expect(products.journal.sessionPermissions(session.sessionId)?.tools).toEqual([
    "memory.read",
    "science.read",
  ]);
  await expect(
    standard.request(acp.methods.agent.session.new, {
      cwd: root,
      mcpServers: [],
      _meta: requestMeta,
    }),
  ).rejects.toThrow(/not negotiated/);
});

it("rejects unsupported versions, malformed restrictions and absent or widened acknowledgements", async () => {
  const { root, native, connect } = await fixture();
  const client = connect();
  await expect(
    client.request(acp.methods.agent.initialize, {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: { _meta: { swarmx: { version: 1, permissions: true } } },
    }),
  ).rejects.toThrow();
  await client.request(acp.methods.agent.initialize, {
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: negotiated,
  });
  await expect(
    client.request(acp.methods.agent.session.new, {
      cwd: root,
      mcpServers: [],
      _meta: { swarmx: { version: 2, permissions: { write: false } } },
    }),
  ).rejects.toThrow();
  expect(native.create).not.toHaveBeenCalled();
  expect(() =>
    acknowledgedPermissions(undefined, { tools: ["memory.read", "science.read"] }),
  ).toThrow();
  expect(() =>
    acknowledgedPermissions(
      {
        swarmx: {
          version: 2,
          permissions: {
            tools: ["memory.read", "memory.write", "science.read", "science.write"],
            delegation: false,
            harnesses: {},
          },
        },
      },
      { tools: ["memory.read", "science.read"] },
    ),
  ).toThrow(/product tool permission/);
  const unprotected = acp.client({ name: "unprotected" }).connect(acpAgent(native, root));
  cleanups.push(async () => {
    await unprotected.close();
  });
  await expect(
    unprotected.agent.request(acp.methods.agent.initialize, {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: negotiated,
    }),
  ).rejects.toThrow(/cannot enforce/);
});

it.each(["end_turn", "cancelled", "max_tokens", "max_turn_requests", "refusal"] as const)(
  "preserves ACP %s instead of collapsing it to success",
  async (stopReason) => {
    const { root, start, connect } = await fixture();
    start.mockResolvedValueOnce({ stopReason });
    const client = connect();
    await client.request(acp.methods.agent.initialize, { protocolVersion: acp.PROTOCOL_VERSION });
    const { sessionId } = await client.request(acp.methods.agent.session.new, {
      cwd: root,
      mcpServers: [],
    });
    expect(
      await client.request(acp.methods.agent.session.prompt, {
        sessionId,
        prompt: [{ type: "resource_link", name: "Source", uri: "file:///research/source.md" }],
      }),
    ).toEqual({ stopReason });
  },
);

it("uses ACP permissions without requiring forms and validates the exact offered option", async () => {
  const { root, start, connect } = await fixture();
  let optionId = "yes";
  const permission = vi.fn(() => ({ outcome: { outcome: "selected" as const, optionId } }));
  const client = connect(
    acp
      .client({ name: "approval" })
      .onRequest(acp.methods.client.session.requestPermission, permission),
  );
  await client.request(acp.methods.agent.initialize, { protocolVersion: acp.PROTOCOL_VERSION });
  const { sessionId } = await client.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
  });
  start.mockImplementation(async (_id, _text, observer) => {
    expect(await observer.interact(question)).toEqual({ allow: true });
    return { stopReason: "end_turn" };
  });
  await client.request(acp.methods.agent.session.prompt, {
    sessionId,
    prompt: [{ type: "text", text: "approve" }],
  });
  expect(permission).toHaveBeenCalledTimes(1);
  optionId = "invented";
  await expect(
    client.request(acp.methods.agent.session.prompt, {
      sessionId,
      prompt: [{ type: "text", text: "approve" }],
    }),
  ).rejects.toThrow(/Unknown permission option/);
});

it("cancels a pending approval and ignores a late allow response", async () => {
  const { root, start, connect } = await fixture();
  const asked = Promise.withResolvers<void>();
  const late = Promise.withResolvers<acp.RequestPermissionResponse>();
  const client = connect(
    acp.client({ name: "late" }).onRequest(acp.methods.client.session.requestPermission, () => {
      asked.resolve();
      return late.promise;
    }),
  );
  await client.request(acp.methods.agent.initialize, { protocolVersion: acp.PROTOCOL_VERSION });
  const { sessionId } = await client.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
  });
  const answers: unknown[] = [];
  start.mockImplementation(async (_id, _text, observer) => {
    answers.push(await observer.interact(question));
    return { stopReason: "cancelled" };
  });
  const running = client.request(acp.methods.agent.session.prompt, {
    sessionId,
    prompt: [{ type: "text", text: "wait" }],
  });
  await asked.promise;
  await client.notify(acp.methods.agent.session.cancel, { sessionId });
  expect(await running).toEqual({ stopReason: "cancelled" });
  late.resolve({ outcome: { outcome: "selected", optionId: "yes" } });
  expect(answers).toEqual([undefined]);
});

it("resumes without replay while load emits native history", async () => {
  const { root, connect } = await fixture();
  const updates: acp.SessionNotification[] = [];
  const client = connect(
    acp
      .client({ name: "resume" })
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        updates.push(params);
      }),
  );
  await client.request(acp.methods.agent.initialize, {
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: negotiated,
  });
  const { sessionId } = await client.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
    _meta: requestMeta,
  });
  const resumed = await client.request(acp.methods.agent.session.resume, { sessionId, cwd: root });
  expect(
    acknowledgedPermissions(resumed._meta, { tools: ["memory.read", "science.read"] }).tools,
  ).toEqual(["memory.read", "science.read"]);
  expect(updates).toEqual([]);
  await client.request(acp.methods.agent.session.load, { sessionId, cwd: root, mcpServers: [] });
  expect(updates).toHaveLength(1);
});

it("uses ACP config options for admitted models and rejects unadvertised settings", async () => {
  const { root, native, start, connect } = await fixture();
  native.models = async () => ({
    models: [
      { id: "small", name: "Small", efforts: [{ id: "high", name: "High" }] },
      { id: "large", name: "Large", efforts: [] },
    ],
    current: {},
  });
  const client = connect();
  await client.request(acp.methods.agent.initialize, {
    protocolVersion: acp.PROTOCOL_VERSION,
    clientCapabilities: negotiated,
  });
  const session = await client.request(acp.methods.agent.session.new, {
    cwd: root,
    mcpServers: [],
    _meta: { swarmx: { version: 2, permissions: { harnesses: { codex: ["small"] } } } },
  });
  expect(session.configOptions).toMatchObject([
    { id: "model", options: [{ value: "" }, { value: "small" }] },
  ]);
  await expect(
    client.request(acp.methods.agent.session.setConfigOption, {
      sessionId: session.sessionId,
      configId: "model",
      value: "large",
    }),
  ).rejects.toThrow(/advertised/);
  await client.request(acp.methods.agent.session.setConfigOption, {
    sessionId: session.sessionId,
    configId: "model",
    value: "small",
  });
  await client.request(acp.methods.agent.session.setConfigOption, {
    sessionId: session.sessionId,
    configId: "effort",
    value: "high",
  });
  await client.request(acp.methods.agent.session.prompt, {
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "work" }],
  });
  expect(start).toHaveBeenLastCalledWith(
    session.sessionId,
    "work",
    expect.anything(),
    expect.objectContaining({ model: "small", effort: "high" }),
  );
});
