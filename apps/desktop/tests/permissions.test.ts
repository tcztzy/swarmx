import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";
import {
  narrowPermissions,
  type PermissionRequest,
  projectPermissions,
} from "../src/permissions.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});
const observer: Observer = {
  text() {},
  tool() {},
  raw() {},
  interact: async () => ({ allow: true }),
};
const context = { actorId: "test", callId: "call", signal: new AbortController().signal };

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-permissions-"));
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: { id: "test", label: "Test", root },
  });
  let counter = 0;
  const start = vi.fn<NativeAgent["start"]>(async () => {
    await { stopReason: "end_turn" };
    return { stopReason: "end_turn" as const };
  });
  const create = vi.fn(async () => `codex:${++counter}`);
  const native: NativeAgent = {
    name: "fixture",
    capabilities: HARNESS_CAPABILITIES.codex,
    models: async () => ({
      models: ["small", "large"].map((id) => ({ id, name: id, efforts: [] })),
      current: { model: "large" },
    }),
    list: async () => [],
    create,
    read: async () => {},
    start,
    steer: async () => {},
    interrupt: async () => {},
    dispose: async () => {},
  };
  await products.attachAgents("http://localhost", "test", native);
  cleanups.push(async () => {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  return {
    products,
    native,
    start,
    create,
    async reopen() {
      await products.dispose();
      const reopened = await ProductServices.create(products.options);
      cleanups.push(() => reopened.dispose());
      await reopened.attachAgents("http://localhost", "test", native);
      return reopened;
    },
    call: (args: unknown) => products.callTool("swarm", args, context),
  };
}

it("rejects every form of explicitly widened authority", () => {
  const parent = projectPermissions({
    tools: ["memory.read", "science.read"],
    delegation: false,
    harnesses: { codex: ["small"] },
  });
  for (const request of [
    { tools: ["memory.read", "memory.write", "science.read", "science.write"] },
    { delegation: true },
    { harnesses: { claude: null } },
    { harnesses: { codex: null } },
    { harnesses: { codex: ["large"] } },
  ] as const)
    expect(() => narrowPermissions(parent, request as PermissionRequest)).toThrow(
      "beyond the parent",
    );
  expect(narrowPermissions(parent, { harnesses: {} }).harnesses).toEqual({});
});

it("checks each Host tool family before dispatch even without a parent execution", async () => {
  const { products } = await fixture();
  products.updatePolicy({ ...products.settings.read().policy, tools: [] });
  for (const [name, args, grant] of [
    ["memory", { action: "read_core_memory" }, "memory.read"],
    ["memory", { action: "create_memory" }, "memory.write"],
    ["science_query", {}, "science.read"],
    ["science_notebook", {}, "science.write"],
  ] as const)
    await expect(products.callTool(name, args, context)).rejects.toThrow(grant);
  products.updatePolicy({ ...products.settings.read().policy, tools: ["memory.read"] });
  await expect(
    products.callTool(
      "memory",
      { action: "read_core_memory", request: { target: "user" } },
      context,
    ),
  ).resolves.toBeDefined();
});

it.each(["plan", "full"])("native %s does not change Host grants or delegation", async (mode) => {
  const { products, native, start, call } = await fixture();
  const models = {
    models: [],
    modes: [
      { id: "plan", name: "Plan" },
      { id: "full", name: "Full access" },
    ],
    current: {},
  };
  native.models = async () => models;
  start.mockImplementation(async (id, text, _output, selection) => {
    if (text === "child") return { stopReason: "end_turn" };
    expect(selection?.mode).toBe(mode);
    expect(products.journal.activeSession(id)?.permissions?.tools).toEqual(["memory.read"]);
    await expect(products.callTool("memory", { action: "create_memory" }, context)).rejects.toThrow(
      "memory.write",
    );
    await call({ action: "send_message", agentId: "codex", text: "child" });
    return { stopReason: "end_turn" };
  });
  await products.rootAgent.start("codex:parent", "delegate", observer, {
    mode,
    permissions: { tools: ["memory.read"] },
  });
  expect(start).toHaveBeenCalledTimes(2);
});

it("keeps legacy conversation history readable but rejects execution with changed semantics", async () => {
  const { products, start } = await fixture();
  const legacy = products.journal.append(
    { sessionId: "codex:legacy", runId: "old", causedBy: null, attributes: {} },
    {
      type: "CUSTOM",
      name: "swarmx.session.created",
      value: {
        permissions: { filesystem: "read-only", delegation: false, harnesses: { codex: null } },
      },
    },
  );
  const agent = await products.agent("codex");
  await agent.read("codex:legacy", observer);
  await expect(agent.start("codex:legacy", "continue", observer)).rejects.toThrow(
    "legacy filesystem permissions",
  );
  expect(start).not.toHaveBeenCalled();
  expect(products.journal.read({ session: "codex:legacy" }).events).toEqual([legacy]);
});

it("persists an empty conversation's grant across restart and native/Swarm aliases", async () => {
  const { products, start, reopen } = await fixture();
  const permissions = {
    tools: ["memory.read", "science.read"],
    delegation: false,
    harnesses: { codex: ["small"] },
  } as const;
  const id = await products.rootAgent.create({
    permissions: { ...permissions, harnesses: { codex: ["small"] } },
  });
  expect(start).not.toHaveBeenCalled();
  expect(products.journal.sessionPermissions(id)).toEqual(permissions);
  const reopened = await reopen();
  const native = await reopened.agent("codex");
  expect(await native.models(id)).toMatchObject({ models: [{ id: "small" }], current: {} });
  await expect(
    reopened.rootAgent.start(id, "write", observer, {
      model: "small",
      permissions: { tools: ["memory.read", "memory.write", "science.read", "science.write"] },
    }),
  ).rejects.toThrow("product tool permission");
  await expect(native.start(id, "large model", observer, { model: "large" })).rejects.toThrow(
    "permitted model",
  );
  start.mockImplementation(async (session) => {
    expect(reopened.journal.activeSession(session)?.permissions).toEqual(permissions);

    return { stopReason: "end_turn" as const };
  });
  await native.start(id, "continue", observer, { model: "small" });
});

it("retains turn restrictions after failure, later messages and project permission expansion", async () => {
  const { products, start, reopen } = await fixture();
  const id = await products.rootAgent.create();
  start.mockRejectedValueOnce(new Error("Native failure"));
  await expect(
    products.rootAgent.start(id, "read", observer, {
      permissions: { tools: ["memory.read", "science.read"] },
    }),
  ).rejects.toThrow("Native failure");
  await products.rootAgent.start(id, "continue", observer, { permissions: { delegation: false } });
  const reopened = await reopen();
  reopened.updatePolicy({
    ...reopened.settings.read().policy,
    tools: ["memory.read", "memory.write", "science.read", "science.write"],
    delegation: true,
  });
  start.mockImplementation(async (session) => {
    expect(reopened.journal.activeSession(session)?.permissions).toMatchObject({
      tools: ["memory.read", "science.read"],
      delegation: false,
    });

    return { stopReason: "end_turn" as const };
  });
  await reopened.rootAgent.start(id, "continue again", observer);
  const grant = reopened.journal.sessionPermissions(id);
  await expect(
    reopened.rootAgent.start(id, "delegate", observer, { permissions: { delegation: true } }),
  ).rejects.toThrow("delegation permission");
  expect(reopened.journal.sessionPermissions(id)).toEqual(grant);
  const independent = await reopened.rootAgent.create();
  expect(reopened.journal.sessionPermissions(independent)).toMatchObject({
    tools: ["memory.read", "memory.write", "science.read", "science.write"],
    delegation: true,
  });
});

it("saves inherited child authority when the parent ends before the child's first message", async () => {
  const { products, start, call, reopen } = await fixture();
  let child = "";
  start.mockImplementationOnce(async () => {
    const result = (await call({ action: "new_session", agentId: "codex" })) as {
      sessionId: string;
    };
    child = result.sessionId;

    return { stopReason: "end_turn" as const };
  });
  await products.rootAgent.start("codex:parent", "create child", observer, {
    permissions: { tools: ["memory.read", "science.read"] },
  });
  const reopened = await reopen();
  start.mockImplementationOnce(async (session) => {
    expect(reopened.journal.activeSession(session)?.permissions?.tools).toEqual([
      "memory.read",
      "science.read",
    ]);

    return { stopReason: "end_turn" as const };
  });
  await reopened.rootAgent.start(child, "continue child", observer);
});

it("carries read-only grants through nested Swarms, direct native aliases and bound MCP calls", async () => {
  const { products, start, call } = await fixture();
  await call({
    action: "create",
    id: "reader",
    leadAgentId: "swarm",
    permissions: { tools: ["memory.read", "science.read"] },
  });
  await call({ action: "create", id: "nested", leadAgentId: "reader" });
  start.mockImplementation(async (session, text, output) => {
    expect(products.journal.activeSession(session)?.permissions?.tools).toEqual([
      "memory.read",
      "science.read",
    ]);
    if (text === "child") return { stopReason: "end_turn" as const };
    const parentContext = { ...context, sessionId: session, runId: output.executionId };
    // Simulate a new HTTP carrier, without inherited AsyncLocalStorage.
    await products.journal.scope.exit(async () => {
      await expect(
        products.callTool(
          "swarm",
          {
            action: "create",
            id: "writer",
            leadAgentId: "codex",
            permissions: {
              tools: ["memory.read", "memory.write", "science.read", "science.write"],
            },
          },
          parentContext,
        ),
      ).rejects.toThrow("product tool permission");
      await expect(
        products.callTool("memory", { action: "create_memory", request: {} }, parentContext),
      ).rejects.toThrow("memory.write");
      await products.callTool(
        "swarm",
        { action: "send_message", agentId: "codex", text: "child" },
        parentContext,
      );
    });

    return { stopReason: "end_turn" as const };
  });
  await (await products.agent("nested")).start("codex:parent", "parent", observer);
  expect(start).toHaveBeenCalledTimes(2);
  const runs = products.journal
    .read()
    .events.filter((record) => record.event.type === "RUN_STARTED");
  expect(runs).toHaveLength(2);
  for (const run of runs)
    expect(run.event).toMatchObject({
      input: { forwardedProps: { permissions: { tools: ["memory.read", "science.read"] } } },
    });
});

it("checks project revocation and model admission on cached Agents and resumed sessions", async () => {
  const { products, start, call } = await fixture();
  const cached = await products.agent("codex");
  products.updatePolicy({ ...products.settings.read().policy, harnesses: { codex: ["small"] } });
  expect(await cached.models()).toMatchObject({ models: [{ id: "small" }], current: {} });
  await expect(cached.start("codex:old", "run", observer)).rejects.toThrow(
    "explicit permitted model",
  );
  await expect(
    cached.start("codex:old", "run", observer, { permissions: { harnesses: {} } }),
  ).rejects.toThrow("not permitted");
  await expect(cached.start("codex:old", "run", observer, { model: "large" })).rejects.toThrow(
    "explicit permitted model",
  );
  await call({
    action: "send_message",
    agentId: "swarm",
    sessionId: "codex:old",
    text: "run",
    model: "small",
  });
  expect(start).toHaveBeenCalledOnce();
  products.updatePolicy({ ...products.settings.read().policy, harnesses: {} });
  await expect(cached.create()).rejects.toThrow("not permitted");
  await expect(
    products.rootAgent.start("codex:old", "run", observer, { model: "small" }),
  ).rejects.toThrow("not permitted");
});

it("does not schedule memory writes or another model from a read-only or non-delegating run", async () => {
  const { products } = await fixture();
  const completed = vi.spyOn(products.learning, "completed");
  await products.rootAgent.start("codex:reader", "read", observer, {
    permissions: { tools: ["memory.read", "science.read"] },
  });
  await products.rootAgent.start("codex:leaf", "answer", observer, {
    permissions: { delegation: false },
  });
  expect(completed).not.toHaveBeenCalled();
});

it("denies re-delegation before creating sessions and keeps cancellation available", async () => {
  const { products, call, start, create } = await fixture();
  await call({
    action: "create",
    id: "leaf",
    leadAgentId: "codex",
    permissions: { delegation: false },
  });
  start.mockImplementation(async () => {
    for (const args of [
      { action: "create", id: "child", leadAgentId: "codex" },
      { action: "new_session", agentId: "codex" },
      { action: "send_message", agentId: "codex", text: "child" },
    ])
      await expect(call(args)).rejects.toThrow("delegation permission");
    await expect(
      call({ action: "cancel", agentId: "codex", sessionId: "codex:parent" }),
    ).resolves.toMatchObject({ cancellationRequested: true });

    return { stopReason: "end_turn" as const };
  });
  await (await products.agent("leaf")).start("codex:parent", "read", observer);
  expect(create).not.toHaveBeenCalled();
});

it("keeps parallel sibling grants isolated and does not let an old Swarm escape a tightened project", async () => {
  const { products, call, start } = await fixture();
  await call({ action: "create", id: "writer", leadAgentId: "codex" });
  const observed = new Map<string, string[] | undefined>();
  const gate = Promise.withResolvers<void>();
  start.mockImplementation(async (id) => {
    observed.set(id, products.journal.scope.getStore()?.permissions?.tools);
    if (observed.size === 2) gate.resolve();
    await gate.promise;
    expect(products.journal.scope.getStore()?.permissions?.tools).toBe(observed.get(id));

    return { stopReason: "end_turn" as const };
  });
  await Promise.all([
    products.rootAgent.start("codex:read", "read", observer, {
      permissions: { tools: ["memory.read", "science.read"] },
    }),
    products.rootAgent.start("codex:write", "write", observer),
  ]);
  expect([...observed.values()]).toEqual([
    ["memory.read", "science.read"],
    ["memory.read", "memory.write", "science.read", "science.write"],
  ]);
  products.updatePolicy({
    ...products.settings.read().policy,
    tools: ["memory.read", "science.read"],
  });
  await (await products.agent("writer")).start("codex:later", "read", observer);
  expect(observed.get("codex:later")).toEqual(["memory.read", "science.read"]);
});
