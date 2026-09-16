import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type Context,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxThinking,
  fauxToolCall,
  registerSessionResourceCleanup,
} from "@earendil-works/pi-ai";
import {
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
} from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createPi } from "../src/agents/pi.js";
import type { AgentOptions, NativeAgent, Observer } from "../src/agents/types.js";
import { loadAgUiHistory } from "../src/host/ag-ui.js";
import { ProductServices } from "../src/host/product-services.js";
import { DEFAULT_POLICY } from "../src/settings.js";

let root: string;
let cwd: string;
let agentDir: string;
let faux: ReturnType<typeof fauxProvider>;
const owned: NativeAgent[] = [];
const services: ProductServices[] = [];
const createModelRuntime = ModelRuntime.create.bind(ModelRuntime);
const observer = (): Observer => ({
  text: vi.fn(),
  tool: vi.fn(),
  raw: vi.fn(),
  interact: vi.fn(),
});
const output = (sink: Observer, role = "assistant") =>
  vi
    .mocked(sink.text)
    .mock.calls.filter((call) => (call[2] ?? "assistant") === role)
    .map((call) => call[1])
    .join("");

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "swarmx-pi-"));
  cwd = join(root, "research");
  agentDir = join(root, "pi");
  await Promise.all([mkdir(cwd), mkdir(agentDir)]);
  vi.stubEnv("PI_CODING_AGENT_DIR", agentDir);
  await writeFile(
    join(agentDir, "settings.json"),
    JSON.stringify({
      defaultProvider: "pi-test",
      defaultModel: "first",
      retry: { enabled: false },
      compaction: { enabled: false },
    }),
  );
  faux = fauxProvider({
    provider: "pi-test",
    tokensPerSecond: Infinity,
    models: [
      { id: "first", reasoning: true },
      { id: "second", reasoning: true },
    ],
  });
  const runtime = await createModelRuntime({
    refreshOnCreate: false,
    modelsPath: null,
    authPath: join(agentDir, "auth.json"),
    modelsStorePath: join(agentDir, "models-store.json"),
  });
  runtime.registerNativeProvider(faux.provider);
  vi.spyOn(runtime, "getAvailable").mockResolvedValue(faux.models);
  vi.spyOn(runtime, "getAvailableSnapshot").mockReturnValue(faux.models);
  vi.spyOn(runtime, "hasConfiguredAuth").mockImplementation((provider) => provider === "pi-test");
  vi.spyOn(runtime, "getAuth").mockResolvedValue({ auth: { apiKey: "test-only" } });
  vi.spyOn(ModelRuntime, "create").mockResolvedValue(runtime);
});

afterEach(async () => {
  for (const product of services.splice(0)) await product.dispose();
  for (const agent of owned.splice(0)) await agent.dispose();
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  await rm(root, { recursive: true, force: true });
});

async function agent(extra: Partial<AgentOptions> = {}) {
  const native = await createPi({
    cwd,
    mcp: { command: "node", args: ["/bridge.js"], env: {} },
    ...extra,
  });
  owned.push(native);
  return native;
}

it("releases the Pi session when native resource cleanup fails", async () => {
  const native = await agent();
  const id = await native.create();
  const unregister = registerSessionResourceCleanup((sessionId) => {
    if (sessionId === id) throw new Error("Cleanup failure");
  });
  try {
    faux.setResponses([fauxAssistantMessage("First answer")]);
    await expect(native.start(id, "first", observer())).rejects.toThrow(
      "Failed to cleanup session resources",
    );
  } finally {
    unregister();
  }
  faux.setResponses([fauxAssistantMessage("Retry answer")]);
  const sink = observer();
  await expect(native.start(id, "retry", sink)).resolves.toEqual({ stopReason: "end_turn" });
  expect(output(sink)).toBe("Retry answer");
});

it("streams native text/thinking and restores Pi history and per-session settings", async () => {
  const native = await agent();
  const id = await native.create();
  const before = await readFile(join(agentDir, "settings.json"), "utf8");
  const sink = observer();
  faux.setResponses([fauxAssistantMessage([fauxThinking("Inspect"), fauxText("First answer")])]);
  expect(
    await native.start(id, "First question", sink, {
      model: "pi-test/second",
      effort: "low",
      instructions: "Frozen memory rule",
    }),
  ).toEqual({ stopReason: "end_turn" });
  expect(output(sink)).toBe("First answer");
  expect(output(sink, "reasoning")).toBe("Inspect");
  expect(await readFile(join(agentDir, "settings.json"), "utf8")).toBe(before);
  await native.dispose();
  const restored = await agent();
  expect(await restored.list()).toEqual([expect.objectContaining({ sessionId: id })]);
  expect((await restored.models(id)).current).toEqual({ model: "pi-test/second", effort: "low" });
  const history = observer();
  await restored.read(id, history);
  expect(output(history)).toBe("First answer");
  expect(output(history, "user")).toBe("First question");
  let context: Context | undefined;
  faux.setResponses([
    (input) => {
      context = input;
      return fauxAssistantMessage("Continued");
    },
  ]);
  await restored.start(id, "Continue", observer(), { instructions: "Frozen memory rule" });
  expect(JSON.stringify(context?.messages)).toContain("First answer");
  expect(context?.systemPrompt).toContain("Frozen memory rule");
});

it("restores empty Host reservations and rejects sessions from another directory", async () => {
  const native = await agent();
  const id = await native.create();
  expect(await SessionManager.list(cwd)).toEqual([]);
  await native.dispose();
  const restored = await agent();
  restored.restoreEmptySessions?.([id]);
  expect(await restored.list()).toContainEqual({ sessionId: id });
  expect((await restored.models(id)).models).toHaveLength(2);
  faux.setResponses([fauxAssistantMessage("Persisted")]);
  await restored.start(id, "Start", observer());
  expect(await SessionManager.list(cwd)).toEqual([expect.objectContaining({ id })]);
  const otherCwd = join(root, "other");
  await mkdir(otherCwd);
  const foreign = await agent({ cwd: otherCwd });
  await expect(foreign.read(id, observer())).rejects.toThrow("does not belong");
});

it("uses Pi's built-in read tool and loads skill bodies only when read", async () => {
  const skill = join(agentDir, "skills", "memory", "SKILL.md");
  await mkdir(join(agentDir, "skills", "memory"), { recursive: true });
  await writeFile(
    skill,
    "---\nname: memory\ndescription: Manage durable memory.\n---\nFULL_SKILL_BODY\n",
  );
  const native = await agent();
  const sink = observer();
  let prompt: string | undefined;
  let toolContext: Context | undefined;
  faux.setResponses([
    (context) => {
      prompt = context.systemPrompt;
      return fauxAssistantMessage(fauxToolCall("read", { path: skill }, { id: "read-skill" }));
    },
    (context) => {
      toolContext = context;
      return fauxAssistantMessage("Skill read");
    },
  ]);
  await native.start(await native.create(), "Read the memory skill", sink);
  expect(prompt).toContain("Manage durable memory.");
  expect(prompt).not.toContain("FULL_SKILL_BODY");
  expect(JSON.stringify(toolContext?.messages)).toContain("FULL_SKILL_BODY");
  expect(sink.tool).toHaveBeenLastCalledWith(
    "read-skill",
    "read",
    { path: skill },
    expect.objectContaining({
      content: [{ type: "text", text: expect.stringContaining("FULL_SKILL_BODY") }],
    }),
  );
});

it("preserves tool failures when native history is reloaded", async () => {
  const native = await agent();
  const id = await native.create();
  const live = { ...observer(), activity: vi.fn() };
  faux.setResponses([
    fauxAssistantMessage(
      fauxToolCall("read", { path: "does-not-exist.txt" }, { id: "failed-read" }),
    ),
    fauxAssistantMessage("The file does not exist."),
  ]);
  await native.start(id, "Read the missing file", live);
  expect(live.activity).toHaveBeenCalledWith({
    type: "tool",
    toolCallId: "failed-read",
    status: "failed",
  });
  await native.dispose();
  const history = await loadAgUiHistory(await agent(), id);
  expect(history.find((message) => message.id === "call:failed-read")?._tool?.status).toBe(
    "failed",
  );
});

it.each([
  ["memory", { action: "read_memory", data: { id: "finding.md" } }],
  ["science_figure", { data: { artifact: { id: "figure", projectId: "study" } } }],
] as const)(
  "projects %s product results in live output and restored history",
  async (name, result) => {
    const call = vi.fn(async () => result);
    const productTools = {
      definitions: [
        {
          name,
          description: name,
          inputSchema: {
            type: "object",
            properties: { action: { type: "string" } },
            required: ["action"],
          },
        },
      ],
      call,
    };
    const native = await agent({ productTools });
    const id = await native.create();
    const sink = observer();
    faux.setResponses([
      fauxAssistantMessage(fauxToolCall(name, { action: "read" }, { id: "product-1" })),
      fauxAssistantMessage("Read"),
    ]);
    await native.start(id, "Read product data", sink);
    expect(call).toHaveBeenCalledWith(
      name,
      { action: "read" },
      "product-1",
      expect.any(AbortSignal),
    );
    expect(sink.tool).toHaveBeenLastCalledWith("product-1", name, { action: "read" }, result);
    const envelope = { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "tool_execution_end", result: envelope }),
      undefined,
    );
    await native.dispose();
    const restored = await agent({ productTools });
    const history = observer();
    await restored.read(id, history);
    expect(history.tool).toHaveBeenLastCalledWith("product-1", name, { action: "read" }, result);
    expect(history.raw).toHaveBeenCalledWith(
      expect.objectContaining({
        message: expect.objectContaining({ role: "toolResult", ...envelope }),
      }),
      undefined,
    );
    expect(call).toHaveBeenCalledTimes(1);
  },
);

it.each(["error", "length"] as const)("preserves the native %s outcome", async (stopReason) => {
  const native = await agent();
  faux.setResponses([fauxAssistantMessage("", { stopReason, errorMessage: "Provider failed" })]);
  const result = native.start(await native.create(), "Run", observer());
  if (stopReason === "error") await expect(result).rejects.toThrow("Provider failed");
  else await expect(result).resolves.toEqual({ stopReason: "max_tokens" });
});

it("rejects unsupported settings before a model request", async () => {
  const native = await agent();
  const id = await native.create();
  await expect(native.start(id, "Run", observer(), { model: "missing" })).rejects.toThrow(
    "unavailable Pi model",
  );
  await expect(native.start(id, "Run", observer(), { effort: "invalid" })).rejects.toThrow(
    "thinking level",
  );
  await expect(native.start(id, "Run", observer(), { mode: "plan" })).rejects.toThrow(
    "permission modes",
  );
  expect(faux.state.callCount).toBe(0);
});

it("reads active model settings without closing that session's SDK resources", async () => {
  const native = await agent();
  const id = await native.create();
  const started = Promise.withResolvers<void>();
  const reply = Promise.withResolvers<ReturnType<typeof fauxAssistantMessage>>();
  faux.setResponses([
    () => {
      started.resolve();
      return reply.promise;
    },
  ]);
  const result = native.start(id, "Wait", observer(), { model: "pi-test/second", effort: "low" });
  const cleaned: (string | undefined)[] = [];
  const unregister = registerSessionResourceCleanup((session) => cleaned.push(session));
  try {
    await started.promise;
    expect((await native.models(id)).current).toEqual({ model: "pi-test/second", effort: "low" });
    expect(cleaned).not.toContain(id);
  } finally {
    reply.resolve(fauxAssistantMessage("Finished"));
    await result;
    unregister();
  }
});

it("cancels preparation without later dispatching a prompt", async () => {
  const native = await agent();
  const id = await native.create();
  const pending = Promise.withResolvers<void>();
  const reload = vi
    .spyOn(DefaultResourceLoader.prototype, "reload")
    .mockReturnValue(pending.promise);
  const result = native.start(id, "Run", observer());
  await vi.waitFor(() => expect(reload).toHaveBeenCalled());
  await native.interrupt(id);
  pending.resolve();
  await expect(result).resolves.toEqual({ stopReason: "cancelled" });
  expect(faux.state.callCount).toBe(0);
});

it.each([
  ["input", "stop"],
  ["before_agent_start", "stop"],
  ["input", "dispose"],
  ["before_agent_start", "dispose"],
] as const)("cancels native %s preparation on %s before dispatch", async (hook, action) => {
  const extensionDir = join(cwd, ".pi", "extensions");
  await mkdir(extensionDir, { recursive: true });
  const entered = join(root, "entered");
  const release = join(root, "release");
  await writeFile(
    join(extensionDir, "preflight.ts"),
    `import { writeFile } from "node:fs/promises";
    import { existsSync } from "node:fs";
    import { setTimeout } from "node:timers/promises";
    export default function(pi) {
      pi.on(${JSON.stringify(hook)}, async () => {
        await writeFile(${JSON.stringify(entered)}, "entered");
        while (!existsSync(${JSON.stringify(release)})) await setTimeout(5);
      });
    }`,
  );
  const native = await agent();
  const id = await native.create();
  const sink = observer();
  faux.setResponses([fauxAssistantMessage("Must not run")]);
  const work = native.start(id, "Cancelled prompt", sink);
  let stopping: Promise<void> | undefined;
  try {
    await vi.waitFor(async () => expect(await readFile(entered, "utf8")).toBe("entered"));
    stopping = action === "stop" ? native.interrupt(id) : native.dispose();
    if (action === "stop") await stopping;
    expect(faux.state.callCount).toBe(0);
  } finally {
    await writeFile(release, "release");
  }
  const [result] = await Promise.all([work, stopping]);
  expect(faux.state.callCount).toBe(0);
  expect(result).toEqual({ stopReason: "cancelled" });
  expect(output(sink)).toBe("");
  expect(sink.tool).not.toHaveBeenCalled();
  if (action === "stop") {
    faux.setResponses([fauxAssistantMessage("Retry answer")]);
    await expect(native.start(id, "Retry", sink)).resolves.toEqual({ stopReason: "end_turn" });
    expect(output(sink)).toBe("Retry answer");
  }
});

it("preserves a native preflight failure when Stop was requested", async () => {
  const native = await agent();
  const id = await native.create();
  const runtime = await ModelRuntime.create();
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const failure = new Error("Native authentication failed");
  vi.mocked(runtime.hasConfiguredAuth).mockReturnValue(false);
  vi.spyOn(runtime, "checkAuth").mockImplementation(async () => {
    entered.resolve();
    await release.promise;
    throw failure;
  });
  const sink = observer();
  const work = native.start(id, "Run", sink);
  await entered.promise;
  try {
    expect(sink.raw).toHaveBeenCalledWith(
      expect.objectContaining({ type: "run_config" }),
      expect.anything(),
    );
    await native.interrupt(id);
  } finally {
    release.resolve();
  }
  await expect(work).rejects.toBe(failure);
  expect(faux.state.callCount).toBe(0);
});

it("steers and cancels one native run without cancelling another", async () => {
  const native = await agent();
  const first = await native.create();
  const second = await native.create();
  const started = Promise.withResolvers<void>();
  faux.setResponses([
    (_context, options) =>
      new Promise((resolve) => {
        started.resolve();
        options?.signal?.addEventListener(
          "abort",
          () => resolve(fauxAssistantMessage("", { stopReason: "aborted" })),
          { once: true },
        );
      }),
    fauxAssistantMessage("Other run"),
  ]);
  const result = native.start(first, "Wait", observer());
  await started.promise;
  await native.steer(first, "Change direction");
  const other = native.start(second, "Independent", observer());
  await native.interrupt(first);
  await expect(result).resolves.toEqual({ stopReason: "cancelled" });
  await expect(other).resolves.toEqual({ stopReason: "end_turn" });
});

it("keeps product calls inside Host permissions and execution history", async () => {
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd,
  });
  services.push(products);
  products.updatePolicy({ ...DEFAULT_POLICY, tools: ["memory.read"], delegation: false });
  await products.attachAgents("http://unused");
  expect(products.defaultHarness).toBe("pi");
  let context: Context | undefined;
  faux.setResponses([
    fauxAssistantMessage(
      fauxToolCall(
        "memory",
        {
          action: "update_core_memory",
          request: { content: "Denied", expectedRevision: "anything" },
        },
        { id: "denied-write" },
      ),
    ),
    (input) => {
      context = input;
      return fauxAssistantMessage("Denied correctly");
    },
  ]);
  const id = await products.rootAgent.create();
  const sink = observer();
  await products.rootAgent.start(id, "Attempt write", sink);
  expect(JSON.stringify(context?.messages)).toContain("memory.write");
  const failure = expect.objectContaining({
    content: [{ type: "text", text: expect.stringContaining("memory.write") }],
  });
  expect(sink.tool).toHaveBeenLastCalledWith("denied-write", "memory", expect.anything(), failure);
  const history = observer();
  await products.rootAgent.read(id, history);
  expect(history.tool).toHaveBeenLastCalledWith(
    "denied-write",
    "memory",
    expect.anything(),
    failure,
  );
  expect((await products.learning.core.read()).content).toBe("");
  expect(products.journal.recall({ sessionId: id })).toEqual(
    expect.arrayContaining([expect.objectContaining({ text: "Denied correctly" })]),
  );
});

it("delegates through nested Swarms into another Pi session with causal history", async () => {
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd,
  });
  services.push(products);
  await products.attachAgents("http://unused");
  let resultContext: Context | undefined;
  faux.setResponses([
    fauxAssistantMessage(
      fauxToolCall("swarm", { action: "create", id: "team", leadAgentId: "swarm" }),
    ),
    fauxAssistantMessage(
      fauxToolCall("swarm", { action: "create", id: "nested", leadAgentId: "team" }),
    ),
    fauxAssistantMessage(
      fauxToolCall(
        "swarm",
        { action: "send_message", agentId: "nested", text: "Child task" },
        { id: "delegate" },
      ),
    ),
    fauxAssistantMessage("Child answer"),
    (context) => {
      resultContext = context;
      return fauxAssistantMessage("Parent answer");
    },
  ]);
  const id = await products.rootAgent.create();
  const sink = observer();
  await products.rootAgent.start(id, "Delegate", sink);
  expect(output(sink)).toBe("Parent answer");
  expect(JSON.stringify(resultContext?.messages)).toContain("Child answer");
  const events = products.journal.read({ limit: 1000 }).events;
  const runs = events.filter((record) => record.event.type === "RUN_STARTED");
  expect(runs).toHaveLength(2);
  expect(runs[1]?.sessionId).not.toBe(id);
  expect(events.find((record) => record.id === runs[1]?.causedBy)?.event).toMatchObject({
    type: "TOOL_CALL_START",
    toolCallId: "delegate",
    toolCallName: "swarm",
  });
});
