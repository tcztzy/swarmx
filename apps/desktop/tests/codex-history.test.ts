import { mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import * as acp from "@agentclientprotocol/sdk";
import { afterEach, expect, it, vi } from "vitest";
import { scopeSessions } from "../src/agent.js";
import { createCodex } from "../src/agents/codex.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { acpAgent } from "../src/host/acp.js";
import { ProductServices } from "../src/host/product-services.js";

const require = createRequire(new URL("../package.json", import.meta.url));
const fixture = String.raw`#!@NODE@
import { appendFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { JSONRPCClient, JSONRPCServer, JSONRPCServerAndClient } from '@RPC@';
const send = ({jsonrpc, ...message}) => process.stdout.write(JSON.stringify(message) + '\n');
const rpc = new JSONRPCServerAndClient(new JSONRPCServer(), new JSONRPCClient(send));
let ownsFresh = false;
const model = id => ({ model: id, displayName: id, description: id, defaultReasoningEffort: 'low', supportedReasoningEfforts: [{ reasoningEffort: 'low', description: 'Low' }, { reasoningEffort: 'high', description: 'High' }] });
const thread = id => ({ id, historyMode: 'legacy', name: 'Fixture', cwd: id === 'foreign' ? '/different-directory' : process.cwd(), model: id === 'unknown-model' ? null : 'model-b', reasoningEffort: id === 'unknown-model' ? null : 'high', cliVersion:'fixture', turns: [] });
const handlers = {
  initialize: () => ({ userAgent: 'SwarmX test', codexHome: process.cwd() }),
  'config/read': () => ({ config: { model_provider: 'openai' } }),
  'permissionProfile/list': () => ({data: [{id:':workspace',description:'Workspace',allowed:true}],nextCursor:null}),
  'model/list': ({ cursor }) => ({ data: [model(cursor ? 'model-b' : 'model-a')], nextCursor: cursor ? null : 'next' }),
  'thread/start': () => {
    ownsFresh = true;
    return { thread: thread('fresh') };
  },
  'thread/resume': ({ threadId }) => {
    if (threadId === 'live') return { thread: thread(threadId) };
    throw new Error('thread ' + threadId + ' already has an active writer');
  },
  'turn/start': ({ threadId }) => {
    if (threadId === 'fresh' && !ownsFresh) throw new Error('thread not loaded: fresh');
    const turn = { id: 'live-turn', items: [], status: 'inProgress', error: null, startedAt: 1789013235, completedAt: null, durationMs: null };
    setTimeout(async () => {
      await rpc.client.notify('turn/started', { threadId, turn });
      for (const phase of ['commentary', 'final_answer']) {
        const item = { type: 'agentMessage', id: phase, phase, text: phase };
        await rpc.client.notify('item/started', { threadId, turnId: turn.id, item });
        await rpc.client.notify('item/agentMessage/delta', { threadId, turnId: turn.id, itemId: item.id, delta: item.text });
        await rpc.client.notify('item/completed', { threadId, turnId: turn.id, item });
      }
      await rpc.client.notify('turn/completed', { threadId, turn: { ...turn, status: 'completed', completedAt: 1789013310, durationMs: 74809 } });
    }, 0);
    return { turn };
  },
  'thread/read': ({ threadId, includeTurns }) => {
    if (threadId === 'fresh' && !ownsFresh) throw new Error('thread not loaded: fresh');
    if (threadId === 'missing') throw new Error('Thread not found');
    return { thread: {...thread(threadId), turns: includeTurns && threadId !== 'live' ? [
      { id: 'turn', startedAt: 1789013486, durationMs: 1027731, items: [
        { type: 'userMessage', id: 'question', content: [{ type: 'text', text: 'Saved question' }] },
        { type: 'agentMessage', id: 'answer', text: 'Saved answer', phase: 'final_answer' },
        { type: 'webSearch', id: 'search', query: 'Saved search', action: { type: 'search', query: 'Saved search' } },
      ] }
    ] : [] } };
  },
};
for (const [method, handler] of Object.entries(handlers))
  rpc.addMethod(method, params => {
    appendFileSync(process.env.SWARMX_TEST_TRACE, JSON.stringify({ method, params }) + '\n');
    return handler(params);
  });
createInterface({input:process.stdin}).on('line',line => rpc.receiveAndSend({jsonrpc:'2.0',...JSON.parse(line)}));
`
  .replace("@NODE@", process.execPath)
  .replace("@RPC@", pathToFileURL(require.resolve("json-rpc-2.0")).href);

const directories: string[] = [];
const agents: NativeAgent[] = [];
afterEach(async () => {
  await Promise.all(agents.splice(0).map((agent) => agent.dispose()));
  vi.unstubAllEnvs();
  await Promise.all(directories.splice(0).map((directory) => rm(directory, { recursive: true })));
});

async function open() {
  const cwd = await realpath(await mkdtemp(join(tmpdir(), "swarmx-codex-history-")));
  directories.push(cwd);
  const executable = join(cwd, "codex.mjs");
  const trace = join(cwd, "requests.jsonl");
  await writeFile(executable, fixture, { mode: 0o755 });
  vi.stubEnv("CODEX_PATH", executable);
  vi.stubEnv("SWARMX_TEST_TRACE", trace);
  const agent = await createCodex({ cwd, mcp: { command: "node", args: ["/bridge.js"], env: {} } });
  agents.push(agent);
  const observer: Observer = {
    text: vi.fn(),
    tool: vi.fn(),
    activity: vi.fn(),
    raw: vi.fn(),
    interact: vi.fn(),
  };
  return {
    cwd,
    agent,
    observer,
    async calls(): Promise<{ method: string; params: Record<string, unknown> }[]> {
      return (await readFile(trace, "utf8"))
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
    },
  };
}

it("reads fresh Codex settings repeatedly without releasing the first-turn runtime", async () => {
  const { agent, observer, calls } = await open();
  const id = await agent.create();
  await agent.read(id, observer);
  for (let i = 0; i < 2; i++) {
    const catalog = await agent.models(id);
    expect(catalog.current).toEqual({ model: "model-b", effort: "high" });
    expect(catalog.models.map((model) => model.id)).toEqual(["model-a", "model-b"]);
  }
  expect((await calls()).some(({ method }) => method === "turn/start")).toBe(false);
  await expect(
    agent.start(id, "First turn", observer, {
      model: "model-b",
      effort: "high",
      mode: ":workspace",
    }),
  ).resolves.toEqual({ stopReason: "end_turn" });
  expect((await calls()).some(({ method }) => method === "thread/resume")).toBe(false);
});

it("creates a Codex-backed Swarm through ACP and validates explicit first-turn settings", async () => {
  const { cwd, agent, calls } = await open();
  const products = await ProductServices.create({
    productHome: join(cwd, "home"),
    cwd,
  });
  try {
    await products.attachAgents("http://localhost", scopeSessions("codex", agent), "codex");
    const connection = acp.client().connect(acpAgent(products.rootAgent, cwd));
    try {
      const remote = connection.agent;
      await remote.request(acp.methods.agent.initialize, { protocolVersion: acp.PROTOCOL_VERSION });
      const session = await remote.request(acp.methods.agent.session.new, { cwd, mcpServers: [] });
      expect(session.configOptions).toContainEqual(
        expect.objectContaining({ id: "model", currentValue: "model-b" }),
      );
      for (const [configId, value] of [
        ["model", "model-b"],
        ["effort", "high"],
        ["mode", ":workspace"],
      ])
        await remote.request(acp.methods.agent.session.setConfigOption, {
          sessionId: session.sessionId,
          configId,
          value,
        });
      expect((await calls()).some(({ method }) => method === "turn/start")).toBe(false);
      await expect(
        remote.request(acp.methods.agent.session.prompt, {
          sessionId: session.sessionId,
          prompt: [{ type: "text", text: "First turn" }],
        }),
      ).resolves.toMatchObject({ stopReason: "end_turn" });
      expect((await calls()).filter(({ method }) => method === "turn/start")).toEqual([
        {
          method: "turn/start",
          params: expect.objectContaining({
            threadId: "fresh",
            model: "model-b",
            effort: "high",
            permissions: ":workspace",
          }),
        },
      ]);
    } finally {
      connection.close();
    }
  } finally {
    await products.dispose();
  }
});

it("reads externally owned native Codex history without acquiring its writer", async () => {
  const { agent, observer, calls } = await open();
  await agent.read("saved", observer);
  const catalog = await agent.models("saved");
  expect(catalog.current).toMatchObject({ model: "model-b", effort: "high" });
  expect(catalog.models.map((model) => model.id)).toEqual(["model-a", "model-b"]);
  expect(observer.text).toHaveBeenCalledWith("question", "Saved question", "user");
  expect(observer.text).toHaveBeenCalledWith("answer", "Saved answer", "assistant");
  expect(observer.activity).toHaveBeenCalledWith(
    expect.objectContaining({
      messageId: "answer",
      phase: "final_answer",
      turnId: "turn",
      startedAt: 1789013486,
      durationMs: 1027731,
    }),
  );
  expect(
    (await calls())
      .filter(({ method }) => method.startsWith("thread/"))
      .every(({ method }) => method === "thread/read"),
  ).toBe(true);
  await expect(agent.start("saved", "Must not dispatch", observer)).rejects.toThrow(
    "already has an active writer",
  );
  expect((await calls()).some(({ method }) => method === "turn/start")).toBe(false);
});

it("leaves unknown native model settings unresolved and rejects missing or foreign history", async () => {
  const { agent, observer, calls } = await open();
  expect((await agent.models("unknown-model")).current).toEqual({});
  for (const id of ["missing", "foreign"]) await expect(agent.read(id, observer)).rejects.toThrow();
  expect(observer.text).not.toHaveBeenCalled();
  expect(
    (await calls())
      .filter(({ method }) => method.startsWith("thread/"))
      .every(({ method }) => method === "thread/read"),
  ).toBe(true);
});

it("streams native message phases and timing before terminal completion without extra text", async () => {
  const { agent, observer, calls } = await open();
  await expect(agent.start("live", "Fixture prompt", observer)).resolves.toEqual({
    stopReason: "end_turn",
  });
  expect(vi.mocked(observer.text).mock.calls).toEqual([
    ["commentary", "commentary", "assistant"],
    ["final_answer", "final_answer", "assistant"],
  ]);
  expect(observer.activity).toHaveBeenCalledWith(
    expect.objectContaining({ messageId: "commentary", phase: "commentary", turnId: "live-turn" }),
  );
  expect(observer.activity).toHaveBeenCalledWith(
    expect.objectContaining({
      messageId: "final_answer",
      phase: "final_answer",
      turnId: "live-turn",
    }),
  );
  expect(observer.activity).toHaveBeenCalledWith({
    type: "message",
    turnId: "live-turn",
    startedAt: 1789013235,
    durationMs: 74809,
  });
  expect((await calls()).filter(({ method }) => method === "turn/start")).toHaveLength(1);
});
