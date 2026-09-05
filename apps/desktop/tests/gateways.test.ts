import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AgentCard, Role, TaskState } from "@a2a-js/sdk";
import { ClientFactory, JsonRpcTransportFactory } from "@a2a-js/sdk/client";
import { EventSchemas } from "@ag-ui/core";
import * as acp from "@agentclientprotocol/sdk";
import { createSwarm } from "@swarmx/swarm";
import { describe, expect, it, vi } from "vitest";
import { scopeSessions } from "../src/agent.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { acpAgent } from "../src/host/acp.js";
import { loadAgUiHistory, parseAgUiInput } from "../src/host/ag-ui.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";

describe("external gateways", () => {
  it("official ACP client reaches a recursive Swarm, native history, forms and cancellation", async () => {
    const leaf = fakeAgent();
    const nested = createSwarm("parent", createSwarm("child", leaf.agent));
    const updates: acp.SessionNotification[] = [];
    const client = acp
      .client({ name: "test" })
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        updates.push(params);
      })
      .onRequest(acp.methods.client.elicitation.create, () => ({
        action: "accept",
        content: { allow: true },
      }));
    const connection = client.connect(acpAgent(nested, process.cwd()));
    try {
      await connection.agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: { elicitation: { form: {} } },
      });
      const session = await connection.agent.request(acp.methods.agent.session.new, {
        cwd: process.cwd(),
        mcpServers: [],
      });
      await connection.agent.request(acp.methods.agent.session.load, {
        ...session,
        cwd: process.cwd(),
        mcpServers: [],
      });
      await connection.agent.request(acp.methods.agent.session.prompt, {
        ...session,
        prompt: [{ type: "text", text: "approve" }],
      });
      expect(leaf.answers).toEqual([{ allow: true }]);
      expect(updates.some(({ update }) => update.sessionUpdate === "agent_message_chunk")).toBe(
        true,
      );
      const pending = connection.agent.request(acp.methods.agent.session.prompt, {
        ...session,
        prompt: [{ type: "text", text: "wait" }],
      });
      await leaf.started.promise;
      await connection.agent.notify(acp.methods.agent.session.cancel, session);
      await expect(pending).resolves.toMatchObject({ stopReason: "cancelled" });
      expect(leaf.interrupt).toHaveBeenCalledOnce();
    } finally {
      connection.close();
    }
  });

  it("AG-UI uses official schemas and rejects foreign native session ids", async () => {
    const leaf = fakeAgent();
    const session = await leaf.agent.create();
    expect(parseAgUiInput(runInput(session, "hello")).threadId).toBe(session);
    expect(() => parseAgUiInput({ threadId: session })).toThrow();
    await expect(loadAgUiHistory(leaf.agent, session)).resolves.toMatchObject([
      { role: "user", content: "restored question" },
      { role: "assistant", content: "restored answer" },
    ]);
    expect(() => leaf.agent.start("claude:same-id", "wrong", observer)).toThrow(/does not belong/);
    expect(leaf.prompts).toEqual([]);
  });

  it("browser AG-UI resumes native interaction and preserves the local security boundary", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      expect((await fetch(`${gateway.origin}/api/v1/bootstrap`)).status).toBe(401);
      expect((await fetch(gateway.launchUrl, { redirect: "manual" })).status).toBe(401);
      expect(
        (
          await fetch(`${gateway.origin}/api/v1/sessions`, {
            headers: { cookie: browser.cookie, origin: "https://foreign.example" },
          })
        ).status,
      ).toBe(403);
      const first = events(await agUi(gateway, browser, runInput(browser.sessionId, "approve")));
      const finished = first.at(-1);
      if (finished?.type !== "RUN_FINISHED" || finished.outcome?.type !== "interrupt")
        throw new Error("No AG-UI interrupt");
      const interrupt = finished.outcome.interrupts[0];
      expect(interrupt).toMatchObject({ id: "permission-1", message: "Write result" });
      const second = events(
        await agUi(gateway, browser, {
          ...runInput(browser.sessionId, "approve"),
          resume: [{ interruptId: interrupt?.id, status: "resolved", payload: { allow: true } }],
        }),
      );
      expect(second.at(-1)).toMatchObject({ type: "RUN_FINISHED", outcome: { type: "success" } });
      expect(gateway.leaf.answers).toEqual([{ allow: true }]);
      const foreign = events(await agUi(gateway, browser, runInput("claude:same-id", "wrong")));
      expect(foreign.at(-1)).toMatchObject({
        type: "RUN_ERROR",
        message: expect.stringMatching(/does not belong/),
      });
    } finally {
      await gateway.dispose();
    }
  });

  it("disconnecting an active AG-UI stream interrupts the native Agent", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      const controller = new AbortController();
      const response = await fetch(`${gateway.origin}/api/ag-ui`, {
        method: "POST",
        headers: headers(gateway, browser),
        signal: controller.signal,
        body: JSON.stringify(runInput(browser.sessionId, "wait")),
      });
      await gateway.leaf.started.promise;
      controller.abort();
      await expect(response.text()).rejects.toThrow();
      await vi.waitFor(() => expect(gateway.leaf.interrupt).toHaveBeenCalledOnce());
    } finally {
      await gateway.dispose();
    }
  });

  it("official A2A client discovers the Card, sends to the same native Swarm, reads and cancels Tasks", async () => {
    const gateway = await createGateway();
    try {
      const card = AgentCard.fromJSON(
        await (await fetch(`${gateway.origin}/a2a/swarm/.well-known/agent-card.json`)).json(),
      );
      expect(card.supportedInterfaces[0]).toMatchObject({
        protocolBinding: "JSONRPC",
        protocolVersion: "1.0",
      });
      const client = await new ClientFactory({
        transports: [
          new JsonRpcTransportFactory({
            fetchImpl: (input, init) => {
              const headers = new Headers(init?.headers);
              headers.set("authorization", `Bearer ${gateway.token}`);
              return fetch(input, { ...init, headers });
            },
          }),
        ],
        preferredTransports: ["JSONRPC"],
      }).createFromAgentCard(card);
      const result = await client.sendMessage(message("hello"));
      expect(result).toMatchObject({
        status: { state: TaskState.TASK_STATE_COMPLETED },
        history: [{ role: Role.ROLE_USER }],
      });
      expect(JSON.stringify(result)).not.toContain("restored answer");
      expect(gateway.leaf.prompts).toEqual(["hello"]);
      const task = await client.sendMessage(message("wait", true));
      if (!("status" in task)) throw new Error("Expected a Task");
      await gateway.leaf.started.promise;
      await client.getTask({ tenant: "", id: task.id });
      await client.cancelTask({ tenant: "", id: task.id });
      expect(gateway.leaf.interrupt).toHaveBeenCalledOnce();
      await vi.waitFor(async () =>
        expect(await client.getTask({ tenant: "", id: task.id })).toMatchObject({
          status: { state: TaskState.TASK_STATE_CANCELED },
        }),
      );
    } finally {
      await gateway.dispose();
    }
  });
});

const observer: Observer = { text() {}, tool() {}, raw() {}, interact: async () => undefined };
function fakeAgent() {
  const prompts: string[] = [];
  const answers: unknown[] = [];
  const started = Promise.withResolvers<void>();
  const stopped = Promise.withResolvers<void>();
  const interrupt = vi.fn(async () => stopped.resolve());
  const agent = scopeSessions("codex", {
    name: "native",
    list: async () => [],
    create: async () => randomUUID(),
    read: async (_id, observer) => {
      observer.text("old-user", "restored question", "user");
      observer.text("old-answer", "restored answer");
    },
    start: async (_id, text, observer) => {
      prompts.push(text);
      if (text === "wait") {
        started.resolve();
        await stopped.promise;
      }
      if (text === "approve")
        answers.push(
          await observer.interact({
            id: "permission-1",
            title: "Write result",
            schema: {
              type: "object",
              properties: { allow: { type: "boolean" } },
              required: ["allow"],
            },
          }),
        );
      observer.text("native-answer", `answer:${text}`);
    },
    steer: async () => {},
    interrupt,
    dispose: async () => stopped.resolve(),
  } satisfies NativeAgent);
  return { agent, prompts, answers, started, interrupt };
}

async function createGateway() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-gateway-"));
  const leaf = fakeAgent();
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: { id: "test", label: "Test", root },
  });
  const host = await startHost({
    products,
    rendererRoot: root,
    workspace: products.options.workspace,
  });
  await products.attachAgents(host.internalUrl, host.internalToken, leaf.agent, "codex");
  return {
    leaf,
    origin: host.internalUrl,
    token: host.internalToken,
    launchUrl: host.issueLaunchUrl(),
    async dispose() {
      await host.dispose();
      await products.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}
async function browserSession(gateway: { launchUrl: string; origin: string }) {
  const response = await fetch(gateway.launchUrl, { redirect: "manual" });
  expect(response.headers.get("set-cookie")).toMatch(/HttpOnly.*SameSite=Strict/i);
  const cookie = response.headers.get("set-cookie")?.split(";", 1)[0];
  if (!cookie) throw new Error("No cookie");
  const session = await (
    await fetch(`${gateway.origin}/api/v1/sessions`, {
      method: "POST",
      headers: { cookie, origin: gateway.origin },
    })
  ).json();
  return { cookie, sessionId: session.sessionId as string };
}
function headers(gateway: { origin: string }, browser: { cookie: string }) {
  return { "content-type": "application/json", cookie: browser.cookie, origin: gateway.origin };
}
async function agUi(gateway: { origin: string }, browser: { cookie: string }, body: object) {
  return (
    await fetch(`${gateway.origin}/api/ag-ui`, {
      method: "POST",
      headers: headers(gateway, browser),
      body: JSON.stringify(body),
    })
  ).text();
}
function runInput(threadId: string, text: string) {
  return {
    threadId,
    runId: randomUUID(),
    state: {},
    messages: [{ id: randomUUID(), role: "user", content: text }],
    tools: [],
    context: [],
    forwardedProps: {},
  };
}
function events(body: string) {
  return body
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => EventSchemas.parse(JSON.parse(line.slice(6))));
}
function message(text: string, returnImmediately = false) {
  return {
    tenant: "",
    message: {
      messageId: randomUUID(),
      contextId: "",
      taskId: "",
      role: Role.ROLE_USER,
      parts: [
        { content: { $case: "text" as const, value: text }, filename: "", mediaType: "text/plain" },
      ],
      extensions: [],
      referenceTaskIds: [],
    },
    configuration: { acceptedOutputModes: ["text/plain"], returnImmediately },
  };
}
