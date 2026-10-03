import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AgentCard, Role, TaskState } from "@a2a-js/sdk";
import { ClientFactory, JsonRpcTransportFactory } from "@a2a-js/sdk/client";
import { type AGUIEvent, EventType } from "@ag-ui/core";
import * as acp from "@agentclientprotocol/sdk";
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import type { RunOptions } from "@swarmx/swarm";
import { createSwarm } from "@swarmx/swarm";
import { describe, expect, it, vi } from "vitest";
import manifest from "../package.json";
import { scopeSessions } from "../src/agent.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import { LogsQuerySchema } from "../src/bridge-contract.js";
import { acpAgent } from "../src/host/acp.js";
import { loadAgUiHistory, parseAgUiInput } from "../src/host/ag-ui.js";
import { HostOperations } from "../src/host/operations.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { SettingsStore } from "../src/host/settings-store.js";
import { DEFAULT_POLICY } from "../src/settings.js";
import { bridgeCall, bridgeClient, socketOf } from "./mcp-bridge-support.js";

describe("external gateways", () => {
  it("persists settings and keeps evidence bound to its execution directory", async () => {
    const gateway = await createGateway();
    try {
      const policy = { ...DEFAULT_POLICY, filesystem: "read-only" as const };
      expect((await gateway.operations.settings()).policy).toEqual(DEFAULT_POLICY);
      await gateway.operations.writeLanguage("en");
      expect(new SettingsStore(gateway.products.options.productHome).readLanguage()).toBe("en");
      expect((await gateway.operations.bootstrap()).language).toBe("en");
      await expect(gateway.operations.updateSettings({ ...policy, cpus: 0 })).rejects.toThrow();
      await gateway.operations.updateSettings(policy);
      expect(new SettingsStore(gateway.products.options.productHome).read().policy).toEqual(policy);
      const session = await gateway.operations.createSession("swarm");
      await gateway.products.rootAgent.start(session.sessionId, "Verified execution", observer);
      const run = gateway.products.journal
        .read({ session: session.sessionId })
        .events.find(({ event }) => event.type === EventType.RUN_STARTED);
      if (!run) throw new Error("Missing recorded execution");
      const sources = [`urn:swarmx:execution:${run.id}` as const];
      const evidence = await gateway.operations.evidence({ sources });
      expect(JSON.stringify(evidence)).toContain("Verified execution");
      await expect(
        gateway.operations.callTool("unknown", {}, randomUUID(), new AbortController().signal),
      ).rejects.toThrow("Unknown SwarmX product tool");
      const other = await createGateway();
      try {
        await expect(other.operations.evidence({ sources })).rejects.toThrow("another directory");
        expect((await other.operations.settings()).policy).toEqual(DEFAULT_POLICY);
        expect(await gateway.operations.evidence({ sources })).toEqual(evidence);
        expect((await gateway.operations.settings()).policy).toEqual(policy);
      } finally {
        await other.dispose();
      }
    } finally {
      await gateway.dispose();
    }
  });

  it("rejects permission changes while a product tool is active and settles it before journal shutdown", async () => {
    const gateway = await createGateway();
    const ready = Promise.withResolvers<void>();
    try {
      vi.spyOn(gateway.products.learning, "call").mockImplementation(
        async (_request, { signal }) => {
          ready.resolve();
          await new Promise<void>((_resolve, reject) =>
            signal.addEventListener("abort", () => reject(signal.reason), { once: true }),
          );
          throw new Error("unreachable");
        },
      );
      const operation = gateway.operations.callTool(
        "memory",
        { action: "read_core_memory", request: {} },
        randomUUID(),
        new AbortController().signal,
      );
      const rejected = expect(operation).rejects.toThrow("closing");
      await ready.promise;
      await expect(gateway.operations.updateSettings(DEFAULT_POLICY)).rejects.toThrow(
        "Stop active executions",
      );
      await gateway.host.dispose();
      await rejected;
    } finally {
      await gateway.dispose();
    }
  });

  it("steers and stops the exact active child without affecting its parent or a later run", async () => {
    const gateway = await createGateway();
    const ready = Promise.withResolvers<void>();
    const stopped = Promise.withResolvers<void>();
    let operation: Promise<void> | undefined;
    try {
      const session = await gateway.products.rootAgent.create();
      const childSession = await gateway.products.rootAgent.create();
      const steer = vi.spyOn(gateway.leaf.agent, "steer");
      const interrupt = vi.spyOn(gateway.leaf.agent, "interrupt").mockImplementation(async () => {
        stopped.resolve();
        return;
      });
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, text) => {
        if (text === "parent") {
          const context = {
            actorId: "parent",
            callId: "dispatch",
            signal: new AbortController().signal,
          };
          const { preparationId } = (await gateway.products.callTool(
            "swarm",
            { action: "prepare", task: "child", queries: ["agent-selection"] },
            context,
          )) as { preparationId: string };
          await gateway.products.callTool(
            "swarm",
            {
              action: "send_message",
              agentId: "codex",
              sessionId: childSession,
              text: "child",
              preparationId,
              reason: "Continue the selected native child session.",
            },
            context,
          );
        } else {
          ready.resolve();
          await stopped.promise;
        }

        return { stopReason: "end_turn" as const };
      });
      operation = gateway.products.rootAgent.start(session, "parent", observer);
      await ready.promise;
      const snapshot = await gateway.operations.logs(
        LogsQuerySchema.parse({ session, descendants: "true" }),
      );
      const child = snapshot.events.find(
        (record) =>
          record.sessionId === childSession && record.event.type === EventType.RUN_STARTED,
      );
      expect(child).toBeDefined();
      expect(snapshot.activeRunIds).toContain(child?.runId);
      const runId = child?.runId ?? "";
      await expect(
        gateway.operations.controlRun(runId, { action: "steer", text: " " }),
      ).rejects.toThrow();
      await gateway.operations.controlRun(runId, {
        action: "steer",
        text: "check the methods",
      });
      expect(steer).toHaveBeenCalledWith(childSession, "check the methods");
      await gateway.operations.controlRun(runId, { action: "cancel" });
      await operation;
      expect(interrupt).toHaveBeenCalledExactlyOnceWith(childSession);
      const laterStarted = Promise.withResolvers<void>();
      const laterDone = Promise.withResolvers<void>();
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async () => {
        laterStarted.resolve();
        await laterDone.promise;

        return { stopReason: "end_turn" as const };
      });
      const later = gateway.products.rootAgent.start(childSession, "later", observer);
      await laterStarted.promise;
      try {
        await expect(gateway.operations.controlRun(runId, { action: "cancel" })).rejects.toThrow(
          "no longer active",
        );
        expect(interrupt).toHaveBeenCalledTimes(1);
      } finally {
        laterDone.resolve();
        await later;
      }
      const log = gateway.products.journal.read({ run: runId }).events;
      expect(
        log.some(
          ({ event }) => event.type === EventType.CUSTOM && event.name === "swarmx.input.steered",
        ),
      ).toBe(true);
      expect(
        log.findLast(({ event }) => event.type === EventType.RUN_FINISHED)?.event,
      ).toMatchObject({
        type: EventType.RUN_FINISHED,
        result: { interruptionRequested: true },
      });
    } finally {
      stopped.resolve();
      await operation;
      await gateway.dispose();
    }
  });

  it("queues parallel delegated confirmations through the parent's AG-UI interaction flow", async () => {
    const gateway = await createGateway();
    let client: Client | undefined;
    const answers: unknown[] = [];
    try {
      const session = await gateway.operations.createSession("swarm");
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (sessionId, text, output) => {
        if (text === "parent") {
          const token = randomUUID();
          gateway.products.mcpExecutions.set(token, { sessionId, runId: output.executionId });
          client = await bridgeClient(socketOf(gateway.products), token);
          const results = await Promise.all(
            ["first", "second"].map(async (text) => {
              const preparation = await client?.callTool({
                name: "swarm",
                arguments: { action: "prepare", task: text, queries: ["agent-selection"] },
              });
              expect(preparation?.isError).not.toBe(true);
              return client?.callTool({
                name: "swarm",
                arguments: {
                  action: "send_message",
                  agentId: "codex",
                  text,
                  preparationId: preparation?.structuredContent?.preparationId,
                  reason: "Use the selected native agent for the parallel child task.",
                },
              });
            }),
          );
          expect(results.every((result) => !result.isError)).toBe(true);
          output.text("summary", "both children finished");
        } else {
          answers.push(
            await output.interact({
              id: "same-native-id",
              title: text,
              schema: { type: "object", properties: { allow: { type: "boolean" } } },
            }),
          );
        }

        return { stopReason: "end_turn" as const };
      });
      let stream = await runAgUi(gateway, input(session.sessionId, "parent"));
      const ids = new Set<string>();
      for (const allow of [false, true]) {
        const finished = stream.find((event) => event.type === EventType.RUN_FINISHED);
        if (finished?.type !== EventType.RUN_FINISHED || finished.outcome?.type !== "interrupt")
          throw new Error("Expected child confirmation");
        const pending = finished.outcome.interrupts[0];
        if (!pending) throw new Error("Missing confirmation");
        expect(pending.message).toMatch(/codex.*codex:/);
        ids.add(pending.id);
        const child = gateway.products.journal
          .activeRuns()
          .find(
            (run) =>
              run.sessionId !== session.sessionId && pending.id.startsWith(`${run.sessionId}:`),
          );
        expect(child).toBeDefined();
        await expect(
          gateway.operations.controlRun(child?.runId ?? "", { action: "cancel" }),
        ).rejects.toThrow("pending confirmation");
        stream = await runAgUi(gateway, {
          ...input(session.sessionId, ""),
          resume: [{ interruptId: pending.id, status: "resolved", payload: { allow } }],
        });
      }
      expect(ids.size).toBe(2);
      expect(answers).toEqual([{ allow: false }, { allow: true }]);
      expect(stream.map(streamText).join("")).toContain("both children finished");
      expect(gateway.products.journal.activeRuns()).toEqual([]);
      const interactions = gateway.products.journal
        .read()
        .events.filter(
          ({ event }) =>
            event.type === EventType.CUSTOM && event.name === "swarmx.interaction.answered",
        );
      expect(interactions).toHaveLength(2);
    } finally {
      await client?.close();
      await gateway.dispose();
    }
  });

  it("delivers sensitive interaction answers without putting them in the execution journal", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      const answers: unknown[] = [];
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, _text, output) => {
        answers.push(
          await output.interact({
            id: "credential",
            title: "Native credential",
            sensitive: true,
            schema: {
              type: "object",
              properties: { answer: { type: "string", format: "password" } },
            },
          }),
        );
        return { stopReason: "end_turn" };
      });
      const first = await runAgUi(gateway, input(session.sessionId, "work"));
      const finished = first.find((event) => event.type === EventType.RUN_FINISHED);
      if (finished?.type !== EventType.RUN_FINISHED || finished.outcome?.type !== "interrupt")
        throw new Error("Expected credential request");
      const pending = finished.outcome.interrupts[0];
      if (!pending) throw new Error("Missing credential request");
      await runAgUi(gateway, {
        ...input(session.sessionId, ""),
        resume: [
          {
            interruptId: pending.id,
            status: "resolved",
            payload: { answer: "private-secret-value" },
          },
        ],
      });
      expect(answers).toEqual([{ answer: "private-secret-value" }]);
      const records = gateway.products.journal.read().events;
      expect(JSON.stringify(records)).not.toContain("private-secret-value");
      expect(
        records.find(
          ({ event }) =>
            event.type === EventType.CUSTOM && event.name === "swarmx.interaction.answered",
        )?.event,
      ).toMatchObject({ value: { id: "credential", status: "answered", redacted: true } });
    } finally {
      await gateway.dispose();
    }
  });

  it("binds MCP bridge credentials to one Host run and rejects stale or inactive endpoints", async () => {
    const gateway = await createGateway();
    gateway.products.mcpExecutions.set("test-process", null);
    const client = await bridgeClient(socketOf(gateway.products), "test-process");
    try {
      const session = await gateway.operations.createSession("swarm");
      const invoke = vi.spyOn(gateway.products, "callTool");
      const call = () =>
        client.callTool({
          name: "swarm",
          arguments: { action: "status" },
        });
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, _text, output) => {
        expect((await call()).isError).toBe(true);
        if (!output.executionId) throw new Error("Missing execution identity");
        gateway.products.mcpExecutions.set("test-process", {
          sessionId: session.sessionId,
          runId: "stale-run",
        });
        expect((await call()).isError).toBe(true);
        gateway.products.mcpExecutions.set("test-process", {
          sessionId: session.sessionId,
          runId: output.executionId,
        });
        expect((await call()).isError).not.toBe(true);
        gateway.products.mcpExecutions.set("test-process", null);
        return { stopReason: "end_turn" as const };
      });
      for (const turnId of ["first", "second"])
        await gateway.products.rootAgent.start(session.sessionId, turnId, observer);
      expect((await call()).isError).toBe(true);
      expect(invoke).toHaveBeenCalledTimes(4);
      const records = gateway.products.journal.read({ session: session.sessionId }).events;
      const runs = records.filter(({ event }) => event.type === EventType.RUN_STARTED);
      const calls = records.filter(({ event }) => event.type === EventType.TOOL_CALL_START);
      expect(calls.map(({ runId }) => runId)).toEqual(runs.map(({ runId }) => runId));
      gateway.products.mcpExecutions.delete("test-process");
      await expect(bridgeClient(socketOf(gateway.products), "test-process")).rejects.toThrow();
    } finally {
      await client.close();
      await gateway.dispose();
    }
  });

  it("preserves causal MCP Memory/delegation records and reads them after restart without a Harness", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, text, output) => {
        if (text === "parent") {
          if (!output.executionId) throw new Error("Missing execution ID");
          const token = randomUUID();
          gateway.products.mcpExecutions.set(token, {
            sessionId: session.sessionId,
            runId: output.executionId,
          });
          const client = await bridgeClient(socketOf(gateway.products), token);
          try {
            const memory = await client.callTool({
              name: "memory",
              arguments: { action: "read_core_memory", request: {} },
            });
            expect(memory.isError).not.toBe(true);
            const preparation = await client.callTool({
              name: "swarm",
              arguments: { action: "prepare", task: "child", queries: ["agent-selection"] },
            });
            expect(preparation.isError).not.toBe(true);
            const delegated = await client.callTool({
              name: "swarm",
              arguments: {
                action: "send_message",
                agentId: "codex",
                text: "child",
                preparationId: preparation.structuredContent?.preparationId,
                reason: "Use the selected native agent for the child task.",
              },
            });
            expect(delegated.isError).not.toBe(true);
          } finally {
            await client.close();
          }
        }
        output.raw({ nativeId: text, extra: { untouched: [null, "原始记录"] } });
        output.text(text, `${text} answer`);

        return { stopReason: "end_turn" as const };
      });
      await gateway.products.rootAgent.start(session.sessionId, "parent", observer);
      const saved = await gateway.operations.logs(LogsQuerySchema.parse({}));
      const starts = saved.events.filter(({ event }) => event.type === EventType.RUN_STARTED);
      expect(starts).toHaveLength(2);
      const delegated = saved.events.findLast(
        ({ event }) => event.type === EventType.TOOL_CALL_START && event.toolCallName === "swarm",
      );
      expect(starts[1]?.causedBy).toBe(delegated?.id);
      const result = saved.events.find(
        ({ event }) =>
          event.type === EventType.TOOL_CALL_RESULT &&
          event.content.includes('"action":"read_core_memory"'),
      );
      if (result?.event.type !== EventType.TOOL_CALL_RESULT)
        throw new Error("Missing Memory result");
      expect(JSON.parse(result.event.content)).toMatchObject({
        action: "read_core_memory",
        data: { content: "" },
      });
      expect(result.runId).toBe(starts[0]?.runId);
      expect(result.sessionId).toBe(session.sessionId);
      expect(JSON.stringify(saved)).not.toContain(gateway.token);
      expect(
        saved.events.some(
          ({ event }) =>
            event.type === EventType.RAW && JSON.stringify(event.event).includes("原始记录"),
        ),
      ).toBe(true);
      expect(LogsQuerySchema.safeParse({ limit: 0 }).success).toBe(false);

      await gateway.host.dispose();
      await gateway.products.dispose();
      const reopened = await ProductServices.create(gateway.products.options);
      try {
        expect({
          ...reopened.journal.read(),
          activeRunIds: reopened.journal.activeRuns().map((run) => run.runId),
        }).toEqual(saved);
      } finally {
        await reopened.dispose();
      }
    } finally {
      await gateway.dispose();
    }
  });

  it("restores memory snapshots and pending writes after restart and requires explicit approval", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      gateway.products.settings.writeMemory({ writeApproval: true });
      const note = await gateway.products.learning.core.read();
      const frozen = await gateway.products.learning.snapshot(session.sessionId);
      await gateway.operations.callTool(
        "memory",
        {
          action: "update_core_memory",
          request: { content: "中文优先", expectedRevision: note.revision },
        },
        "memory-proposal",
        new AbortController().signal,
      );
      await gateway.host.dispose();
      await gateway.products.dispose();
      const products = await ProductServices.create(gateway.products.options);
      const host = await startHost({
        products,
        agent: fakeAgent().agent,
        agentId: "codex",
      });
      try {
        const operations = new HostOperations(host);
        expect(await products.learning.snapshot(session.sessionId)).toBe(frozen);
        const [pending] = (await products.learning.status()).pending;
        if (!pending) throw new Error("Pending memory did not survive restart");
        products.mcpExecutions.set("restart-process", null);
        const unbound = await bridgeCall(socketOf(products), "restart-process", "memory", {
          action: "memory_decide",
          request: { id: pending.id, decision: "approve" },
        });
        expect(unbound.isError).toBe(true);
        expect(JSON.stringify(unbound)).toContain("not bound to an active execution");
        await expect(bridgeClient(socketOf(products), "unknown-process")).rejects.toThrow();
        expect((await products.learning.core.read()).content).toBe("");
        await operations.callTool(
          "memory",
          {
            action: "memory_decide",
            request: { id: pending.id, decision: "approve" },
          },
          randomUUID(),
          new AbortController().signal,
        );
        expect((await products.learning.core.read()).content).toBe("中文优先");
        expect((await products.learning.status()).pending).toEqual([]);
        expect(await products.learning.snapshot(session.sessionId)).toBe(frozen);
        expect(await products.learning.snapshot("codex:new-session")).toContain("中文优先");
      } finally {
        await host.dispose();
        await products.dispose();
      }
    } finally {
      await gateway.dispose();
    }
  });

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

  it("resumes the same native session after an external ACP client reconnects", async () => {
    const leaf = fakeAgent();
    const create = vi.spyOn(leaf.agent, "create");
    const start = vi.spyOn(leaf.agent, "start");
    const updates: acp.SessionNotification[] = [];
    let sessionId = "";
    for (const text of ["first analysis", "continue the saved analysis"]) {
      const connection = acp
        .client({ name: "domain-client" })
        .onNotification(acp.methods.client.session.update, ({ params }) => updates.push(params))
        .connect(acpAgent(leaf.agent, process.cwd()));
      try {
        await connection.agent.request(acp.methods.agent.initialize, {
          protocolVersion: acp.PROTOCOL_VERSION,
        });
        if (!sessionId) {
          const session = await connection.agent.request(acp.methods.agent.session.new, {
            cwd: process.cwd(),
            mcpServers: [],
          });
          sessionId = session.sessionId;
        } else {
          await expect(
            connection.agent.request(acp.methods.agent.session.prompt, {
              sessionId,
              prompt: [{ type: "text", text }],
            }),
          ).rejects.toThrow("Create, load or resume");
          updates.length = 0;
          await connection.agent.request(acp.methods.agent.session.resume, {
            sessionId,
            cwd: process.cwd(),
            mcpServers: [],
          });
          expect(updates).toEqual([]);
        }
        await expect(
          connection.agent.request(acp.methods.agent.session.prompt, {
            sessionId,
            prompt: [{ type: "text", text }],
          }),
        ).resolves.toMatchObject({ stopReason: "end_turn" });
      } finally {
        connection.close();
        await connection.closed;
      }
    }
    expect(create).toHaveBeenCalledOnce();
    expect(start.mock.calls.map(([id, text]) => [id, text])).toEqual([
      [sessionId, "first analysis"],
      [sessionId, "continue the saved analysis"],
    ]);
  });

  it("AG-UI uses official schemas and rejects foreign native session ids", async () => {
    const leaf = fakeAgent();
    const session = await leaf.agent.create();
    expect(parseAgUiInput(input(session, "hello")).threadId).toBe(session);
    expect(() => parseAgUiInput({ threadId: session })).toThrow();
    await expect(loadAgUiHistory(leaf.agent, session)).resolves.toMatchObject([
      { role: "user", content: "restored question" },
      { role: "assistant", content: "restored answer" },
    ]);
    expect(() => leaf.agent.start("claude:same-id", "wrong", observer)).toThrow(/does not belong/);
    expect(leaf.prompts).toEqual([]);
  });

  it("preserves native phases, duration and tool kind/status through direct Swarm history", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.products.rootAgent.create();
      vi.spyOn(gateway.leaf.agent, "read").mockImplementation(async (_id, observer) => {
        observer.text("question", "Saved question", "user");
        for (const phase of ["commentary", "final_answer"] as const) {
          observer.activity?.({
            type: "message",
            messageId: phase,
            phase,
            turnId: "turn",
            startedAt: 1789013486,
            durationMs: 1027731,
          });
          observer.text(phase, phase);
        }
        observer.activity?.({
          type: "tool",
          toolCallId: "shell",
          kind: "execute",
          status: "in_progress",
        });
        observer.tool("shell", "exit 2", { command: "exit 2" });
        observer.activity?.({ type: "tool", toolCallId: "shell", status: "failed" });
        observer.tool(
          "shell",
          "exit 2",
          { command: "exit 2" },
          { formatted_output: "failed command", exit_code: 2 },
        );
      });
      const history = await loadAgUiHistory(gateway.products.rootAgent, session);
      expect(history).toEqual([
        { id: "question", role: "user", content: "Saved question" },
        ...["commentary", "final_answer"].map((phase) => ({
          id: phase,
          role: "assistant",
          content: phase,
          _meta: { phase, turnId: "turn", startedAt: 1789013486, durationMs: 1027731 },
        })),
        {
          id: "call:shell",
          role: "assistant",
          _tool: { kind: "execute", status: "failed" },
          toolCalls: [
            {
              id: "shell",
              type: "function",
              function: { name: "exit 2", arguments: JSON.stringify({ command: "exit 2" }) },
            },
          ],
        },
        {
          id: "result:shell",
          role: "tool",
          toolCallId: "shell",
          content: JSON.stringify({ formatted_output: "failed command", exit_code: 2 }),
        },
      ]);
    } finally {
      await gateway.dispose();
    }
  });

  it("AG-UI resumes native interaction and streams foreign session errors", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      const first = await runAgUi(gateway, input(session.sessionId, "approve"));
      const finished = first.find((event) => event.type === EventType.RUN_FINISHED);
      if (finished?.type !== EventType.RUN_FINISHED || finished.outcome?.type !== "interrupt")
        throw new Error("No AG-UI interrupt");
      const interrupt = finished.outcome.interrupts[0];
      expect(interrupt).toMatchObject({ id: "permission-1", message: "Write result" });
      const second = await runAgUi(gateway, {
        ...input(session.sessionId, "approve"),
        resume: [
          {
            interruptId: interrupt?.id,
            status: "resolved",
            payload: { allow: true },
          },
        ],
      });
      expect(second.at(-1)).toMatchObject({
        type: EventType.RUN_FINISHED,
        outcome: { type: "success" },
      });
      expect(gateway.leaf.answers).toEqual([{ allow: true }]);
      const foreign = await runAgUi(gateway, input("claude:same-id", "wrong"));
      expect(foreign.at(-1)).toMatchObject({
        type: EventType.RUN_ERROR,
        message: expect.stringMatching(/does not belong/),
      });
    } finally {
      await gateway.dispose();
    }
  });

  it("cancelling an active AG-UI bridge run interrupts the native Agent", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      const bridge = await gateway.products.agUi("swarm");
      const events: AGUIEvent[] = [];
      const running = bridge.run(parseAgUiInput(input(session.sessionId, "wait")), {
        event: (event) => events.push(event),
      });
      await gateway.leaf.started.promise;
      await bridge.cancel(session.sessionId);
      await running;
      await vi.waitFor(() => expect(gateway.leaf.interrupt).toHaveBeenCalledOnce());
      expect(events.at(-1)).toMatchObject({
        type: EventType.RUN_FINISHED,
        result: { stopReason: "cancelled" },
      });
    } finally {
      await gateway.dispose();
    }
  });

  it("serves native model catalogs and forwards only per-turn model settings", async () => {
    const gateway = await createGateway();
    try {
      const session = await gateway.operations.createSession("swarm");
      expect(await gateway.operations.bootstrap()).toMatchObject({ defaultHarness: "codex" });
      expect(await gateway.operations.models("swarm", session.sessionId)).toEqual({
        models: [{ id: "native-model", name: "Native", efforts: [{ id: "high", name: "High" }] }],
        current: {},
      });
      await expect(gateway.operations.models("swarm", "claude:foreign")).rejects.toThrow();
      const forwarded = {
        ...input(session.sessionId, "hello"),
        forwardedProps: { modelName: "native-model", reasoningEffort: "high" },
      };
      expect((await runAgUi(gateway, forwarded)).at(-1)).toMatchObject({
        type: EventType.RUN_FINISHED,
      });
      expect(gateway.leaf.settings).toEqual([
        {
          model: "native-model",
          effort: "high",
          instructions: expect.stringContaining("read_memory_guide"),
        },
      ]);
      for (const forwardedProps of [
        { modelName: "native-model --global" },
        { modelName: "native-model", approvalPolicy: "never" },
      ]) {
        const result = await runAgUi(gateway, { ...forwarded, forwardedProps });
        expect(result.at(-1)?.type).toBe(EventType.RUN_ERROR);
      }
      expect(gateway.leaf.prompts).toEqual(["hello"]);
    } finally {
      await gateway.dispose();
    }
  });

  it.each([
    ["Codex connection closed", "Internal error: Codex connection closed"],
    [{ details: "Codex connection closed" }, "Internal error: Codex connection closed"],
    [{ message: "Codex connection closed" }, "Internal error: Codex connection closed"],
    [{ opaque: "not a user-facing message" }, "Internal error"],
  ])(
    "preserves readable structured errors when native history fails (%j)",
    async (data, message) => {
      const gateway = await createGateway();
      try {
        const session = await gateway.operations.createSession("swarm");
        const read = vi
          .spyOn(gateway.leaf.agent, "read")
          .mockRejectedValueOnce(Object.assign(new Error("Internal error"), { data }));
        await expect(gateway.operations.history("swarm", session.sessionId)).rejects.toThrow(
          message,
        );
        await expect(gateway.operations.history("swarm", session.sessionId)).resolves.toEqual({
          supported: true,
          messages: [
            { id: "old-user", role: "user", content: "restored question" },
            { id: "old-answer", role: "assistant", content: "restored answer" },
          ],
        });
        expect(read).toHaveBeenCalledTimes(2);
        expect(gateway.leaf.prompts).toEqual([]);
      } finally {
        await gateway.dispose();
      }
    },
  );

  it("official A2A client follows the advertised endpoint and uses the Host directory", async () => {
    const gateway = await createGateway();
    try {
      const card = AgentCard.fromJSON(
        await (await fetch(`${gateway.origin}/a2a/swarm/.well-known/agent-card.json`)).json(),
      );
      expect(card.version).toBe(manifest.version);
      expect(card.supportedInterfaces[0]).toMatchObject({
        protocolBinding: "JSONRPC",
        protocolVersion: "1.0",
        url: `${gateway.origin}/a2a/swarm`,
      });
      const client = await new ClientFactory({
        transports: [
          new JsonRpcTransportFactory({
            fetchImpl: (url, init) => {
              const headers = new Headers(init?.headers);
              headers.set("authorization", `Bearer ${gateway.token}`);
              return fetch(url, { ...init, headers });
            },
          }),
        ],
        preferredTransports: ["JSONRPC"],
      }).createFromAgentCard(card);
      const directory = gateway.products.options.cwd;
      const result = await client.sendMessage(message("hello"));
      expect(result).toMatchObject({
        status: { state: TaskState.TASK_STATE_COMPLETED },
        history: [{ role: Role.ROLE_USER }],
      });
      expect(JSON.stringify(result)).not.toContain("restored answer");
      expect(gateway.leaf.prompts).toEqual(["hello"]);
      if (!("contextId" in result)) throw new Error("Expected a Task");
      const resumed = await client.sendMessage(message("again", undefined, result.contextId));
      expect(resumed).toMatchObject({ status: { state: TaskState.TASK_STATE_COMPLETED } });
      expect(gateway.leaf.prompts).toEqual(["hello", "again"]);
      const outside = await mkdtemp(join(tmpdir(), "swarmx-outside-"));
      try {
        await expect(
          client.sendMessage(message("outside", { directory: outside })),
        ).rejects.toThrow(/does not match the Host working directory/);
      } finally {
        await rm(outside, { recursive: true, force: true });
      }
      const task = await client.sendMessage(message("wait", { directory }, undefined, true));
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

function fakeAgent(historySupported = true) {
  const prompts: string[] = [];
  const answers: unknown[] = [];
  const settings: Array<RunOptions | undefined> = [];
  const started = Promise.withResolvers<void>();
  const stopped = Promise.withResolvers<void>();
  const interrupt = vi.fn(async () => {
    stopped.resolve();
    return;
  });
  const agent = scopeSessions("codex", {
    name: "native",
    capabilities: { ...HARNESS_CAPABILITIES.codex, history: historySupported },
    models: async () => ({
      models: [{ id: "native-model", name: "Native", efforts: [{ id: "high", name: "High" }] }],
      current: {},
    }),
    list: async () => [],
    create: async () => randomUUID(),
    read: async (_id, observer) => {
      observer.text("old-user", "restored question", "user");
      observer.text("old-answer", "restored answer");
    },
    start: async (_id, text, observer, options) => {
      prompts.push(text);
      settings.push(options);
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
      return { stopReason: text === "wait" ? "cancelled" : "end_turn" };
    },
    steer: async () => {},
    interrupt,
    dispose: async () => stopped.resolve(),
  } satisfies NativeAgent);
  return { agent, prompts, answers, settings, started, interrupt };
}

async function createGateway(historySupported = true) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-gateway-"));
  const leaf = fakeAgent(historySupported);
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
  });
  products.settings.writeMemory({ autoReview: false });
  const host = await startHost({
    products,
    agent: leaf.agent,
    agentId: "codex",
  });
  return {
    leaf,
    products,
    host,
    operations: new HostOperations(host),
    origin: host.origin,
    token: host.token,
    async dispose() {
      await host.dispose();
      await products.dispose();
      await rm(root, { recursive: true, force: true });
    },
  };
}

async function runAgUi(
  gateway: Awaited<ReturnType<typeof createGateway>>,
  body: object,
): Promise<AGUIEvent[]> {
  const bridge = await gateway.products.agUi("swarm");
  const events: AGUIEvent[] = [];
  await bridge.run(parseAgUiInput(body), { event: (event) => events.push(event) });
  return events;
}

function streamText(event: AGUIEvent): string {
  return event.type === EventType.TEXT_MESSAGE_CONTENT ? event.delta : "";
}

function input(threadId: string, text: string) {
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

function message(
  text: string,
  directory?: { directory: string },
  contextId = "",
  returnImmediately = false,
) {
  return {
    tenant: "",
    message: {
      messageId: randomUUID(),
      contextId,
      taskId: "",
      role: Role.ROLE_USER,
      parts: [
        {
          content: { $case: "text" as const, value: text },
          filename: "",
          mediaType: "text/plain",
        },
      ],
      extensions: [],
      referenceTaskIds: [],
      ...(directory === undefined ? {} : { metadata: { swarmx: directory } }),
    },
    configuration: { acceptedOutputModes: ["text/plain"], returnImmediately },
  };
}

it("reports unsupported native history without fabricating a transcript", async () => {
  const gateway = await createGateway(false);
  try {
    const { sessionId } = await gateway.operations.createSession("swarm");
    const read = vi.spyOn(gateway.leaf.agent, "read");
    const models = vi.spyOn(gateway.leaf.agent, "models");
    expect(await gateway.operations.history("swarm", sessionId)).toEqual({ supported: false });
    expect(models).toHaveBeenCalledWith(sessionId);
    expect(read).not.toHaveBeenCalled();
    expect(gateway.leaf.prompts).toEqual([]);
    models.mockRejectedValueOnce(new Error("Native session missing"));
    await expect(gateway.operations.history("swarm", "codex:missing")).rejects.toThrow(
      "Native session missing",
    );
  } finally {
    await gateway.dispose();
  }
});
