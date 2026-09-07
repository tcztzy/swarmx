import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AgentCard, Role, TaskState } from "@a2a-js/sdk";
import { ClientFactory, JsonRpcTransportFactory } from "@a2a-js/sdk/client";
import { EventSchemas, EventType } from "@ag-ui/core";
import * as acp from "@agentclientprotocol/sdk";
import { Client as McpClient } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import type { RunOptions } from "@swarmx/swarm";
import { createSwarm } from "@swarmx/swarm";
import { describe, expect, it, vi } from "vitest";
import { scopeSessions } from "../src/agent.js";
import type { NativeAgent, Observer } from "../src/agents/types.js";
import { HARNESS_CAPABILITIES } from "../src/agents/types.js";
import type { ExecutionRecord } from "../src/execution-record.js";
import { acpAgent } from "../src/host/acp.js";
import { loadAgUiHistory, parseAgUiInput } from "../src/host/ag-ui.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";
import { resolveWorkspace, SettingsStore } from "../src/host/workspace-settings.js";
import { DEFAULT_POLICY } from "../src/settings.js";

describe("external gateways", () => {
  it("keeps running projects, browser APIs and MCP bound to their own directory", async () => {
    const gateway = await createGateway();
    let running: Promise<void> | undefined;
    try {
      const browser = await browserSession(gateway);
      const first = gateway.products.options.workspace;
      const directory = join(first.root, "second-project");
      await mkdir(directory);
      const request = async (project: string, path: string, body?: object, method = "POST") =>
        fetch(`${gateway.origin}/projects/${project}/api/v1/${path}`, {
          headers: headers(gateway, browser),
          ...(body ? { method, body: JSON.stringify(body) } : {}),
        });
      expect(
        (
          await fetch(`${gateway.origin}/projects/${first.id}/api/v1/projects`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ label: "Unauthorized", root: directory }),
          })
        ).status,
      ).toBe(401);
      const added = await request(first.id, "projects", {
        label: "Second project",
        root: directory,
      });
      expect(added.status).toBe(201);
      const second = await added.json();
      expect(
        (await request(first.id, "settings", { ...DEFAULT_POLICY, cpus: 3 }, "PUT")).status,
      ).toBe(200);
      const collection = await request(first.id, "tools/science_notebook", {
        action: "create_project",
        request: { requestId: randomUUID(), title: "Only first project" },
      });
      expect(collection.status).toBe(200);
      running = gateway.products.rootAgent.start(browser.sessionId, "wait", {
        text() {},
        tool() {},
        raw() {},
        interact: async () => undefined,
      });
      await gateway.leaf.started.promise;
      expect((await request(first.id, `projects/${second.id}/open`, {})).status).toBe(200);
      expect(gateway.products.journal.activeRuns()).toHaveLength(1);
      expect(gateway.leaf.interrupt).not.toHaveBeenCalled();
      expect((await (await request(first.id, "science")).json()).projects).toHaveLength(1);
      expect((await (await request(second.id, "science")).json()).projects).toEqual([]);
      expect((await (await request(first.id, "settings")).json()).policy.cpus).toBe(3);
      expect((await (await request(second.id, "settings")).json()).policy.cpus).toBe(
        DEFAULT_POLICY.cpus,
      );
      const before = await (await request(second.id, "memory")).json();
      const note = before.notes.find((note: { target: string }) => note.target === "workspace");
      expect(
        (
          await request(
            second.id,
            "memory/notes",
            {
              target: "workspace",
              content: "Second project only",
              expectedRevision: note.revision,
            },
            "PUT",
          )
        ).status,
      ).toBe(200);
      expect((await gateway.products.learning.core.read("workspace")).content).toBe("");
      expect((await request("unknown", "settings")).status).toBe(404);
      const catalog = await (await request(first.id, "projects")).json();
      expect(catalog.activeId).toBe(second.id);
      expect(catalog.projects.map((project: { id: string }) => project.id)).toEqual([
        first.id,
        second.id,
      ]);
      const client = new McpClient({ name: "project-test", version: "1.0.0" });
      await client.connect(
        new StreamableHTTPClientTransport(new URL(`${gateway.origin}/projects/${second.id}/mcp`), {
          requestInit: { headers: { authorization: `Bearer ${gateway.token}` } },
        }),
      );
      try {
        const recalled = await client.callTool({
          name: "memory",
          arguments: { action: "read_core_memory", request: { target: "workspace" } },
        });
        expect(recalled.isError).toBe(true);
        expect(JSON.stringify(recalled)).toContain("active session and execution identity");
      } finally {
        await client.close();
      }
      await gateway.products.rootAgent.interrupt(browser.sessionId);
      await running;
    } finally {
      await gateway.dispose();
      await running;
    }
  });

  it("authenticates research mutations, persists policy and isolates artifacts across workspace switches", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      const send = (
        path: string,
        body: object,
        method = "POST",
        auth = headers(gateway, browser),
      ) =>
        fetch(`${gateway.origin}/api/v1/${path}`, {
          method,
          headers: auth,
          body: JSON.stringify(body),
        });
      const get = (path: string) =>
        fetch(`${gateway.origin}/api/v1/${path}`, { headers: { cookie: browser.cookie } });
      const policy = { ...DEFAULT_POLICY, filesystem: "read-only" };
      expect(
        (
          await send("language", { language: "en" }, "PUT", {
            ...headers(gateway, browser),
            cookie: "",
          })
        ).status,
      ).toBe(401);
      expect(
        (
          await send("language", { language: "en" }, "PUT", {
            ...headers(gateway, browser),
            origin: "https://untrusted.invalid",
          })
        ).status,
      ).toBe(403);
      expect((await send("language", { language: "invalid" }, "PUT")).status).toBe(400);
      expect((await send("language", { language: "en" }, "PUT")).status).toBe(200);
      expect(
        new SettingsStore(gateway.products.options.productHome, "other-workspace").readLanguage(),
      ).toBe("en");
      expect((await (await get("bootstrap")).json()).language).toBe("en");
      expect(
        (await send("settings", policy, "PUT", { ...headers(gateway, browser), cookie: "" }))
          .status,
      ).toBe(401);
      expect(
        (
          await send("settings", policy, "PUT", {
            ...headers(gateway, browser),
            origin: "https://untrusted.invalid",
          })
        ).status,
      ).toBe(403);
      expect((await send("settings", { ...policy, cpus: 0 }, "PUT")).status).toBe(400);
      expect((await send("settings", policy, "PUT")).status).toBe(200);
      expect(
        new SettingsStore(
          gateway.products.options.productHome,
          gateway.products.options.workspace.id,
        ).read().policy,
      ).toEqual(policy);
      const response = await send("tools/science_notebook", {
        action: "create_project",
        request: { requestId: randomUUID(), title: "Verified project" },
      });
      expect(response.status).toBe(200);
      const { data: project } = await response.json();
      const source = "sample,value\nA,42\n";
      const imported = await send("artifacts", {
        requestId: randomUUID(),
        projectId: project.id,
        name: "input.csv",
        dataBase64: Buffer.from(source).toString("base64"),
      });
      expect(imported.status).toBe(200);
      const artifact = await imported.json();
      const content = await get(`artifacts/${artifact.id}/content`);
      expect(content.headers.get("content-disposition")).toContain("attachment;");
      expect(await content.text()).toBe(source);
      expect((await get(`artifact-preview?id=${artifact.id}`)).status).toBe(200);
      expect(
        JSON.stringify(await (await get(`research-object?project=${project.id}`)).json()),
      ).toContain(`urn:uuid:${artifact.id}`);
      expect((await send("tools/swarm", {})).status).toBe(404);
      expect((await send("environment", { action: "inspect" })).status).toBe(500);
      const next = join(gateway.products.options.workspace.root, "next-workspace");
      await mkdir(next);
      expect((await send("workspace", { root: next }, "PUT")).status).toBe(200);
      expect(gateway.host.products.options.workspace.root).toBe(await realpath(next));
      expect((await (await get("science")).json()).projects).toEqual([]);
      expect((await get(`artifacts/${artifact.id}/content`)).status).toBe(404);
      expect((await (await get("settings")).json()).policy).toEqual(DEFAULT_POLICY);
      expect(
        (await send("workspace", { root: gateway.products.options.workspace.root }, "PUT")).status,
      ).toBe(200);
      expect((await (await get("settings")).json()).policy).toEqual(policy);
      expect(await (await get(`artifacts/${artifact.id}/content`)).text()).toBe(source);
    } finally {
      await gateway.dispose();
    }
  });

  it("rejects permission and workspace changes while science work is active and settles work before journal shutdown", async () => {
    const gateway = await createGateway();
    const ready = Promise.withResolvers<void>();
    try {
      const browser = await browserSession(gateway);
      vi.spyOn(gateway.products.science, "executeNotebookCell").mockImplementation(
        async (_session, _request, signal) => {
          ready.resolve();
          await new Promise<void>((_resolve, reject) =>
            signal?.addEventListener("abort", () => reject(signal.reason), { once: true }),
          );
          throw new Error("unreachable");
        },
      );
      const operation = gateway.products.callTool(
        "science_notebook",
        {
          action: "execute",
          request: {
            requestId: randomUUID(),
            notebookId: randomUUID(),
            source: "print(1)",
            outputArtifact: null,
          },
        },
        { actorId: "renderer", callId: randomUUID(), signal: new AbortController().signal },
      );
      const rejected = expect(operation).rejects.toThrow("closing");
      await ready.promise;
      for (const [path, body] of [
        ["workspace", { root: gateway.products.options.workspace.root }],
        ["settings", DEFAULT_POLICY],
      ] as const) {
        expect(
          (
            await fetch(`${gateway.origin}/api/v1/${path}`, {
              method: "PUT",
              headers: headers(gateway, browser),
              body: JSON.stringify(body),
            })
          ).status,
        ).toBe(409);
      }
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
      const browser = await browserSession(gateway);
      const childSession = await gateway.products.rootAgent.create();
      const steer = vi.spyOn(gateway.leaf.agent, "steer");
      const interrupt = vi.spyOn(gateway.leaf.agent, "interrupt").mockImplementation(async () => {
        stopped.resolve();
        return;
      });
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, text) => {
        if (text === "parent") {
          await gateway.products.callTool(
            "swarm",
            { action: "send_message", agentId: "codex", sessionId: childSession, text: "child" },
            {
              actorId: "parent",
              callId: "dispatch",
              signal: new AbortController().signal,
            },
          );
        } else {
          ready.resolve();
          await stopped.promise;
        }

        return { stopReason: "end_turn" as const };
      });
      operation = gateway.products.rootAgent.start(browser.sessionId, "parent", observer);
      await ready.promise;
      const snapshot = await (
        await fetch(
          `${gateway.origin}/api/v1/logs?session=${encodeURIComponent(browser.sessionId)}&descendants=true`,
          { headers: { cookie: browser.cookie } },
        )
      ).json();
      const child = (snapshot.events as ExecutionRecord[]).find(
        (record) =>
          record.sessionId === childSession && record.event.type === EventType.RUN_STARTED,
      );
      expect(child).toBeDefined();
      expect(snapshot.activeRunIds).toContain(child?.runId);
      const url = `${gateway.origin}/api/v1/runs/${child?.runId}`;
      const post = (body: object, auth = headers(gateway, browser)) =>
        fetch(url, { method: "POST", headers: auth, body: JSON.stringify(body) });
      expect(
        (await post({ action: "cancel" }, { ...headers(gateway, browser), cookie: "" })).status,
      ).toBe(401);
      expect(
        (
          await post(
            { action: "cancel" },
            { ...headers(gateway, browser), origin: "https://other.invalid" },
          )
        ).status,
      ).toBe(403);
      expect((await post({ action: "steer", text: " " })).status).toBe(400);
      expect((await post({ action: "steer", text: "check the methods" })).status).toBe(200);
      expect(steer).toHaveBeenCalledWith(childSession, "check the methods");
      expect((await post({ action: "cancel" })).status).toBe(200);
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
        expect((await post({ action: "cancel" })).status).toBe(409);
        expect(interrupt).toHaveBeenCalledTimes(1);
      } finally {
        laterDone.resolve();
        await later;
      }
      const log = gateway.products.journal.read({ run: child?.runId ?? "" }).events;
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
    const mcp = new McpClient({ name: "delegation-interactions", version: "1" });
    const answers: unknown[] = [];
    try {
      const browser = await browserSession(gateway);
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (sessionId, text, output) => {
        if (text === "parent") {
          await mcp.connect(
            new StreamableHTTPClientTransport(
              new URL(
                `${gateway.origin}/mcp?session=${encodeURIComponent(sessionId)}&run=${output.executionId}`,
              ),
              {
                requestInit: { headers: { authorization: `Bearer ${gateway.token}` } },
              },
            ),
          );
          const results = await Promise.all(
            ["first", "second"].map((text) =>
              mcp.callTool({
                name: "swarm",
                arguments: { action: "send_message", agentId: "codex", text },
              }),
            ),
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
      let stream = await agUi(gateway, browser, runInput(browser.sessionId, "parent"));
      const ids = new Set<string>();
      for (const allow of [false, true]) {
        const finished = events(stream).find((event) => event.type === EventType.RUN_FINISHED);
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
              run.sessionId !== browser.sessionId && pending.id.startsWith(`${run.sessionId}:`),
          );
        expect(child).toBeDefined();
        const control = await fetch(`${gateway.origin}/api/v1/runs/${child?.runId}`, {
          method: "POST",
          headers: headers(gateway, browser),
          body: JSON.stringify({ action: "cancel" }),
        });
        expect(control.status).toBe(409);
        stream = await agUi(gateway, browser, {
          ...runInput(browser.sessionId, ""),
          resume: [{ interruptId: pending.id, status: "resolved", payload: { allow } }],
        });
      }
      expect(ids.size).toBe(2);
      expect(answers).toEqual([{ allow: false }, { allow: true }]);
      expect(stream).toContain("both children finished");
      expect(gateway.products.journal.activeRuns()).toEqual([]);
      const interactions = gateway.products.journal
        .read()
        .events.filter(
          ({ event }) =>
            event.type === EventType.CUSTOM && event.name === "swarmx.interaction.answered",
        );
      expect(interactions).toHaveLength(2);
    } finally {
      await mcp.close();
      await gateway.dispose();
    }
  });

  it("binds ACP MCP endpoints to one Host run and rejects stale or inactive endpoints", async () => {
    const gateway = await createGateway();
    const client = new McpClient({ name: "acp-binding-test", version: "1" });
    try {
      const browser = await browserSession(gateway);
      const invoke = vi.spyOn(gateway.products, "callTool");
      const call = () =>
        client.callTool({
          name: "swarm",
          arguments: { action: "status" },
        });
      gateway.products.acpExecutions.set("test-process", null);
      await client.connect(
        new StreamableHTTPClientTransport(new URL(`${gateway.origin}/mcp?acp=test-process`), {
          requestInit: { headers: { authorization: "Bearer test-process" } },
        }),
      );
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, _text, output) => {
        expect((await call()).isError).toBe(true);
        if (!output.executionId) throw new Error("Missing execution identity");
        gateway.products.acpExecutions.set("test-process", {
          sessionId: browser.sessionId,
          runId: "stale-run",
        });
        expect((await call()).isError).toBe(true);
        gateway.products.acpExecutions.set("test-process", {
          sessionId: browser.sessionId,
          runId: output.executionId,
        });
        expect((await call()).isError).not.toBe(true);
        gateway.products.acpExecutions.set("test-process", null);
        return { stopReason: "end_turn" as const };
      });
      for (const turnId of ["first", "second"])
        await gateway.products.rootAgent.start(browser.sessionId, turnId, observer);
      expect((await call()).isError).toBe(true);
      expect(invoke).toHaveBeenCalledTimes(4);
      const records = gateway.products.journal.read({ session: browser.sessionId }).events;
      const runs = records.filter(({ event }) => event.type === EventType.RUN_STARTED);
      const calls = records.filter(({ event }) => event.type === EventType.TOOL_CALL_START);
      expect(calls.map(({ runId }) => runId)).toEqual(runs.map(({ runId }) => runId));
      expect(
        (
          await fetch(`${gateway.origin}/mcp?session=${browser.sessionId}&run=${runs[0]?.runId}`, {
            headers: { authorization: "Bearer test-process" },
          })
        ).status,
      ).toBe(403);
      gateway.products.acpExecutions.delete("test-process");
      expect(
        (
          await fetch(`${gateway.origin}/mcp?acp=test-process`, {
            headers: { authorization: "Bearer test-process" },
          })
        ).status,
      ).toBe(401);
    } finally {
      await client.close();
      await gateway.dispose();
    }
  });

  it("journals a browser DVC mutation before dispatch and retains its failure", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      vi.spyOn(gateway.products.dvc, "pull").mockImplementation(async () => {
        expect(gateway.products.journal.read().events.at(-1)?.event.type).toBe(
          EventType.TOOL_CALL_END,
        );
        throw new Error("DVC command failed");
      });
      const response = await fetch(`${gateway.origin}/api/v1/dvc`, {
        method: "POST",
        headers: headers(gateway, browser),
        body: JSON.stringify({ action: "pull", request: {} }),
      });
      expect(response.status).toBe(500);
      expect(gateway.products.journal.read().events.at(-1)?.event).toMatchObject({
        type: EventType.CUSTOM,
        name: "swarmx.tool.failed",
        value: { message: "DVC command failed" },
      });
    } finally {
      await gateway.dispose();
    }
  });

  it("preserves causal MCP science/delegation records and reads them after restart without a Harness", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      vi.spyOn(gateway.leaf.agent, "start").mockImplementation(async (_id, text, output) => {
        if (text === "parent") {
          if (!output.executionId) throw new Error("Missing execution ID");
          const client = new McpClient({ name: "journal-test", version: "1" });
          try {
            await client.connect(
              new StreamableHTTPClientTransport(
                new URL(
                  `${gateway.origin}/mcp?session=${encodeURIComponent(browser.sessionId)}&run=${encodeURIComponent(output.executionId)}`,
                ),
                { requestInit: { headers: { authorization: `Bearer ${gateway.token}` } } },
              ),
            );
            const science = await client.callTool({
              name: "science_notebook",
              arguments: {
                action: "create_project",
                request: { requestId: randomUUID(), title: "Traceable project" },
              },
            });
            expect(science.isError).not.toBe(true);
            const delegated = await client.callTool({
              name: "swarm",
              arguments: {
                action: "send_message",
                agentId: "codex",
                text: "child",
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
      await gateway.products.rootAgent.start(browser.sessionId, "parent", observer);
      const read = await fetch(`${gateway.origin}/api/v1/logs`, {
        headers: headers(gateway, browser),
      });
      expect(read.status).toBe(200);
      const saved = (await read.json()) as { events: ExecutionRecord[]; nextAfter: number };
      const starts = saved.events.filter(({ event }) => event.type === EventType.RUN_STARTED);
      expect(starts).toHaveLength(2);
      const delegated = saved.events.find(
        ({ event }) => event.type === EventType.TOOL_CALL_START && event.toolCallName === "swarm",
      );
      expect(starts[1]?.causedBy).toBe(delegated?.id);
      const result = saved.events.find(
        ({ event }) =>
          event.type === EventType.TOOL_CALL_RESULT &&
          event.content.includes("Created science project"),
      );
      if (result?.event.type !== EventType.TOOL_CALL_RESULT)
        throw new Error("Missing science result");
      expect(JSON.parse(result.event.content).locator).toMatchObject({
        sessionId: browser.sessionId,
        journalSeq: 1,
      });
      expect(result.runId).toBe(starts[0]?.runId);
      expect(JSON.stringify(saved)).not.toContain(gateway.token);
      expect(
        saved.events.some(
          ({ event }) =>
            event.type === EventType.RAW && JSON.stringify(event.event).includes("原始记录"),
        ),
      ).toBe(true);
      expect((await fetch(`${gateway.origin}/api/v1/logs`)).status).toBe(401);
      expect(
        (
          await fetch(`${gateway.origin}/api/v1/logs?limit=0`, {
            headers: headers(gateway, browser),
          })
        ).status,
      ).toBe(400);
      expect(
        (
          await fetch(`${gateway.origin}/api/v1/logs`, {
            headers: { ...headers(gateway, browser), origin: "https://foreign.example" },
          })
        ).status,
      ).toBe(403);
      const page = await (
        await fetch(`${gateway.origin}/api/v1/logs?limit=1`, { headers: headers(gateway, browser) })
      ).json();
      expect(page.events).toEqual(saved.events.slice(0, 1));
      expect(page.nextAfter).toBe(saved.events[0]?.seq);

      await gateway.host.dispose();
      await gateway.products.dispose();
      const reopened = await ProductServices.create(gateway.products.options);
      const host = await startHost({
        products: reopened,
        workspace: reopened.options.workspace,
        rendererRoot: reopened.options.workspace.root,
      });
      try {
        const launch = await fetch(host.issueLaunchUrl(), { redirect: "manual" });
        const cookie = launch.headers.get("set-cookie")?.split(";", 1)[0];
        if (!cookie) throw new Error("Missing restored browser cookie");
        const restored = await fetch(`${host.internalUrl}/api/v1/logs`, { headers: { cookie } });
        expect(await restored.json()).toEqual(saved);
      } finally {
        await host.dispose();
        await reopened.dispose();
      }
    } finally {
      await gateway.dispose();
    }
  });

  it("restores memory snapshots and pending writes after restart and requires browser approval", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      gateway.products.settings.writeMemory({ writeApproval: true });
      const note = await gateway.products.learning.core.read("user");
      const frozen = await gateway.products.learning.snapshot(browser.sessionId);
      await gateway.products.callTool(
        "memory",
        {
          action: "update_core_memory",
          request: { target: "user", content: "中文优先", expectedRevision: note.revision },
        },
        { actorId: "model", callId: "memory-proposal", signal: new AbortController().signal },
      );
      await gateway.host.dispose();
      await gateway.products.dispose();
      const products = await ProductServices.create(gateway.products.options);
      const host = await startHost({
        products,
        workspace: products.options.workspace,
        rendererRoot: products.options.workspace.root,
      });
      try {
        expect(await products.learning.snapshot(browser.sessionId)).toBe(frozen);
        const [pending] = (await products.learning.status()).pending;
        if (!pending) throw new Error("Pending memory did not survive restart");
        const url = `${host.internalUrl}/api/v1/memory/pending/${pending.id}`;
        const launch = await fetch(host.issueLaunchUrl(), { redirect: "manual" });
        const cookie = launch.headers.get("set-cookie")?.split(";", 1)[0];
        if (!cookie) throw new Error("Missing browser cookie");
        const request = { method: "POST", body: JSON.stringify({ action: "approve" }) };
        expect(
          (
            await fetch(url, {
              ...request,
              headers: { authorization: `Bearer ${host.internalToken}`, origin: host.internalUrl },
            })
          ).status,
        ).toBe(401);
        expect(
          (await fetch(url, { ...request, headers: { cookie, origin: "https://foreign.example" } }))
            .status,
        ).toBe(403);
        expect((await products.learning.core.read("user")).content).toBe("");
        expect(
          (
            await fetch(url, {
              ...request,
              headers: { cookie, origin: host.internalUrl, "content-type": "application/json" },
            })
          ).status,
        ).toBe(200);
        expect((await products.learning.core.read("user")).content).toBe("中文优先");
        expect((await products.learning.status()).pending).toEqual([]);
        expect(await products.learning.snapshot(browser.sessionId)).toBe(frozen);
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
    const nested = createSwarm("parent", (client) =>
      client.connect(
        createSwarm("child", (child) => child.connect(acpAgent(leaf.agent, process.cwd()))),
      ),
    );
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
    const connection = client.connect(nested);
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

  it("serves native model catalogs and forwards only per-turn model settings", async () => {
    const gateway = await createGateway();
    try {
      const browser = await browserSession(gateway);
      const init = { headers: headers(gateway, browser) };
      expect((await fetch(`${gateway.origin}/api/v1/models`)).status).toBe(401);
      expect(await (await fetch(`${gateway.origin}/api/v1/bootstrap`, init)).json()).toMatchObject({
        defaultHarness: "codex",
      });
      expect(
        await (
          await fetch(
            `${gateway.origin}/api/v1/models?agent=swarm&session=${encodeURIComponent(browser.sessionId)}`,
            init,
          )
        ).json(),
      ).toEqual({
        models: [{ id: "native-model", name: "Native", efforts: [{ id: "high", name: "High" }] }],
        current: {},
      });
      expect(
        (await fetch(`${gateway.origin}/api/v1/models?agent=swarm&session=claude:foreign`, init))
          .ok,
      ).toBe(false);
      const input = {
        ...runInput(browser.sessionId, "hello"),
        forwardedProps: { modelName: "native-model", reasoningEffort: "high" },
      };
      expect(events(await agUi(gateway, browser, input)).at(-1)).toMatchObject({
        type: "RUN_FINISHED",
      });
      expect(gateway.leaf.settings).toEqual([
        {
          model: "native-model",
          effort: "high",
          instructions: expect.stringContaining("SwarmX memory"),
        },
      ]);
      for (const forwardedProps of [
        { modelName: "native-model --global" },
        { modelName: "native-model", approvalPolicy: "never" },
      ]) {
        const result = await agUi(gateway, browser, { ...input, forwardedProps });
        expect(result).toContain("RUN_ERROR");
      }
      expect(gateway.leaf.prompts).toEqual(["hello"]);
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
  const settings: Array<RunOptions | undefined> = [];
  const started = Promise.withResolvers<void>();
  const stopped = Promise.withResolvers<void>();
  const interrupt = vi.fn(async () => {
    stopped.resolve();
    return;
  });
  const agent = scopeSessions("codex", {
    name: "native",
    capabilities: HARNESS_CAPABILITIES.codex,
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

async function createGateway() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-gateway-"));
  const leaf = fakeAgent();
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    workspace: await resolveWorkspace(root),
  });
  const host = await startHost({
    products,
    rendererRoot: root,
    workspace: products.options.workspace,
  });
  await products.attachAgents(host.internalUrl, host.internalToken, leaf.agent, "codex");
  return {
    leaf,
    products,
    host,
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
