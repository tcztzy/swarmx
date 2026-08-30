import { mkdtempSync, rmSync } from "node:fs";
import type { ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Context } from "@deepseek-ai/cordis";
import WebServer from "@deepseek-ai/dsh-host-webserver";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ConversationRuntime } from "../src/runtime/contracts.js";
import { ConversationRuntimeRegistry } from "../src/runtime/registry.js";
import ConversationRuntimeService from "../src/runtime/web-plugin.js";
import { WorkspaceAuthority } from "../src/runtime/workspace.js";

const scratch: string[] = [];

function fixtureRuntime(): ConversationRuntime {
  return {
    kind: "codex",
    list: vi.fn(async () => [
      {
        runtime: "codex",
        conversationId: "native-thread-1",
        workspace: { id: "workspace", label: "workspace" },
        title: "Native thread",
        archived: false,
        updatedAt: 1,
      },
    ]),
    create: vi.fn(async ({ workspace }) => ({
      runtime: "codex",
      conversationId: "native-thread-2",
      workspace,
      title: "New thread",
      archived: false,
      updatedAt: 2,
    })),
    read: vi.fn(async (conversationId) => ({
      runtime: "codex",
      conversationId,
      workspace: { id: "workspace", label: "workspace" },
      title: "Native thread",
      archived: false,
      turns: [
        {
          id: "native-turn-1",
          status: "completed",
          items: [
            {
              type: "user_message",
              id: "native-user-1",
              turnId: "native-turn-1",
              text: "original",
              createdAt: 1,
            },
          ],
        },
      ],
    })),
    start: vi.fn(async () => ({ turnId: "native-turn-1" })),
    steer: vi.fn(async () => {}),
    interrupt: vi.fn(async () => {}),
    revise: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "native-thread-1",
      workspace: { id: "workspace", label: "workspace" },
      title: "Revised",
      archived: false,
      updatedAt: 4,
    })),
    fork: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "native-thread-3",
      workspace: { id: "workspace", label: "workspace" },
      title: "Fork",
      archived: false,
      updatedAt: 3,
    })),
    archive: vi.fn(async () => {}),
    subscribe: vi.fn(() => () => {}),
    respondToApproval: vi.fn(async () => {}),
    dispose: vi.fn(async () => {}),
  };
}

function rejectOnAbort(signal: AbortSignal | undefined): Promise<never> {
  if (signal === undefined) return Promise.reject(new Error("runtime signal is missing"));
  return new Promise((_, reject) => {
    const abort = () => reject(signal.reason ?? new Error("runtime signal aborted"));
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });
}

async function readStreamChunk(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<string> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const result = await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("timed out waiting for SSE data")), 1_000);
      }),
    ]);
    return new TextDecoder().decode(result.value);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

afterEach(() => {
  for (const path of scratch.splice(0)) rmSync(path, { recursive: true, force: true });
});

describe("DSH Web conversation runtime plugin", () => {
  it("reuses the DSH listener and exposes only the bounded runtime protocol", async () => {
    const context = new Context();
    const webFiber = await context.plugin(WebServer, { host: "127.0.0.1", port: 0 });
    const runtime = fixtureRuntime();
    const registry = new ConversationRuntimeRegistry([runtime], "codex");
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-web-"));
    scratch.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtimeFiber = await context.plugin(ConversationRuntimeService, { registry, workspace });
    const origin = `http://${context.webServer.host}:${String(context.webServer.port)}`;
    try {
      const metadata = await fetch(`${origin}/api/swarmx/conversation-runtimes`);
      expect(metadata.status).toBe(200);
      await expect(metadata.json()).resolves.toEqual({
        defaultRuntimeKind: "codex",
        runtimeKinds: ["codex"],
      });

      const list = await fetch(
        `${origin}/api/swarmx/conversation-runtimes/conversations?runtimeKind=codex`,
      );
      await expect(list.json()).resolves.toMatchObject([
        { runtime: "codex", conversationId: "native-thread-1" },
      ]);

      const start = await fetch(`${origin}/api/swarmx/conversation-runtimes/start`, {
        method: "POST",
        headers: { "content-type": "application/json", origin },
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "native-thread-1",
          text: "continue",
        }),
      });
      expect(start.status).toBe(200);
      expect(runtime.start).toHaveBeenCalledWith(
        { conversationId: "native-thread-1", text: "continue" },
        expect.any(AbortSignal),
      );

      const edit = await fetch(`${origin}/api/swarmx/conversation-runtimes/edit`, {
        method: "POST",
        headers: { "content-type": "application/json", origin },
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "native-thread-1",
          userItemId: "native-user-1",
          text: "replacement",
        }),
      });
      expect(edit.status).toBe(200);
      expect(runtime.revise).toHaveBeenCalledWith(
        {
          conversationId: "native-thread-1",
          beforeTurnId: "native-turn-1",
          text: "replacement",
        },
        expect.any(AbortSignal),
      );

      const approval = await fetch(`${origin}/api/swarmx/conversation-runtimes/approval`, {
        method: "POST",
        headers: { "content-type": "application/json", origin },
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "native-thread-1",
          turnId: "native-turn-1",
          itemId: "native-item-1",
          approvalId: "native-approval-1",
          decision: "accept",
          form: { confirm: true, count: 2 },
        }),
      });
      expect(approval.status).toBe(200);
      expect(runtime.respondToApproval).toHaveBeenCalledWith({
        runtime: "codex",
        conversationId: "native-thread-1",
        turnId: "native-turn-1",
        itemId: "native-item-1",
        approvalId: "native-approval-1",
        decision: "accept",
        form: { confirm: true, count: 2 },
      });

      expect((await fetch(origin)).status).toBe(404);
    } finally {
      await runtimeFiber.dispose();
      await registry.dispose();
      await webFiber.dispose();
    }
  });

  it("rejects cross-origin mutation and an unregistered runtime", async () => {
    const context = new Context();
    const webFiber = await context.plugin(WebServer, { host: "127.0.0.1", port: 0 });
    const registry = new ConversationRuntimeRegistry([fixtureRuntime()], "codex");
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-web-"));
    scratch.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtimeFiber = await context.plugin(ConversationRuntimeService, { registry, workspace });
    const origin = `http://${context.webServer.host}:${String(context.webServer.port)}`;
    try {
      const crossOrigin = await fetch(`${origin}/api/swarmx/conversation-runtimes/create`, {
        method: "POST",
        headers: { "content-type": "application/json", origin: "https://invalid.example" },
        body: JSON.stringify({ runtimeKind: "codex" }),
      });
      expect(crossOrigin.status).toBe(403);

      const missing = await fetch(
        `${origin}/api/swarmx/conversation-runtimes/conversations?runtimeKind=dsh`,
      );
      expect(missing.status).toBe(404);
      await expect(missing.json()).resolves.toMatchObject({
        error: 'Conversation runtime "dsh" is not registered.',
      });
    } finally {
      await runtimeFiber.dispose();
      await registry.dispose();
      await webFiber.dispose();
    }
  });

  it("V211 cancels waiting GET and complete-body POST work on response disconnect", async () => {
    const context = new Context();
    const webFiber = await context.plugin(WebServer, { host: "127.0.0.1", port: 0 });
    const runtime = fixtureRuntime();
    let emit: Parameters<ConversationRuntime["subscribe"]>[0] | undefined;
    vi.mocked(runtime.subscribe).mockImplementation((listener) => {
      emit = listener;
      return () => {
        emit = undefined;
      };
    });
    const registry = new ConversationRuntimeRegistry([runtime], "codex");
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-web-"));
    scratch.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtimeFiber = await context.plugin(ConversationRuntimeService, { registry, workspace });
    const origin = `http://${context.webServer.host}:${String(context.webServer.port)}`;
    const listUrl = `${origin}/api/swarmx/conversation-runtimes/conversations?runtimeKind=codex`;
    const sseController = new AbortController();
    try {
      const normal = await fetch(listUrl);
      expect(normal.status).toBe(200);
      await normal.arrayBuffer();
      const normalSignal = vi.mocked(runtime.list).mock.calls.at(-1)?.[0];
      await new Promise((resolve) => setImmediate(resolve));
      expect(normalSignal?.aborted).toBe(false);

      const sse = await fetch(`${origin}/api/swarmx/conversation-runtimes/events`, {
        signal: sseController.signal,
      });
      expect(sse.status).toBe(200);
      if (sse.body === null) throw new Error("SSE response body is missing");
      const reader = sse.body.getReader();
      await expect(readStreamChunk(reader)).resolves.toContain(": connected");
      emit?.({
        seq: 1,
        runtime: "codex",
        conversationId: "native-thread-1",
        type: "turn_status",
        turnId: "native-turn-1",
        status: "running",
      });
      await expect(readStreamChunk(reader)).resolves.toContain(
        '"conversationId":"native-thread-1"',
      );

      let getSignal: AbortSignal | undefined;
      vi.mocked(runtime.list).mockImplementationOnce(async (signal) => {
        getSignal = signal;
        return rejectOnAbort(signal);
      });
      const getController = new AbortController();
      const getRequest = fetch(listUrl, { signal: getController.signal });
      const getRejected = expect(getRequest).rejects.toMatchObject({ name: "AbortError" });
      await vi.waitFor(() => expect(getSignal).toBeInstanceOf(AbortSignal));
      getController.abort();
      await getRejected;
      await vi.waitFor(() => expect(getSignal?.aborted).toBe(true));

      let postSignal: AbortSignal | undefined;
      vi.mocked(runtime.start).mockImplementationOnce(async (_request, signal) => {
        postSignal = signal;
        return rejectOnAbort(signal);
      });
      const postController = new AbortController();
      const postRequest = fetch(`${origin}/api/swarmx/conversation-runtimes/start`, {
        method: "POST",
        headers: { "content-type": "application/json", origin },
        body: JSON.stringify({
          runtimeKind: "codex",
          conversationId: "native-thread-1",
          text: "continue",
        }),
        signal: postController.signal,
      });
      const postRejected = expect(postRequest).rejects.toMatchObject({ name: "AbortError" });
      await vi.waitFor(() => expect(postSignal).toBeInstanceOf(AbortSignal));
      postController.abort();
      await postRejected;
      await vi.waitFor(() => expect(postSignal?.aborted).toBe(true));

      sseController.abort();
      await reader.cancel().catch(() => {});
    } finally {
      sseController.abort();
      await runtimeFiber.dispose();
      await registry.dispose();
      await webFiber.dispose();
    }
  });

  it("V211 drops an SSE consumer as soon as its response buffer fills", async () => {
    const context = new Context();
    const webFiber = await context.plugin(WebServer, { host: "127.0.0.1", port: 0 });
    const runtime = fixtureRuntime();
    let emit: Parameters<ConversationRuntime["subscribe"]>[0] | undefined;
    vi.mocked(runtime.subscribe).mockImplementation((listener) => {
      emit = listener;
      return () => {
        emit = undefined;
      };
    });
    const registry = new ConversationRuntimeRegistry([runtime], "codex");
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-web-"));
    scratch.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtimeFiber = await context.plugin(ConversationRuntimeService, { registry, workspace });
    const origin = `http://${context.webServer.host}:${String(context.webServer.port)}`;
    const sseController = new AbortController();
    try {
      const sse = await fetch(`${origin}/api/swarmx/conversation-runtimes/events`, {
        signal: sseController.signal,
      });
      if (sse.body === null) throw new Error("SSE response body is missing");
      const reader = sse.body.getReader();
      await expect(readStreamChunk(reader)).resolves.toContain(": connected");

      const service = context.conversationRuntimes as unknown as {
        streams: Set<ServerResponse>;
      };
      const stream = [...service.streams][0];
      if (stream === undefined) throw new Error("SSE server response is missing");
      const write = vi.spyOn(stream, "write").mockReturnValueOnce(false);
      const destroy = vi.spyOn(stream, "destroy");

      emit?.({
        seq: 1,
        runtime: "codex",
        conversationId: "native-thread-1",
        type: "turn_status",
        turnId: "native-turn-1",
        status: "running",
      });

      expect(write).toHaveBeenCalledOnce();
      expect(destroy).toHaveBeenCalledOnce();
      expect(service.streams).toHaveLength(0);
      await reader.cancel().catch(() => {});
    } finally {
      sseController.abort();
      await runtimeFiber.dispose();
      await registry.dispose();
      await webFiber.dispose();
    }
  });

  it("V211/V231 exposes a bounded non-cooperative shutdown before later disposal", async () => {
    const context = new Context();
    const webFiber = await context.plugin(WebServer, { host: "127.0.0.1", port: 0 });
    const runtime = fixtureRuntime();
    let createSignal: AbortSignal | undefined;
    vi.mocked(runtime.create).mockImplementationOnce(async (_request, signal) => {
      createSignal = signal;
      return new Promise<never>(() => {});
    });
    const registry = new ConversationRuntimeRegistry([runtime], "codex");
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-web-"));
    scratch.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const runtimeFiber = await context.plugin(ConversationRuntimeService, {
      registry,
      routeDrainTimeoutMs: 25,
      workspace,
    });
    const origin = `http://${context.webServer.host}:${String(context.webServer.port)}`;
    const createRequest = fetch(`${origin}/api/swarmx/conversation-runtimes/create`, {
      method: "POST",
      headers: { "content-type": "application/json", origin },
      body: JSON.stringify({ runtimeKind: "codex" }),
    });
    void createRequest.catch(() => undefined);
    try {
      await vi.waitFor(() => expect(createSignal).toBeInstanceOf(AbortSignal));
      const shutdown = context.conversationRuntimes.shutdown();
      void shutdown.catch(() => undefined);
      expect(context.conversationRuntimes.shutdown()).toBe(shutdown);
      await vi.waitFor(() => expect(createSignal?.aborted).toBe(true));
      await expect(shutdown).rejects.toThrow(
        "1 conversation runtime route(s) did not settle within 25ms.",
      );
      await expect(createRequest).rejects.toBeDefined();
      await registry.dispose();
      expect(runtime.dispose).toHaveBeenCalledOnce();
    } finally {
      await context.conversationRuntimes.shutdown().catch(() => {});
      await runtimeFiber.dispose();
      await registry.dispose();
      await webFiber.dispose();
    }
  });
});
