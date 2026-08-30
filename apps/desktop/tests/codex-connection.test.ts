import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { describe, expect, it, vi } from "vitest";
import {
  CodexJsonRpcConnection,
  type CodexProcess,
  CodexRpcError,
} from "../src/runtime/codex/connection.js";
import { startCodexRuntime } from "../src/runtime/codex/index.js";
import { CodexMemberBindingStore } from "../src/runtime/codex/member-bindings.js";
import { WorkspaceAuthority } from "../src/runtime/workspace.js";

function fakeProcess(
  exitMode: "sigterm" | "sigkill" = "sigterm",
  pid: number | null = 1234,
): CodexProcess & {
  stdout: PassThrough;
  stderr: PassThrough;
  stdin: PassThrough;
  pid: number | undefined;
  emitError(error: Error): void;
} {
  const events = new EventEmitter();
  let exited = false;
  const child: CodexProcess & {
    stdout: PassThrough;
    stderr: PassThrough;
    stdin: PassThrough;
    pid: number | undefined;
    emitError(error: Error): void;
  } = {
    stdout: new PassThrough(),
    stderr: new PassThrough(),
    stdin: new PassThrough(),
    pid: pid ?? undefined,
    emitError: (error) => events.emit("error", error),
    kill: vi.fn((signal?: NodeJS.Signals) => {
      if (!exited && (exitMode === "sigterm" || signal === "SIGKILL")) {
        exited = true;
        queueMicrotask(() => events.emit("exit", 0, signal));
      }
      return true;
    }),
    once: (event, listener) => events.once(event, listener),
  };
  return child;
}

async function nextMessage(stream: PassThrough): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    stream.once("data", (chunk: Buffer) => resolve(JSON.parse(chunk.toString("utf8").trim())));
  });
}

describe("Codex JSON-RPC connection", () => {
  it("times out startup and terminates a silent live child", async () => {
    await expect(
      startCodexRuntime({
        command: process.execPath,
        args: ["-e", "setInterval(() => {}, 1000)"],
        startupTimeoutMs: 25,
      }),
    ).rejects.toThrow("startup timed out after 25ms");
    await expect(
      startCodexRuntime({ command: process.execPath, startupTimeoutMs: 300_001 }),
    ).rejects.toThrow("integer from 1 through 300000ms");
  });

  it("applies the startup deadline to persisted binding reconciliation", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-startup-deadline-"));
    const workspaceRoot = join(root, "workspace");
    const productHome = join(root, "product");
    mkdirSync(workspaceRoot);
    const workspace = new WorkspaceAuthority().mint(workspaceRoot);
    const journal = new SwarmJournal(join(productHome, "swarm"));
    const teamId = `codex-mcp-thread:${"d".repeat(64)}`;
    const memberId = randomUUID();
    journal.append(teamId, {
      type: "team/created",
      data: {
        createdAt: 1,
        lead: {
          createdAt: 1,
          description: "Lead",
          id: teamId,
          name: "lead",
          phase: "active",
          role: "lead",
          modelPolicy: { source: "observed" },
        },
        name: "Startup deadline team",
        workspaceKey: journal.workspaceKey(workspace.root),
      },
    });
    journal.append(teamId, {
      type: "member/updated",
      data: {
        createdAt: 2,
        description: "Persisted member",
        id: memberId,
        name: "worker",
        phase: "active",
        role: "legacy",
        modelPolicy: { source: "observed" },
      },
    });
    new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root)).claim({
      id: memberId,
      conversationId: "codex:persisted-member",
    });
    journal.close();

    try {
      await expect(
        startCodexRuntime({
          command: process.execPath,
          args: [
            "-e",
            `const readline = require("node:readline").createInterface({ input: process.stdin });
readline.on("line", (line) => {
  const request = JSON.parse(line);
  if (request.method === "initialize") {
    process.stdout.write(JSON.stringify({ id: request.id, result: {} }) + "\\n");
  }
});`,
          ],
          bridgeToken: "startup-deadline-test",
          bridgeUrl: "http://127.0.0.1:1/",
          productHome,
          startupTimeoutMs: 100,
          workspace,
        }),
      ).rejects.toThrow("startup timed out after 100ms");
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it("initializes exactly once and opts into the allowlisted revert API", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    const initializing = connection.initialize();
    const request = await nextMessage(child.stdin);
    expect(request).toMatchObject({
      id: 1,
      method: "initialize",
      params: {
        clientInfo: { name: "swarmx", title: "SwarmX", version: "0.1.0" },
        capabilities: { experimentalApi: true },
      },
    });
    child.stdout.write(`${JSON.stringify({ id: 1, result: { userAgent: "codex-test" } })}\n`);
    await initializing;
    expect(await nextMessage(child.stdin)).toEqual({ method: "initialized" });

    const pending = connection.request("thread/list", { limit: 1 });
    const listRequest = await nextMessage(child.stdin);
    expect(listRequest).toEqual({ id: 2, method: "thread/list", params: { limit: 1 } });
    child.stdout.write(`${JSON.stringify({ id: 2, result: { data: [], nextCursor: null } })}\n`);
    await expect(pending).resolves.toEqual({ data: [], nextCursor: null });
    await connection.initialize();
    expect(connection.initialized).toBe(true);
  });

  it("answers server requests through one registered handler", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    connection.onRequest("item/fileChange/requestApproval", async (params, requestId) => ({
      decision: params.itemId === "item-1" && requestId === "server-1" ? "accept" : "decline",
    }));
    child.stdout.write(
      `${JSON.stringify({
        id: "server-1",
        method: "item/fileChange/requestApproval",
        params: { itemId: "item-1" },
      })}\n`,
    );
    expect(await nextMessage(child.stdin)).toEqual({
      id: "server-1",
      result: { decision: "accept" },
    });
  });

  it("preserves structured JSON-RPC errors for fail-closed recovery decisions", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    const pending = connection.request("thread/read", { threadId: "missing" });
    await nextMessage(child.stdin);
    child.stdout.write(
      `${JSON.stringify({
        id: 1,
        error: {
          code: -32601,
          message: "Method not found",
          data: { method: "thread/read" },
        },
      })}\n`,
    );

    const failure = await pending.then(
      () => undefined,
      (error: unknown) => error,
    );
    expect(failure).toBeInstanceOf(CodexRpcError);
    expect(failure).toMatchObject({
      code: -32601,
      data: { method: "thread/read" },
      message: 'Method not found {"method":"thread/read"}',
    });
  });

  it("drops late server-request replies after disposal", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    let resolveHandler: ((value: unknown) => void) | undefined;
    const handled = new Promise<unknown>((resolve) => {
      resolveHandler = resolve;
    });
    connection.onRequest("mcpServer/elicitation/request", async () => handled);
    child.stdout.write(
      `${JSON.stringify({
        id: "server-late",
        method: "mcpServer/elicitation/request",
        params: { message: "confirm" },
      })}\n`,
    );
    await new Promise<void>((resolve) => setImmediate(resolve));

    await connection.dispose();
    resolveHandler?.({ action: "cancel" });
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(child.kill).toHaveBeenCalledOnce();
  });

  it("fails safely when peer stdin closes before a server-request handler settles", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    let resolveHandler: ((value: unknown) => void) | undefined;
    const handled = new Promise<unknown>((resolve) => {
      resolveHandler = resolve;
    });
    connection.onRequest("mcpServer/elicitation/request", async () => handled);
    child.stdout.write(
      `${JSON.stringify({
        id: "server-peer-closed",
        method: "mcpServer/elicitation/request",
        params: { message: "confirm" },
      })}\n`,
    );
    await new Promise<void>((resolve) => setImmediate(resolve));

    child.stdin.destroy();
    resolveHandler?.({ action: "cancel" });
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(child.kill).toHaveBeenCalledOnce();
  });

  it("rejects a request without losing its Promise when stdin has already ended", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child);
    child.stdin.end();

    await expect(connection.request("thread/list", {})).rejects.toThrow("stdin is not writable");
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(child.kill).toHaveBeenCalledOnce();
  });

  it("waits for App Server exit and escalates an unresponsive child", async () => {
    const child = fakeProcess("sigkill");
    const connection = new CodexJsonRpcConnection(child, { shutdownTimeoutMs: 5 });

    await connection.dispose();
    expect(child.kill).toHaveBeenNthCalledWith(1, "SIGTERM");
    expect(child.kill).toHaveBeenNthCalledWith(2, "SIGKILL");
  });

  it("does not treat a post-spawn process error as an exit", async () => {
    const child = fakeProcess("sigkill");
    const connection = new CodexJsonRpcConnection(child, { shutdownTimeoutMs: 5 });

    child.emitError(new Error("process operation failed"));
    await connection.dispose();

    expect(child.kill).toHaveBeenNthCalledWith(1, "SIGTERM");
    expect(child.kill).toHaveBeenNthCalledWith(2, "SIGKILL");
  });

  it("treats a spawn error without a pid as terminal", async () => {
    const child = fakeProcess("sigkill", null);
    const connection = new CodexJsonRpcConnection(child, { shutdownTimeoutMs: 5 });

    child.emitError(new Error("spawn failed"));
    await connection.dispose();

    expect(child.kill).not.toHaveBeenCalled();
  });

  it("rejects oversized frames and pending calls when disposed", async () => {
    const child = fakeProcess();
    const connection = new CodexJsonRpcConnection(child, { maxFrameBytes: 64 });
    const pending = connection.request("thread/list", {});
    await nextMessage(child.stdin);
    child.stdout.write(
      `${JSON.stringify({ method: "event", params: { text: "x".repeat(100) } })}\n`,
    );
    await expect(pending).rejects.toThrow("exceeds 64 bytes");
    expect(child.kill).toHaveBeenCalledOnce();
  });
});
