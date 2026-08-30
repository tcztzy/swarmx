import type { Readable, Writable } from "node:stream";
import { z } from "zod";

const RpcIdSchema = z.union([z.string(), z.number().int()]);
const RpcMessageSchema = z
  .object({
    id: RpcIdSchema.optional(),
    method: z.string().min(1).optional(),
    params: z.unknown().optional(),
    result: z.unknown().optional(),
    error: z
      .object({
        code: z.number().int().optional(),
        message: z.string(),
        data: z.unknown().optional(),
      })
      .optional(),
  })
  .passthrough();

type RpcId = z.infer<typeof RpcIdSchema>;
type RpcMessage = z.infer<typeof RpcMessageSchema>;

export interface CodexProcess {
  readonly pid?: number | undefined;
  stdout: Readable;
  stderr: Readable;
  stdin: Writable;
  kill(signal?: NodeJS.Signals): boolean | undefined;
  once(event: "exit" | "error", listener: (...args: unknown[]) => void): unknown;
}

export interface CodexConnectionOptions {
  maxFrameBytes?: number;
  maxStderrBytes?: number;
  shutdownTimeoutMs?: number;
}

interface PendingCall {
  resolve(value: unknown): void;
  reject(error: Error): void;
}

type NotificationHandler = (params: unknown) => void;
type RequestHandler = (params: Record<string, unknown>, requestId: RpcId) => Promise<unknown>;

const DEFAULT_MAX_FRAME_BYTES = 2 * 1024 * 1024;
const DEFAULT_MAX_STDERR_BYTES = 32 * 1024;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 5_000;

export class CodexRpcError extends Error {
  readonly code: number | undefined;
  readonly data: unknown;

  constructor(message: string, code?: number, data?: unknown) {
    super(`${message}${data === undefined ? "" : ` ${JSON.stringify(data)}`}`);
    this.name = "CodexRpcError";
    this.code = code;
    this.data = data;
  }
}

export class CodexJsonRpcConnection {
  private readonly pending = new Map<number, PendingCall>();
  private readonly notificationHandlers = new Map<string, Set<NotificationHandler>>();
  private readonly requestHandlers = new Map<string, RequestHandler>();
  private readonly maxFrameBytes: number;
  private readonly maxStderrBytes: number;
  private readonly shutdownTimeoutMs: number;
  private readonly exitedPromise: Promise<void>;
  private resolveExited!: () => void;
  private buffer = "";
  private stderr = "";
  private nextId = 1;
  private initialization?: Promise<void>;
  private closed = false;
  private exited = false;
  private _initialized = false;

  constructor(
    private readonly child: CodexProcess,
    options: CodexConnectionOptions = {},
  ) {
    this.maxFrameBytes = options.maxFrameBytes ?? DEFAULT_MAX_FRAME_BYTES;
    this.maxStderrBytes = options.maxStderrBytes ?? DEFAULT_MAX_STDERR_BYTES;
    this.shutdownTimeoutMs = options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS;
    this.exitedPromise = new Promise<void>((resolve) => {
      this.resolveExited = resolve;
    });
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => this.consume(chunk));
    child.stderr.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-this.maxStderrBytes);
    });
    child.stdin.once("error", (error) =>
      this.fail(
        new Error(
          this.closedMessage(
            `Codex App Server stdin failed: ${error instanceof Error ? error.message : String(error)}`,
          ),
        ),
      ),
    );
    child.stdin.once("close", () =>
      this.fail(new Error(this.closedMessage("Codex App Server stdin closed"))),
    );
    child.once("exit", () => {
      this.markExited();
      this.fail(new Error(this.closedMessage("Codex App Server exited")));
    });
    child.once("error", (error) => {
      if (child.pid === undefined) this.markExited();
      this.fail(
        new Error(
          this.closedMessage(
            `Codex App Server failed: ${error instanceof Error ? error.message : String(error)}`,
          ),
        ),
      );
    });
  }

  get initialized(): boolean {
    return this._initialized;
  }

  initialize(): Promise<void> {
    this.initialization ??= this.request("initialize", {
      clientInfo: { name: "swarmx", title: "SwarmX", version: "0.1.0" },
      capabilities: { experimentalApi: true },
    }).then(() => {
      this.notify("initialized");
      this._initialized = true;
    });
    return this.initialization;
  }

  request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    if (this.closed) return Promise.reject(new Error("Codex App Server connection is closed."));
    const id = this.nextId++;
    const result = new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    try {
      this.write({ id, method, params });
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
    }
    return result;
  }

  notify(method: string, params?: Record<string, unknown>): void {
    if (this.closed) throw new Error("Codex App Server connection is closed.");
    try {
      this.write(params === undefined ? { method } : { method, params });
    } catch (error) {
      const failure = error instanceof Error ? error : new Error(String(error));
      this.fail(failure);
      throw failure;
    }
  }

  onNotification(method: string, handler: NotificationHandler): () => void {
    const handlers = this.notificationHandlers.get(method) ?? new Set();
    handlers.add(handler);
    this.notificationHandlers.set(method, handlers);
    return () => {
      handlers.delete(handler);
      if (handlers.size === 0) this.notificationHandlers.delete(method);
    };
  }

  onRequest(method: string, handler: RequestHandler): () => void {
    if (this.requestHandlers.has(method)) {
      throw new Error(`Codex App Server request handler "${method}" is already registered.`);
    }
    this.requestHandlers.set(method, handler);
    return () => {
      if (this.requestHandlers.get(method) === handler) this.requestHandlers.delete(method);
    };
  }

  async dispose(): Promise<void> {
    if (!this.closed) this.fail(new Error("Codex App Server connection was disposed."));
    if (await this.waitForExit()) return;
    this.child.kill("SIGKILL");
    if (!(await this.waitForExit())) {
      throw new Error("Codex App Server did not exit after SIGKILL.");
    }
  }

  private consume(chunk: string): void {
    if (this.closed) return;
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer) > this.maxFrameBytes && !this.buffer.includes("\n")) {
      this.fail(new Error(`Codex App Server frame exceeds ${String(this.maxFrameBytes)} bytes.`));
      return;
    }
    for (;;) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      if (Buffer.byteLength(line) > this.maxFrameBytes) {
        this.fail(new Error(`Codex App Server frame exceeds ${String(this.maxFrameBytes)} bytes.`));
        return;
      }
      try {
        this.handle(RpcMessageSchema.parse(JSON.parse(line)));
      } catch (error) {
        this.fail(
          new Error(
            `Invalid Codex App Server frame: ${error instanceof Error ? error.message : String(error)}`,
          ),
        );
        return;
      }
    }
  }

  private handle(message: RpcMessage): void {
    if (message.id !== undefined && ("result" in message || message.error !== undefined)) {
      if (typeof message.id !== "number") {
        throw new Error("Codex response id must match a numeric client request id.");
      }
      const pending = this.pending.get(message.id);
      if (pending === undefined) return;
      this.pending.delete(message.id);
      if (message.error !== undefined) {
        pending.reject(
          new CodexRpcError(message.error.message, message.error.code, message.error.data),
        );
      } else pending.resolve(message.result);
      return;
    }

    if (message.method === undefined)
      throw new Error("Codex frame has neither response nor method.");
    if (message.id !== undefined) {
      const handler = this.requestHandlers.get(message.method);
      if (handler === undefined) {
        this.write({
          id: message.id,
          error: { code: -32601, message: `Unsupported Codex request "${message.method}".` },
        });
        return;
      }
      const params = recordParams(message.params);
      void handler(params, message.id).then(
        (result) => this.reply({ id: message.id, result }),
        (error: unknown) =>
          this.reply({
            id: message.id,
            error: {
              code: -32603,
              message: error instanceof Error ? error.message : String(error),
            },
          }),
      );
      return;
    }
    for (const handler of this.notificationHandlers.get(message.method) ?? []) {
      handler(message.params);
    }
  }

  private write(message: Record<string, unknown>): void {
    if (
      !this.child.stdin.writable ||
      this.child.stdin.writableEnded ||
      this.child.stdin.destroyed
    ) {
      throw new Error("Codex App Server stdin is not writable.");
    }
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private reply(message: Record<string, unknown>): void {
    if (this.closed) return;
    if (
      !this.child.stdin.writable ||
      this.child.stdin.writableEnded ||
      this.child.stdin.destroyed
    ) {
      this.fail(new Error("Codex App Server stdin closed before server reply."));
      return;
    }
    try {
      this.write(message);
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private fail(error: Error): void {
    if (this.closed) return;
    this.closed = true;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.notificationHandlers.clear();
    this.requestHandlers.clear();
    if (!this.child.stdin.writableEnded && !this.child.stdin.destroyed) this.child.stdin.end();
    if (!this.exited) this.child.kill("SIGTERM");
  }

  private markExited(): void {
    if (this.exited) return;
    this.exited = true;
    this.resolveExited();
  }

  private async waitForExit(): Promise<boolean> {
    if (this.exited) return true;
    return new Promise<boolean>((resolve) => {
      const timeout = setTimeout(() => resolve(false), this.shutdownTimeoutMs);
      void this.exitedPromise.then(() => {
        clearTimeout(timeout);
        resolve(true);
      });
    });
  }

  private closedMessage(prefix: string): string {
    const detail = this.stderr.trim();
    return detail ? `${prefix}: ${detail}` : `${prefix}.`;
  }
}

function recordParams(value: unknown): Record<string, unknown> {
  if (value === undefined) return {};
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Codex request params must be an object.");
  }
  return value as Record<string, unknown>;
}
