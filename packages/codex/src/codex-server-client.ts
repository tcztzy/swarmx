import { type ChildProcess, spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

export interface CodexServerLaunchSpec {
  command: string;
  args: string[];
  env: Record<string, string>;
}

export interface CodexServerClientOptions {
  command: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
  clearEnv?: boolean;
  model?: string;
  effort?: string;
  preferredMode?: string;
  requestPermission?: CodexPermissionHandler;
  onSessionId?: (sessionId: string) => void | Promise<void>;
  signal?: AbortSignal;
}

export interface CodexMessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_progress" | "tool_result";
  agent?: string;
  toolName?: string;
}

export interface CodexPromptInput {
  text: string;
  attachments?: readonly CodexAttachment[];
}

export interface CodexAttachment {
  uri: string;
  kind: string;
  mimeType: string;
  name: string;
}

interface CodexPermissionRequest {
  sessionId: string;
  toolCall: {
    toolCallId: string;
    kind: "execute" | "edit" | "other";
    status: "pending";
    title?: string;
    content?: { type: string; text: string }[];
    rawInput: unknown;
  };
  options: Array<{
    optionId: string;
    name: string;
    kind: "allow_once" | "allow_always" | "reject_once";
  }>;
}

export type CodexPermissionHandler = (request: CodexPermissionRequest) => Promise<{
  outcome:
    | { outcome: "cancelled" }
    | { outcome: "selected"; optionId: string }
    | { outcome: "approved" }
    | { outcome: "rejected" };
}>;

interface PendingRequest {
  resolve(value: unknown): void;
  reject(error: Error): void;
}

type JsonRpcMessage = {
  id?: number;
  method?: string;
  params?: Record<string, unknown>;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
};

const COMMAND_APPROVAL_METHOD = "item/commandExecution/requestApproval";
const FILE_CHANGE_APPROVAL_METHOD = "item/fileChange/requestApproval";
const PERMISSIONS_APPROVAL_METHOD = "item/permissions/requestApproval";
const MCP_ELICITATION_METHOD = "mcpServer/elicitation/request";
const TOOL_USER_INPUT_METHOD = "item/tool/requestUserInput";

class JsonLineConnection {
  private nextId = 1;
  private readonly pending = new Map<number, PendingRequest>();
  private readonly notifications = new Map<string, Set<(params: unknown) => void>>();
  private readonly requests = new Map<
    string,
    (params: Record<string, unknown>) => Promise<unknown> | unknown
  >();
  private readonly notificationWaiters = new Set<{ reject(error: Error): void }>();
  private disposed = false;
  private buffer = "";

  constructor(private readonly child: ChildProcess) {
    child.stdout?.setEncoding("utf-8");
    child.stdout?.on("data", (chunk: string) => this.consume(chunk));
    child.stderr?.setEncoding("utf-8");
    child.stderr?.on("data", () => {});
  }

  sendRequest(method: string, params?: Record<string, unknown>): Promise<unknown> {
    if (this.disposed) return Promise.reject(new Error("Codex server connection is closed."));
    const id = this.nextId++;
    const response = new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.write({ id, method, ...(params ? { params } : {}) });
    return response;
  }

  sendNotification(method: string, params?: Record<string, unknown>): void {
    if (this.disposed) return;
    this.write({ method, ...(params ? { params } : {}) });
  }

  onNotification(method: string, listener: (params: unknown) => void): () => void {
    const listeners = this.notifications.get(method) ?? new Set();
    listeners.add(listener);
    this.notifications.set(method, listeners);
    return () => {
      listeners.delete(listener);
      if (listeners.size === 0) this.notifications.delete(method);
    };
  }

  onRequest(
    method: string,
    handler: (params: Record<string, unknown>) => Promise<unknown> | unknown,
  ): void {
    this.requests.set(method, handler);
  }

  addNotificationWaiter(waiter: { reject(error: Error): void }): void {
    this.notificationWaiters.add(waiter);
  }

  removeNotificationWaiter(waiter: { reject(error: Error): void }): void {
    this.notificationWaiters.delete(waiter);
  }

  rejectAll(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    for (const waiter of this.notificationWaiters) waiter.reject(error);
    this.notificationWaiters.clear();
  }

  close(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.rejectAll(new Error("Codex server connection is closed."));
    this.child.stdin?.end();
  }

  private consume(chunk: string): void {
    this.buffer += chunk;
    for (;;) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let message: JsonRpcMessage;
      try {
        message = JSON.parse(line) as JsonRpcMessage;
      } catch {
        continue;
      }
      this.handle(message);
    }
  }

  private handle(message: JsonRpcMessage): void {
    if (typeof message.id === "number" && ("result" in message || "error" in message)) {
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) {
        const detail = message.error.message ?? "Codex server request failed.";
        pending.reject(
          new Error(
            `${detail}${message.error.data ? ` ${JSON.stringify(message.error.data)}` : ""}`,
          ),
        );
      } else {
        pending.resolve(message.result);
      }
      return;
    }

    if (message.method) {
      const requestHandler = this.requests.get(message.method);
      if (requestHandler) {
        void Promise.resolve(requestHandler(message.params ?? {}))
          .then((result) => {
            if (typeof message.id === "number") {
              this.write({ id: message.id, result });
            }
          })
          .catch((error: unknown) => {
            if (typeof message.id === "number") {
              this.write({
                id: message.id,
                error: {
                  code: -32603,
                  message: error instanceof Error ? error.message : String(error),
                },
              });
            }
          });
        return;
      }
      const listeners = this.notifications.get(message.method);
      if (listeners) {
        for (const listener of listeners) listener(message.params);
        return;
      }
      if (typeof message.id === "number") {
        this.write({
          id: message.id,
          error: {
            code: -32601,
            message: `Unsupported Codex app-server request "${message.method}".`,
          },
        });
      }
    }
  }

  private write(message: Record<string, unknown>): void {
    if (this.child.stdin?.writable) this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }
}

interface NotificationHandle {
  method: string;
  reject(error: Error): void;
  dispose(): void;
}

/**
 * Direct Codex App Server JSON-RPC client. It intentionally does not import or
 * implement ACP: it is the `codex_server` transport behind `swarmx-codex`.
 */
export class CodexServerClient {
  private child: ChildProcess | null = null;
  private connection: JsonLineConnection | null = null;
  private stderr = "";
  private sessionId: string | null = null;
  private requestPermission?: CodexPermissionHandler;
  private notificationHandles: NotificationHandle[] = [];

  constructor(
    private readonly launch: CodexServerLaunchSpec,
    private readonly defaultPermissionHandler?: CodexPermissionHandler,
  ) {}

  async prompt(
    options: CodexServerClientOptions,
    input: string | CodexPromptInput,
    _swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: CodexMessageChunk) => void,
  ): Promise<{ messages: CodexMessageChunk[] }> {
    await this.ensureStarted(options);
    const thread =
      sessionId && this.sessionId !== sessionId
        ? await this.request("thread/resume", {
            threadId: sessionId,
            cwd: options.cwd ?? process.cwd(),
            model: options.model ?? null,
          })
        : sessionId
          ? { thread: { id: sessionId } }
          : await this.request("thread/start", {
              cwd: options.cwd ?? process.cwd(),
              model: options.model ?? null,
            });
    const threadId = (thread as { thread: { id: string } }).thread.id;
    this.sessionId = threadId;
    if (!sessionId) await options.onSessionId?.(threadId);

    const inputItems = codexUserInput(input);
    const matcher = { turnId: "" };
    const completion = this.waitForTurnCompletion(threadId, matcher);
    let turnStarted: unknown;
    try {
      turnStarted = await this.request("turn/start", {
        threadId,
        input: inputItems,
        model: options.model ?? null,
        effort: options.effort ?? null,
        approvalPolicy: this.requestPermission ? "on-request" : "never",
      });
      matcher.turnId = (turnStarted as { turn: { id: string } }).turn.id;
      completion.flush();
    } catch (error) {
      completion.reject(error instanceof Error ? error : new Error(String(error)));
      throw error;
    }

    const messages = await completion.promise;
    const chunks = messagesFromTurnItems(
      ((messages as { turn?: { items?: unknown[] } })?.turn?.items ?? []) as Array<{
        type?: string;
        id?: string;
        text?: string;
        command?: string;
        aggregatedOutput?: string | null;
        exitCode?: number | null;
        tool?: string;
        server?: string;
        result?: unknown;
        content?: string[];
        summary?: string[];
      }>,
    );
    for (const chunk of chunks) onChunk?.(chunk);
    return { messages: chunks };
  }

  async listSessions(
    options: CodexServerClientOptions,
    cwd?: string,
  ): Promise<Array<{ sessionId: string; cwd: string; title?: string; updatedAt?: string }>> {
    await this.ensureStarted(options);
    const response = (await this.request("thread/list", {
      limit: 100,
      ...(cwd || options.cwd ? { cwd: cwd ?? options.cwd } : {}),
    })) as {
      data?: Array<{
        id: string;
        cwd?: string;
        name?: string | null;
        preview?: string;
        updatedAt?: number;
      }>;
    };
    return (response.data ?? []).map((thread) => ({
      sessionId: thread.id,
      cwd: thread.cwd ?? "",
      title: thread.name ?? thread.preview ?? thread.id,
      ...(thread.updatedAt ? { updatedAt: new Date(thread.updatedAt * 1000).toISOString() } : {}),
    }));
  }

  async loadSession(
    options: CodexServerClientOptions,
    sessionId: string,
    cwd: string,
  ): Promise<{ messages: CodexMessageChunk[] }> {
    await this.ensureStarted(options);
    await this.request("thread/resume", {
      threadId: sessionId,
      cwd,
      model: options.model ?? null,
    });
    const response = (await this.request("thread/read", {
      threadId: sessionId,
      includeTurns: true,
    })) as { thread?: { turns?: Array<{ items?: unknown[] }> } };
    const items = (response.thread?.turns ?? []).flatMap(
      (turn) => (turn.items ?? []) as Array<{ type?: string }>,
    );
    return { messages: messagesFromTurnItems(items) };
  }

  stderrOutput(): string {
    return this.stderr;
  }

  kill(): void {
    this.connection?.rejectAll(new Error("Codex server was stopped."));
    for (const handle of this.notificationHandles.splice(0)) handle.dispose();
    this.connection?.close();
    this.connection = null;
    if (this.child && this.child.exitCode === null && this.child.signalCode === null) {
      this.child.kill("SIGTERM");
      setTimeout(() => {
        if (this.child && this.child.exitCode === null && this.child.signalCode === null) {
          this.child.kill("SIGKILL");
        }
      }, 500).unref();
    }
    this.child = null;
  }

  private async ensureStarted(options: CodexServerClientOptions): Promise<void> {
    if (this.connection && this.child) return;
    this.requestPermission = options.requestPermission ?? this.defaultPermissionHandler;
    const env: Record<string, string> = options.clearEnv
      ? {}
      : Object.fromEntries(
          Object.entries(process.env).filter(
            (entry): entry is [string, string] => entry[1] !== undefined,
          ),
        );
    Object.assign(env, options.env ?? {}, this.launch.env);
    const child = spawn(this.launch.command, this.launch.args, {
      cwd: options.cwd,
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stderr = "";
    await waitForSpawn(child);
    const connection = new JsonLineConnection(child);
    this.connection = connection;
    child.stderr?.setEncoding("utf-8");
    child.stderr?.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-4000);
    });
    child.once("exit", () => {
      connection.rejectAll(new Error("Codex server process exited."));
      this.child = null;
    });
    if (options.signal) {
      const abort = (): void => this.kill();
      if (options.signal.aborted) abort();
      else options.signal.addEventListener("abort", abort, { once: true });
    }

    this.registerApprovalHandlers();
    const initialized = await connection.sendRequest("initialize", {
      clientInfo: { name: "swarmx", title: "SwarmX", version: "4.0.0" },
      capabilities: null,
    });
    if (!initialized) throw new Error("Codex app-server did not initialize.");
    connection.sendNotification("initialized");
  }

  private registerApprovalHandlers(): void {
    const connection = this.connection;
    if (!connection) return;
    connection.onRequest(COMMAND_APPROVAL_METHOD, (params) =>
      this.approvalResponse("execute", params),
    );
    connection.onRequest(FILE_CHANGE_APPROVAL_METHOD, (params) =>
      this.approvalResponse("edit", params),
    );
    connection.onRequest(PERMISSIONS_APPROVAL_METHOD, async () => ({
      permissions: {},
      scope: "turn",
      strictAutoReview: false,
    }));
    connection.onRequest(MCP_ELICITATION_METHOD, async () => ({
      action: "cancel",
      content: null,
      _meta: null,
    }));
    connection.onRequest(TOOL_USER_INPUT_METHOD, async () => ({ answers: {} }));
  }

  private async approvalResponse(
    kind: "execute" | "edit",
    params: Record<string, unknown>,
  ): Promise<{ decision: "accept" | "acceptForSession" | "decline" | "cancel" }> {
    const handler = this.requestPermission;
    if (!handler) return { decision: "cancel" };
    const toolCallId = typeof params.itemId === "string" ? params.itemId : "codex-approval";
    const response = await handler({
      sessionId: this.sessionId ?? String(params.threadId ?? ""),
      toolCall: {
        toolCallId,
        kind,
        status: "pending",
        rawInput: params,
      },
      options: [
        { optionId: "allow_once", name: "Allow Once", kind: "allow_once" },
        { optionId: "allow_always", name: "Allow for Session", kind: "allow_always" },
        { optionId: "reject_once", name: "Reject", kind: "reject_once" },
      ],
    });
    if (response.outcome.outcome === "cancelled") return { decision: "cancel" };
    if (response.outcome.outcome === "selected") {
      if (response.outcome.optionId === "allow_once") return { decision: "accept" };
      if (response.outcome.optionId === "allow_always") return { decision: "acceptForSession" };
    }
    return { decision: "decline" };
  }

  private waitForTurnCompletion(
    threadId: string,
    matcher: { turnId: string },
  ): { promise: Promise<unknown>; flush(): void; reject(error: Error): void } {
    const activeConnection = this.connection;
    if (!activeConnection) {
      return {
        promise: Promise.reject(new Error("Codex server connection is unavailable.")),
        flush: () => {},
        reject: () => {},
      };
    }
    const connection = activeConnection;

    const queued: unknown[] = [];
    let settled = false;
    let resolvePromise!: (value: unknown) => void;
    let rejectPromise!: (error: Error) => void;
    const promise = new Promise<unknown>((resolve, reject) => {
      resolvePromise = resolve;
      rejectPromise = reject;
    });

    const matches = (params: unknown): boolean => {
      const candidate = params as { threadId?: string; turn?: { id?: string } };
      return (
        candidate.threadId === threadId &&
        matcher.turnId !== "" &&
        (candidate.turn?.id ?? "") === matcher.turnId
      );
    };

    const notificationHandles = this.notificationHandles;
    let handle!: NotificationHandle;
    let dispose!: () => void;

    function remove(): void {
      dispose();
      const index = notificationHandles.indexOf(handle);
      if (index >= 0) notificationHandles.splice(index, 1);
      connection.removeNotificationWaiter(handle);
    }
    function settle(params: unknown): void {
      if (settled) return;
      settled = true;
      remove();
      resolvePromise(params);
    }
    function reject(error: Error): void {
      if (settled) return;
      settled = true;
      remove();
      rejectPromise(error);
    }

    dispose = connection.onNotification("turn/completed", (params) => {
      if (matches(params)) {
        settle(params);
        return;
      }
      if (matcher.turnId === "" && (params as { threadId?: string }).threadId === threadId) {
        queued.push(params);
      }
    });
    handle = { method: "turn/completed", reject, dispose };
    this.notificationHandles.push(handle);
    connection.addNotificationWaiter(handle);

    return {
      promise,
      flush() {
        for (const params of queued.splice(0)) {
          if (matches(params)) {
            settle(params);
            return;
          }
        }
      },
      reject,
    };
  }

  private request(method: string, params: Record<string, unknown>): Promise<unknown> {
    if (!this.connection)
      return Promise.reject(new Error("Codex server connection is unavailable."));
    return this.connection.sendRequest(method, params);
  }
}

function waitForSpawn(child: ChildProcess): Promise<void> {
  return new Promise((resolve, reject) => {
    child.once("spawn", resolve);
    child.once("error", reject);
  });
}

function codexUserInput(input: string | CodexPromptInput): Array<Record<string, unknown>> {
  const items: Array<Record<string, unknown>> = [];
  const text = typeof input === "string" ? input : input.text;
  if (text) items.push({ type: "text", text, text_elements: [] });
  if (typeof input !== "string") {
    for (const attachment of input.attachments ?? []) {
      if (attachment.kind !== "image") continue;
      const path = attachment.uri.startsWith("file://")
        ? fileURLToPath(attachment.uri)
        : attachment.uri;
      items.push({ type: "localImage", path });
    }
  }
  return items.length > 0 ? items : [{ type: "text", text: "", text_elements: [] }];
}

function messagesFromTurnItems(
  items: Array<{
    type?: string;
    id?: string;
    text?: string;
    command?: string;
    aggregatedOutput?: string | null;
    exitCode?: number | null;
    tool?: string;
    server?: string;
    result?: unknown;
    content?: string[];
    summary?: string[];
  }>,
): CodexMessageChunk[] {
  const messages: CodexMessageChunk[] = [];
  for (const item of items) {
    switch (item.type) {
      case "agentMessage":
        if (item.text) {
          messages.push({ role: "assistant", content: item.text, kind: "message" });
        }
        break;
      case "reasoning": {
        const text = [...(item.summary ?? []), ...(item.content ?? [])].join("\n");
        if (text) messages.push({ role: "assistant", content: text, kind: "thinking" });
        break;
      }
      case "commandExecution":
        if (item.command) {
          messages.push({ role: "assistant", content: item.command, kind: "tool_call" });
        }
        if (item.aggregatedOutput !== null && item.aggregatedOutput !== undefined) {
          messages.push({
            role: "tool",
            content:
              item.aggregatedOutput ||
              (item.exitCode === 0
                ? "Command completed."
                : `Exit code ${item.exitCode ?? "unknown"}.`),
            kind: "tool_result",
            toolName: "execute",
          });
        }
        break;
      case "mcpToolCall":
        messages.push({
          role: "assistant",
          content: `${item.server ?? "mcp"}.${item.tool ?? ""}`,
          kind: "tool_call",
          toolName: item.tool,
        });
        if (item.result) {
          messages.push({
            role: "tool",
            content: JSON.stringify(item.result),
            kind: "tool_result",
            toolName: item.tool,
          });
        }
        break;
      default:
        break;
    }
  }
  return messages;
}
