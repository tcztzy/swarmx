import { AsyncLocalStorage } from "node:async_hooks";
import { type ChildProcess, spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import type {
  SessionInfo as AcpSessionInfo,
  Client,
  ClientSideConnection,
  ListSessionsRequest,
  LoadSessionRequest,
  LoadSessionResponse,
  NewSessionRequest,
  PromptRequest,
  RequestPermissionRequest,
  RequestPermissionResponse,
  SessionConfigOption,
  SessionNotification,
  SessionUpdate,
} from "@agentclientprotocol/sdk";
import { SWARMX_VERSION } from "./version.js";

let _acp: typeof import("@agentclientprotocol/sdk") | null = null;
const ACP_CANCEL_GRACE_MS = 500;
const CHILD_KILL_GRACE_MS = 500;

interface RequestContext {
  id: string;
  controller: AbortController;
  clients: Set<AcpClient>;
  cancelPromise: Promise<void> | null;
}

let requestScope: AsyncLocalStorage<RequestContext> | null = null;
let activeRequests: Map<string, RequestContext> | null = null;

export class RequestCancelledError extends Error {
  constructor(requestId?: string) {
    super(requestId ? `Request "${requestId}" was cancelled.` : "Request was cancelled.");
    this.name = "RequestCancelledError";
  }
}

/**
 * Run one request in a cancellation scope. Request IDs are exclusive while the
 * request is active so a stale stop can never be redirected to a newer run.
 */
export async function withAcpRequest<T>(requestId: string, run: () => Promise<T>): Promise<T> {
  const id = validateRequestId(requestId);
  const requests = getActiveRequests();
  if (requests.has(id)) {
    throw new Error(`Request "${id}" is already active.`);
  }

  const context: RequestContext = {
    id,
    controller: new AbortController(),
    clients: new Set(),
    cancelPromise: null,
  };
  requests.set(id, context);

  return getRequestScope().run(context, async () => {
    try {
      const result = await run();
      if (context.controller.signal.aborted) {
        throw context.controller.signal.reason instanceof Error
          ? context.controller.signal.reason
          : new RequestCancelledError(context.id);
      }
      return result;
    } catch (error) {
      if (context.controller.signal.aborted) {
        throw context.controller.signal.reason instanceof Error
          ? context.controller.signal.reason
          : new RequestCancelledError(context.id);
      }
      throw error;
    } finally {
      for (const client of [...context.clients]) client.kill();
      if (requests.get(id) === context) requests.delete(id);
    }
  });
}

/**
 * Cancel an active request. The AbortSignal is tripped synchronously; ACP
 * clients then send session/cancel and retain process termination as fallback.
 * Repeated cancellation of the same live request is idempotent.
 */
export async function cancelAcpRequest(requestId: string): Promise<boolean> {
  const context = getActiveRequests().get(requestId);
  if (!context) return false;

  if (!context.cancelPromise) {
    context.controller.abort(new RequestCancelledError(context.id));
    context.cancelPromise = Promise.allSettled(
      [...context.clients].map((client) => client.cancel()),
    ).then(() => undefined);
  }

  await context.cancelPromise;
  return true;
}

/** The signal for the request currently executing in this async context. */
export function currentRequestSignal(): AbortSignal | undefined {
  return getRequestScope().getStore()?.controller.signal;
}

/** Throw the request's stable cancellation reason at cooperative boundaries. */
export function throwIfCurrentRequestCancelled(): void {
  const signal = currentRequestSignal();
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new RequestCancelledError();
  }
}

async function loadAcp(): Promise<typeof import("@agentclientprotocol/sdk")> {
  if (!_acp) {
    _acp = await import("@agentclientprotocol/sdk");
  }
  return _acp;
}

export interface AcpClientOptions {
  command: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
  clearEnv?: boolean;
  /** Requested ACP session model. Applied only when the server advertises it. */
  model?: string;
  /** Requested ACP reasoning/thought level. Applied after model selection. */
  effort?: string;
  /** Optional host-owned interactive authorization bridge. Missing handlers fail closed. */
  requestPermission?: AcpPermissionHandler;
}

export type AcpPermissionRequest = RequestPermissionRequest;
export type AcpPermissionResponse = RequestPermissionResponse;
export type AcpPermissionHandler = (
  request: AcpPermissionRequest,
) => Promise<AcpPermissionResponse>;

export interface AcpPromptResult {
  sessionId: string;
  messages: MessageChunk[];
  stopReason: string;
}

export interface MessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  swarmEvent?: string;
  toolName?: string;
}

export class AcpClient {
  private child: ChildProcess | null = null;
  private connection: ClientSideConnection | null = null;
  private requestContext: RequestContext | null = null;
  private requestId: string | undefined;
  private sessionId: string | null = null;
  private promptActive = false;
  private cancellationRequested = false;
  private cancelNotificationSent = false;
  private cancelFallback: ReturnType<typeof setTimeout> | null = null;
  private stderr = "";

  private buildEnv(opts: AcpClientOptions): Record<string, string> {
    const env: Record<string, string> = {};
    if (!opts.clearEnv) {
      Object.assign(env, process.env as Record<string, string>);
    }
    if (opts.env) {
      Object.assign(env, opts.env);
    }
    return env;
  }

  private async spawnAndConnect(
    opts: AcpClientOptions,
    onSessionUpdate: (update: SessionUpdate) => void,
  ): Promise<{
    connection: ClientSideConnection;
    acp: typeof import("@agentclientprotocol/sdk");
  }> {
    this.beginOperation();
    const acp = await loadAcp();
    this.throwIfCancelled();
    const env = this.buildEnv(opts);

    const child = spawn(opts.command, opts.args, {
      cwd: opts.cwd,
      env,
      detached: process.platform !== "win32",
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stderr = "";

    await waitForChildSpawn(child);
    this.throwIfCancelled();

    child.once("exit", () => {
      if (this.child === child) this.child = null;
    });

    child.stderr?.setEncoding("utf-8");
    child.stderr?.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-4000);
    });

    if (!child.stdin || !child.stdout) {
      throw new Error("ACP child process stdio is unavailable");
    }

    const transport = acp.ndJsonStream(Writable.toWeb(child.stdin), Readable.toWeb(child.stdout));

    const clientStubs: Client = {
      async requestPermission(
        request: RequestPermissionRequest,
      ): Promise<RequestPermissionResponse> {
        if (!opts.requestPermission) return { outcome: { outcome: "cancelled" } };
        const response = await opts.requestPermission(request);
        const outcome = response.outcome;
        if (
          outcome.outcome === "selected" &&
          !request.options.some((option) => option.optionId === outcome.optionId)
        ) {
          return { outcome: { outcome: "cancelled" } };
        }
        return response;
      },
      async sessionUpdate(notification: SessionNotification): Promise<void> {
        onSessionUpdate(notification.update);
      },
    };

    const connection = new acp.ClientSideConnection(() => clientStubs, transport);
    this.connection = connection;

    return { connection, acp };
  }

  async prompt(
    opts: AcpClientOptions,
    userText: string,
    swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<AcpPromptResult> {
    const chunks: MessageChunk[] = [];

    try {
      const { connection, acp } = await this.spawnAndConnect(opts, (update) => {
        const msg = sessionUpdateToChunk(update);
        if (msg) {
          chunks.push(msg);
          onChunk?.(msg);
        }
      });

      const initialized = await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });
      this.throwIfCancelled();

      let sid: string;
      let advertisedModels: {
        currentModelId: string;
        availableModels: Array<{ modelId: string }>;
      } | null = null;
      let configOptions: SessionConfigOption[] | null = null;
      if (sessionId) {
        sid = sessionId;
        if (initialized.agentCapabilities?.loadSession) {
          const loaded = await connection.loadSession({
            sessionId,
            cwd: opts.cwd ?? process.cwd(),
            mcpServers: [],
          });
          advertisedModels = loaded.models ?? null;
          configOptions = loaded.configOptions ?? null;
        }
      } else {
        const resp = await connection.newSession({
          cwd: opts.cwd ?? process.cwd(),
          mcpServers: [],
        });
        sid = resp.sessionId;
        advertisedModels = resp.models ?? null;
        configOptions = resp.configOptions ?? null;
      }
      this.sessionId = sid;
      this.throwIfCancelled();

      await applySessionSelections(
        connection,
        sid,
        configOptions,
        advertisedModels,
        opts.model,
        opts.effort,
        () => this.throwIfCancelled(),
      );

      const meta: Record<string, unknown> = {};
      if (swarmConfig) {
        meta.swarmConfig = swarmConfig;
      }

      const promptBlock: PromptRequest["prompt"][number] = {
        type: "text",
        text: userText,
        ...(Object.keys(meta).length > 0 ? { _meta: meta } : {}),
      };

      const promptReq: PromptRequest = {
        sessionId: sid,
        prompt: [promptBlock],
      };

      this.promptActive = true;
      const promptResp = await connection.prompt(promptReq);
      this.promptActive = false;

      return {
        sessionId: sid,
        messages: mergeChunks(chunks),
        stopReason: promptResp.stopReason ?? "end_turn",
      };
    } catch (error) {
      if (this.cancellationRequested) {
        throw new RequestCancelledError(this.requestId);
      }
      throw error;
    } finally {
      this.promptActive = false;
      this.kill();
    }
  }

  async listSessions(opts: AcpClientOptions, cwd?: string): Promise<AcpSessionInfo[]> {
    try {
      const { connection, acp } = await this.spawnAndConnect(opts, () => {});
      const initialized = await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });
      if (!initialized.agentCapabilities?.sessionCapabilities?.list) {
        throw new Error("ACP backend does not advertise session/list support.");
      }

      const req: ListSessionsRequest = cwd ? { cwd } : {};
      const resp = await connection.listSessions(req);
      return resp.sessions ?? [];
    } finally {
      this.kill();
    }
  }

  async loadSession(
    opts: AcpClientOptions,
    sessionId: string,
    cwd: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<{ response: LoadSessionResponse; messages: MessageChunk[] }> {
    const chunks: MessageChunk[] = [];

    try {
      const { connection, acp } = await this.spawnAndConnect(opts, (update) => {
        const msg = sessionUpdateToChunk(update);
        if (msg) {
          chunks.push(msg);
          onChunk?.(msg);
        }
      });
      const initialized = await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });
      if (!initialized.agentCapabilities?.loadSession) {
        throw new Error("ACP backend does not advertise session/load support.");
      }

      const req: LoadSessionRequest = {
        sessionId,
        cwd,
        mcpServers: [],
      };
      const resp = await connection.loadSession(req);
      return { response: resp, messages: mergeChunks(chunks) };
    } finally {
      this.kill();
    }
  }

  async newSession(opts: AcpClientOptions, cwd: string): Promise<string> {
    try {
      const { connection, acp } = await this.spawnAndConnect(opts, () => {});
      await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });

      const req: NewSessionRequest = { cwd, mcpServers: [] };
      const resp = await connection.newSession(req);
      return resp.sessionId;
    } finally {
      this.kill();
    }
  }

  /** Request protocol-level cancellation, then terminate the process if it stalls. */
  async cancel(): Promise<void> {
    if (this.cancellationRequested) return;
    this.cancellationRequested = true;

    if (this.child && !this.cancelFallback) {
      this.cancelFallback = setTimeout(() => this.kill(), ACP_CANCEL_GRACE_MS);
      this.cancelFallback.unref?.();
    }

    if (this.connection && this.sessionId && this.promptActive && !this.cancelNotificationSent) {
      this.cancelNotificationSent = true;
      try {
        const settled = await settleWithin(
          this.connection.cancel({ sessionId: this.sessionId }),
          ACP_CANCEL_GRACE_MS,
        );
        if (!settled) this.kill();
      } catch {
        this.kill();
      }
    }
  }

  kill(): void {
    if (this.cancelFallback) clearTimeout(this.cancelFallback);
    this.cancelFallback = null;

    const child = this.child;
    this.child = null;
    if (child && child.exitCode === null && child.signalCode === null) {
      killChildTree(child, "SIGTERM");
      const forceKill = setTimeout(() => {
        killChildTree(child, "SIGKILL");
      }, CHILD_KILL_GRACE_MS);
      forceKill.unref?.();
      child.once("exit", () => {
        clearTimeout(forceKill);
        killChildTree(child, "SIGKILL");
      });
    }

    this.connection = null;
    this.sessionId = null;
    this.promptActive = false;
    this.requestContext?.clients.delete(this);
    this.requestContext = null;
  }

  stderrOutput(): string {
    return this.stderr.trim();
  }

  private beginOperation(): void {
    if (this.child || this.connection) {
      throw new Error("ACP client already has an active operation.");
    }

    this.requestContext = getRequestScope().getStore() ?? null;
    this.requestId = this.requestContext?.id;
    this.requestContext?.clients.add(this);
    this.cancellationRequested = this.requestContext?.controller.signal.aborted ?? false;
    this.cancelNotificationSent = false;
    this.sessionId = null;
    this.promptActive = false;
    this.throwIfCancelled();
  }

  private throwIfCancelled(): void {
    if (!this.cancellationRequested && !this.requestContext?.controller.signal.aborted) return;
    this.cancellationRequested = true;
    throw new RequestCancelledError(this.requestContext?.id);
  }
}

type LegacyModelState = {
  currentModelId: string;
  availableModels: Array<{ modelId: string }>;
};

async function applySessionSelections(
  connection: ClientSideConnection,
  sessionId: string,
  initialConfigOptions: SessionConfigOption[] | null,
  legacyModels: LegacyModelState | null,
  model: string | undefined,
  effort: string | undefined,
  checkCancelled: () => void,
): Promise<void> {
  let configOptions = initialConfigOptions ?? [];

  if (model) {
    const modelOption = findSessionConfigSelect(configOptions, "model", [
      "model",
      "models",
      "model_id",
    ]);
    if (modelOption) {
      const modelValue = findSessionConfigValue(modelOption, model);
      if (!modelValue) {
        throw new Error(`ACP backend cannot run configured model "${model}".`);
      }
      if (modelOption.currentValue !== modelValue) {
        const response = await connection.setSessionConfigOption({
          sessionId,
          configId: modelOption.id,
          value: modelValue,
        });
        configOptions = response.configOptions;
        checkCancelled();
      }
    } else {
      if (!legacyModels) {
        throw new Error(
          `ACP backend did not advertise session model selection; cannot apply model "${model}".`,
        );
      }
      if (!legacyModels.availableModels.some((available) => available.modelId === model)) {
        throw new Error(`ACP backend cannot run configured model "${model}".`);
      }
      if (legacyModels.currentModelId !== model) {
        await connection.unstable_setSessionModel({ sessionId, modelId: model });
        checkCancelled();
      }
    }
  }

  if (!effort) return;
  const effortOption = findSessionConfigSelect(configOptions, "thought_level", [
    "thought_level",
    "reasoning_effort",
    "reasoning",
    "effort",
  ]);
  if (!effortOption) {
    throw new Error(
      `ACP backend did not advertise reasoning effort selection; cannot apply effort "${effort}".`,
    );
  }
  const effortValue = findSessionConfigValue(effortOption, effort);
  if (!effortValue) {
    throw new Error(`ACP backend cannot apply configured effort "${effort}".`);
  }
  if (effortOption.currentValue !== effortValue) {
    await connection.setSessionConfigOption({
      sessionId,
      configId: effortOption.id,
      value: effortValue,
    });
    checkCancelled();
  }
}

/** Find a stable ACP select option by category, with id/name fallback for category-less agents. */
export function findSessionConfigSelect(
  configOptions: readonly SessionConfigOption[],
  category: "model" | "thought_level",
  fallbackNames: readonly string[],
): Extract<SessionConfigOption, { type: "select" }> | undefined {
  const selects = configOptions.filter(
    (option): option is Extract<SessionConfigOption, { type: "select" }> =>
      option.type === "select",
  );
  const categorized = selects.filter((option) => option.category === category);
  const normalizedFallbacks = new Set(fallbackNames.map(normalizeConfigName));
  const matches =
    categorized.length > 0
      ? categorized
      : selects.filter(
          (option) =>
            normalizedFallbacks.has(normalizeConfigName(option.id)) ||
            normalizedFallbacks.has(normalizeConfigName(option.name)),
        );
  if (matches.length > 1) {
    throw new Error(`ACP backend advertised ambiguous ${category} configuration options.`);
  }
  return matches[0];
}

/** Resolve both flat and grouped ACP select values without interpreting Provider metadata. */
export function findSessionConfigValue(
  option: Extract<SessionConfigOption, { type: "select" }>,
  requested: string,
): string | undefined {
  const values = option.options.flatMap((entry) => ("group" in entry ? entry.options : [entry]));
  const exact = values.find((value) => value.value === requested);
  if (exact) return exact.value;
  const normalized = normalizeConfigName(requested);
  const matches = values.filter(
    (value) =>
      normalizeConfigName(value.value) === normalized ||
      normalizeConfigName(value.name) === normalized,
  );
  return matches.length === 1 ? matches[0]?.value : undefined;
}

function normalizeConfigName(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
}

function validateRequestId(requestId: string): string {
  if (typeof requestId !== "string" || requestId.trim().length === 0) {
    throw new Error("Request ID must be a non-empty string.");
  }
  if (requestId.length > 256) throw new Error("Request ID must be at most 256 characters.");
  return requestId;
}

function getRequestScope(): AsyncLocalStorage<RequestContext> {
  requestScope ??= new AsyncLocalStorage<RequestContext>();
  return requestScope;
}

function getActiveRequests(): Map<string, RequestContext> {
  activeRequests ??= new Map<string, RequestContext>();
  return activeRequests;
}

function waitForChildSpawn(child: ChildProcess): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const onError = (error: Error): void => {
      child.on("error", () => {});
      reject(error);
    };
    child.once("error", onError);
    child.once("spawn", () => {
      child.removeListener("error", onError);
      // Keep a listener installed because later ChildProcess errors must never
      // become process-level uncaught exceptions during cleanup.
      child.on("error", () => {});
      resolve();
    });
  });
}

function killChildTree(child: ChildProcess, signal: NodeJS.Signals): void {
  if (process.platform !== "win32" && child.pid) {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch {
      // Fall through when the process group has already exited or was unavailable.
    }
  }
  child.kill(signal);
}

function settleWithin(promise: Promise<unknown>, timeoutMs: number): Promise<boolean> {
  return new Promise<boolean>((resolve, reject) => {
    const timer = setTimeout(() => resolve(false), timeoutMs);
    timer.unref?.();
    promise.then(
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

function sessionUpdateToChunk(update: SessionUpdate): MessageChunk | null {
  const u = update as Record<string, unknown>;
  const updateKind = stringValue(u.sessionUpdate) ?? stringValue(u.updateType);

  switch (updateKind) {
    case "user_message_chunk":
    case "agent_message_chunk": {
      const content = (u.content as Record<string, unknown> | undefined) ?? {};
      const text = String(content.text ?? "");
      if (!text) return null;
      const meta = recordValue(content._meta) ?? recordValue(content.meta) ?? {};
      return {
        role:
          stringValue(meta.role) ?? (updateKind === "user_message_chunk" ? "user" : "assistant"),
        content: text,
        kind: "message",
        agent: stringValue(meta.agent),
        swarmEvent: stringValue(meta.swarmEvent),
      };
    }
    case "agent_thought_chunk": {
      const content = (u.content as Record<string, unknown> | undefined) ?? {};
      const text = String(content.text ?? "");
      if (!text) return null;
      return { role: "assistant", content: text, kind: "thinking" };
    }
    case "tool_call": {
      const args = u.rawInput ? JSON.stringify(u.rawInput) : "";
      return {
        role: "assistant",
        content: args,
        kind: "tool_call",
        toolName: stringValue(u.title),
      };
    }
    case "tool_call_update": {
      const fields = recordValue(u.fields);
      const rawOutput = u.rawOutput ?? fields?.rawOutput;
      const status = u.status ?? fields?.status;
      const result =
        (rawOutput ? JSON.stringify(rawOutput) : "") || (status ? JSON.stringify(status) : "");
      if (!result) return null;
      return {
        role: "assistant",
        content: result,
        kind: "tool_result",
        toolName: stringValue(u.title) ?? stringValue(fields?.title) ?? "tool",
      };
    }
    default:
      return null;
  }
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function mergeChunks(chunks: MessageChunk[]): MessageChunk[] {
  function key(c: MessageChunk): string {
    return `${c.role ?? ""}|${c.agent ?? ""}|${c.swarmEvent ?? ""}|${c.kind}`;
  }

  const merged: MessageChunk[] = [];
  for (const chunk of chunks) {
    const ck = key(chunk);
    const last = merged[merged.length - 1];
    if (last && key(last) === ck) {
      last.content += chunk.content;
    } else {
      merged.push({ ...chunk });
    }
  }
  return merged;
}
