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
  SessionModeState,
  SessionNotification,
  SessionUpdate,
} from "@agentclientprotocol/sdk";
import { buildAcpPromptContent } from "./media.js";
import {
  cancelRequest,
  RequestCancelledError,
  type RequestParticipantRegistration,
  registerCurrentRequestParticipant,
  withRequestScope,
} from "./request-scope.js";
import type { MediaAttachment } from "./types.js";
import { SWARMX_VERSION } from "./version.js";

export {
  currentRequestSignal,
  RequestCancelledError,
  throwIfCurrentRequestCancelled,
} from "./request-scope.js";

let _acp: typeof import("@agentclientprotocol/sdk") | null = null;
const ACP_CANCEL_GRACE_MS = 500;
const CHILD_KILL_GRACE_MS = 500;

export class AcpSessionUnavailableError extends Error {
  readonly sessionId: string;

  constructor(sessionId: string, cause?: unknown) {
    const detail = cause instanceof Error && cause.message ? `: ${cause.message}` : "";
    super(`ACP session "${sessionId}" is unavailable${detail}`, { cause });
    this.name = "AcpSessionUnavailableError";
    this.sessionId = sessionId;
  }
}

export interface AcpPromptInput {
  text: string;
  attachments?: readonly MediaAttachment[];
}

/**
 * Run one request in a cancellation scope. Request IDs are exclusive while the
 * request is active so a stale stop can never be redirected to a newer run.
 */
export async function withAcpRequest<T>(requestId: string, run: () => Promise<T>): Promise<T> {
  return withRequestScope(requestId, run);
}

/**
 * Cancel an active request. The AbortSignal is tripped synchronously; ACP
 * clients then send session/cancel and retain process termination as fallback.
 * Repeated cancellation of the same live request is idempotent.
 */
export async function cancelAcpRequest(requestId: string): Promise<boolean> {
  return cancelRequest(requestId);
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
  /** Preferred ACP session mode. Applied when the server advertises a matching mode. */
  preferredMode?: string;
  /** Optional host-owned interactive authorization bridge. Missing handlers fail closed. */
  requestPermission?: AcpPermissionHandler;
  /** Called after a new Session exists and before its first prompt begins. */
  onSessionId?: (sessionId: string) => void | Promise<void>;
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
  kind: "message" | "thinking" | "tool_call" | "tool_progress" | "tool_result";
  agent?: string;
  render?: {
    invocationId?: string;
    status?: "queued" | "running" | "succeeded" | "failed" | "canceled" | "skipped" | "completed";
  };
  swarmEvent?: string;
  structuredContent?: unknown;
  toolName?: string;
}

export class AcpClient {
  private child: ChildProcess | null = null;
  private connection: ClientSideConnection | null = null;
  private requestRegistration: RequestParticipantRegistration | undefined;
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
    input: string | AcpPromptInput,
    swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<AcpPromptResult> {
    const chunks: MessageChunk[] = [];
    let promptUpdatesActive = false;

    try {
      const { connection, acp } = await this.spawnAndConnect(opts, (update) => {
        if (!promptUpdatesActive) return;
        const msg = sessionUpdateToChunk(update);
        if (msg) {
          if (msg.kind !== "tool_progress") chunks.push(msg);
          onChunk?.(msg);
        }
      });

      const initialized = await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });
      this.throwIfCancelled();
      const promptInput = typeof input === "string" ? { text: input } : input;

      let sid: string;
      let advertisedModels: {
        currentModelId: string;
        availableModels: Array<{ modelId: string }>;
      } | null = null;
      let advertisedModes: SessionModeState | null = null;
      let configOptions: SessionConfigOption[] | null = null;
      if (sessionId) {
        sid = sessionId;
        if (!initialized.agentCapabilities?.loadSession) {
          throw new AcpSessionUnavailableError(
            sessionId,
            new Error("ACP backend does not advertise session/load support."),
          );
        }
        try {
          const loaded = await connection.loadSession({
            sessionId,
            cwd: opts.cwd ?? process.cwd(),
            mcpServers: [],
          });
          advertisedModels =
            (loaded as LoadSessionResponse & { models?: LegacyModelState }).models ?? null;
          advertisedModes = loaded.modes ?? null;
          configOptions = loaded.configOptions ?? null;
        } catch (error) {
          if (error instanceof AcpSessionUnavailableError) throw error;
          throw new AcpSessionUnavailableError(sessionId, error);
        }
      } else {
        const resp = await connection.newSession({
          cwd: opts.cwd ?? process.cwd(),
          mcpServers: [],
        });
        sid = resp.sessionId;
        advertisedModels = (resp as typeof resp & { models?: LegacyModelState }).models ?? null;
        advertisedModes = resp.modes ?? null;
        configOptions = resp.configOptions ?? null;
        await opts.onSessionId?.(sid);
      }
      this.sessionId = sid;
      this.throwIfCancelled();

      await applySessionSelections(
        connection,
        sid,
        configOptions,
        advertisedModels,
        advertisedModes,
        opts.preferredMode,
        opts.model,
        opts.effort,
        () => this.throwIfCancelled(),
      );

      const meta: Record<string, unknown> = {};
      if (swarmConfig) {
        meta.swarmConfig = swarmConfig;
      }

      const promptReq: PromptRequest = {
        sessionId: sid,
        prompt: await buildAcpPromptContent({
          text: promptInput.text,
          attachments: promptInput.attachments,
          promptCapabilities: initialized.agentCapabilities?.promptCapabilities,
          meta,
        }),
      };

      promptUpdatesActive = true;
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
          if (msg.kind !== "tool_progress") chunks.push(msg);
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
    this.requestRegistration?.unregister();
    this.requestRegistration = undefined;
  }

  stderrOutput(): string {
    return this.stderr.trim();
  }

  private beginOperation(): void {
    if (this.child || this.connection) {
      throw new Error("ACP client already has an active operation.");
    }

    this.requestRegistration = registerCurrentRequestParticipant({
      cancel: () => this.cancel(),
      cleanup: () => this.kill(),
    });
    this.requestId = this.requestRegistration?.requestId;
    this.cancellationRequested = this.requestRegistration?.signal.aborted ?? false;
    this.cancelNotificationSent = false;
    this.sessionId = null;
    this.promptActive = false;
    this.throwIfCancelled();
  }

  private throwIfCancelled(): void {
    if (!this.cancellationRequested && !this.requestRegistration?.signal.aborted) return;
    this.cancellationRequested = true;
    throw new RequestCancelledError(this.requestRegistration?.requestId);
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
  legacyModes: SessionModeState | null,
  preferredMode: string | undefined,
  model: string | undefined,
  effort: string | undefined,
  checkCancelled: () => void,
): Promise<void> {
  let configOptions = initialConfigOptions ?? [];

  if (preferredMode) {
    const normalizedMode = normalizeConfigName(preferredMode);
    const advertisedMode = legacyModes?.availableModes.find(
      (mode) =>
        normalizeConfigName(mode.id) === normalizedMode ||
        normalizeConfigName(mode.name) === normalizedMode,
    );
    if (advertisedMode && legacyModes?.currentModeId !== advertisedMode.id) {
      await connection.setSessionMode({ sessionId, modeId: advertisedMode.id });
      checkCancelled();
    } else if (!advertisedMode) {
      const modeOption = findSessionConfigSelect(configOptions, "mode", [
        "mode",
        "session_mode",
        "permission_mode",
      ]);
      const modeValue = modeOption ? findSessionConfigValue(modeOption, preferredMode) : undefined;
      if (modeOption && modeValue && modeOption.currentValue !== modeValue) {
        const response = await connection.setSessionConfigOption({
          sessionId,
          configId: modeOption.id,
          value: modeValue,
        });
        configOptions = response.configOptions;
        checkCancelled();
      }
    }
  }

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
        const setSessionModel = (
          connection as ClientSideConnection & {
            unstable_setSessionModel?: (params: {
              sessionId: string;
              modelId: string;
            }) => Promise<unknown>;
          }
        ).unstable_setSessionModel;
        if (!setSessionModel) {
          throw new Error(
            `ACP backend did not expose a compatible session model selection method; cannot apply model "${model}".`,
          );
        }
        await setSessionModel.call(connection, { sessionId, modelId: model });
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
  category: "mode" | "model" | "thought_level",
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
      const invocationId = stringValue(u.toolCallId);
      return {
        role: "assistant",
        content: args,
        kind: "tool_call",
        toolName: stringValue(u.title),
        ...(invocationId ? { render: { invocationId, status: "running" } } : {}),
      };
    }
    case "tool_call_update": {
      const fields = recordValue(u.fields);
      const rawOutput = u.rawOutput ?? fields?.rawOutput;
      const status = u.status ?? fields?.status;
      const invocationId = stringValue(u.toolCallId) ?? stringValue(fields?.toolCallId);
      const terminalProgress = acpTerminalProgress(u, fields, invocationId);
      if (terminalProgress) return terminalProgress;
      const renderStatus = acpRenderStatus(status);
      const result =
        (rawOutput ? JSON.stringify(rawOutput) : "") || (status ? JSON.stringify(status) : "");
      if (!result) return null;
      return {
        role: "assistant",
        content: result,
        kind: "tool_result",
        toolName: stringValue(u.title) ?? stringValue(fields?.title) ?? "tool",
        ...(invocationId || renderStatus
          ? {
              render: {
                ...(invocationId ? { invocationId } : {}),
                ...(renderStatus ? { status: renderStatus } : {}),
              },
            }
          : {}),
      };
    }
    default:
      return null;
  }
}

function acpTerminalProgress(
  update: Record<string, unknown>,
  fields: Record<string, unknown> | undefined,
  invocationId: string | undefined,
): MessageChunk | null {
  if (!invocationId) return null;
  const meta = recordValue(update._meta) ?? recordValue(fields?._meta);
  if (!meta) return null;
  const delta = recordValue(meta.terminal_output_delta);
  const snapshot = recordValue(meta.terminal_output);
  const terminal = delta ?? snapshot;
  const content = stringValue(terminal?.data);
  if (!terminal || !content) return null;
  const terminalId = stringValue(terminal.terminal_id);
  return {
    role: "tool",
    content,
    kind: "tool_progress",
    toolName: stringValue(update.title) ?? stringValue(fields?.title) ?? "terminal",
    structuredContent: {
      output: content,
      stream: "combined",
      mode: delta ? "append" : "replace",
      ...(terminalId ? { terminal_id: terminalId } : {}),
    },
    render: { invocationId, status: "running" },
  };
}

function acpRenderStatus(
  value: unknown,
): NonNullable<MessageChunk["render"]>["status"] | undefined {
  if (typeof value !== "string") return undefined;
  if (["queued", "pending"].includes(value)) return "queued";
  if (["running", "in_progress"].includes(value)) return "running";
  if (["completed", "succeeded"].includes(value)) return "succeeded";
  if (["failed", "error"].includes(value)) return "failed";
  if (["canceled", "cancelled"].includes(value)) return "canceled";
  return value === "skipped" ? "skipped" : undefined;
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
