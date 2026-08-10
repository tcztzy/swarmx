import { createHash, randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import * as fs from "node:fs";
import { createConnection, createServer, type Server, type Socket } from "node:net";
import { homedir } from "node:os";
import path from "node:path";
import { z } from "zod";
import {
  AppAttachedTaskControlService,
  type CreateTaskWorkItemInput,
} from "./task-control-service.js";
import {
  isTaskWorkItemRunnable,
  TaskApprovalSchema,
  TaskApprovalStatusSchema,
  TaskBudgetSchema,
  type TaskWorkItem,
  TaskWorkItemSchema,
} from "./task-runtime.js";
import { TaskRuntimeStore } from "./task-runtime-store.js";
import { TaskWorkerLaunchSpecSchema } from "./task-worker-process.js";
import {
  TaskWorkerCapabilityGrantSchema,
  TaskWorkerPayloadSchema,
} from "./task-worker-protocol.js";

const DEFAULT_TASK_RUNTIME_ROOT = path.join(homedir(), ".swarmx", "task-runtime");
const SUPERVISOR_TOKEN_FILE = "supervisor-token";
const SUPERVISOR_SOCKET_FILE = "supervisor.sock";
const MAX_PROTOCOL_LINE_BYTES = 4 * 1024 * 1024;
const DEFAULT_REQUEST_TIMEOUT_MS = 5_000;
const WorkItemIdSchema = z.string().regex(/^awi_[A-Za-z0-9][A-Za-z0-9_-]*$/);
const RequestIdSchema = z.uuid();
const TokenSchema = z.string().regex(/^[a-f0-9]{64}$/);

export const TaskSupervisorCreateWorkItemSchema = z
  .object({
    id: WorkItemIdSchema.optional(),
    backend: z.string().min(1).max(160),
    operation: z.string().min(1).max(160),
    input: TaskWorkerPayloadSchema,
    priority: z.number().int().optional(),
    owner: z.string().min(1).max(256).optional(),
    budget: TaskBudgetSchema.optional(),
    maxAttempts: z.number().int().positive().max(100).optional(),
    creatorSessionId: z.string().min(1).max(256).optional(),
  })
  .strict();

const PingCommandSchema = z.object({ operation: z.literal("ping") }).strict();
const ListCommandSchema = z.object({ operation: z.literal("list") }).strict();
const CreateCommandSchema = z
  .object({
    operation: z.literal("create"),
    workItem: TaskSupervisorCreateWorkItemSchema,
  })
  .strict();
const RunCommandSchema = z
  .object({
    operation: z.literal("run"),
    workItemId: WorkItemIdSchema,
    launch: TaskWorkerLaunchSpecSchema,
    grants: z.array(TaskWorkerCapabilityGrantSchema).max(256).default([]),
  })
  .strict();
const CancelCommandSchema = z
  .object({
    operation: z.literal("cancel"),
    workItemId: WorkItemIdSchema,
    reason: z.string().min(1).max(512).optional(),
  })
  .strict();
const DecideCommandSchema = z
  .object({
    operation: z.literal("decide"),
    approvalId: z.string().regex(/^apr_[A-Za-z0-9][A-Za-z0-9_-]*$/),
    status: TaskApprovalStatusSchema.exclude(["requested"]),
    decidedBy: z.string().min(1).max(256),
    reason: z.string().min(1).max(512).optional(),
    response: TaskWorkerPayloadSchema.optional(),
  })
  .strict();

export const TaskSupervisorCommandSchema = z.discriminatedUnion("operation", [
  PingCommandSchema,
  ListCommandSchema,
  CreateCommandSchema,
  RunCommandSchema,
  CancelCommandSchema,
  DecideCommandSchema,
]);

const authShape = {
  requestId: RequestIdSchema,
  token: TokenSchema,
};
export const TaskSupervisorRequestSchema = z.discriminatedUnion("operation", [
  PingCommandSchema.extend(authShape),
  ListCommandSchema.extend(authShape),
  CreateCommandSchema.extend(authShape),
  RunCommandSchema.extend(authShape),
  CancelCommandSchema.extend(authShape),
  DecideCommandSchema.extend(authShape),
]);

const responseBase = { requestId: RequestIdSchema, ok: z.literal(true) };
export const TaskSupervisorResponseSchema = z.discriminatedUnion("operation", [
  z
    .object({
      ...responseBase,
      operation: z.literal("ping"),
      supervisorPid: z.number().int().positive(),
      activeWorkItemIds: z.array(WorkItemIdSchema),
    })
    .strict(),
  z
    .object({
      ...responseBase,
      operation: z.literal("list"),
      workItems: z.array(TaskWorkItemSchema).max(10_000),
      approvals: z.array(TaskApprovalSchema).max(10_000),
      activeWorkItemIds: z.array(WorkItemIdSchema),
    })
    .strict(),
  z
    .object({
      ...responseBase,
      operation: z.literal("create"),
      workItem: TaskWorkItemSchema,
    })
    .strict(),
  z
    .object({
      ...responseBase,
      operation: z.literal("run"),
      workItemId: WorkItemIdSchema,
      accepted: z.literal(true),
    })
    .strict(),
  z
    .object({
      ...responseBase,
      operation: z.literal("cancel"),
      workItem: TaskWorkItemSchema,
    })
    .strict(),
  z
    .object({
      ...responseBase,
      operation: z.literal("decide"),
      workItem: TaskWorkItemSchema,
    })
    .strict(),
  z
    .object({
      requestId: RequestIdSchema,
      ok: z.literal(false),
      operation: z.literal("error"),
      error: z
        .object({
          code: z.enum(["authentication", "validation", "conflict", "internal"]),
          message: z.string().min(1).max(1_024),
        })
        .strict(),
    })
    .strict(),
]);

export type TaskSupervisorCommand = z.infer<typeof TaskSupervisorCommandSchema>;
export type TaskSupervisorRequest = z.infer<typeof TaskSupervisorRequestSchema>;
export type TaskSupervisorResponse = z.infer<typeof TaskSupervisorResponseSchema>;

export interface TaskSupervisorPaths {
  rootDir: string;
  tokenPath: string;
  socketPath: string;
}

export function taskSupervisorPaths(rootDir = DEFAULT_TASK_RUNTIME_ROOT): TaskSupervisorPaths {
  const resolved = path.resolve(rootDir);
  return {
    rootDir: resolved,
    tokenPath: path.join(resolved, SUPERVISOR_TOKEN_FILE),
    socketPath:
      process.platform === "win32"
        ? `\\\\.\\pipe\\swarmx-task-${createHash("sha256").update(resolved).digest("hex").slice(0, 24)}`
        : path.join(resolved, SUPERVISOR_SOCKET_FILE),
  };
}

export function ensureTaskSupervisorToken(rootDir = DEFAULT_TASK_RUNTIME_ROOT): string {
  const paths = taskSupervisorPaths(rootDir);
  fs.mkdirSync(paths.rootDir, { recursive: true, mode: 0o700 });
  fs.chmodSync(paths.rootDir, 0o700);
  try {
    const descriptor = fs.openSync(paths.tokenPath, "wx", 0o600);
    try {
      fs.writeFileSync(descriptor, `${randomBytes(32).toString("hex")}\n`, "utf8");
      fs.fsyncSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
  } catch (error) {
    if (!isFileExists(error)) throw error;
  }
  fs.chmodSync(paths.tokenPath, 0o600);
  return TokenSchema.parse(fs.readFileSync(paths.tokenPath, "utf8").trim());
}

export interface TaskSupervisorServerOptions {
  rootDir?: string;
  socketPath?: string;
  token?: string;
  service?: AppAttachedTaskControlService;
}

/** Local authenticated authority whose worker runs are independent from any one Desktop client. */
export class TaskSupervisorServer {
  readonly service: AppAttachedTaskControlService;
  readonly socketPath: string;
  private readonly token: string;
  private readonly activeRuns = new Map<string, Promise<unknown>>();
  private server?: Server;

  constructor(options: TaskSupervisorServerOptions = {}) {
    const paths = taskSupervisorPaths(options.rootDir);
    this.socketPath = options.socketPath ?? paths.socketPath;
    this.token = TokenSchema.parse(options.token ?? ensureTaskSupervisorToken(paths.rootDir));
    this.service =
      options.service ??
      new AppAttachedTaskControlService({
        store: new TaskRuntimeStore({ rootDir: paths.rootDir }),
        ownerId: `supervisor:${process.pid}`,
      });
  }

  async listen(): Promise<void> {
    if (this.server) throw new Error("Task supervisor is already listening.");
    this.service.recoverOnStartup();
    const server = createServer((socket) => this.accept(socket));
    this.server = server;
    try {
      await new Promise<void>((resolve, reject) => {
        const onError = (error: Error) => {
          server.off("listening", onListening);
          reject(error);
        };
        const onListening = () => {
          server.off("error", onError);
          resolve();
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(this.socketPath);
      });
      if (process.platform !== "win32") fs.chmodSync(this.socketPath, 0o600);
    } catch (error) {
      this.server = undefined;
      throw error;
    }
  }

  async close(): Promise<void> {
    const server = this.server;
    this.server = undefined;
    if (server) {
      await new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      );
    }
    if (process.platform !== "win32") {
      try {
        fs.unlinkSync(this.socketPath);
      } catch (error) {
        if (!isFileMissing(error)) throw error;
      }
    }
  }

  private accept(socket: Socket): void {
    socket.setEncoding("utf8");
    let buffer = "";
    let handled = false;
    socket.on("data", (chunk: string) => {
      if (handled) return;
      buffer += chunk;
      if (Buffer.byteLength(buffer, "utf8") > MAX_PROTOCOL_LINE_BYTES) {
        handled = true;
        socket.destroy(new Error("Task supervisor request exceeds the protocol limit."));
        return;
      }
      const newline = buffer.indexOf("\n");
      if (newline < 0) return;
      handled = true;
      void this.respond(socket, buffer.slice(0, newline));
    });
  }

  private async respond(socket: Socket, line: string): Promise<void> {
    let requestId: string = randomUUID();
    try {
      const decoded: unknown = JSON.parse(line);
      if (isRecord(decoded) && typeof decoded.requestId === "string") requestId = decoded.requestId;
      if (!isRecord(decoded) || !secureTokenEqual(decoded.token, this.token)) {
        this.write(
          socket,
          failureResponse(requestId, "authentication", "Task supervisor authentication failed."),
        );
        return;
      }
      const request = TaskSupervisorRequestSchema.parse(decoded);
      this.write(socket, await this.dispatch(request));
    } catch (error) {
      const code = error instanceof z.ZodError ? "validation" : "internal";
      this.write(socket, failureResponse(requestId, code, boundedErrorMessage(error)));
    }
  }

  private async dispatch(request: TaskSupervisorRequest): Promise<TaskSupervisorResponse> {
    if (request.operation === "ping") {
      return TaskSupervisorResponseSchema.parse({
        requestId: request.requestId,
        ok: true,
        operation: "ping",
        supervisorPid: process.pid,
        activeWorkItemIds: [...this.activeRuns.keys()],
      });
    }
    if (request.operation === "list") {
      const state = this.service.store.state();
      const workItems = Object.values(state.workItems).sort((left, right) =>
        right.updatedAt.localeCompare(left.updatedAt),
      );
      return TaskSupervisorResponseSchema.parse({
        requestId: request.requestId,
        ok: true,
        operation: "list",
        workItems,
        approvals: Object.values(state.approvals),
        activeWorkItemIds: [...this.activeRuns.keys()],
      });
    }
    if (request.operation === "create") {
      const workItem = this.service.createWorkItem(request.workItem as CreateTaskWorkItemInput);
      return TaskSupervisorResponseSchema.parse({
        requestId: request.requestId,
        ok: true,
        operation: "create",
        workItem,
      });
    }
    if (request.operation === "run") {
      if (this.activeRuns.has(request.workItemId)) {
        return failureResponse(
          request.requestId,
          "conflict",
          `Work item "${request.workItemId}" is already active.`,
        );
      }
      const workItem = this.service.store.state().workItems[request.workItemId];
      if (!workItem || !isTaskWorkItemRunnable(workItem, new Date().toISOString())) {
        return failureResponse(
          request.requestId,
          "conflict",
          `Work item "${request.workItemId}" is not runnable.`,
        );
      }
      const active = this.service.runWorkItem(request.workItemId, {
        launch: request.launch,
        grants: request.grants,
      });
      this.activeRuns.set(request.workItemId, active);
      void active.catch(() => undefined).finally(() => this.activeRuns.delete(request.workItemId));
      return TaskSupervisorResponseSchema.parse({
        requestId: request.requestId,
        ok: true,
        operation: "run",
        workItemId: request.workItemId,
        accepted: true,
      });
    }
    if (request.operation === "cancel") {
      const state = this.service.cancelWorkItem(request.workItemId, request.reason);
      return workItemResponse(request.requestId, "cancel", state.workItems[request.workItemId]);
    }
    const state = this.service.decideApproval({
      approvalId: request.approvalId,
      status: request.status,
      decidedBy: request.decidedBy,
      ...(request.reason ? { reason: request.reason } : {}),
      ...(request.response === undefined ? {} : { response: request.response }),
    });
    const approval = state.approvals[request.approvalId];
    return workItemResponse(
      request.requestId,
      "decide",
      state.workItems[approval?.workItemId ?? ""],
    );
  }

  private write(socket: Socket, response: TaskSupervisorResponse): void {
    socket.end(`${JSON.stringify(TaskSupervisorResponseSchema.parse(response))}\n`);
  }
}

export interface TaskSupervisorClientOptions {
  socketPath: string;
  token: string;
  timeoutMs?: number;
}

export class TaskSupervisorClient {
  private readonly socketPath: string;
  private readonly token: string;
  private readonly timeoutMs: number;

  constructor(options: TaskSupervisorClientOptions) {
    this.socketPath = z.string().min(1).max(4_096).parse(options.socketPath);
    this.token = TokenSchema.parse(options.token);
    this.timeoutMs = z
      .number()
      .int()
      .positive()
      .max(60_000)
      .parse(options.timeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS);
  }

  async request(commandInput: unknown): Promise<Exclude<TaskSupervisorResponse, { ok: false }>> {
    const command = TaskSupervisorCommandSchema.parse(commandInput);
    const request = TaskSupervisorRequestSchema.parse({
      ...command,
      requestId: randomUUID(),
      token: this.token,
    });
    const response = await sendRequest(this.socketPath, request, this.timeoutMs);
    if (!response.ok) throw new Error(response.error.message);
    return response;
  }
}

async function sendRequest(
  socketPath: string,
  request: TaskSupervisorRequest,
  timeoutMs: number,
): Promise<TaskSupervisorResponse> {
  return await new Promise((resolve, reject) => {
    const socket = createConnection(socketPath);
    let buffer = "";
    let settled = false;
    const timer = setTimeout(
      () => finish(new Error("Task supervisor request timed out.")),
      timeoutMs,
    );
    const finish = (error?: Error, response?: TaskSupervisorResponse) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      socket.destroy();
      if (error) reject(error);
      else if (response) resolve(response);
    };
    socket.setEncoding("utf8");
    socket.on("connect", () => socket.write(`${JSON.stringify(request)}\n`));
    socket.on("data", (chunk: string) => {
      buffer += chunk;
      if (Buffer.byteLength(buffer, "utf8") > MAX_PROTOCOL_LINE_BYTES) {
        finish(new Error("Task supervisor response exceeds the protocol limit."));
        return;
      }
      const newline = buffer.indexOf("\n");
      if (newline < 0) return;
      try {
        const response = TaskSupervisorResponseSchema.parse(JSON.parse(buffer.slice(0, newline)));
        if (response.requestId !== request.requestId) {
          finish(new Error("Task supervisor response correlation mismatch."));
        } else {
          finish(undefined, response);
        }
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)));
      }
    });
    socket.on("error", (error) => finish(error));
    socket.on("end", () => {
      if (!settled) finish(new Error("Task supervisor closed without a response."));
    });
  });
}

function workItemResponse(
  requestId: string,
  operation: "cancel" | "decide",
  workItem: TaskWorkItem | undefined,
): TaskSupervisorResponse {
  if (!workItem) return failureResponse(requestId, "internal", "Work item state is unavailable.");
  return TaskSupervisorResponseSchema.parse({ requestId, ok: true, operation, workItem });
}

function failureResponse(
  requestIdInput: string,
  code: "authentication" | "validation" | "conflict" | "internal",
  message: string,
): TaskSupervisorResponse {
  const requestId = RequestIdSchema.safeParse(requestIdInput);
  return TaskSupervisorResponseSchema.parse({
    requestId: requestId.success ? requestId.data : randomUUID(),
    ok: false,
    operation: "error",
    error: { code, message: message.slice(0, 1_024) || "Task supervisor request failed." },
  });
}

function secureTokenEqual(candidate: unknown, expected: string): boolean {
  if (typeof candidate !== "string" || candidate.length !== expected.length) return false;
  return timingSafeEqual(Buffer.from(candidate, "utf8"), Buffer.from(expected, "utf8"));
}

function boundedErrorMessage(error: unknown): string {
  if (error instanceof z.ZodError) return z.prettifyError(error).slice(0, 1_024);
  return (error instanceof Error ? error.message : String(error)).slice(0, 1_024);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isFileExists(error: unknown): boolean {
  return isNodeError(error) && error.code === "EEXIST";
}

function isFileMissing(error: unknown): boolean {
  return isNodeError(error) && error.code === "ENOENT";
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}
