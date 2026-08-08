import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { z } from "zod";
import {
  assertTaskWorkerCapabilityCallAllowed,
  parseTaskWorkerEventLine,
  serializeTaskWorkerMessage,
  TASK_WORKER_MAX_JSONL_LINE_BYTES,
  TASK_WORKER_PROTOCOL_VERSION,
  type TaskWorkerCanceledMessage,
  type TaskWorkerCapabilityCallMessage,
  type TaskWorkerCapabilityGrant,
  TaskWorkerCapabilityGrantSchema,
  type TaskWorkerCapabilityResultMessage,
  type TaskWorkerCompleteMessage,
  type TaskWorkerEventMessage,
  type TaskWorkerFailMessage,
  type TaskWorkerFeature,
  type TaskWorkerHelloMessage,
  type TaskWorkerNeedsHumanMessage,
  TaskWorkerStartMessageSchema,
} from "./task-worker-protocol.js";

const DEFAULT_HELLO_TIMEOUT_MS = 8_000;
const DEFAULT_HEARTBEAT_INTERVAL_MS = 5_000;
const DEFAULT_HEARTBEAT_TIMEOUT_MS = 20_000;
const DEFAULT_CANCEL_GRACE_MS = 2_000;
const DEFAULT_TERMINAL_EXIT_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024;
const STDERR_LIMIT_BYTES = 4_096;

export const TaskWorkerLaunchSpecSchema = z
  .object({
    backendId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]*$/),
    program: z.string().min(1).max(4_096),
    args: z.array(z.string().max(32_768)).max(256),
    cwd: z.string().min(1).max(4_096),
    env: z.record(z.string(), z.string().max(32_768)),
    environmentDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    /** Dedicated root for worker-declared artifact paths. Omit to disable artifact ingestion. */
    artifactRoot: z.string().min(1).max(4_096).optional(),
  })
  .strict()
  .superRefine((launch, ctx) => {
    const entries = Object.entries(launch.env);
    if (entries.length > 128) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["env"],
        message: "Task worker environments may contain at most 128 explicit variables.",
      });
    }
    for (const [key] of entries) {
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/u.test(key) || isSecretEnvironmentKey(key)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["env", key],
          message: `Task worker environment variable "${key}" is unsafe or secret-bearing.`,
        });
      }
    }
  });

export type TaskWorkerLaunchSpec = z.infer<typeof TaskWorkerLaunchSpecSchema>;

export type TaskWorkerTerminalMessage =
  | TaskWorkerCompleteMessage
  | TaskWorkerFailMessage
  | TaskWorkerCanceledMessage
  | TaskWorkerNeedsHumanMessage;

export type TaskWorkerCapabilityOutcome = TaskWorkerCapabilityResultMessage["outcome"];

export interface TaskWorkerProcessHandlers {
  onEvent?: (
    event: Exclude<TaskWorkerEventMessage, TaskWorkerHelloMessage>,
  ) => void | Promise<void>;
  onCapabilityCall?: (
    call: TaskWorkerCapabilityCallMessage,
  ) => TaskWorkerCapabilityOutcome | Promise<TaskWorkerCapabilityOutcome>;
}

export interface RunTaskWorkerProcessOptions extends TaskWorkerProcessHandlers {
  launch: TaskWorkerLaunchSpec;
  start: unknown;
  grants?: readonly unknown[];
  enabledFeatures?: readonly TaskWorkerFeature[];
  helloTimeoutMs?: number;
  heartbeatIntervalMs?: number;
  heartbeatTimeoutMs?: number;
  cancelGraceMs?: number;
  terminalExitTimeoutMs?: number;
  maxArtifactBytes?: number;
  signal?: AbortSignal;
}

export interface TaskWorkerProcessResult {
  hello: TaskWorkerHelloMessage;
  terminal: TaskWorkerTerminalMessage;
  exitCode: number | null;
  signalCode: NodeJS.Signals | null;
  stderrSummary?: string;
}

export class TaskWorkerProcessError extends Error {
  readonly code:
    | "spawn_failed"
    | "hello_timeout"
    | "heartbeat_timeout"
    | "budget_exceeded"
    | "protocol_error"
    | "unexpected_exit"
    | "canceled";

  constructor(code: TaskWorkerProcessError["code"], message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "TaskWorkerProcessError";
    this.code = code;
  }
}

/**
 * Runs one leased task in a replaceable JSONL worker process. The child receives only the
 * explicit launch environment; ambient Provider credentials are never inherited.
 */
export async function runTaskWorkerProcess(
  options: RunTaskWorkerProcessOptions,
): Promise<TaskWorkerProcessResult> {
  const launch = TaskWorkerLaunchSpecSchema.parse(options.launch);
  const start = TaskWorkerStartMessageSchema.parse(options.start);
  const availableGrants: TaskWorkerCapabilityGrant[] = (options.grants ?? []).map((grant) =>
    TaskWorkerCapabilityGrantSchema.parse(grant),
  );
  const grantIds = new Set(availableGrants.map((grant) => grant.grantId));
  for (const grantId of start.capabilityGrantIds) {
    if (!grantIds.has(grantId)) {
      throw new TaskWorkerProcessError(
        "protocol_error",
        `Start references unknown capability grant "${grantId}".`,
      );
    }
  }
  const selectedGrantIds = new Set(start.capabilityGrantIds);
  const grants = availableGrants.filter((grant) => selectedGrantIds.has(grant.grantId));
  if (start.environmentDigest !== launch.environmentDigest) {
    throw new TaskWorkerProcessError(
      "protocol_error",
      "Worker launch and start environment digests do not match.",
    );
  }

  return await new Promise<TaskWorkerProcessResult>((resolve, reject) => {
    const child = spawn(launch.program, [...launch.args], {
      cwd: launch.cwd,
      env: { ...launch.env },
      detached: process.platform !== "win32",
      stdio: ["pipe", "pipe", "pipe"],
    });
    let hello: TaskWorkerHelloMessage | undefined;
    let terminal: TaskWorkerTerminalMessage | undefined;
    let exitCode: number | null = null;
    let signalCode: NodeJS.Signals | null = null;
    let stdoutBuffer = Buffer.alloc(0);
    let stderrBuffer = Buffer.alloc(0);
    let expectedSequence = 0;
    let started = false;
    let settled = false;
    let closed = false;
    let pendingFailure: Error | undefined;
    let processing = Promise.resolve();
    let cancelTimer: ReturnType<typeof setTimeout> | undefined;
    let killTimer: ReturnType<typeof setTimeout> | undefined;
    let heartbeatTimer: ReturnType<typeof setTimeout> | undefined;
    let wallTimeTimer: ReturnType<typeof setTimeout> | undefined;
    let terminalExitTimer: ReturnType<typeof setTimeout> | undefined;
    let enabledFeatures = new Set<TaskWorkerFeature>();
    const pendingCapabilityCalls = new Set<Promise<void>>();

    const helloTimer = setTimeout(() => {
      fail(
        new TaskWorkerProcessError(
          "hello_timeout",
          `Task worker did not send hello within ${options.helloTimeoutMs ?? DEFAULT_HELLO_TIMEOUT_MS}ms.`,
        ),
      );
    }, options.helloTimeoutMs ?? DEFAULT_HELLO_TIMEOUT_MS);

    const cleanup = (): void => {
      clearTimeout(helloTimer);
      if (cancelTimer) clearTimeout(cancelTimer);
      if (killTimer) clearTimeout(killTimer);
      if (heartbeatTimer) clearTimeout(heartbeatTimer);
      if (wallTimeTimer) clearTimeout(wallTimeTimer);
      if (terminalExitTimer) clearTimeout(terminalExitTimer);
      options.signal?.removeEventListener("abort", abortListener);
    };

    const rejectAfterClose = (error: Error): void => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    };

    const fail = (error: Error): void => {
      if (settled || pendingFailure) return;
      pendingFailure = error;
      clearTimeout(helloTimer);
      if (heartbeatTimer) clearTimeout(heartbeatTimer);
      if (wallTimeTimer) clearTimeout(wallTimeTimer);
      if (terminalExitTimer) clearTimeout(terminalExitTimer);
      terminateWorkerProcess(child, "SIGTERM");
      killTimer = setTimeout(() => terminateWorkerProcess(child, "SIGKILL"), 250);
      killTimer.unref();
      if (closed) rejectAfterClose(error);
    };

    const armHeartbeatTimer = (): void => {
      if (heartbeatTimer) clearTimeout(heartbeatTimer);
      const timeoutMs = options.heartbeatTimeoutMs ?? DEFAULT_HEARTBEAT_TIMEOUT_MS;
      heartbeatTimer = setTimeout(() => {
        fail(
          new TaskWorkerProcessError(
            "heartbeat_timeout",
            `Task worker missed its ${timeoutMs}ms heartbeat deadline.`,
          ),
        );
      }, timeoutMs);
    };

    if (start.budget?.wallTimeMs) {
      wallTimeTimer = setTimeout(() => {
        fail(
          new TaskWorkerProcessError(
            "budget_exceeded",
            `Task worker exceeded its ${start.budget?.wallTimeMs}ms wall-time budget.`,
          ),
        );
      }, start.budget.wallTimeMs);
    }

    const write = (message: unknown): void => {
      if (!child.stdin.writable) {
        throw new TaskWorkerProcessError("unexpected_exit", "Task worker stdin is unavailable.");
      }
      child.stdin.write(serializeTaskWorkerMessage(message), "utf8");
    };

    const requestCancel = (reason: string): void => {
      if (settled) return;
      if (!started) {
        fail(new TaskWorkerProcessError("canceled", reason));
        return;
      }
      try {
        write({
          protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
          messageId: messageId("cancel"),
          direction: "host_to_worker",
          type: "cancel",
          workItemId: start.workItemId,
          runId: start.runId,
          leaseId: start.leaseId,
          fencingToken: start.fencingToken,
          requestedAt: new Date().toISOString(),
          mode: "cancel",
          reason,
          graceMs: options.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS,
        });
      } catch (error) {
        fail(processError(error, "protocol_error", "Failed to send worker cancellation."));
        return;
      }
      cancelTimer = setTimeout(() => {
        terminateWorkerProcess(child, "SIGTERM");
        killTimer = setTimeout(() => terminateWorkerProcess(child, "SIGKILL"), 250);
        killTimer.unref();
      }, options.cancelGraceMs ?? DEFAULT_CANCEL_GRACE_MS);
    };

    const abortListener = (): void => {
      const reason = options.signal?.reason;
      requestCancel(reason instanceof Error ? reason.message : "Task cancellation requested.");
    };
    options.signal?.addEventListener("abort", abortListener, { once: true });
    if (options.signal?.aborted) abortListener();

    const validateRunEnvelope = (
      event: Exclude<TaskWorkerEventMessage, TaskWorkerHelloMessage>,
    ) => {
      if (
        event.workItemId !== start.workItemId ||
        event.runId !== start.runId ||
        event.leaseId !== start.leaseId ||
        event.fencingToken !== start.fencingToken
      ) {
        throw new TaskWorkerProcessError(
          "protocol_error",
          "Worker event does not match the active run lease and fencing token.",
        );
      }
      if (event.sequence !== expectedSequence) {
        throw new TaskWorkerProcessError(
          "protocol_error",
          `Worker event sequence ${event.sequence} does not match expected ${expectedSequence}.`,
        );
      }
      expectedSequence += 1;
    };

    const handleCapabilityCall = async (call: TaskWorkerCapabilityCallMessage): Promise<void> => {
      let outcome: TaskWorkerCapabilityOutcome;
      try {
        assertTaskWorkerCapabilityCallAllowed(call, grants);
        outcome = options.onCapabilityCall
          ? await options.onCapabilityCall(call)
          : {
              status: "unknown",
              error: {
                code: "gateway_unavailable",
                message: "No capability gateway is attached to this task runtime.",
                retryable: false,
              },
            };
      } catch (error) {
        outcome = {
          status: "failed",
          error: {
            code: "capability_denied",
            message: boundedErrorMessage(error),
            retryable: false,
          },
        };
      }
      write({
        protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
        messageId: messageId("capability-result"),
        direction: "host_to_worker",
        type: "capability_result",
        workItemId: start.workItemId,
        runId: start.runId,
        leaseId: start.leaseId,
        fencingToken: start.fencingToken,
        callId: call.callId,
        grantId: call.grantId,
        capabilityId: call.capabilityId,
        outcome,
      });
    };

    const handleLine = async (line: string): Promise<void> => {
      const event = parseTaskWorkerEventLine(line);
      if (!hello) {
        if (event.type !== "hello") {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "The first task worker message must be hello.",
          );
        }
        if (event.worker.environmentDigest !== launch.environmentDigest) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "Worker hello environment digest does not match the verified launch environment.",
          );
        }
        if (event.worker.backendId !== launch.backendId) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            `Worker advertised backend "${event.worker.backendId}", expected "${launch.backendId}".`,
          );
        }
        if (!event.supportedProtocolVersions.includes(TASK_WORKER_PROTOCOL_VERSION)) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "Worker does not support the Core task worker protocol version.",
          );
        }
        if (!event.operations.includes(start.operation.name)) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            `Worker does not advertise operation "${start.operation.name}".`,
          );
        }
        if (!event.features.includes("heartbeat")) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "Leased task workers must advertise heartbeat support.",
          );
        }
        hello = event;
        clearTimeout(helloTimer);
        enabledFeatures = new Set(
          (options.enabledFeatures ?? event.features).filter((feature) =>
            event.features.includes(feature),
          ),
        );
        if (!enabledFeatures.has("heartbeat")) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "The host must negotiate heartbeat for a leased task worker.",
          );
        }
        write({
          protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
          messageId: messageId("capabilities"),
          direction: "host_to_worker",
          type: "capabilities",
          helloMessageId: event.messageId,
          selectedProtocolVersion: TASK_WORKER_PROTOCOL_VERSION,
          enabledFeatures: [...enabledFeatures],
          grants,
          limits: {
            maxJsonlLineBytes: TASK_WORKER_MAX_JSONL_LINE_BYTES,
            heartbeatIntervalMs: options.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS,
            heartbeatTimeoutMs: options.heartbeatTimeoutMs ?? DEFAULT_HEARTBEAT_TIMEOUT_MS,
            maxArtifactBytes: options.maxArtifactBytes ?? DEFAULT_MAX_ARTIFACT_BYTES,
          },
        });
        write(start);
        started = true;
        armHeartbeatTimer();
        if (options.signal?.aborted) abortListener();
        return;
      }

      if (event.type === "hello") {
        throw new TaskWorkerProcessError("protocol_error", "Worker sent hello more than once.");
      }
      if (terminal) {
        throw new TaskWorkerProcessError(
          "protocol_error",
          `Worker emitted ${event.type} after its terminal task message.`,
        );
      }
      validateRunEnvelope(event);
      const requiredFeature = taskWorkerEventFeature(event);
      if (requiredFeature && !enabledFeatures.has(requiredFeature)) {
        throw new TaskWorkerProcessError(
          "protocol_error",
          `Worker emitted ${event.type} without the negotiated "${requiredFeature}" feature.`,
        );
      }
      if (
        event.type === "artifact" &&
        event.artifact.sizeBytes > (options.maxArtifactBytes ?? DEFAULT_MAX_ARTIFACT_BYTES)
      ) {
        throw new TaskWorkerProcessError(
          "protocol_error",
          "Worker artifact exceeds the negotiated byte limit.",
        );
      }
      if (event.type === "heartbeat") armHeartbeatTimer();
      await options.onEvent?.(event);
      if (event.type === "capability_call") {
        const pending = handleCapabilityCall(event);
        pendingCapabilityCalls.add(pending);
        pending
          .catch((error) => {
            fail(processError(error, "protocol_error", "Capability gateway response failed."));
          })
          .finally(() => pendingCapabilityCalls.delete(pending));
        return;
      }
      if (
        event.type === "complete" ||
        event.type === "fail" ||
        event.type === "canceled" ||
        event.type === "needs_human"
      ) {
        await Promise.all(pendingCapabilityCalls);
        if (terminal) {
          throw new TaskWorkerProcessError(
            "protocol_error",
            "Worker emitted more than one terminal task message.",
          );
        }
        terminal = event;
        if (heartbeatTimer) clearTimeout(heartbeatTimer);
        if (wallTimeTimer) clearTimeout(wallTimeTimer);
        child.stdin.end();
        const exitTimeoutMs = options.terminalExitTimeoutMs ?? DEFAULT_TERMINAL_EXIT_TIMEOUT_MS;
        terminalExitTimer = setTimeout(() => {
          fail(
            new TaskWorkerProcessError(
              "unexpected_exit",
              `Task worker did not exit within ${exitTimeoutMs}ms of its terminal message.`,
            ),
          );
        }, exitTimeoutMs);
      }
    };

    child.once("error", (error) => {
      fail(
        new TaskWorkerProcessError("spawn_failed", "Failed to start task worker.", {
          cause: error,
        }),
      );
    });
    child.stdin.on("error", (error) => {
      if (!closed && !terminal) {
        fail(
          new TaskWorkerProcessError("unexpected_exit", "Task worker stdin failed.", {
            cause: error,
          }),
        );
      }
    });
    child.stdout.on("data", (chunk: Buffer) => {
      if (settled || pendingFailure) return;
      stdoutBuffer = Buffer.concat([stdoutBuffer, chunk]);
      if (
        stdoutBuffer.byteLength > TASK_WORKER_MAX_JSONL_LINE_BYTES &&
        !stdoutBuffer.includes(0x0a)
      ) {
        fail(
          new TaskWorkerProcessError(
            "protocol_error",
            `Worker JSONL line exceeds ${TASK_WORKER_MAX_JSONL_LINE_BYTES} bytes.`,
          ),
        );
        return;
      }
      while (true) {
        const newline = stdoutBuffer.indexOf(0x0a);
        if (newline < 0) break;
        const lineBuffer = stdoutBuffer.subarray(0, newline);
        stdoutBuffer = stdoutBuffer.subarray(newline + 1);
        if (lineBuffer.byteLength > TASK_WORKER_MAX_JSONL_LINE_BYTES) {
          fail(
            new TaskWorkerProcessError(
              "protocol_error",
              `Worker JSONL line exceeds ${TASK_WORKER_MAX_JSONL_LINE_BYTES} bytes.`,
            ),
          );
          return;
        }
        const line = lineBuffer.toString("utf8").replace(/\r$/u, "");
        processing = processing.then(() => handleLine(line));
        processing.catch((error) => {
          fail(processError(error, "protocol_error", "Invalid task worker message."));
        });
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderrBuffer = Buffer.concat([stderrBuffer, chunk]).subarray(-STDERR_LIMIT_BYTES);
    });
    child.once("close", (code, signal) => {
      closed = true;
      exitCode = code;
      signalCode = signal;
      processing
        .then(() => {
          if (settled) return;
          if (pendingFailure) {
            rejectAfterClose(pendingFailure);
            return;
          }
          if (stdoutBuffer.length > 0) {
            fail(
              new TaskWorkerProcessError(
                "protocol_error",
                "Task worker exited with an unterminated JSONL record.",
              ),
            );
            return;
          }
          if (!hello || !terminal) {
            fail(
              new TaskWorkerProcessError(
                options.signal?.aborted ? "canceled" : "unexpected_exit",
                options.signal?.aborted
                  ? "Task worker exited during cancellation."
                  : `Task worker exited before a terminal message (code ${code ?? "none"}). ${sanitizeWorkerStderr(stderrBuffer.toString("utf8"))}`,
              ),
            );
            return;
          }
          if (code !== 0) {
            fail(
              new TaskWorkerProcessError(
                "unexpected_exit",
                `Task worker exited with code ${code ?? "none"} after its terminal message.`,
              ),
            );
            return;
          }
          settled = true;
          cleanup();
          resolve({
            hello,
            terminal,
            exitCode,
            signalCode,
            stderrSummary: sanitizeWorkerStderr(stderrBuffer.toString("utf8")),
          });
        })
        .catch((error) => {
          rejectAfterClose(
            pendingFailure ?? processError(error, "protocol_error", "Invalid task worker message."),
          );
        });
    });
  });
}

function terminateWorkerProcess(
  child: ChildProcessWithoutNullStreams,
  signal: NodeJS.Signals,
): void {
  if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform !== "win32") {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch {
      // Fall through to the direct child when the process group has already exited.
    }
  }
  try {
    child.kill(signal);
  } catch {
    // Process termination races are expected during cancellation and shutdown.
  }
}

function taskWorkerEventFeature(
  event: Exclude<TaskWorkerEventMessage, TaskWorkerHelloMessage>,
): TaskWorkerFeature | undefined {
  switch (event.type) {
    case "heartbeat":
    case "progress":
    case "checkpoint":
    case "artifact":
    case "needs_human":
      return event.type;
    case "capability_call":
      return "capability_gateway";
    case "canceled":
      return "cancel";
    case "complete":
    case "fail":
      return undefined;
  }
}

function messageId(kind: string): string {
  return `${kind}:${randomUUID()}`;
}

function processError(
  error: unknown,
  code: TaskWorkerProcessError["code"],
  fallback: string,
): TaskWorkerProcessError {
  return error instanceof TaskWorkerProcessError
    ? error
    : new TaskWorkerProcessError(code, fallback, {
        cause: error instanceof Error ? error : undefined,
      });
}

function boundedErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.slice(0, 4_096) || "Capability request failed.";
}

function isSecretEnvironmentKey(key: string): boolean {
  const normalized = key.toLowerCase().replace(/[^a-z0-9]/gu, "");
  if (normalized.endsWith("ref") || normalized.endsWith("refid")) return false;
  return (
    normalized.includes("apikey") ||
    normalized.includes("authorization") ||
    normalized.includes("credential") ||
    normalized.includes("password") ||
    normalized.includes("passwd") ||
    normalized.includes("privatekey") ||
    normalized.includes("secret") ||
    normalized === "cookie" ||
    normalized.endsWith("token")
  );
}

function sanitizeWorkerStderr(value: string): string | undefined {
  const sanitized = value
    .split("")
    .filter((character) => {
      const code = character.charCodeAt(0);
      return code === 0x09 || code === 0x0a || code === 0x0d || (code >= 0x20 && code !== 0x7f);
    })
    .join("")
    .replace(/\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{8,}\b/giu, "[REDACTED]")
    .replace(
      /\b(authorization|api[_-]?key|access[_-]?token|password)\s*[:=]\s*\S+/giu,
      "$1=[REDACTED]",
    )
    .trim();
  return sanitized || undefined;
}
