import { createHash, randomUUID } from "node:crypto";
import * as fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  createTaskRuntimeEvent,
  expiredTaskLeases,
  isTaskWorkItemRunnable,
  type TaskApproval,
  TaskApprovalSchema,
  type TaskApprovalStatus,
  type TaskArtifactReference,
  type TaskBudget,
  type TaskCheckpoint,
  type TaskLease,
  TaskRunSchema,
  type TaskRuntimeState,
  type TaskSessionLink,
  type TaskWorkItem,
  TaskWorkItemSchema,
} from "./task-runtime.js";
import { type TaskRuntimeRecoveryResult, TaskRuntimeStore } from "./task-runtime-store.js";
import {
  runTaskWorkerProcess,
  type TaskWorkerCapabilityOutcome,
  type TaskWorkerLaunchSpec,
  TaskWorkerProcessError,
  type TaskWorkerProcessResult,
} from "./task-worker-process.js";
import {
  TASK_WORKER_PROTOCOL_VERSION,
  type TaskWorkerArtifactMessage,
  type TaskWorkerCapabilityCallMessage,
  type TaskWorkerCapabilityGrant,
  TaskWorkerCapabilityOutcomeSchema,
  type TaskWorkerCheckpoint,
  TaskWorkerCheckpointSchema,
  type TaskWorkerEventMessage,
  TaskWorkerStartMessageSchema,
} from "./task-worker-protocol.js";

const DEFAULT_LEASE_DURATION_MS = 30_000;

export interface CreateTaskWorkItemInput {
  id?: string;
  backend: string;
  operation: string;
  input: unknown;
  priority?: number;
  owner?: string;
  budget?: TaskBudget;
  maxAttempts?: number;
  creatorSessionId?: string;
}

export interface RunTaskWorkItemOptions {
  launch: TaskWorkerLaunchSpec;
  grants?: readonly TaskWorkerCapabilityGrant[];
  signal?: AbortSignal;
}

export interface DecideTaskApprovalInput {
  approvalId: string;
  status: Exclude<TaskApprovalStatus, "requested">;
  decidedBy: string;
  reason?: string;
  response?: unknown;
}

export interface TaskCapabilityGatewayContext {
  workItem: TaskWorkItem;
  call: TaskWorkerCapabilityCallMessage;
}

/** Main-owned adapters may resolve secrets internally, but must return only protocol-safe data. */
export interface TaskCapabilityGateway {
  invoke(context: TaskCapabilityGatewayContext): Promise<TaskWorkerCapabilityOutcome>;
}

export interface AppAttachedTaskControlServiceOptions {
  store?: TaskRuntimeStore;
  ownerId?: string;
  leaseDurationMs?: number;
  now?: () => Date;
  capabilityGateway?: TaskCapabilityGateway;
}

export interface TaskRuntimeStartupRecovery {
  log: TaskRuntimeRecoveryResult;
  recoveredLeaseIds: string[];
  state: TaskRuntimeState;
}

export interface TaskWorkItemRunResult {
  process: TaskWorkerProcessResult;
  state: TaskRuntimeState;
}

export class TaskCheckpointEnvironmentMismatchError extends Error {
  constructor(checkpointId: string, checkpointDigest: string, launchDigest: string) {
    super(
      `Checkpoint "${checkpointId}" belongs to environment ${checkpointDigest}, not ${launchDigest}; automatic resume was refused.`,
    );
    this.name = "TaskCheckpointEnvironmentMismatchError";
  }
}

/**
 * Restartable control-plane slice attached to its host process. Its event log survives the host,
 * but a separate swarmxd/service manager is still required to execute while Desktop is closed.
 */
export class AppAttachedTaskControlService {
  readonly lifecycle = "app_attached" as const;
  readonly store: TaskRuntimeStore;
  readonly ownerId: string;
  private readonly leaseDurationMs: number;
  private readonly now: () => Date;
  private readonly capabilityGateway?: TaskCapabilityGateway;
  private readonly activeRuns = new Map<string, AbortController>();

  constructor(options: AppAttachedTaskControlServiceOptions = {}) {
    this.store = options.store ?? new TaskRuntimeStore();
    this.ownerId = options.ownerId ?? `controller:${process.pid}:${randomUUID()}`;
    this.leaseDurationMs = options.leaseDurationMs ?? DEFAULT_LEASE_DURATION_MS;
    if (!Number.isSafeInteger(this.leaseDurationMs) || this.leaseDurationMs <= 0) {
      throw new Error("Task leaseDurationMs must be a positive integer.");
    }
    this.now = options.now ?? (() => new Date());
    this.capabilityGateway = options.capabilityGateway;
  }

  createWorkItem(input: CreateTaskWorkItemInput): TaskWorkItem {
    const timestamp = this.timestamp();
    const inputBlob = this.store.putJson(input.input);
    const workItem = TaskWorkItemSchema.parse({
      id: input.id ?? id("awi_"),
      status: "queued",
      executor: { backend: input.backend, operation: input.operation },
      priority: input.priority ?? 0,
      owner: input.owner,
      createdAt: timestamp,
      updatedAt: timestamp,
      inputRef: inputBlob.ref,
      budget: input.budget,
      retry: { attemptsStarted: 0, maxAttempts: input.maxAttempts ?? 1 },
    });
    const events = [
      createTaskRuntimeEvent({
        eventType: "work_item_created",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `work-item:create:${workItem.id}`,
        workItemId: workItem.id,
        payload: { workItem },
      }),
    ];
    if (input.creatorSessionId) {
      const link = this.sessionLink(workItem.id, input.creatorSessionId, "creator", timestamp);
      events.push(
        createTaskRuntimeEvent({
          eventType: "session_linked",
          timestamp,
          source: this.ownerId,
          idempotencyKey: `session-link:${link.linkId}`,
          workItemId: workItem.id,
          payload: { link },
        }),
      );
    }
    return this.store.append(events).state.workItems[workItem.id] as TaskWorkItem;
  }

  linkSession(workItemId: string, sessionId: string, role: "creator" | "observer" = "observer") {
    const timestamp = this.timestamp();
    const link = this.sessionLink(workItemId, sessionId, role, timestamp);
    return this.store.append(
      createTaskRuntimeEvent({
        eventType: "session_linked",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `session-link:${link.linkId}`,
        workItemId,
        payload: { link },
      }),
    ).state;
  }

  unlinkSession(workItemId: string, linkId: string): TaskRuntimeState {
    const timestamp = this.timestamp();
    return this.store.append(
      createTaskRuntimeEvent({
        eventType: "session_unlinked",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `session-unlink:${linkId}:${timestamp}`,
        workItemId,
        payload: { linkId, unlinkedAt: timestamp },
      }),
    ).state;
  }

  async runWorkItem(
    workItemId: string,
    options: RunTaskWorkItemOptions,
  ): Promise<TaskWorkItemRunResult> {
    if (this.activeRuns.has(workItemId)) throw new Error(`Work item "${workItemId}" is active.`);
    let state = this.store.state();
    const workItem = state.workItems[workItemId];
    const now = this.timestamp();
    if (!workItem || !isTaskWorkItemRunnable(workItem, now)) {
      throw new Error(`Work item "${workItemId}" is not runnable.`);
    }
    const remainingWallTimeMs = remainingBudget(
      workItem.budget?.wallTimeMs,
      workItem.budgetUsage.wallTimeMs,
    );
    if (remainingWallTimeMs === 0) {
      throw new Error(`Work item "${workItemId}" exhausted its wall-time budget.`);
    }
    const resumeFrom = this.resumeCheckpoint(state, workItem, options.launch.environmentDigest);
    const attempt = workItem.retry.attemptsStarted + 1;
    const runId = id("run_");
    const leaseId = id("lease_");
    const fencingToken = workItem.lastFencingToken + 1;
    const expiresAt = addMilliseconds(now, this.leaseDurationMs);
    const run = TaskRunSchema.parse({
      runId,
      workItemId,
      executor: workItem.executor,
      status: "created",
      attempt,
      createdAt: now,
      environmentDigest: options.launch.environmentDigest,
      resumeFromCheckpointId: resumeFrom?.checkpointId,
    });
    const lease: TaskLease = {
      leaseId,
      workItemId,
      runId,
      workerId: this.ownerId,
      fencingToken,
      acquiredAt: now,
      heartbeatAt: now,
      expiresAt,
      budgetSnapshot: workItem.budget,
    };
    const claim = { leaseId, fencingToken };
    state = this.store.append([
      createTaskRuntimeEvent({
        eventType: "run_created",
        timestamp: now,
        source: this.ownerId,
        idempotencyKey: `run:create:${runId}`,
        workItemId,
        runId,
        payload: { run },
      }),
      createTaskRuntimeEvent({
        eventType: "lease_acquired",
        timestamp: now,
        source: this.ownerId,
        idempotencyKey: `lease:acquire:${leaseId}`,
        workItemId,
        runId,
        payload: { lease },
      }),
      createTaskRuntimeEvent({
        eventType: "run_started",
        timestamp: now,
        source: this.ownerId,
        idempotencyKey: `run:start:${runId}`,
        workItemId,
        runId,
        payload: { claim, startedAt: now },
      }),
    ]).state;

    const input = workItem.inputRef ? this.store.readJson(workItem.inputRef) : null;
    const start = TaskWorkerStartMessageSchema.parse({
      protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
      messageId: `start:${runId}`,
      direction: "host_to_worker",
      type: "start",
      workItemId,
      runId,
      leaseId,
      fencingToken,
      attempt,
      operation: { name: workItem.executor.operation, input },
      environmentDigest: options.launch.environmentDigest,
      resumeFrom,
      humanDecisions: workItem.approvalIds.flatMap((approvalId) => {
        const approval = state.approvals[approvalId];
        if (
          !approval ||
          approval.status === "requested" ||
          !approval.decidedAt ||
          !approval.decidedBy
        ) {
          return [];
        }
        return [
          {
            approvalId,
            status: approval.status,
            decidedAt: approval.decidedAt,
            decidedBy: approval.decidedBy,
            reason: approval.reason,
            response: approval.decisionRef ? this.store.readJson(approval.decisionRef) : undefined,
          },
        ];
      }),
      capabilityGrantIds: (options.grants ?? []).map((grant) => grant.grantId),
      budget: workItem.budget
        ? {
            wallTimeMs: remainingWallTimeMs,
            outputBytes: remainingBudget(
              workItem.budget.maxArtifactBytes,
              workItem.budgetUsage.artifactBytes,
            ),
            capabilityCalls: remainingCapabilityCallsByKind(workItem),
          }
        : undefined,
    });
    const controller = new AbortController();
    let detachSignal = () => {};
    this.activeRuns.set(workItemId, controller);

    try {
      detachSignal = relayAbort(options.signal, () => {
        const reason = abortReason(options.signal?.reason);
        this.persistCancellationRequest(workItemId, runId, reason);
        controller.abort(options.signal?.reason);
      });
      const processResult = await runTaskWorkerProcess({
        launch: options.launch,
        start,
        grants: options.grants,
        signal: controller.signal,
        heartbeatIntervalMs: boundedHeartbeatInterval(this.leaseDurationMs),
        heartbeatTimeoutMs: boundedHeartbeatTimeout(this.leaseDurationMs),
        maxArtifactBytes: remainingBudget(
          workItem.budget?.maxArtifactBytes,
          workItem.budgetUsage.artifactBytes,
        ),
        onEvent: async (event) => {
          await this.recordWorkerEvent(event, lease, options.launch.artifactRoot);
        },
        onCapabilityCall: this.capabilityGateway
          ? async (call) => await this.invokeCapability(workItemId, call, lease)
          : undefined,
      });
      state = this.finishRun(processResult, lease);
      return { process: processResult, state };
    } catch (error) {
      state = this.failInterruptedRun(lease, error);
      throw Object.assign(error instanceof Error ? error : new Error(String(error)), {
        taskRuntimeState: state,
      });
    } finally {
      detachSignal();
      this.activeRuns.delete(workItemId);
    }
  }

  cancelWorkItem(workItemId: string, reason = "Canceled by the user."): TaskRuntimeState {
    const state = this.store.state();
    const workItem = state.workItems[workItemId];
    const run = workItem?.activeRunId ? state.runs[workItem.activeRunId] : undefined;
    if (!workItem || !run || !workItem.lease) {
      throw new Error(`Work item "${workItemId}" has no active leased run.`);
    }
    const next = this.persistCancellationRequest(workItemId, run.runId, reason);
    this.activeRuns.get(workItemId)?.abort(new Error(reason));
    return next;
  }

  private persistCancellationRequest(
    workItemId: string,
    runId: string,
    reason: string,
  ): TaskRuntimeState {
    const state = this.store.state();
    const workItem = state.workItems[workItemId];
    const run = state.runs[runId];
    if (
      !workItem ||
      !run ||
      !workItem.lease ||
      workItem.activeRunId !== runId ||
      run.workItemId !== workItemId
    ) {
      throw new Error(`Work item "${workItemId}" has no active leased run.`);
    }
    if (run.status === "cancel_requested") return state;
    if (run.status !== "leased" && run.status !== "running") {
      throw new Error(`Work item "${workItemId}" has no active leased run.`);
    }
    const timestamp = this.timestamp();
    return this.store.append(
      createTaskRuntimeEvent({
        eventType: "cancel_requested",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `cancel:request:${run.runId}`,
        workItemId,
        runId: run.runId,
        payload: { requestedAt: timestamp, reason },
      }),
    ).state;
  }

  decideApproval(input: DecideTaskApprovalInput): TaskRuntimeState {
    let state = this.store.state();
    const approval = state.approvals[input.approvalId];
    if (approval?.status !== "requested") {
      throw new Error(`Approval "${input.approvalId}" is not awaiting a decision.`);
    }
    const timestamp = this.timestamp();
    const decisionRef =
      input.response === undefined ? undefined : this.store.putJson(input.response).ref;
    const decided = TaskApprovalSchema.parse({
      ...approval,
      status: input.status,
      decidedAt: timestamp,
      decidedBy: input.decidedBy,
      reason: input.reason ?? approval.reason,
      decisionRef,
    });
    state = this.store.append(
      createTaskRuntimeEvent({
        eventType: "approval_recorded",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `approval:decision:${approval.approvalId}:${input.status}`,
        workItemId: approval.workItemId,
        payload: { approval: decided },
      }),
    ).state;
    const workItem = state.workItems[approval.workItemId];
    if (workItem?.status === "needs_human") {
      if (
        ["approved", "waived"].includes(input.status) &&
        workItem.retry.attemptsStarted < workItem.retry.maxAttempts
      ) {
        state = this.scheduleRetry(
          workItem,
          timestamp,
          input.reason ?? "Resume after human decision.",
          approval.approvalId,
        );
      } else {
        state = this.blockAfterApproval(
          workItem,
          decided,
          input.status === "rejected"
            ? (input.reason ?? "Human approval was rejected.")
            : "Human decision was recorded after the run-attempt budget was exhausted.",
        );
      }
    }
    return state;
  }

  recoverOnStartup(): TaskRuntimeStartupRecovery {
    const log = this.store.recoverTornTail();
    let state = log.state;
    const recoveredLeaseIds: string[] = [];
    const now = this.timestamp();
    for (const lease of expiredTaskLeases(state, now)) {
      state = this.store.append(
        createTaskRuntimeEvent({
          eventType: "lease_expired",
          timestamp: now,
          source: this.ownerId,
          idempotencyKey: `lease:expired:${lease.leaseId}`,
          workItemId: lease.workItemId,
          runId: lease.runId,
          payload: {
            claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
            expiredAt: now,
            reason: "The app-attached controller recovered an expired worker lease.",
          },
        }),
      ).state;
      recoveredLeaseIds.push(lease.leaseId);
      const recovered = state.workItems[lease.workItemId];
      if (
        recovered?.status === "failed" &&
        !recovered.cancellation &&
        recovered.retry.attemptsStarted < recovered.retry.maxAttempts
      ) {
        state = this.scheduleRetry(recovered, now, "Retry after expired worker lease.");
      }
    }
    for (const workItem of Object.values(state.workItems)) {
      const run = workItem.activeRunId ? state.runs[workItem.activeRunId] : undefined;
      if (
        workItem.status === "failed" &&
        !workItem.cancellation &&
        run?.failure?.retryable &&
        workItem.retry.attemptsStarted < workItem.retry.maxAttempts
      ) {
        state = this.scheduleRetry(
          workItem,
          now,
          workItem.retry.lastFailure ?? "Recover retryable worker failure.",
        );
        continue;
      }
      if (
        workItem.status === "needs_human" &&
        (run?.status === "needs_human" || run?.failure?.code === "LEASE_EXPIRED_AWAITING_HUMAN")
      ) {
        const approvalId = [...workItem.approvalIds].reverse().find((candidate) => {
          const approval = state.approvals[candidate];
          return approval?.runId === run.runId && approval.status !== "requested";
        });
        if (approvalId) {
          const approval = state.approvals[approvalId];
          if (
            approval &&
            ["approved", "waived"].includes(approval.status) &&
            workItem.retry.attemptsStarted < workItem.retry.maxAttempts
          ) {
            state = this.scheduleRetry(
              workItem,
              now,
              "Recover resume after persisted human decision.",
              approvalId,
            );
          } else if (approval) {
            state = this.blockAfterApproval(
              workItem,
              approval,
              approval.status === "rejected"
                ? (approval.reason ?? "Human approval was rejected.")
                : "Human decision was recovered after the run-attempt budget was exhausted.",
            );
          }
        }
      }
    }
    return { log, recoveredLeaseIds, state };
  }

  private async recordWorkerEvent(
    event: Exclude<TaskWorkerEventMessage, { type: "hello" }>,
    lease: TaskLease,
    artifactRoot: string | undefined,
  ): Promise<void> {
    const claim = { leaseId: lease.leaseId, fencingToken: lease.fencingToken };
    const observedAt = this.timestamp();
    if (event.type === "heartbeat") {
      this.store.append(
        createTaskRuntimeEvent({
          eventType: "lease_heartbeat",
          timestamp: observedAt,
          source: `worker:${lease.workerId}`,
          idempotencyKey: `heartbeat:${event.runId}:${event.sequence}`,
          workItemId: event.workItemId,
          runId: event.runId,
          payload: {
            claim,
            heartbeatAt: observedAt,
            expiresAt: addMilliseconds(observedAt, this.leaseDurationMs),
          },
        }),
      );
    } else if (event.type === "progress") {
      this.store.append(
        createTaskRuntimeEvent({
          eventType: "progress_recorded",
          timestamp: observedAt,
          source: `worker:${lease.workerId}`,
          idempotencyKey: `progress:${event.runId}:${event.sequence}`,
          workItemId: event.workItemId,
          runId: event.runId,
          payload: {
            claim,
            progress: {
              sequence: event.sequence,
              recordedAt: observedAt,
              message: event.message,
              completedUnits: event.fraction === undefined ? undefined : event.fraction * 1_000,
              totalUnits: event.fraction === undefined ? undefined : 1_000,
            },
          },
        }),
      );
    } else if (event.type === "checkpoint") {
      this.recordCheckpoint(event.checkpoint, event, lease, observedAt);
    } else if (event.type === "artifact") {
      this.recordArtifact(event, lease, artifactRoot, observedAt);
    } else if (event.type === "needs_human") {
      const approvalId = id("apr_");
      const requestRef = this.store.putJson(event.request).ref;
      this.store.append([
        createTaskRuntimeEvent({
          eventType: "approval_recorded",
          timestamp: observedAt,
          source: `worker:${lease.workerId}`,
          idempotencyKey: `approval:request:${approvalId}`,
          workItemId: event.workItemId,
          payload: {
            approval: {
              approvalId,
              workItemId: event.workItemId,
              runId: event.runId,
              kind: event.request.kind,
              status: "requested",
              requestedAt: observedAt,
              requestedBy: `worker:${lease.workerId}`,
              reason: event.request.prompt,
              requestRef,
            },
          },
        }),
        createTaskRuntimeEvent({
          eventType: "needs_human",
          timestamp: observedAt,
          source: `worker:${lease.workerId}`,
          idempotencyKey: event.idempotencyKey,
          workItemId: event.workItemId,
          runId: event.runId,
          payload: { claim, requestedAt: observedAt, reason: event.request.prompt, approvalId },
        }),
      ]);
    }
  }

  private recordCheckpoint(
    workerCheckpoint: TaskWorkerCheckpoint,
    event: Extract<TaskWorkerEventMessage, { type: "checkpoint" }>,
    lease: TaskLease,
    observedAt: string,
  ): TaskCheckpoint {
    if (workerCheckpoint.artifact) {
      throw new Error(
        "Artifact-backed execution checkpoints require a durable materializer and are disabled in the app-attached runtime.",
      );
    }
    const blob = this.store.putJson(workerCheckpoint);
    const state = this.store.state();
    const run = state.runs[event.runId];
    if (!run || workerCheckpoint.environmentDigest !== run.environmentDigest) {
      throw new Error("Worker checkpoint environment does not match its authoritative run.");
    }
    const checkpoint: TaskCheckpoint = {
      checkpointId: workerCheckpoint.checkpointId,
      workItemId: event.workItemId,
      runId: event.runId,
      sequence: event.sequence,
      createdAt: observedAt,
      resumeRef: blob.ref,
      checksum: `sha256:${blob.sha256}`,
      environmentDigest: workerCheckpoint.environmentDigest,
      parentCheckpointId: run.latestCheckpointId ?? run.resumeFromCheckpointId,
      artifactIds: [],
    };
    this.store.append(
      createTaskRuntimeEvent({
        eventType: "checkpoint_recorded",
        timestamp: observedAt,
        source: `worker:${lease.workerId}`,
        idempotencyKey: event.idempotencyKey,
        workItemId: event.workItemId,
        runId: event.runId,
        payload: {
          claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
          checkpoint,
        },
      }),
    );
    return checkpoint;
  }

  private recordArtifact(
    event: TaskWorkerArtifactMessage,
    lease: TaskLease,
    artifactRoot: string | undefined,
    observedAt: string,
  ): TaskArtifactReference {
    if (!artifactRoot) {
      throw new Error("Worker artifact ingestion requires a dedicated artifactRoot.");
    }
    const realRoot = fs.realpathSync(path.resolve(artifactRoot));
    const candidatePath = path.resolve(realRoot, event.artifact.relativePath);
    const candidateLstat = fs.lstatSync(candidatePath);
    if (candidateLstat.isSymbolicLink()) {
      throw new Error("Worker artifact paths must not be symbolic links.");
    }
    const filePath = fs.realpathSync(candidatePath);
    const relative = path.relative(realRoot, filePath);
    if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
      throw new Error("Worker artifact escaped its configured root.");
    }
    const noFollow = "O_NOFOLLOW" in fs.constants ? fs.constants.O_NOFOLLOW : 0;
    const descriptor = fs.openSync(filePath, fs.constants.O_RDONLY | noFollow);
    let bytes: Buffer;
    let stat: fs.Stats;
    try {
      stat = fs.fstatSync(descriptor);
      if (!stat.isFile() || stat.size !== event.artifact.sizeBytes) {
        throw new Error(
          `Worker artifact size does not match its receipt: ${event.artifact.relativePath}`,
        );
      }
      const state = this.store.state();
      const workItem = state.workItems[event.workItemId];
      const maxBytes = workItem?.budget?.maxArtifactBytes;
      if (maxBytes !== undefined && workItem.budgetUsage.artifactBytes + stat.size > maxBytes) {
        throw new Error("Worker artifact exceeds the WorkItem artifact budget.");
      }
      bytes = fs.readFileSync(descriptor);
    } finally {
      fs.closeSync(descriptor);
    }
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    if (`sha256:${sha256}` !== event.artifact.sha256) {
      throw new Error(`Worker artifact checksum does not match: ${event.artifact.relativePath}`);
    }
    const blob = this.store.putBytes(bytes);
    const artifact: TaskArtifactReference = {
      artifactId: event.artifact.artifactId,
      workItemId: event.workItemId,
      runId: event.runId,
      kind: event.artifact.kind,
      uri: pathToFileURL(this.store.pathForBlob(blob.ref)).href,
      createdAt: observedAt,
      mediaType: event.artifact.mediaType,
      sha256,
      sizeBytes: stat.size,
      immutable: true,
    };
    this.store.append(
      createTaskRuntimeEvent({
        eventType: "artifact_recorded",
        timestamp: observedAt,
        source: `worker:${lease.workerId}`,
        idempotencyKey: event.idempotencyKey,
        workItemId: event.workItemId,
        runId: event.runId,
        payload: {
          claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
          artifact,
        },
      }),
    );
    return artifact;
  }

  private async invokeCapability(
    workItemId: string,
    call: TaskWorkerCapabilityCallMessage,
    lease: TaskLease,
  ): Promise<TaskWorkerCapabilityOutcome> {
    const state = this.store.state();
    const workItem = state.workItems[workItemId];
    if (!workItem || !this.capabilityGateway) {
      return {
        status: "unknown",
        error: {
          code: "gateway_unavailable",
          message: "Capability gateway unavailable.",
          retryable: false,
        },
      };
    }
    const existingReceipt = Object.values(state.sideEffectReceipts).find(
      (receipt) => receipt.effectIdempotencyKey === call.idempotencyKey,
    );
    const effectKind = `${call.capabilityId}:${call.operation}`;
    if (
      existingReceipt &&
      (existingReceipt.workItemId !== workItemId || existingReceipt.effectKind !== effectKind)
    ) {
      return {
        status: "failed",
        error: {
          code: "idempotency_scope_collision",
          message: "The capability idempotency key is already bound to different work.",
          retryable: false,
        },
      };
    }
    if (existingReceipt?.status === "committed") {
      if (!existingReceipt.detailRef) {
        return {
          status: "unknown",
          error: {
            code: "committed_result_unavailable",
            message: "The committed effect receipt lacks a replayable result payload.",
            retryable: false,
          },
        };
      }
      try {
        const outcome = TaskWorkerCapabilityOutcomeSchema.parse(
          this.store.readJson(existingReceipt.detailRef),
        );
        if (outcome.status !== "succeeded") throw new Error("Committed outcome is not success.");
        return {
          ...outcome,
          receipt: {
            receiptId: existingReceipt.receiptId,
            idempotencyKey: call.idempotencyKey,
            externalRef: existingReceipt.externalRef,
          },
        };
      } catch {
        return {
          status: "unknown",
          error: {
            code: "committed_result_unavailable",
            message: "The committed effect result failed replay validation.",
            retryable: false,
          },
        };
      }
    }
    if (existingReceipt) {
      return {
        status: "unknown",
        error: {
          code: "effect_outcome_requires_reconciliation",
          message: "A prior capability call lacks a committed receipt and requires reconciliation.",
          retryable: false,
        },
      };
    }
    const receiptId = deterministicReceiptId(call.idempotencyKey);
    const recordedAt = this.timestamp();
    const baseReceipt = {
      receiptId,
      workItemId,
      runId: call.runId,
      effectKind,
      effectIdempotencyKey: call.idempotencyKey,
      recordedAt,
      delivery: "at_least_once" as const,
      exactlyOnce: false as const,
    };
    this.store.append(
      createTaskRuntimeEvent({
        eventType: "side_effect_receipt_recorded",
        timestamp: recordedAt,
        source: this.ownerId,
        idempotencyKey: `side-effect:started:${call.idempotencyKey}`,
        workItemId,
        runId: call.runId,
        payload: {
          claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
          receipt: { ...baseReceipt, status: "uncertain" },
        },
      }),
    );
    try {
      const outcome = TaskWorkerCapabilityOutcomeSchema.parse(
        await this.capabilityGateway.invoke({ workItem, call }),
      );
      if ("error" in outcome) {
        if (outcome.status === "failed") {
          const failedAt = this.timestamp();
          this.store.append(
            createTaskRuntimeEvent({
              eventType: "side_effect_receipt_recorded",
              timestamp: failedAt,
              source: this.ownerId,
              idempotencyKey: `side-effect:not-committed:${call.idempotencyKey}`,
              workItemId,
              runId: call.runId,
              payload: {
                claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
                receipt: { ...baseReceipt, status: "not_committed", recordedAt: failedAt },
              },
            }),
          );
        }
        return outcome;
      }
      const committedAt = this.timestamp();
      const replayableOutcome: TaskWorkerCapabilityOutcome = {
        ...outcome,
        receipt: {
          receiptId,
          idempotencyKey: call.idempotencyKey,
          ...(outcome.receipt?.externalRef ? { externalRef: outcome.receipt.externalRef } : {}),
        },
      };
      const detailRef = this.store.putJson(replayableOutcome).ref;
      this.store.append(
        createTaskRuntimeEvent({
          eventType: "side_effect_receipt_recorded",
          timestamp: committedAt,
          source: this.ownerId,
          idempotencyKey: `side-effect:committed:${call.idempotencyKey}`,
          workItemId,
          runId: call.runId,
          payload: {
            claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
            receipt: {
              ...baseReceipt,
              status: "committed",
              recordedAt: committedAt,
              externalRef: outcome.receipt?.externalRef,
              detailRef,
            },
          },
        }),
      );
      return replayableOutcome;
    } catch (error) {
      return {
        status: "unknown",
        error: {
          code: "effect_outcome_unknown",
          message: boundedErrorMessage(
            error,
            "Capability execution ended without a durable completion receipt.",
          ),
          retryable: false,
        },
      };
    }
  }

  private finishRun(result: TaskWorkerProcessResult, lease: TaskLease): TaskRuntimeState {
    const terminal = result.terminal;
    const claim = { leaseId: lease.leaseId, fencingToken: lease.fencingToken };
    const observedAt = this.timestamp();
    this.assertTerminalReferences(terminal.workItemId, terminal.runId, terminal);
    if (terminal.type === "complete") {
      const resultRef =
        terminal.result === undefined ? undefined : this.store.putJson(terminal.result).ref;
      return this.store.append(
        createTaskRuntimeEvent({
          eventType: "run_completed",
          timestamp: observedAt,
          source: `worker:${result.hello.worker.instanceId}`,
          idempotencyKey: terminal.idempotencyKey,
          workItemId: terminal.workItemId,
          runId: terminal.runId,
          payload: { claim, completedAt: observedAt, resultRef },
        }),
      ).state;
    }
    if (terminal.type === "fail") {
      const currentRun = this.store.state().runs[terminal.runId];
      if (currentRun?.status === "cancel_requested") {
        return this.store.append(
          createTaskRuntimeEvent({
            eventType: "cancel_acknowledged",
            timestamp: observedAt,
            source: this.ownerId,
            idempotencyKey: `cancel:host-ack-after-failure:${terminal.runId}`,
            workItemId: terminal.workItemId,
            runId: terminal.runId,
            payload: {
              claim,
              acknowledgedAt: observedAt,
              reason: `Worker exited after cancellation: ${terminal.failure.message}`,
            },
          }),
        ).state;
      }
      let state = this.store.append(
        createTaskRuntimeEvent({
          eventType: "run_failed",
          timestamp: observedAt,
          source: `worker:${result.hello.worker.instanceId}`,
          idempotencyKey: terminal.idempotencyKey,
          workItemId: terminal.workItemId,
          runId: terminal.runId,
          payload: {
            claim,
            failure: {
              occurredAt: observedAt,
              message: terminal.failure.message,
              code: terminal.failure.code,
              retryable: terminal.failure.retryable,
            },
          },
        }),
      ).state;
      const workItem = state.workItems[terminal.workItemId];
      if (
        terminal.failure.retryable &&
        workItem &&
        !workItem.cancellation &&
        workItem.status === "failed" &&
        workItem.retry.attemptsStarted < workItem.retry.maxAttempts
      ) {
        state = this.scheduleRetry(workItem, observedAt, terminal.failure.message);
      }
      return state;
    }
    if (terminal.type === "canceled") {
      let state = this.store.state();
      const run = state.runs[terminal.runId];
      if (run?.status !== "cancel_requested") {
        state = this.store.append(
          createTaskRuntimeEvent({
            eventType: "cancel_requested",
            timestamp: observedAt,
            source: this.ownerId,
            idempotencyKey: `cancel:implicit:${terminal.runId}`,
            workItemId: terminal.workItemId,
            runId: terminal.runId,
            payload: { requestedAt: observedAt, reason: terminal.reason },
          }),
        ).state;
      }
      return this.store.append(
        createTaskRuntimeEvent({
          eventType: "cancel_acknowledged",
          timestamp: observedAt,
          source: `worker:${result.hello.worker.instanceId}`,
          idempotencyKey: terminal.idempotencyKey,
          workItemId: terminal.workItemId,
          runId: terminal.runId,
          payload: { claim, acknowledgedAt: observedAt, reason: terminal.reason },
        }),
      ).state;
    }
    return this.store.state();
  }

  private assertTerminalReferences(
    workItemId: string,
    runId: string,
    terminal: TaskWorkerProcessResult["terminal"],
  ): void {
    const state = this.store.state();
    const artifactIds = terminal.type === "complete" ? terminal.artifactIds : [];
    for (const artifactId of artifactIds) {
      const artifact = state.artifacts[artifactId];
      if (!artifact || artifact.workItemId !== workItemId || artifact.runId !== runId) {
        throw new Error(`Terminal message references unknown artifact "${artifactId}".`);
      }
    }
    const checkpointId =
      terminal.type === "needs_human" ? terminal.request.checkpointId : terminal.checkpointId;
    if (!checkpointId) return;
    const checkpoint = state.checkpoints[checkpointId];
    const run = state.runs[runId];
    if (
      !checkpoint ||
      !run ||
      checkpoint.workItemId !== workItemId ||
      (checkpoint.runId !== runId &&
        (run.latestCheckpointId !== undefined || run.resumeFromCheckpointId !== checkpointId))
    ) {
      throw new Error(`Terminal message references unknown checkpoint "${checkpointId}".`);
    }
  }

  private failInterruptedRun(lease: TaskLease, error: unknown): TaskRuntimeState {
    let state = this.store.state();
    let run = state.runs[lease.runId];
    if (
      !run ||
      ["failed", "succeeded", "canceled", "needs_human", "interrupted"].includes(run.status)
    ) {
      return state;
    }
    const currentLease = state.workItems[lease.workItemId]?.lease;
    if (
      !currentLease ||
      currentLease.leaseId !== lease.leaseId ||
      currentLease.fencingToken !== lease.fencingToken
    ) {
      return state;
    }
    const occurredAt = this.timestamp();
    if (Date.parse(occurredAt) >= Date.parse(currentLease.expiresAt)) {
      state = this.store.append(
        createTaskRuntimeEvent({
          eventType: "lease_expired",
          timestamp: occurredAt,
          source: this.ownerId,
          idempotencyKey: `lease:expired:${currentLease.leaseId}`,
          workItemId: currentLease.workItemId,
          runId: currentLease.runId,
          payload: {
            claim: {
              leaseId: currentLease.leaseId,
              fencingToken: currentLease.fencingToken,
            },
            expiredAt: occurredAt,
            reason: "Worker process ended after its fenced lease expired.",
          },
        }),
      ).state;
      return this.scheduleRetryIfAllowed(state, currentLease.workItemId, occurredAt);
    }
    if (
      error instanceof TaskWorkerProcessError &&
      error.code === "canceled" &&
      run.status !== "cancel_requested"
    ) {
      state = this.store.append(
        createTaskRuntimeEvent({
          eventType: "cancel_requested",
          timestamp: occurredAt,
          source: this.ownerId,
          idempotencyKey: `cancel:host-request:${lease.runId}`,
          workItemId: lease.workItemId,
          runId: lease.runId,
          payload: { requestedAt: occurredAt, reason: error.message },
        }),
      ).state;
      run = state.runs[lease.runId];
    }
    if (run?.status === "cancel_requested") {
      return this.store.append(
        createTaskRuntimeEvent({
          eventType: "cancel_acknowledged",
          timestamp: occurredAt,
          source: this.ownerId,
          idempotencyKey: `cancel:host-ack:${lease.runId}`,
          workItemId: lease.workItemId,
          runId: lease.runId,
          payload: {
            claim: {
              leaseId: currentLease.leaseId,
              fencingToken: currentLease.fencingToken,
            },
            acknowledgedAt: occurredAt,
            reason: "The worker process exited after cancellation.",
          },
        }),
      ).state;
    }
    const message =
      error instanceof Error ? error.message.slice(0, 16_384) : "Worker process failed.";
    const retryable =
      !(error instanceof TaskWorkerProcessError) ||
      ["spawn_failed", "heartbeat_timeout", "unexpected_exit"].includes(error.code);
    state = this.store.append(
      createTaskRuntimeEvent({
        eventType: "run_failed",
        timestamp: occurredAt,
        source: this.ownerId,
        idempotencyKey: `run:host-failure:${lease.runId}`,
        workItemId: lease.workItemId,
        runId: lease.runId,
        payload: {
          claim: {
            leaseId: currentLease.leaseId,
            fencingToken: currentLease.fencingToken,
          },
          failure: { occurredAt, message, code: "WORKER_PROCESS_FAILED", retryable },
        },
      }),
    ).state;
    return retryable ? this.scheduleRetryIfAllowed(state, lease.workItemId, occurredAt) : state;
  }

  private scheduleRetryIfAllowed(
    state: TaskRuntimeState,
    workItemId: string,
    timestamp: string,
  ): TaskRuntimeState {
    const workItem = state.workItems[workItemId];
    const run = workItem?.activeRunId ? state.runs[workItem.activeRunId] : undefined;
    return workItem?.status === "failed" &&
      !workItem.cancellation &&
      run?.failure?.retryable &&
      workItem.retry.attemptsStarted < workItem.retry.maxAttempts
      ? this.scheduleRetry(workItem, timestamp, workItem.retry.lastFailure ?? "Retry worker run.")
      : state;
  }

  private scheduleRetry(
    workItem: TaskWorkItem,
    timestamp: string,
    reason: string,
    approvalId?: string,
  ): TaskRuntimeState {
    return this.store.append(
      createTaskRuntimeEvent({
        eventType: "retry_scheduled",
        timestamp,
        source: this.ownerId,
        idempotencyKey: `retry:${workItem.id}:${workItem.retry.attemptsStarted}`,
        workItemId: workItem.id,
        payload: {
          scheduledAt: timestamp,
          nextAttemptAt: timestamp,
          reason,
          resumeFromCheckpointId: workItem.latestCheckpointId,
          approvalId,
        },
      }),
    ).state;
  }

  private blockAfterApproval(
    workItem: TaskWorkItem,
    approval: TaskApproval,
    reason: string,
  ): TaskRuntimeState {
    const timestamp = approval.decidedAt ?? this.timestamp();
    return this.store.append(
      createTaskRuntimeEvent({
        eventType: "work_item_blocked",
        timestamp,
        source: "task-runtime:approval-reconciliation",
        idempotencyKey: `approval:block:${approval.approvalId}:${approval.status}`,
        workItemId: workItem.id,
        payload: { reason },
      }),
    ).state;
  }

  private resumeCheckpoint(
    state: TaskRuntimeState,
    workItem: TaskWorkItem,
    environmentDigest: string,
  ): TaskWorkerCheckpoint | undefined {
    if (!workItem.latestCheckpointId) return undefined;
    const checkpoint = state.checkpoints[workItem.latestCheckpointId];
    if (!checkpoint) throw new Error(`Checkpoint "${workItem.latestCheckpointId}" is missing.`);
    if (checkpoint.environmentDigest !== environmentDigest) {
      throw new TaskCheckpointEnvironmentMismatchError(
        checkpoint.checkpointId,
        checkpoint.environmentDigest,
        environmentDigest,
      );
    }
    if (checkpoint.checksum && checkpoint.checksum !== checkpoint.resumeRef) {
      throw new Error(
        `Checkpoint "${checkpoint.checkpointId}" metadata checksum does not match its resume ref.`,
      );
    }
    const workerCheckpoint = TaskWorkerCheckpointSchema.parse(
      this.store.readJson(checkpoint.resumeRef),
    );
    if (workerCheckpoint.artifact) {
      throw new Error(
        "Artifact-backed execution checkpoints require a durable materializer and are disabled in the app-attached runtime.",
      );
    }
    if (workerCheckpoint.checkpointId !== checkpoint.checkpointId) {
      throw new Error(
        `Checkpoint "${checkpoint.checkpointId}" payload identifies "${workerCheckpoint.checkpointId}".`,
      );
    }
    if (
      workerCheckpoint.environmentDigest !== checkpoint.environmentDigest ||
      workerCheckpoint.environmentDigest !== environmentDigest
    ) {
      throw new TaskCheckpointEnvironmentMismatchError(
        checkpoint.checkpointId,
        workerCheckpoint.environmentDigest,
        environmentDigest,
      );
    }
    return workerCheckpoint;
  }

  private sessionLink(
    workItemId: string,
    sessionId: string,
    role: "creator" | "observer",
    linkedAt: string,
  ): TaskSessionLink {
    return {
      linkId: id("slnk_"),
      workItemId,
      sessionId,
      role,
      linkedAt,
    };
  }

  private timestamp(): string {
    return this.now().toISOString();
  }
}

function id(prefix: string): string {
  return `${prefix}${randomUUID().replaceAll("-", "")}`;
}

function deterministicReceiptId(idempotencyKey: string): string {
  return `rcpt_${createHash("sha256").update(idempotencyKey).digest("hex").slice(0, 32)}`;
}

function addMilliseconds(timestamp: string, milliseconds: number): string {
  return new Date(Date.parse(timestamp) + milliseconds).toISOString();
}

function relayAbort(signal: AbortSignal | undefined, relay: () => void): () => void {
  if (!signal) return () => {};
  if (signal.aborted) {
    relay();
    return () => {};
  }
  signal.addEventListener("abort", relay, { once: true });
  return () => signal.removeEventListener("abort", relay);
}

function abortReason(reason: unknown): string {
  if (reason instanceof Error && reason.message) return reason.message;
  if (typeof reason === "string" && reason) return reason;
  return "Task cancellation requested.";
}

function remainingBudget(limit: number | undefined, used: number): number | undefined {
  return limit === undefined ? undefined : Math.max(0, limit - used);
}

function boundedHeartbeatInterval(leaseDurationMs: number): number {
  return Math.min(60_000, Math.max(1, Math.floor(leaseDurationMs / 3)));
}

function boundedHeartbeatTimeout(leaseDurationMs: number): number {
  return Math.min(60_000, Math.max(1, Math.floor((leaseDurationMs * 2) / 3)));
}

function remainingCapabilityCallsByKind(workItem: TaskWorkItem): Record<string, number> {
  const budget = workItem.budget?.capabilityCalls ?? {};
  const result: Record<string, number> = {};
  for (const [kind, limit] of Object.entries(budget)) {
    const remaining = Math.max(0, limit - (workItem.budgetUsage.capabilityCalls[kind] ?? 0));
    if (remaining > 0) result[kind] = remaining;
  }
  return result;
}

function boundedErrorMessage(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : String(error);
  return message && message.length <= 4_096 ? message : fallback;
}
