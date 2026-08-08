import { describe, expect, it } from "vitest";
import {
  applyTaskRuntimeEvent,
  type CreateTaskRuntimeEventInput,
  createTaskRuntimeEvent,
  emptyTaskRuntimeState,
  evaluateTaskSchedule,
  expiredTaskLeases,
  isTaskWorkItemRunnable,
  replayTaskRuntimeEvents,
  TASK_RUNTIME_DELIVERY_SEMANTICS,
  TaskLeaseSchema,
  TaskRunSchema,
  type TaskRuntimeEvent,
  TaskRuntimeEventSchema,
  TaskRuntimeIdempotencyCollisionError,
  TaskRuntimeInvariantError,
  type TaskRuntimeState,
  TaskRuntimeStateSchema,
  TaskWorkItemSchema,
} from "../src/task-runtime.js";

const t0 = "2026-08-05T00:00:00.000Z";
const t1 = "2026-08-05T00:01:00.000Z";
const t2 = "2026-08-05T00:02:00.000Z";
const t3 = "2026-08-05T00:03:00.000Z";
const t4 = "2026-08-05T00:04:00.000Z";
const t5 = "2026-08-05T00:05:00.000Z";
const t10 = "2026-08-05T00:10:00.000Z";
const t15 = "2026-08-05T00:15:00.000Z";
const t16 = "2026-08-05T00:16:00.000Z";
const t20 = "2026-08-05T00:20:00.000Z";

const workItemId = "awi_portable_task";
const runId = "run_attempt_1";
const leaseId = "lease_attempt_1";
const executor = { backend: "portable-worker", operation: "transform-records" };
const claim = { leaseId, fencingToken: 1 };
const environmentDigest = `sha256:${"a".repeat(64)}`;

function workItem(overrides: Record<string, unknown> = {}) {
  return TaskWorkItemSchema.parse({
    id: workItemId,
    status: "queued",
    executor,
    createdAt: t0,
    updatedAt: t0,
    retry: { maxAttempts: 2 },
    ...overrides,
  });
}

function run(overrides: Record<string, unknown> = {}) {
  return TaskRunSchema.parse({
    runId,
    workItemId,
    executor,
    status: "created",
    attempt: 1,
    createdAt: t1,
    environmentDigest,
    ...overrides,
  });
}

function event(input: CreateTaskRuntimeEventInput): TaskRuntimeEvent {
  return createTaskRuntimeEvent(input);
}

function runningEvents(): TaskRuntimeEvent[] {
  return [
    event({
      eventType: "work_item_created",
      timestamp: t0,
      source: "test.control",
      idempotencyKey: "create:portable-task",
      workItemId,
      payload: { workItem: workItem() },
    }),
    event({
      eventType: "run_created",
      timestamp: t1,
      source: "test.control",
      idempotencyKey: "run:create:1",
      workItemId,
      runId,
      payload: { run: run() },
    }),
    event({
      eventType: "lease_acquired",
      timestamp: t1,
      source: "test.control",
      idempotencyKey: "lease:acquire:1",
      workItemId,
      runId,
      payload: {
        lease: {
          leaseId,
          workItemId,
          runId,
          workerId: "worker-1",
          fencingToken: 1,
          acquiredAt: t1,
          heartbeatAt: t1,
          expiresAt: t10,
        },
      },
    }),
    event({
      eventType: "run_started",
      timestamp: t2,
      source: "test.worker",
      idempotencyKey: "run:start:1",
      workItemId,
      runId,
      payload: { claim, startedAt: t2 },
    }),
  ];
}

function runningState(): TaskRuntimeState {
  return replayTaskRuntimeEvents(runningEvents());
}

describe("task runtime kernel", () => {
  it("exposes strict language-neutral boundaries and honest delivery semantics", () => {
    const parsed = workItem();
    expect(parsed.executor).toEqual(executor);
    expect(parsed.status).toBe("queued");
    expect(parsed.retry).toEqual({ attemptsStarted: 0, maxAttempts: 2 });
    expect(parsed.budgetUsage).toEqual({
      wallTimeMs: 0,
      artifactBytes: 0,
      checkpoints: 0,
      progressEvents: 0,
      capabilityCalls: {},
    });
    expect(TASK_RUNTIME_DELIVERY_SEMANTICS).toEqual({
      delivery: "at_least_once",
      exactlyOnce: false,
      externalEffects: "idempotency_key_with_durable_receipt",
    });

    expect(() => TaskWorkItemSchema.parse({ ...parsed, unexpected: true })).toThrow();
    expect(() => TaskRunSchema.parse({ ...run(), environmentDigest: "unversioned" })).toThrow(
      /sha256/,
    );
    expect(() =>
      TaskLeaseSchema.parse({
        leaseId,
        workItemId,
        runId,
        workerId: "worker-1",
        fencingToken: 1,
        acquiredAt: t1,
        heartbeatAt: t2,
        expiresAt: t2,
      }),
    ).toThrow(/expiry/);
    expect(() =>
      TaskRuntimeEventSchema.parse({
        schemaVersion: 1,
        eventId: "evt_invalid_payload",
        eventType: "run_started",
        timestamp: t2,
        source: "test.worker",
        idempotencyKey: "invalid:payload",
        workItemId,
        runId,
        payload: { claim, startedAt: t2, patch: { status: "succeeded" } },
      }),
    ).toThrow();
    expect(() =>
      TaskRuntimeEventSchema.parse({
        ...runningEvents()[0],
        schemaVersion: 2,
      }),
    ).toThrow();
  });

  it("keeps an unleased run proposal recoverable after a torn multi-event append", () => {
    let state = replayTaskRuntimeEvents(runningEvents().slice(0, 2));
    expect(state.workItems[workItemId]).toMatchObject({
      status: "queued",
      retry: { attemptsStarted: 0, maxAttempts: 2 },
    });
    expect(state.workItems[workItemId]?.activeRunId).toBeUndefined();
    expect(isTaskWorkItemRunnable(state.workItems[workItemId], t2)).toBe(true);

    const recoveredRunId = "run_attempt_1_recovered";
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "run_created",
        timestamp: t2,
        source: "test.control",
        idempotencyKey: "run:create:1:recovered",
        workItemId,
        runId: recoveredRunId,
        payload: {
          run: run({ runId: recoveredRunId, createdAt: t2 }),
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "lease_acquired",
        timestamp: t2,
        source: "test.control",
        idempotencyKey: "lease:acquire:1:recovered",
        workItemId,
        runId: recoveredRunId,
        payload: {
          lease: {
            leaseId: "lease_attempt_1_recovered",
            workItemId,
            runId: recoveredRunId,
            workerId: "worker-recovered",
            fencingToken: 1,
            acquiredAt: t2,
            heartbeatAt: t2,
            expiresAt: t10,
          },
        },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({
      status: "leased",
      activeRunId: recoveredRunId,
      retry: { attemptsStarted: 1, maxAttempts: 2 },
    });
  });

  it("recovers a persisted human request when its terminal event was torn", () => {
    let state = runningState();
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "approval_recorded",
        timestamp: t3,
        source: "test.worker",
        idempotencyKey: "approval:request:torn",
        workItemId,
        payload: {
          approval: {
            approvalId: "apr_torn_request",
            workItemId,
            runId,
            kind: "approval",
            status: "requested",
            requestedAt: t3,
            requestedBy: "test.worker",
            reason: "Continue?",
          },
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "lease_expired",
        timestamp: t15,
        source: "test.control",
        idempotencyKey: "lease:expire:torn-human",
        workItemId,
        runId,
        payload: { claim, expiredAt: t15 },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({
      status: "needs_human",
      needsHumanReason: "Lease expired while awaiting a persisted human decision.",
    });
    expect(state.runs[runId]).toMatchObject({
      status: "interrupted",
      failure: { code: "LEASE_EXPIRED_AWAITING_HUMAN", retryable: false },
    });
  });

  it("replays typed progress, artifact, checkpoint, and completion events", () => {
    const artifactEvent = event({
      eventType: "artifact_recorded",
      timestamp: t3,
      source: "test.worker",
      idempotencyKey: "artifact:result:1",
      workItemId,
      runId,
      payload: {
        claim,
        artifact: {
          artifactId: "art_result_1",
          workItemId,
          runId,
          kind: "result",
          uri: "artifact://result-1",
          createdAt: t3,
          sizeBytes: 128,
        },
      },
    });
    const checkpointEvent = event({
      eventType: "checkpoint_recorded",
      timestamp: t4,
      source: "test.worker",
      idempotencyKey: "checkpoint:1",
      workItemId,
      runId,
      payload: {
        claim,
        checkpoint: {
          checkpointId: "ckp_attempt_1",
          workItemId,
          runId,
          sequence: 1,
          createdAt: t4,
          resumeRef: "checkpoint://attempt-1",
          environmentDigest,
          artifactIds: ["art_result_1"],
        },
      },
    });
    const progressEvent = event({
      eventType: "progress_recorded",
      timestamp: t3,
      source: "test.worker",
      idempotencyKey: "progress:1",
      workItemId,
      runId,
      payload: {
        claim,
        progress: { sequence: 1, recordedAt: t3, completedUnits: 1, totalUnits: 2 },
      },
    });
    const completedEvent = event({
      eventType: "run_completed",
      timestamp: t5,
      source: "test.worker",
      idempotencyKey: "complete:1",
      workItemId,
      runId,
      payload: { claim, completedAt: t5, resultRef: "artifact://result-1" },
    });

    const state = replayTaskRuntimeEvents([
      ...runningEvents(),
      progressEvent,
      artifactEvent,
      checkpointEvent,
      completedEvent,
    ]);
    expect(state.workItems[workItemId]).toMatchObject({
      status: "succeeded",
      latestCheckpointId: "ckp_attempt_1",
      artifactIds: ["art_result_1"],
      budgetUsage: { progressEvents: 1, checkpoints: 1, artifactBytes: 128 },
    });
    expect(state.workItems[workItemId]?.lease).toBeUndefined();
    expect(state.runs[runId]).toMatchObject({
      status: "succeeded",
      latestCheckpointId: "ckp_attempt_1",
      resultRef: "artifact://result-1",
    });
    expect(state.checkpoints.ckp_attempt_1?.resumeRef).toBe("checkpoint://attempt-1");
    expect(state.artifacts.art_result_1?.immutable).toBe(true);
    expect(TaskRuntimeStateSchema.parse(state)).toEqual(state);

    const duplicate = applyTaskRuntimeEvent(state, completedEvent);
    expect(duplicate).toBe(state);
  });

  it("rejects a checkpoint whose environment differs from its authoritative run", () => {
    expect(() =>
      applyTaskRuntimeEvent(
        runningState(),
        event({
          eventType: "checkpoint_recorded",
          timestamp: t3,
          source: "test.worker",
          idempotencyKey: "checkpoint:wrong-environment",
          workItemId,
          runId,
          payload: {
            claim,
            checkpoint: {
              checkpointId: "ckp_wrong_environment",
              workItemId,
              runId,
              sequence: 1,
              createdAt: t3,
              resumeRef: "checkpoint://wrong-environment",
              environmentDigest: `sha256:${"b".repeat(64)}`,
            },
          },
        }),
      ),
    ).toThrow(/checkpoint environment/i);
  });

  it("detects idempotency collisions instead of treating unlike events as duplicates", () => {
    const first = event({
      eventType: "progress_recorded",
      timestamp: t3,
      source: "test.worker",
      idempotencyKey: "progress:collision",
      workItemId,
      runId,
      payload: { claim, progress: { sequence: 1, recordedAt: t3, completedUnits: 1 } },
    });
    const collision = event({
      eventType: "progress_recorded",
      timestamp: t4,
      source: "test.worker",
      idempotencyKey: "progress:collision",
      workItemId,
      runId,
      payload: { claim, progress: { sequence: 2, recordedAt: t4, completedUnits: 2 } },
    });
    const state = applyTaskRuntimeEvent(runningState(), first);

    expect(() => applyTaskRuntimeEvent(state, collision)).toThrow(
      TaskRuntimeIdempotencyCollisionError,
    );
    expect(state.events).toHaveLength(runningEvents().length + 1);
  });

  it("renews live leases, fences stale workers, expires interrupted runs, and schedules retry", () => {
    let state = runningState();
    const heartbeat = event({
      eventType: "lease_heartbeat",
      timestamp: t5,
      source: "test.worker",
      idempotencyKey: "heartbeat:1",
      workItemId,
      runId,
      payload: { claim, heartbeatAt: t5, expiresAt: t15 },
    });
    state = applyTaskRuntimeEvent(state, heartbeat);
    expect(expiredTaskLeases(state, "2026-08-05T00:14:59.000Z")).toEqual([]);
    expect(expiredTaskLeases(state, t15)).toMatchObject([{ leaseId }]);

    const staleProgress = event({
      eventType: "progress_recorded",
      timestamp: "2026-08-05T00:06:00.000Z",
      source: "test.worker",
      idempotencyKey: "stale:progress",
      workItemId,
      runId,
      payload: {
        claim: { leaseId, fencingToken: 2 },
        progress: { sequence: 1, recordedAt: "2026-08-05T00:06:00.000Z" },
      },
    });
    expect(() => applyTaskRuntimeEvent(state, staleProgress)).toThrow(/fenced lease/);

    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "lease_expired",
        timestamp: t15,
        source: "test.control",
        idempotencyKey: "lease:expire:1",
        workItemId,
        runId,
        payload: { claim, expiredAt: t15 },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({ status: "failed", lease: undefined });
    expect(state.runs[runId]).toMatchObject({
      status: "interrupted",
      failure: { code: "LEASE_EXPIRED", retryable: true },
    });

    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "retry_scheduled",
        timestamp: t16,
        source: "test.control",
        idempotencyKey: "retry:schedule:2",
        workItemId,
        payload: { scheduledAt: t16, nextAttemptAt: t20, reason: "Retry interrupted work." },
      }),
    );
    const queued = state.workItems[workItemId];
    expect(queued).toMatchObject({ status: "queued", activeRunId: undefined });
    expect(isTaskWorkItemRunnable(queued, "2026-08-05T00:19:59.000Z")).toBe(false);
    expect(isTaskWorkItemRunnable(queued, t20)).toBe(true);
  });

  it("resumes a retry from an execution checkpoint with a higher fencing token", () => {
    let state = runningState();
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "checkpoint_recorded",
        timestamp: t3,
        source: "test.worker",
        idempotencyKey: "checkpoint:resume",
        workItemId,
        runId,
        payload: {
          claim,
          checkpoint: {
            checkpointId: "ckp_resume_1",
            workItemId,
            runId,
            sequence: 1,
            createdAt: t3,
            resumeRef: "checkpoint://resume-1",
            environmentDigest,
          },
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "checkpoint_recorded",
        timestamp: t4,
        source: "test.worker",
        idempotencyKey: "checkpoint:resume:child",
        workItemId,
        runId,
        payload: {
          claim,
          checkpoint: {
            checkpointId: "ckp_resume_2",
            workItemId,
            runId,
            sequence: 2,
            createdAt: t4,
            resumeRef: "checkpoint://resume-2",
            environmentDigest,
            parentCheckpointId: "ckp_resume_1",
          },
        },
      }),
    );
    expect(state.checkpoints.ckp_resume_2?.parentCheckpointId).toBe("ckp_resume_1");
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "run_failed",
        timestamp: t5,
        source: "test.worker",
        idempotencyKey: "run:fail:1",
        workItemId,
        runId,
        payload: {
          claim,
          failure: { occurredAt: t5, message: "Transient failure.", retryable: true },
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "retry_scheduled",
        timestamp: t5,
        source: "test.control",
        idempotencyKey: "retry:checkpoint:2",
        workItemId,
        payload: {
          scheduledAt: t5,
          nextAttemptAt: t5,
          reason: "Resume from durable checkpoint.",
          resumeFromCheckpointId: "ckp_resume_2",
        },
      }),
    );

    const secondRunId = "run_attempt_2";
    expect(() =>
      applyTaskRuntimeEvent(
        state,
        event({
          eventType: "run_created",
          timestamp: t5,
          source: "test.control",
          idempotencyKey: "run:create:2:wrong-environment",
          workItemId,
          runId: "run_attempt_2_wrong_environment",
          payload: {
            run: run({
              runId: "run_attempt_2_wrong_environment",
              attempt: 2,
              createdAt: t5,
              environmentDigest: `sha256:${"b".repeat(64)}`,
              resumeFromCheckpointId: "ckp_resume_2",
            }),
          },
        }),
      ),
    ).toThrow(/checkpoint environment/i);
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "run_created",
        timestamp: t5,
        source: "test.control",
        idempotencyKey: "run:create:2",
        workItemId,
        runId: secondRunId,
        payload: {
          run: run({
            runId: secondRunId,
            attempt: 2,
            createdAt: t5,
            resumeFromCheckpointId: "ckp_resume_2",
          }),
        },
      }),
    );
    const secondLeaseId = "lease_attempt_2";
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "lease_acquired",
        timestamp: t5,
        source: "test.control",
        idempotencyKey: "lease:acquire:2",
        workItemId,
        runId: secondRunId,
        payload: {
          lease: {
            leaseId: secondLeaseId,
            workItemId,
            runId: secondRunId,
            workerId: "worker-2",
            fencingToken: 2,
            acquiredAt: t5,
            heartbeatAt: t5,
            expiresAt: t15,
          },
        },
      }),
    );
    expect(state.runs[secondRunId]).toMatchObject({
      status: "leased",
      resumeFromCheckpointId: "ckp_resume_2",
      fencingToken: 2,
    });
    expect(state.workItems[workItemId]?.lastFencingToken).toBe(2);

    expect(() =>
      applyTaskRuntimeEvent(
        state,
        event({
          eventType: "progress_recorded",
          timestamp: "2026-08-05T00:06:00.000Z",
          source: "test.worker",
          idempotencyKey: "old-worker:progress",
          workItemId,
          runId,
          payload: {
            claim,
            progress: { sequence: 2, recordedAt: "2026-08-05T00:06:00.000Z" },
          },
        }),
      ),
    ).toThrow(TaskRuntimeInvariantError);
  });

  it("keeps session observation independent from cancellation and requires worker acknowledgement", () => {
    let state = runningState();
    for (const [linkId, sessionId, role] of [
      ["slnk_creator", "session-1", "creator"],
      ["slnk_observer", "session-2", "observer"],
    ] as const) {
      state = applyTaskRuntimeEvent(
        state,
        event({
          eventType: "session_linked",
          timestamp: t3,
          source: "test.control",
          idempotencyKey: `session:link:${sessionId}`,
          workItemId,
          payload: { link: { linkId, workItemId, sessionId, role, linkedAt: t3 } },
        }),
      );
    }
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "session_unlinked",
        timestamp: t4,
        source: "test.control",
        idempotencyKey: "session:unlink:1",
        workItemId,
        payload: { linkId: "slnk_creator", unlinkedAt: t4 },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({
      status: "running",
      sessionLinkIds: ["slnk_observer"],
    });

    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "cancel_requested",
        timestamp: t4,
        source: "test.control",
        idempotencyKey: "cancel:request:1",
        workItemId,
        runId,
        payload: { requestedAt: t4, reason: "User stopped the task." },
      }),
    );
    expect(state.workItems[workItemId]?.status).toBe("running");
    expect(state.runs[runId]?.status).toBe("cancel_requested");

    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "cancel_acknowledged",
        timestamp: t5,
        source: "test.worker",
        idempotencyKey: "cancel:ack:1",
        workItemId,
        runId,
        payload: { claim, acknowledgedAt: t5 },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({
      status: "canceled",
      cancellation: { status: "acknowledged" },
      sessionLinkIds: ["slnk_observer"],
    });
    expect(state.workItems[workItemId]?.lease).toBeUndefined();
    expect(state.runs[runId]?.status).toBe("canceled");

    let expiredCancellation = runningState();
    expiredCancellation = applyTaskRuntimeEvent(
      expiredCancellation,
      event({
        eventType: "cancel_requested",
        timestamp: t4,
        source: "test.control",
        idempotencyKey: "cancel:request:expiry",
        workItemId,
        runId,
        payload: { requestedAt: t4 },
      }),
    );
    expiredCancellation = applyTaskRuntimeEvent(
      expiredCancellation,
      event({
        eventType: "lease_expired",
        timestamp: t10,
        source: "test.control",
        idempotencyKey: "lease:expire:cancellation",
        workItemId,
        runId,
        payload: { claim, expiredAt: t10 },
      }),
    );
    expect(expiredCancellation.workItems[workItemId]).toMatchObject({
      status: "canceled",
      cancellation: { status: "acknowledged", acknowledgedAt: t10 },
    });
    expect(expiredCancellation.runs[runId]?.status).toBe("canceled");
    expect(expiredCancellation.runs[runId]?.failure).toBeUndefined();
    expect(() =>
      applyTaskRuntimeEvent(
        expiredCancellation,
        event({
          eventType: "retry_scheduled",
          timestamp: t16,
          source: "test.control",
          idempotencyKey: "retry:canceled",
          workItemId,
          payload: { scheduledAt: t16, nextAttemptAt: t16, reason: "Must not retry." },
        }),
      ),
    ).toThrow(/Only failed/);
  });

  it("records human approval before explicitly scheduling gated work", () => {
    let state = runningState();
    const requestedApproval = {
      approvalId: "apr_resume",
      workItemId,
      runId,
      kind: "resume",
      status: "requested" as const,
      requestedAt: t3,
      requestedBy: "worker-1",
    };
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "approval_recorded",
        timestamp: t3,
        source: "test.control",
        idempotencyKey: "approval:request:resume",
        workItemId,
        payload: { approval: requestedApproval },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "needs_human",
        timestamp: t3,
        source: "test.worker",
        idempotencyKey: "needs-human:1",
        workItemId,
        runId,
        payload: { claim, requestedAt: t3, reason: "Review required.", approvalId: "apr_resume" },
      }),
    );
    expect(state.workItems[workItemId]?.status).toBe("needs_human");
    expect(() =>
      applyTaskRuntimeEvent(
        state,
        event({
          eventType: "retry_scheduled",
          timestamp: t4,
          source: "test.control",
          idempotencyKey: "retry:without-approval",
          workItemId,
          payload: { scheduledAt: t4, nextAttemptAt: t4, reason: "Resume." },
        }),
      ),
    ).toThrow(/approved/i);

    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "approval_recorded",
        timestamp: t4,
        source: "test.control",
        idempotencyKey: "approval:decide:resume",
        workItemId,
        payload: {
          approval: {
            ...requestedApproval,
            status: "approved",
            decidedAt: t4,
            decidedBy: "reviewer",
          },
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "retry_scheduled",
        timestamp: t5,
        source: "test.control",
        idempotencyKey: "retry:with-approval",
        workItemId,
        payload: {
          scheduledAt: t5,
          nextAttemptAt: t5,
          reason: "Approved resume.",
          approvalId: "apr_resume",
        },
      }),
    );
    expect(state.workItems[workItemId]?.status).toBe("queued");
  });

  it("treats external receipts as evidence, preserving uncertainty across lease recovery", () => {
    let state = runningState();
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "side_effect_receipt_recorded",
        timestamp: t3,
        source: "test.worker",
        idempotencyKey: "receipt:event:uncertain",
        workItemId,
        runId,
        payload: {
          claim,
          receipt: {
            receiptId: "rcpt_publish_1",
            workItemId,
            runId,
            effectKind: "publish",
            effectIdempotencyKey: "publish:portable-task:1",
            status: "uncertain",
            recordedAt: t3,
          },
        },
      }),
    );
    expect(state.sideEffectReceipts.rcpt_publish_1).toMatchObject({
      status: "uncertain",
      delivery: "at_least_once",
      exactlyOnce: false,
    });
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "lease_expired",
        timestamp: t10,
        source: "test.control",
        idempotencyKey: "lease:expire:uncertain",
        workItemId,
        runId,
        payload: { claim, expiredAt: t10 },
      }),
    );
    expect(state.workItems[workItemId]).toMatchObject({
      status: "needs_human",
      needsHumanReason: "Lease expired with an uncertain external side effect.",
    });
    expect(state.runs[runId]?.failure?.retryable).toBe(false);
  });

  it("allows an uncertain receipt to become committed but never regress", () => {
    let state = runningState();
    const receiptBase = {
      receiptId: "rcpt_write_1",
      workItemId,
      runId,
      effectKind: "write",
      effectIdempotencyKey: "write:portable-task:1",
    };
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "side_effect_receipt_recorded",
        timestamp: t3,
        source: "test.worker",
        idempotencyKey: "receipt:write:uncertain",
        workItemId,
        runId,
        payload: {
          claim,
          receipt: { ...receiptBase, status: "uncertain", recordedAt: t3 },
        },
      }),
    );
    state = applyTaskRuntimeEvent(
      state,
      event({
        eventType: "side_effect_receipt_recorded",
        timestamp: t4,
        source: "test.worker",
        idempotencyKey: "receipt:write:committed",
        workItemId,
        runId,
        payload: {
          claim,
          receipt: {
            ...receiptBase,
            status: "committed",
            recordedAt: t4,
            externalRef: "external://write-1",
          },
        },
      }),
    );
    expect(state.sideEffectReceipts.rcpt_write_1?.status).toBe("committed");

    expect(() =>
      applyTaskRuntimeEvent(
        state,
        event({
          eventType: "side_effect_receipt_recorded",
          timestamp: t5,
          source: "test.worker",
          idempotencyKey: "receipt:write:regression",
          workItemId,
          runId,
          payload: {
            claim,
            receipt: { ...receiptBase, status: "uncertain", recordedAt: t5 },
          },
        }),
      ),
    ).toThrow(/cannot become uncertain/);
  });

  it("evaluates interval schedules deterministically", () => {
    const schedule = {
      scheduleId: "maintenance",
      enabled: true,
      cadence: { kind: "interval" as const, everySeconds: 60 },
      lastTriggeredAt: t0,
    };
    expect(evaluateTaskSchedule(schedule, "2026-08-05T00:00:59.000Z")).toMatchObject({
      due: false,
      nextDueAt: t1,
    });
    expect(evaluateTaskSchedule(schedule, t1)).toEqual({
      scheduleId: "maintenance",
      due: true,
      disabled: false,
      now: t1,
      dueAt: t1,
      nextDueAt: t2,
      idempotencyKey: `schedule:maintenance:${t1}`,
    });
    expect(emptyTaskRuntimeState().semantics.exactlyOnce).toBe(false);
  });
});
