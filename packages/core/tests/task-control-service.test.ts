import { existsSync, readFileSync, rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AppAttachedTaskControlService } from "../src/task-control-service.js";
import {
  createTaskRuntimeEvent,
  type TaskCheckpoint,
  type TaskLease,
  TaskRunSchema,
} from "../src/task-runtime.js";
import { TaskRuntimeStore } from "../src/task-runtime-store.js";
import type { TaskWorkerCapabilityCallMessage } from "../src/task-worker-protocol.js";

const DIGEST = `sha256:${"a".repeat(64)}`;
const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("AppAttachedTaskControlService", () => {
  it("keeps WorkItem authority independent while multiple Sessions link and unlink", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:test",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_session_independent",
      backend: "python",
      operation: "swarmx.echo",
      input: { message: "durable" },
      creatorSessionId: "session_creator",
    });

    nowMs += 1_000;
    let state = service.linkSession(workItem.id, "session_observer");
    const creatorLink = Object.values(state.sessionLinks).find(
      (link) => link.sessionId === "session_creator",
    );
    const observerLink = Object.values(state.sessionLinks).find(
      (link) => link.sessionId === "session_observer",
    );
    expect(creatorLink?.role).toBe("creator");
    expect(observerLink?.role).toBe("observer");
    expect(state.workItems[workItem.id]?.sessionLinkIds).toHaveLength(2);

    nowMs += 1_000;
    if (!creatorLink) throw new Error("Expected a creator Session link.");
    state = service.unlinkSession(workItem.id, creatorLink.linkId);
    expect(state.workItems[workItem.id]).toMatchObject({ status: "queued", runIds: [] });
    expect(state.workItems[workItem.id]?.sessionLinkIds).toEqual([observerLink?.linkId]);
    expect(state.sessionLinks[creatorLink.linkId]?.unlinkedAt).toBeDefined();

    const reopened = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:reopened",
      now: () => new Date(nowMs),
    }).store.state();
    expect(reopened.workItems[workItem.id]).toEqual(state.workItems[workItem.id]);
    const reopenedObserver = reopened.sessionLinks[observerLink?.linkId ?? ""];
    expect(reopenedObserver).toMatchObject({ sessionId: "session_observer" });
    expect(reopenedObserver).not.toHaveProperty("unlinkedAt");
  });

  it("recovers an expired lease and schedules a checkpoint-backed retry once", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:recovery",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_expired_retry",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 4 },
      maxAttempts: 2,
    });
    const seeded = seedActiveRun(service.store, workItem.id, nowMs, { checkpoint: true });

    nowMs += 20_000;
    const recovered = service.recoverOnStartup();
    expect(recovered.recoveredLeaseIds).toEqual([seeded.lease.leaseId]);
    expect(recovered.state.runs[seeded.runId]).toMatchObject({
      status: "interrupted",
      latestCheckpointId: seeded.checkpoint?.checkpointId,
      failure: { code: "LEASE_EXPIRED", retryable: true },
    });
    expect(recovered.state.workItems[workItem.id]).toMatchObject({
      status: "queued",
      activeRunId: undefined,
      latestCheckpointId: seeded.checkpoint?.checkpointId,
      retry: { attemptsStarted: 1, maxAttempts: 2 },
    });

    const secondRecovery = service.recoverOnStartup();
    expect(secondRecovery.recoveredLeaseIds).toEqual([]);
    expect(secondRecovery.state.events).toHaveLength(recovered.state.events.length);
  });

  it("recovers a retryable failure when the retry event was lost after a crash", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:failure-recovery",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_failed_retry_recovery",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 4 },
      maxAttempts: 2,
    });
    const seeded = seedActiveRun(service.store, workItem.id, nowMs);
    const failedAt = timestamp(nowMs + 4_000);
    service.store.append(
      createTaskRuntimeEvent({
        eventType: "run_failed",
        timestamp: failedAt,
        source: "test",
        idempotencyKey: `run:failed:${seeded.runId}`,
        workItemId: workItem.id,
        runId: seeded.runId,
        payload: {
          claim: {
            leaseId: seeded.lease.leaseId,
            fencingToken: seeded.lease.fencingToken,
          },
          failure: {
            occurredAt: failedAt,
            message: "Retry event was not persisted before the host crash.",
            retryable: true,
          },
        },
      }),
    );

    nowMs += 5_000;
    const recovered = service.recoverOnStartup();
    expect(recovered.recoveredLeaseIds).toEqual([]);
    expect(recovered.state.workItems[workItem.id]).toMatchObject({
      status: "queued",
      activeRunId: undefined,
      retry: { attemptsStarted: 1, maxAttempts: 2 },
    });
    expect(
      recovered.state.events.filter((event) => event.eventType === "retry_scheduled"),
    ).toHaveLength(1);
  });

  it("recovers an approved human decision when its resume event was lost", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:approval-recovery",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_approval_retry_recovery",
      backend: "python",
      operation: "swarmx.count",
      input: { prompt: "Continue?" },
      maxAttempts: 2,
    });
    const seeded = seedActiveRun(service.store, workItem.id, nowMs);
    const approvalId = "apr_recover_decision";
    const requestedAt = timestamp(nowMs + 3_000);
    service.store.append([
      createTaskRuntimeEvent({
        eventType: "approval_recorded",
        timestamp: requestedAt,
        source: "test",
        idempotencyKey: `approval:request:${approvalId}`,
        workItemId: workItem.id,
        payload: {
          approval: {
            approvalId,
            workItemId: workItem.id,
            runId: seeded.runId,
            kind: "approval",
            status: "requested",
            requestedAt,
            requestedBy: "test.worker",
            reason: "Continue?",
          },
        },
      }),
      createTaskRuntimeEvent({
        eventType: "needs_human",
        timestamp: requestedAt,
        source: "test",
        idempotencyKey: `needs-human:${seeded.runId}`,
        workItemId: workItem.id,
        runId: seeded.runId,
        payload: {
          claim: {
            leaseId: seeded.lease.leaseId,
            fencingToken: seeded.lease.fencingToken,
          },
          requestedAt,
          reason: "Continue?",
          approvalId,
        },
      }),
    ]);
    const decidedAt = timestamp(nowMs + 4_000);
    service.store.append(
      createTaskRuntimeEvent({
        eventType: "approval_recorded",
        timestamp: decidedAt,
        source: "test",
        idempotencyKey: `approval:decision:${approvalId}:approved`,
        workItemId: workItem.id,
        payload: {
          approval: {
            approvalId,
            workItemId: workItem.id,
            runId: seeded.runId,
            kind: "approval",
            status: "approved",
            requestedAt,
            requestedBy: "test.worker",
            decidedAt,
            decidedBy: "test.user",
            reason: "Continue?",
          },
        },
      }),
    );

    nowMs += 5_000;
    const recovered = service.recoverOnStartup();
    expect(recovered.state.workItems[workItem.id]).toMatchObject({
      status: "queued",
      activeRunId: undefined,
      retry: { attemptsStarted: 1, maxAttempts: 2 },
    });
    const retry = recovered.state.events.find((event) => event.eventType === "retry_scheduled");
    expect(retry?.eventType === "retry_scheduled" ? retry.payload.approvalId : undefined).toBe(
      approvalId,
    );
  });

  it("blocks work durably when a pending approval is rejected", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:approval-rejection",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_rejected_approval",
      backend: "python",
      operation: "swarmx.count",
      input: { prompt: "Continue?" },
      maxAttempts: 2,
    });
    const seeded = seedActiveRun(service.store, workItem.id, nowMs);
    const approvalId = "apr_rejected_decision";
    const requestedAt = timestamp(nowMs + 3_000);
    service.store.append([
      createTaskRuntimeEvent({
        eventType: "approval_recorded",
        timestamp: requestedAt,
        source: "test",
        idempotencyKey: `approval:request:${approvalId}`,
        workItemId: workItem.id,
        payload: {
          approval: {
            approvalId,
            workItemId: workItem.id,
            runId: seeded.runId,
            kind: "approval",
            status: "requested",
            requestedAt,
            requestedBy: "test.worker",
            reason: "Continue?",
          },
        },
      }),
      createTaskRuntimeEvent({
        eventType: "needs_human",
        timestamp: requestedAt,
        source: "test",
        idempotencyKey: `needs-human:${seeded.runId}`,
        workItemId: workItem.id,
        runId: seeded.runId,
        payload: {
          claim: {
            leaseId: seeded.lease.leaseId,
            fencingToken: seeded.lease.fencingToken,
          },
          requestedAt,
          reason: "Continue?",
          approvalId,
        },
      }),
    ]);

    nowMs += 4_000;
    const state = service.decideApproval({
      approvalId,
      status: "rejected",
      decidedBy: "test.user",
      reason: "Do not continue.",
    });
    expect(state.approvals[approvalId]).toMatchObject({
      status: "rejected",
      decidedBy: "test.user",
    });
    expect(state.workItems[workItem.id]).toMatchObject({
      status: "blocked",
      blockedReason: "Do not continue.",
    });
    expect(state.events.filter((event) => event.eventType === "retry_scheduled")).toHaveLength(0);

    const recovered = service.recoverOnStartup();
    expect(recovered.state.workItems[workItem.id]?.status).toBe("blocked");
    expect(
      recovered.state.events.filter((event) => event.eventType === "retry_scheduled"),
    ).toHaveLength(0);
  });

  it.each([
    {
      label: "checkpoint identity",
      suffix: "identity",
      checkpointPayloadId: "ckp_payload_from_other_run",
      expected: /payload identifies/i,
    },
    {
      label: "checkpoint environment",
      suffix: "environment",
      checkpointPayloadEnvironmentDigest: `sha256:${"b".repeat(64)}`,
      expected: /belongs to environment/i,
    },
    {
      label: "checkpoint checksum",
      suffix: "checksum",
      checkpointMetadataChecksum: `sha256:${"b".repeat(64)}`,
      expected: /metadata checksum does not match/i,
    },
    {
      label: "artifact-backed representation",
      suffix: "artifact",
      checkpointPayloadArtifact: true,
      expected: /artifact-backed execution checkpoints/i,
    },
  ])("rejects a persisted resume blob with mismatched $label", async (testCase) => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: `controller:resume-${testCase.suffix}`,
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: `awi_resume_${testCase.suffix}`,
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 4 },
      maxAttempts: 2,
    });
    seedActiveRun(service.store, workItem.id, nowMs, {
      checkpoint: true,
      checkpointPayloadId: testCase.checkpointPayloadId,
      checkpointPayloadEnvironmentDigest: testCase.checkpointPayloadEnvironmentDigest,
      checkpointMetadataChecksum: testCase.checkpointMetadataChecksum,
      checkpointPayloadArtifact: testCase.checkpointPayloadArtifact,
    });
    nowMs += 20_000;
    service.recoverOnStartup();

    await expect(
      service.runWorkItem(workItem.id, {
        launch: {
          backendId: "python",
          program: "must-not-run",
          args: [],
          cwd: rootDir,
          env: {},
          environmentDigest: DIGEST,
        },
      }),
    ).rejects.toThrow(testCase.expected);
  });

  it("replays committed capability outcomes without leaking across WorkItems", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    let gatewayCalls = 0;
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:capability-replay",
      now: () => new Date(nowMs),
      capabilityGateway: {
        invoke: async ({ call }) => {
          gatewayCalls += 1;
          return {
            status: "succeeded" as const,
            value: { durableValue: "replayed" },
            artifactIds: [],
            receipt: {
              receiptId: "rcpt_gateway_original",
              idempotencyKey: call.idempotencyKey,
              externalRef: "gateway:receipt:private-to-this-work-item",
            },
          };
        },
      },
    });
    const first = service.createWorkItem({
      id: "awi_capability_first",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 1 },
    });
    const firstRun = seedActiveRun(service.store, first.id, nowMs);
    nowMs += 3_000;
    const call = capabilityCall(first.id, firstRun, "effect:shared-key");
    const invoke = service as unknown as {
      invokeCapability(
        workItemId: string,
        call: TaskWorkerCapabilityCallMessage,
        lease: TaskLease,
      ): Promise<unknown>;
    };
    const initial = await invoke.invokeCapability(first.id, call, firstRun.lease);
    const replayed = await invoke.invokeCapability(first.id, call, firstRun.lease);
    expect(initial).toEqual(replayed);
    expect(initial).toMatchObject({
      status: "succeeded",
      value: { durableValue: "replayed" },
      receipt: { externalRef: "gateway:receipt:private-to-this-work-item" },
    });
    expect(gatewayCalls).toBe(1);

    const second = service.createWorkItem({
      id: "awi_capability_second",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 1 },
    });
    const secondRun = seedActiveRun(service.store, second.id, nowMs);
    const collision = await invoke.invokeCapability(
      second.id,
      capabilityCall(second.id, secondRun, call.idempotencyKey),
      secondRun.lease,
    );
    expect(collision).toEqual({
      status: "failed",
      error: {
        code: "idempotency_scope_collision",
        message: "The capability idempotency key is already bound to different work.",
        retryable: false,
      },
    });
    expect(JSON.stringify(collision)).not.toContain("gateway:receipt");
    expect(gatewayCalls).toBe(1);
  });

  it("persists an already-aborted external signal before canceling worker startup", async () => {
    const rootDir = await temporaryRoot();
    const service = cancellationService(rootDir, "pre-aborted");
    const workItem = service.createWorkItem({
      id: "awi_pre_aborted_signal",
      backend: "test-backend",
      operation: "test.run",
      input: null,
    });
    const external = new AbortController();
    external.abort(new Error("Canceled before worker startup."));
    const worker = cancellationWorker(rootDir, "acknowledge");

    let failure: unknown;
    try {
      await service.runWorkItem(workItem.id, { launch: worker.launch, signal: external.signal });
    } catch (error) {
      failure = error;
    }

    expect(failure).toMatchObject({ code: "canceled" });
    const state = service.store.state();
    const requestIndex = state.events.findIndex((event) => event.eventType === "cancel_requested");
    const acknowledgementIndex = state.events.findIndex(
      (event) => event.eventType === "cancel_acknowledged",
    );
    expect(requestIndex).toBeGreaterThanOrEqual(0);
    expect(acknowledgementIndex).toBeGreaterThan(requestIndex);
    expect(state.events[requestIndex]).toMatchObject({
      eventType: "cancel_requested",
      payload: { reason: "Canceled before worker startup." },
    });
  });

  it("persists a running external abort before the worker observes its cancel signal", async () => {
    const rootDir = await temporaryRoot();
    const service = cancellationService(rootDir, "running-abort");
    const workItem = service.createWorkItem({
      id: "awi_running_abort_signal",
      backend: "test-backend",
      operation: "test.run",
      input: null,
    });
    const external = new AbortController();
    const worker = cancellationWorker(rootDir, "acknowledge");
    const running = service.runWorkItem(workItem.id, {
      launch: worker.launch,
      signal: external.signal,
    });
    await waitForFile(worker.readyPath);

    external.abort(new Error("Canceled while worker was running."));
    const duplicate = service.cancelWorkItem(workItem.id, "Duplicate cancellation request.");
    expect(duplicate.events.filter((event) => event.eventType === "cancel_requested")).toHaveLength(
      1,
    );

    const result = await running;
    expect(result.process.terminal).toMatchObject({ type: "canceled" });
    expect(readFileSync(worker.observationPath, "utf8")).toBe("visible");
    expect(
      result.state.events.filter((event) => event.eventType === "cancel_requested"),
    ).toHaveLength(1);
    expect(result.state.workItems[workItem.id]?.status).toBe("canceled");
  });

  it("keeps cancellation durable when the worker crashes after receiving the signal", async () => {
    const rootDir = await temporaryRoot();
    const service = cancellationService(rootDir, "crashing-abort");
    const workItem = service.createWorkItem({
      id: "awi_crashing_abort_signal",
      backend: "test-backend",
      operation: "test.run",
      input: null,
    });
    const external = new AbortController();
    const worker = cancellationWorker(rootDir, "crash");
    const running = service.runWorkItem(workItem.id, {
      launch: worker.launch,
      signal: external.signal,
    });
    await waitForFile(worker.readyPath);

    external.abort(new Error("Cancel before simulated worker crash."));
    await expect(running).rejects.toMatchObject({ code: "canceled" });

    expect(readFileSync(worker.observationPath, "utf8")).toBe("visible");
    const state = service.store.state();
    const requestIndex = state.events.findIndex((event) => event.eventType === "cancel_requested");
    const acknowledgementIndex = state.events.findIndex(
      (event) => event.eventType === "cancel_acknowledged",
    );
    expect(requestIndex).toBeGreaterThanOrEqual(0);
    expect(acknowledgementIndex).toBeGreaterThan(requestIndex);
    expect(state.workItems[workItem.id]?.status).toBe("canceled");
  });

  it("finishes a requested cancellation when its worker lease expires without retrying", async () => {
    const rootDir = await temporaryRoot();
    let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir }),
      ownerId: "controller:cancel-recovery",
      now: () => new Date(nowMs),
    });
    const workItem = service.createWorkItem({
      id: "awi_expired_cancel",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 100 },
      maxAttempts: 2,
    });
    const seeded = seedActiveRun(service.store, workItem.id, nowMs);
    const requestedAt = timestamp(nowMs + 4_000);
    service.store.append(
      createTaskRuntimeEvent({
        eventType: "cancel_requested",
        timestamp: requestedAt,
        source: "test",
        idempotencyKey: `cancel:request:${seeded.runId}`,
        workItemId: workItem.id,
        runId: seeded.runId,
        payload: { requestedAt, reason: "Stop after controller restart." },
      }),
    );

    nowMs += 20_000;
    const recovered = service.recoverOnStartup();
    expect(recovered.state.runs[seeded.runId]).toMatchObject({
      status: "canceled",
      cancellation: { status: "acknowledged" },
    });
    expect(recovered.state.workItems[workItem.id]).toMatchObject({
      status: "canceled",
      retry: { attemptsStarted: 1, maxAttempts: 2 },
    });
    expect(
      recovered.state.events.filter((event) => event.eventType === "retry_scheduled"),
    ).toHaveLength(0);
  });
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-task-control-"));
  temporaryRoots.push(root);
  return root;
}

function cancellationService(rootDir: string, label: string): AppAttachedTaskControlService {
  let nowMs = Date.parse("2026-08-05T08:00:00.000Z");
  return new AppAttachedTaskControlService({
    store: new TaskRuntimeStore({ rootDir }),
    ownerId: `controller:${label}`,
    leaseDurationMs: 3_000,
    now: () => new Date(nowMs++),
  });
}

function cancellationWorker(rootDir: string, mode: "acknowledge" | "crash") {
  const readyPath = path.join(rootDir, `worker-${mode}-ready`);
  const observationPath = path.join(rootDir, `worker-${mode}-observation`);
  const eventLogPath = path.join(rootDir, "events.jsonl");
  const crash = mode === "crash" ? "process.exit(17); return;" : "";
  const source = `
    const fs = require("node:fs");
    const readline = require("node:readline");
    const emit = (message) => process.stdout.write(JSON.stringify(message) + "\\n");
    emit({
      protocolVersion: 1,
      messageId: "hello:cancellation-test",
      direction: "worker_to_host",
      type: "hello",
      worker: {
        instanceId: "cancellation-test-worker",
        backendId: "test-backend",
        backendVersion: "1",
        language: "javascript",
        languageVersion: process.version,
        environmentDigest: "${DIGEST}"
      },
      supportedProtocolVersions: [1],
      operations: ["test.run"],
      features: ["heartbeat", "cancel"]
    });
    readline.createInterface({ input: process.stdin }).on("line", (line) => {
      const control = JSON.parse(line);
      if (control.type === "start") {
        fs.writeFileSync(${JSON.stringify(readyPath)}, "ready");
        emit({
          protocolVersion: 1,
          messageId: "heartbeat:cancellation-test",
          direction: "worker_to_host",
          type: "heartbeat",
          workItemId: control.workItemId,
          runId: control.runId,
          leaseId: control.leaseId,
          fencingToken: control.fencingToken,
          sequence: 0,
          emittedAt: new Date().toISOString()
        });
        return;
      }
      if (control.type !== "cancel") return;
      const eventLog = fs.readFileSync(${JSON.stringify(eventLogPath)}, "utf8");
      const visible = eventLog.includes('"eventType":"cancel_requested"');
      fs.writeFileSync(${JSON.stringify(observationPath)}, visible ? "visible" : "missing");
      ${crash}
      emit({
        protocolVersion: 1,
        messageId: "canceled:cancellation-test",
        direction: "worker_to_host",
        type: "canceled",
        workItemId: control.workItemId,
        runId: control.runId,
        leaseId: control.leaseId,
        fencingToken: control.fencingToken,
        sequence: 1,
        emittedAt: new Date().toISOString(),
        idempotencyKey: "canceled:" + control.runId,
        mode: "cancel",
        reason: visible ? "Cancellation event was durable." : "Cancellation event was missing."
      });
    });
  `;
  return {
    readyPath,
    observationPath,
    launch: {
      backendId: "test-backend",
      program: process.execPath,
      args: ["-e", source],
      cwd: rootDir,
      env: {},
      environmentDigest: DIGEST,
    },
  };
}

async function waitForFile(filePath: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!existsSync(filePath)) {
    if (Date.now() >= deadline) throw new Error(`Timed out waiting for ${filePath}.`);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function seedActiveRun(
  store: TaskRuntimeStore,
  workItemId: string,
  baseMs: number,
  options: {
    checkpoint?: boolean;
    checkpointPayloadId?: string;
    checkpointPayloadEnvironmentDigest?: string;
    checkpointMetadataChecksum?: string;
    checkpointPayloadArtifact?: boolean;
  } = {},
): { runId: string; lease: TaskLease; checkpoint?: TaskCheckpoint } {
  const runId = `run_seed_${workItemId}`;
  const lease: TaskLease = {
    leaseId: `lease_seed_${workItemId}`,
    workItemId,
    runId,
    workerId: "controller:seed",
    fencingToken: 1,
    acquiredAt: timestamp(baseMs + 1_000),
    heartbeatAt: timestamp(baseMs + 1_000),
    expiresAt: timestamp(baseMs + 10_000),
  };
  const run = TaskRunSchema.parse({
    runId,
    workItemId,
    executor: { backend: "python", operation: "swarmx.count" },
    status: "created",
    attempt: 1,
    createdAt: timestamp(baseMs + 1_000),
    environmentDigest: DIGEST,
  });
  store.append([
    createTaskRuntimeEvent({
      eventType: "run_created",
      timestamp: timestamp(baseMs + 1_000),
      source: "test",
      idempotencyKey: `run:create:${runId}`,
      workItemId,
      runId,
      payload: { run },
    }),
    createTaskRuntimeEvent({
      eventType: "lease_acquired",
      timestamp: timestamp(baseMs + 1_000),
      source: "test",
      idempotencyKey: `lease:acquire:${lease.leaseId}`,
      workItemId,
      runId,
      payload: { lease },
    }),
    createTaskRuntimeEvent({
      eventType: "run_started",
      timestamp: timestamp(baseMs + 2_000),
      source: "test",
      idempotencyKey: `run:start:${runId}`,
      workItemId,
      runId,
      payload: {
        claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
        startedAt: timestamp(baseMs + 2_000),
      },
    }),
  ]);

  let checkpoint: TaskCheckpoint | undefined;
  if (options.checkpoint) {
    const checkpointId = `ckp_seed_${workItemId}`;
    const workerCheckpoint = {
      checkpointId: options.checkpointPayloadId ?? checkpointId,
      format: "swarmx.python.count",
      formatVersion: 1,
      environmentDigest: options.checkpointPayloadEnvironmentDigest ?? DIGEST,
      ...(options.checkpointPayloadArtifact
        ? {
            artifact: {
              artifactId: `art_checkpoint_${workItemId}`,
              kind: "checkpoint",
              relativePath: "checkpoints/resume.json",
              sha256: DIGEST,
              sizeBytes: 1,
            },
          }
        : { state: { nextStep: 2, totalSteps: 4 } }),
    };
    const blob = store.putJson(workerCheckpoint);
    checkpoint = {
      checkpointId,
      workItemId,
      runId,
      sequence: 2,
      createdAt: timestamp(baseMs + 5_000),
      resumeRef: blob.ref,
      checksum: options.checkpointMetadataChecksum ?? `sha256:${blob.sha256}`,
      environmentDigest: DIGEST,
      artifactIds: [],
    };
    store.append(
      createTaskRuntimeEvent({
        eventType: "checkpoint_recorded",
        timestamp: checkpoint.createdAt,
        source: "test",
        idempotencyKey: `checkpoint:${checkpoint.checkpointId}`,
        workItemId,
        runId,
        payload: {
          claim: { leaseId: lease.leaseId, fencingToken: lease.fencingToken },
          checkpoint,
        },
      }),
    );
  }
  return { runId, lease, checkpoint };
}

function timestamp(milliseconds: number): string {
  return new Date(milliseconds).toISOString();
}

function capabilityCall(
  workItemId: string,
  seeded: { runId: string; lease: TaskLease },
  idempotencyKey: string,
): TaskWorkerCapabilityCallMessage {
  return {
    protocolVersion: 1,
    messageId: `capability:${workItemId}`,
    direction: "worker_to_host",
    type: "capability_call",
    workItemId,
    runId: seeded.runId,
    leaseId: seeded.lease.leaseId,
    fencingToken: seeded.lease.fencingToken,
    sequence: 0,
    emittedAt: seeded.lease.heartbeatAt,
    callId: `call:${workItemId}`,
    grantId: "grant:test",
    capabilityId: "provider.request",
    operation: "responses.create",
    idempotencyKey,
    arguments: { promptRef: "prompt:durable" },
  };
}
