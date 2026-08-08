import { rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { AppAttachedTaskControlService } from "../../core/src/task-control-service.js";
import { TaskRuntimeStore } from "../../core/src/task-runtime-store.js";
import {
  runTaskWorkerProcess,
  type TaskWorkerLaunchSpec,
} from "../../core/src/task-worker-process.js";
import { TASK_WORKER_PROTOCOL_VERSION } from "../../core/src/task-worker-protocol.js";

const DIGEST = `sha256:${"b".repeat(64)}`;
const WORKER_PATH = fileURLToPath(new URL("../python/swarmx_worker.py", import.meta.url));
const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("Python task worker", () => {
  it("runs checkpoint-to-completion through the durable app-attached control service", async () => {
    const rootDir = await temporaryRoot();
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir: path.join(rootDir, "authority") }),
      ownerId: "controller:python-smoke",
    });
    const workItem = service.createWorkItem({
      id: "awi_python_echo",
      backend: "python",
      operation: "swarmx.echo",
      input: { message: "hello from Core" },
    });

    const result = await service.runWorkItem(workItem.id, {
      launch: pythonLaunch(rootDir),
    });
    const completedWorkItem = result.state.workItems[workItem.id];
    const run = completedWorkItem?.activeRunId
      ? result.state.runs[completedWorkItem.activeRunId]
      : undefined;

    expect(result.process.hello.worker).toMatchObject({
      backendId: "python",
      language: "python",
      environmentDigest: DIGEST,
    });
    expect(result.process.terminal).toMatchObject({
      type: "complete",
      result: { message: "hello from Core" },
    });
    expect(completedWorkItem?.status).toBe("succeeded");
    expect(completedWorkItem?.latestCheckpointId).toMatch(/^ckp_/u);
    expect(Object.values(result.state.checkpoints)).toEqual([
      expect.objectContaining({
        checkpointId: completedWorkItem?.latestCheckpointId,
        environmentDigest: DIGEST,
      }),
    ]);
    expect(run?.resultRef ? service.store.readJson(run.resultRef) : undefined).toEqual({
      message: "hello from Core",
    });
  }, 10_000);

  it("acknowledges a durable cancel request while a Python operation is active", async () => {
    const rootDir = await temporaryRoot();
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir: path.join(rootDir, "authority") }),
      ownerId: "controller:python-cancel",
    });
    const workItem = service.createWorkItem({
      id: "awi_python_cancel",
      backend: "python",
      operation: "swarmx.count",
      input: { steps: 100, delayMs: 25 },
    });
    const runPromise = service.runWorkItem(workItem.id, {
      launch: pythonLaunch(rootDir),
    });

    await waitFor(
      () => Object.values(service.store.state().checkpoints).length > 0,
      "Python worker did not emit its first checkpoint.",
    );
    const requested = service.cancelWorkItem(workItem.id, "Smoke-test cancellation.");
    expect(requested.workItems[workItem.id]?.cancellation?.status).toBe("requested");

    const result = await runPromise;
    expect(result.process.terminal).toMatchObject({
      type: "canceled",
      reason: "Smoke-test cancellation.",
    });
    expect(result.state.workItems[workItem.id]).toMatchObject({
      status: "canceled",
      cancellation: { status: "acknowledged", reason: "Smoke-test cancellation." },
    });
    expect(result.state.events.map((event) => event.eventType)).toEqual(
      expect.arrayContaining(["cancel_requested", "cancel_acknowledged"]),
    );
  }, 10_000);

  it("resumes the replaceable Python executor from a Core-supplied checkpoint", async () => {
    const rootDir = await temporaryRoot();
    const eventTypes: string[] = [];
    const result = await runTaskWorkerProcess({
      launch: pythonLaunch(rootDir),
      start: {
        protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
        messageId: "start:python-resume",
        direction: "host_to_worker",
        type: "start",
        workItemId: "awi_python_resume",
        runId: "run_python_resume",
        leaseId: "lease_python_resume",
        fencingToken: 1,
        attempt: 2,
        operation: { name: "swarmx.count", input: { steps: 4, delayMs: 0 } },
        environmentDigest: DIGEST,
        resumeFrom: {
          checkpointId: "ckp_previous_run_2",
          format: "swarmx.python.count",
          formatVersion: 1,
          environmentDigest: DIGEST,
          state: { nextStep: 2, totalSteps: 4 },
        },
        capabilityGrantIds: [],
      },
      onEvent: (event) => {
        eventTypes.push(event.type);
      },
    });

    expect(result.terminal).toMatchObject({
      type: "complete",
      result: { count: 4, resumedFrom: 2 },
    });
    expect(eventTypes.filter((type) => type === "checkpoint")).toHaveLength(2);
    expect(eventTypes.at(-1)).toBe("complete");

    const finalCheckpointId = "ckp_previous_run_4";
    const alreadyFinished = await runTaskWorkerProcess({
      launch: pythonLaunch(rootDir),
      start: {
        protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
        messageId: "start:python-final-checkpoint-resume",
        direction: "host_to_worker",
        type: "start",
        workItemId: "awi_python_final_resume",
        runId: "run_python_final_resume",
        leaseId: "lease_python_final_resume",
        fencingToken: 1,
        attempt: 2,
        operation: { name: "swarmx.count", input: { steps: 4, delayMs: 0 } },
        environmentDigest: DIGEST,
        resumeFrom: {
          checkpointId: finalCheckpointId,
          format: "swarmx.python.count",
          formatVersion: 1,
          environmentDigest: DIGEST,
          state: { nextStep: 4, totalSteps: 4 },
        },
        capabilityGrantIds: [],
      },
    });
    expect(alreadyFinished.terminal).toMatchObject({
      type: "complete",
      checkpointId: finalCheckpointId,
      result: { count: 4, resumedFrom: 4 },
    });
  }, 10_000);

  it("persists a human request and resumes with a bounded decision payload", async () => {
    const rootDir = await temporaryRoot();
    const service = new AppAttachedTaskControlService({
      store: new TaskRuntimeStore({ rootDir: path.join(rootDir, "authority") }),
      ownerId: "controller:python-human",
    });
    const workItem = service.createWorkItem({
      id: "awi_python_human",
      backend: "python",
      operation: "swarmx.needs_human",
      input: { prompt: "Approve the durable continuation?" },
      maxAttempts: 2,
    });

    const gated = await service.runWorkItem(workItem.id, { launch: pythonLaunch(rootDir) });
    expect(gated.state.workItems[workItem.id]?.status).toBe("needs_human");
    const approval = Object.values(gated.state.approvals)[0];
    if (!approval?.requestRef) throw new Error("Expected a durable human request payload.");
    expect(service.store.readJson(approval.requestRef)).toMatchObject({
      prompt: "Approve the durable continuation?",
      options: expect.arrayContaining([expect.objectContaining({ optionId: "continue" })]),
    });

    const queued = service.decideApproval({
      approvalId: approval.approvalId,
      status: "approved",
      decidedBy: "user:test",
      response: { selectedOptionId: "continue" },
    });
    expect(queued.workItems[workItem.id]?.status).toBe("queued");

    const resumed = await service.runWorkItem(workItem.id, { launch: pythonLaunch(rootDir) });
    expect(resumed.state.workItems[workItem.id]?.status).toBe("succeeded");
    expect(resumed.process.terminal).toMatchObject({
      type: "complete",
      result: {
        approvalId: approval.approvalId,
        status: "approved",
        response: { selectedOptionId: "continue" },
      },
    });
  }, 10_000);
});

function pythonLaunch(cwd: string): TaskWorkerLaunchSpec {
  return {
    backendId: "python",
    program: process.env.SWARMX_TEST_PYTHON ?? "python3",
    args: ["-I", "-B", "-u", WORKER_PATH, "--environment-digest", DIGEST],
    cwd,
    env: {
      PATH: process.env.PATH ?? "",
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONUTF8: "1",
    },
    environmentDigest: DIGEST,
  };
}

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-python-worker-"));
  temporaryRoots.push(root);
  return root;
}

async function waitFor(predicate: () => boolean, failureMessage: string): Promise<void> {
  const deadline = Date.now() + 5_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(failureMessage);
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}
