import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { runTaskWorkerProcess, type TaskWorkerLaunchSpec } from "../src/task-worker-process.js";
import { TASK_WORKER_PROTOCOL_VERSION } from "../src/task-worker-protocol.js";

const DIGEST = `sha256:${"d".repeat(64)}`;

describe("task worker process host", () => {
  it("rejects explicit secret-bearing worker environment variables", async () => {
    await expect(
      runTaskWorkerProcess({
        launch: {
          ...nodeLaunch("setInterval(() => {}, 1000);"),
          env: { OPENAI_API_KEY: "must-not-cross-the-worker-boundary" },
        },
        start: startMessage("secret-environment"),
      }),
    ).rejects.toThrow(/secret-bearing/);
  });

  it("counts worker startup and handshake time against the wall-time budget", async () => {
    await expect(
      runTaskWorkerProcess({
        launch: nodeLaunch("setInterval(() => {}, 1000);"),
        start: { ...startMessage("startup-budget"), budget: { wallTimeMs: 20 } },
        helloTimeoutMs: 1_000,
      }),
    ).rejects.toMatchObject({ code: "budget_exceeded" });
  });

  it("terminates a worker that misses its heartbeat deadline", async () => {
    await expect(
      runTaskWorkerProcess({
        launch: nodeWorker(`setInterval(() => {}, 1000);`),
        start: startMessage("heartbeat-timeout"),
        heartbeatIntervalMs: 10,
        heartbeatTimeoutMs: 40,
      }),
    ).rejects.toMatchObject({ code: "heartbeat_timeout" });
  });

  it("rejects output after a terminal event instead of accepting late side effects", async () => {
    const afterStart = `
      emit({
        protocolVersion: 1,
        messageId: "complete:1",
        direction: "worker_to_host",
        type: "complete",
        workItemId: "awi_process_test",
        runId: "run_process_test",
        leaseId: "lease_process_test",
        fencingToken: 1,
        sequence: 0,
        emittedAt: new Date().toISOString(),
        idempotencyKey: "complete:process-test",
        artifactIds: []
      });
      emit({
        protocolVersion: 1,
        messageId: "heartbeat:late",
        direction: "worker_to_host",
        type: "heartbeat",
        workItemId: "awi_process_test",
        runId: "run_process_test",
        leaseId: "lease_process_test",
        fencingToken: 1,
        sequence: 1,
        emittedAt: new Date().toISOString()
      });
      setTimeout(() => process.exit(0), 20);
    `;

    await expect(
      runTaskWorkerProcess({
        launch: nodeWorker(afterStart),
        start: startMessage("post-terminal"),
        heartbeatTimeoutMs: 1_000,
      }),
    ).rejects.toMatchObject({ code: "protocol_error" });
  });
});

function startMessage(label: string) {
  return {
    protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
    messageId: `start:${label}`,
    direction: "host_to_worker",
    type: "start",
    workItemId: "awi_process_test",
    runId: "run_process_test",
    leaseId: "lease_process_test",
    fencingToken: 1,
    attempt: 1,
    operation: { name: "test.run", input: null },
    environmentDigest: DIGEST,
    capabilityGrantIds: [],
  };
}

function nodeWorker(afterStart: string): TaskWorkerLaunchSpec {
  const source = `
    const readline = require("node:readline");
    const emit = (message) => process.stdout.write(JSON.stringify(message) + "\\n");
    emit({
      protocolVersion: 1,
      messageId: "hello:1",
      direction: "worker_to_host",
      type: "hello",
      worker: {
        instanceId: "test-worker-1",
        backendId: "test-backend",
        backendVersion: "1",
        language: "javascript",
        languageVersion: process.version,
        environmentDigest: "${DIGEST}"
      },
      supportedProtocolVersions: [1],
      operations: ["test.run"],
      features: ["heartbeat"]
    });
    let controls = 0;
    readline.createInterface({ input: process.stdin }).on("line", () => {
      controls += 1;
      if (controls === 2) { ${afterStart} }
    });
  `;
  return nodeLaunch(source);
}

function nodeLaunch(source: string): TaskWorkerLaunchSpec {
  return {
    backendId: "test-backend",
    program: process.execPath,
    args: ["-e", source],
    cwd: tmpdir(),
    env: {},
    environmentDigest: DIGEST,
  };
}
