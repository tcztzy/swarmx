import { rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  ensureTaskSupervisorToken,
  TaskSupervisorClient,
  TaskSupervisorServer,
  taskSupervisorPaths,
} from "../src/task-supervisor.js";
import type { TaskWorkerLaunchSpec } from "../src/task-worker-process.js";

const DIGEST = `sha256:${"e".repeat(64)}`;
const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("authenticated task supervisor", () => {
  it("keeps an accepted WorkItem running after the requesting client disconnects", async () => {
    const rootDir = await temporaryRoot();
    const paths = taskSupervisorPaths(rootDir);
    const token = ensureTaskSupervisorToken(rootDir);
    const server = new TaskSupervisorServer({ rootDir, socketPath: paths.socketPath, token });
    await server.listen();

    try {
      const firstClient = new TaskSupervisorClient({ socketPath: paths.socketPath, token });
      const created = await firstClient.request({
        operation: "create",
        workItem: {
          id: "awi_supervisor_detached",
          backend: "test-backend",
          operation: "test.run",
          input: { durable: true },
        },
      });
      expect(created).toMatchObject({ operation: "create", workItem: { status: "queued" } });

      await expect(
        firstClient.request({
          operation: "run",
          workItemId: "awi_supervisor_detached",
          launch: nodeWorker(),
          grants: [],
        }),
      ).resolves.toMatchObject({ operation: "run", accepted: true });

      const reattachedClient = new TaskSupervisorClient({ socketPath: paths.socketPath, token });
      await expectCompleted(reattachedClient, "awi_supervisor_detached");
    } finally {
      await server.close();
    }
  });

  it("rejects unauthenticated clients and unsafe launch input at the protocol boundary", async () => {
    const rootDir = await temporaryRoot();
    const paths = taskSupervisorPaths(rootDir);
    const token = ensureTaskSupervisorToken(rootDir);
    const server = new TaskSupervisorServer({ rootDir, socketPath: paths.socketPath, token });
    await server.listen();

    try {
      await expect(
        new TaskSupervisorClient({ socketPath: paths.socketPath, token: "f".repeat(64) }).request({
          operation: "ping",
        }),
      ).rejects.toThrow(/authentication/i);
      const client = new TaskSupervisorClient({ socketPath: paths.socketPath, token });
      await expect(
        client.request({
          operation: "run",
          workItemId: "awi_missing",
          launch: { ...nodeWorker(), env: { OPENAI_API_KEY: "must-not-cross" } },
          grants: [],
        }),
      ).rejects.toThrow(/secret-bearing/);
    } finally {
      await server.close();
    }
  });
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-task-supervisor-"));
  temporaryRoots.push(root);
  return root;
}

async function expectCompleted(client: TaskSupervisorClient, workItemId: string): Promise<void> {
  const deadline = Date.now() + 3_000;
  let lastStatus = "missing";
  while (Date.now() < deadline) {
    const response = await client.request({ operation: "list" });
    if (response.operation !== "list") throw new Error("Expected a list response.");
    const workItem = response.workItems.find((candidate) => candidate.id === workItemId);
    if (workItem?.status === "succeeded") return;
    lastStatus = workItem ? JSON.stringify(workItem) : "missing";
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`Work item ${workItemId} did not complete: ${lastStatus}`);
}

function nodeWorker(): TaskWorkerLaunchSpec {
  const source = `
    const readline = require("node:readline");
    const emit = (message) => process.stdout.write(JSON.stringify(message) + "\\n");
    emit({
      protocolVersion: 1,
      messageId: "hello:1",
      direction: "worker_to_host",
      type: "hello",
      worker: {
        instanceId: "test-supervisor-worker",
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
    readline.createInterface({ input: process.stdin }).on("line", (line) => {
      const message = JSON.parse(line);
      if (message.type !== "start") return;
      setTimeout(() => {
        emit({
          protocolVersion: 1,
          messageId: "complete:1",
          direction: "worker_to_host",
          type: "complete",
          workItemId: message.workItemId,
          runId: message.runId,
          leaseId: message.leaseId,
          fencingToken: message.fencingToken,
          sequence: 0,
          emittedAt: new Date().toISOString(),
          idempotencyKey: "complete:supervisor-test",
          artifactIds: []
        });
        setTimeout(() => process.exit(0), 20);
      }, 80);
    });
  `;
  return {
    backendId: "test-backend",
    program: process.execPath,
    args: ["-e", source],
    cwd: tmpdir(),
    env: {},
    environmentDigest: DIGEST,
  };
}
