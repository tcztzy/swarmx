import { describe, expect, it } from "vitest";
import {
  assertTaskWorkerCapabilityCallAllowed,
  parseTaskWorkerControlLine,
  parseTaskWorkerEventLine,
  parseTaskWorkerJsonl,
  parseTaskWorkerMessage,
  serializeTaskWorkerMessage,
  TASK_WORKER_MAX_JSONL_LINE_BYTES,
  TASK_WORKER_PROTOCOL_VERSION,
  TaskWorkerArtifactMessageSchema,
  TaskWorkerCheckpointMessageSchema,
  TaskWorkerControlMessageSchema,
  TaskWorkerEventMessageSchema,
  TaskWorkerProtocolMessageSchema,
} from "../src/task-worker-protocol.js";

const NOW = "2026-08-05T08:00:00.000Z";
const DIGEST = `sha256:${"a".repeat(64)}`;

const runFields = {
  workItemId: "awi_1",
  runId: "run_1",
  leaseId: "lease_1",
  fencingToken: 7,
};

const eventFields = {
  protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
  direction: "worker_to_host" as const,
  ...runFields,
  sequence: 1,
  emittedAt: NOW,
};

describe("task worker protocol", () => {
  it("round-trips strict worker events and host control messages as JSONL", () => {
    const hello = {
      protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
      messageId: "msg_hello",
      direction: "worker_to_host" as const,
      type: "hello" as const,
      worker: {
        instanceId: "python_worker_1",
        backendId: "python",
        backendVersion: "1.0.0",
        language: "python",
        languageVersion: "3.12.4",
        environmentDigest: DIGEST,
      },
      supportedProtocolVersions: [TASK_WORKER_PROTOCOL_VERSION],
      operations: ["smoke.echo"],
      features: ["heartbeat", "checkpoint", "cancel", "capability_gateway"],
    };
    const helloLine = serializeTaskWorkerMessage(hello);
    expect(helloLine.endsWith("\n")).toBe(true);
    expect(parseTaskWorkerEventLine(helloLine.slice(0, -1))).toEqual(hello);
    expect(TaskWorkerEventMessageSchema.parse(hello)).toEqual(hello);

    const start = {
      protocolVersion: TASK_WORKER_PROTOCOL_VERSION,
      messageId: "msg_start",
      direction: "host_to_worker" as const,
      type: "start" as const,
      ...runFields,
      attempt: 2,
      operation: { name: "smoke.echo", input: { value: "hello" } },
      environmentDigest: DIGEST,
      resumeFrom: {
        checkpointId: "ckp_1",
        format: "json",
        formatVersion: 1,
        environmentDigest: DIGEST,
        state: { offset: 4 },
      },
      capabilityGrantIds: ["grant_model"],
      budget: { wallTimeMs: 30_000, outputBytes: 1_024, capabilityCalls: { "cap:op": 1 } },
    };
    const startLine = serializeTaskWorkerMessage(start);
    expect(parseTaskWorkerControlLine(startLine.slice(0, -1))).toEqual(start);
    expect(TaskWorkerControlMessageSchema.parse(start)).toEqual(start);
  });

  it("accepts every required worker lifecycle event with fencing data", () => {
    const artifact = {
      artifactId: "art_1",
      kind: "result",
      relativePath: "results/output.json",
      sha256: DIGEST,
      sizeBytes: 12,
      mediaType: "application/json",
    };
    const messages = [
      { ...eventFields, messageId: "msg_heartbeat", type: "heartbeat" },
      {
        ...eventFields,
        messageId: "msg_progress",
        type: "progress",
        sequence: 2,
        message: "Running",
        fraction: 0.5,
        counters: { records: 2 },
      },
      {
        ...eventFields,
        messageId: "msg_checkpoint",
        type: "checkpoint",
        sequence: 3,
        idempotencyKey: "checkpoint:1",
        checkpoint: {
          checkpointId: "ckp_1",
          format: "json",
          formatVersion: 1,
          environmentDigest: DIGEST,
          state: { offset: 2 },
        },
      },
      {
        ...eventFields,
        messageId: "msg_artifact",
        type: "artifact",
        sequence: 4,
        idempotencyKey: "artifact:1",
        artifact,
      },
      {
        ...eventFields,
        messageId: "msg_human",
        type: "needs_human",
        sequence: 5,
        idempotencyKey: "human:1",
        request: {
          requestId: "approval_1",
          kind: "approval",
          prompt: "Approve the bounded capability call?",
          options: [{ optionId: "approve", label: "Approve" }],
          checkpointId: "ckp_1",
        },
      },
      {
        ...eventFields,
        messageId: "msg_complete",
        type: "complete",
        sequence: 6,
        idempotencyKey: "complete:1",
        summary: "Completed.",
        result: { count: 2 },
        artifactIds: ["art_1"],
        checkpointId: "ckp_1",
      },
      {
        ...eventFields,
        messageId: "msg_fail",
        type: "fail",
        sequence: 6,
        idempotencyKey: "fail:1",
        failure: { code: "worker_error", message: "Failed safely.", retryable: true },
        checkpointId: "ckp_1",
      },
      {
        ...eventFields,
        messageId: "msg_canceled",
        type: "canceled",
        sequence: 6,
        idempotencyKey: "canceled:1",
        mode: "cancel",
        reason: "Requested by the control plane.",
        checkpointId: "ckp_1",
      },
      {
        ...eventFields,
        messageId: "msg_capability",
        type: "capability_call",
        sequence: 6,
        callId: "call_1",
        grantId: "grant_model",
        capabilityId: "model_gateway",
        operation: "invoke",
        idempotencyKey: "effect:stable-1",
        arguments: { modelRef: "model_1", promptRef: "artifact_prompt" },
      },
    ];

    for (const message of messages) {
      expect(parseTaskWorkerMessage(message, "worker_to_host")).toEqual(message);
    }
  });

  it("accepts capabilities, cancel, and capability results only from the host", () => {
    const controls = [
      {
        protocolVersion: 1,
        messageId: "msg_capabilities",
        direction: "host_to_worker",
        type: "capabilities",
        helloMessageId: "msg_hello",
        selectedProtocolVersion: 1,
        enabledFeatures: ["heartbeat", "capability_gateway"],
        grants: [
          {
            grantId: "grant_model",
            capabilityId: "model_gateway",
            operations: ["invoke"],
            expiresAt: "2026-08-05T09:00:00.000Z",
          },
        ],
        limits: {
          maxJsonlLineBytes: TASK_WORKER_MAX_JSONL_LINE_BYTES,
          heartbeatIntervalMs: 1_000,
          heartbeatTimeoutMs: 5_000,
          maxArtifactBytes: 10_000_000,
        },
      },
      {
        protocolVersion: 1,
        messageId: "msg_cancel",
        direction: "host_to_worker",
        type: "cancel",
        ...runFields,
        requestedAt: NOW,
        mode: "cancel",
        reason: "User requested cancellation.",
        graceMs: 500,
      },
      {
        protocolVersion: 1,
        messageId: "msg_capability_result",
        direction: "host_to_worker",
        type: "capability_result",
        ...runFields,
        callId: "call_1",
        grantId: "grant_model",
        capabilityId: "model_gateway",
        outcome: {
          status: "succeeded",
          value: { responseRef: "artifact_response" },
          artifactIds: ["art_response"],
          receipt: {
            receiptId: "rcpt_1",
            idempotencyKey: "effect:stable-1",
            externalRef: "provider-request-42",
          },
        },
      },
    ];

    for (const message of controls) {
      expect(parseTaskWorkerMessage(message, "host_to_worker")).toEqual(message);
      expect(() => parseTaskWorkerMessage(message, "worker_to_host")).toThrow();
    }
  });

  it("rejects wrong directions, versions, unknown fields, and missing fencing tokens", () => {
    const heartbeat = {
      ...eventFields,
      messageId: "msg_heartbeat",
      type: "heartbeat",
    };

    expect(() => parseTaskWorkerMessage(heartbeat, "host_to_worker")).toThrow();
    expect(() =>
      TaskWorkerProtocolMessageSchema.parse({ ...heartbeat, protocolVersion: 2 }),
    ).toThrow();
    expect(() => TaskWorkerProtocolMessageSchema.parse({ ...heartbeat, extra: true })).toThrow(
      /unrecognized/i,
    );
    const { fencingToken: _fencingToken, ...unfenced } = heartbeat;
    expect(() => TaskWorkerProtocolMessageSchema.parse(unfenced)).toThrow();

    expect(() =>
      parseTaskWorkerMessage(
        {
          protocolVersion: 1,
          messageId: "msg_start_mismatched_resume",
          direction: "host_to_worker",
          type: "start",
          ...runFields,
          attempt: 2,
          operation: { name: "smoke.echo", input: null },
          environmentDigest: DIGEST,
          resumeFrom: {
            checkpointId: "ckp_wrong_environment",
            format: "json",
            formatVersion: 1,
            environmentDigest: `sha256:${"b".repeat(64)}`,
            state: { offset: 1 },
          },
          capabilityGrantIds: [],
        },
        "host_to_worker",
      ),
    ).toThrow(/resume checkpoint environment/i);
  });

  it("rejects inline secrets recursively while allowing opaque secret references", () => {
    const start = {
      protocolVersion: 1,
      messageId: "msg_start",
      direction: "host_to_worker",
      type: "start",
      ...runFields,
      attempt: 1,
      operation: {
        name: "smoke.echo",
        input: { nested: { apiKey: "sk-inline" } },
      },
      environmentDigest: DIGEST,
      capabilityGrantIds: [],
    };
    expect(() => parseTaskWorkerMessage(start, "host_to_worker")).toThrow(
      /inline secret field.*apiKey/,
    );

    expect(
      parseTaskWorkerMessage(
        {
          ...start,
          operation: { name: "smoke.echo", input: { secretRef: "provider-auth:local" } },
        },
        "host_to_worker",
      ),
    ).toMatchObject({ operation: { input: { secretRef: "provider-auth:local" } } });
  });

  it("bounds JSONL lines and reports malformed or empty records", () => {
    expect(() => parseTaskWorkerEventLine("{bad json}")).toThrow(/Invalid task worker protocol/);
    expect(() => parseTaskWorkerJsonl("\n", "worker_to_host")).toThrow(/line 1 is empty/);
    expect(() =>
      parseTaskWorkerEventLine(`{"padding":"${"界".repeat(TASK_WORKER_MAX_JSONL_LINE_BYTES)}"}`),
    ).toThrow(/maximum is/);
  });

  it("rejects unsafe artifacts and ambiguous checkpoints", () => {
    expect(() =>
      TaskWorkerArtifactMessageSchema.parse({
        ...eventFields,
        messageId: "msg_artifact",
        type: "artifact",
        idempotencyKey: "artifact:unsafe",
        artifact: {
          artifactId: "art_unsafe",
          kind: "result",
          relativePath: "../provider-auth.json",
          sha256: DIGEST,
          sizeBytes: 1,
        },
      }),
    ).toThrow(/safe relative paths/);

    expect(() =>
      TaskWorkerCheckpointMessageSchema.parse({
        ...eventFields,
        messageId: "msg_checkpoint",
        type: "checkpoint",
        idempotencyKey: "checkpoint:ambiguous",
        checkpoint: {
          checkpointId: "ckp_ambiguous",
          format: "json",
          formatVersion: 1,
          environmentDigest: DIGEST,
          state: { offset: 1 },
          artifact: {
            artifactId: "art_checkpoint",
            kind: "checkpoint",
            relativePath: "checkpoints/1.json",
            sha256: DIGEST,
            sizeBytes: 1,
          },
        },
      }),
    ).toThrow(/exactly one/);
  });

  it("authorizes capability calls against operation-scoped, expiring grants", () => {
    const call = {
      ...eventFields,
      messageId: "msg_call",
      type: "capability_call",
      callId: "call_1",
      grantId: "grant_model",
      capabilityId: "model_gateway",
      operation: "invoke",
      idempotencyKey: "effect:stable-1",
      arguments: { modelRef: "model_1" },
    };
    const grant = {
      grantId: "grant_model",
      capabilityId: "model_gateway",
      operations: ["invoke"],
      expiresAt: "2026-08-05T09:00:00.000Z",
    };

    expect(assertTaskWorkerCapabilityCallAllowed(call, [grant], new Date(NOW))).toEqual(call);
    expect(() =>
      assertTaskWorkerCapabilityCallAllowed(
        { ...call, operation: "admin" },
        [grant],
        new Date(NOW),
      ),
    ).toThrow(/does not authorize operation/);
    expect(() =>
      assertTaskWorkerCapabilityCallAllowed(call, [grant], new Date("2026-08-05T10:00:00Z")),
    ).toThrow(/expired/);
  });
});
