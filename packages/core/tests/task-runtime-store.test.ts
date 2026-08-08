import { appendFileSync, readFileSync, rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  createTaskRuntimeEvent,
  TaskRuntimeIdempotencyCollisionError,
  TaskWorkItemSchema,
} from "../src/task-runtime.js";
import { TaskRuntimeStore, TaskRuntimeTornTailError } from "../src/task-runtime-store.js";

const NOW = "2026-08-05T08:00:00.000Z";
const temporaryRoots: string[] = [];

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("TaskRuntimeStore", () => {
  it("replays appended events and content-addressed payloads after reopening", async () => {
    const rootDir = await temporaryRoot();
    const first = new TaskRuntimeStore({ rootDir });
    const payload = first.putJson({ steps: 3, labels: ["one", "two"] });
    const event = workItemCreatedEvent("awi_replay", payload.ref);

    const appended = first.append(event);
    expect(appended.appended).toBe(true);
    expect(appended.state.workItems.awi_replay?.inputRef).toBe(payload.ref);

    const reopened = new TaskRuntimeStore({ rootDir });
    expect(reopened.state()).toEqual(appended.state);
    expect(reopened.readJson(payload.ref)).toEqual({ steps: 3, labels: ["one", "two"] });
    expect(readFileSync(reopened.eventLogPath, "utf8").trim().split("\n")).toHaveLength(1);
  });

  it("refuses to persist recursively embedded plaintext secret fields", async () => {
    const store = new TaskRuntimeStore({ rootDir: await temporaryRoot() });

    expect(() => store.putJson({ request: { metadata: [{ apiKey: "do-not-persist" }] } })).toThrow(
      /must not contain inline secret field/u,
    );
    expect(() => store.putJson({ credentialRef: "provider-auth:local" })).not.toThrow();
  });

  it("suppresses exact retries and rejects idempotency-key collisions", async () => {
    const store = new TaskRuntimeStore({ rootDir: await temporaryRoot() });
    const event = workItemCreatedEvent("awi_duplicate", "sha256:input", {
      eventId: "evt_original",
      idempotencyKey: "work-item:create:stable",
    });

    expect(store.append(event).appended).toBe(true);
    expect(store.append(event)).toMatchObject({ appended: false, events: [] });
    expect(store.append({ ...event, eventId: "evt_equivalent_delivery" })).toMatchObject({
      appended: false,
      events: [],
    });
    expect(store.state().events).toHaveLength(1);

    const collision = workItemCreatedEvent("awi_collision", "sha256:different", {
      eventId: "evt_collision",
      idempotencyKey: event.idempotencyKey,
    });
    expect(() => store.append(collision)).toThrow(TaskRuntimeIdempotencyCollisionError);
    expect(store.state().events).toHaveLength(1);
  });

  it("detects and explicitly truncates only an unterminated final record", async () => {
    const store = new TaskRuntimeStore({ rootDir: await temporaryRoot() });
    const event = workItemCreatedEvent("awi_torn_tail", "sha256:input");
    store.append(event);
    appendFileSync(store.eventLogPath, '{"schemaVersion":1,"eventId":"evt_torn"', "utf8");

    const inspection = store.inspect();
    expect(inspection.tornTail).toBe(true);
    expect(inspection.tornTailBytes).toBeGreaterThan(0);
    expect(inspection.state.workItems.awi_torn_tail).toBeDefined();
    expect(() => store.append(event)).toThrow(TaskRuntimeTornTailError);

    const recovery = store.recoverTornTail();
    expect(recovery).toMatchObject({
      recovered: true,
      tornTail: false,
      removedBytes: inspection.tornTailBytes,
    });
    expect(store.inspect()).toMatchObject({ tornTail: false, tornTailBytes: 0 });
    expect(store.state().events).toHaveLength(1);
    expect(readFileSync(store.eventLogPath, "utf8").endsWith("\n")).toBe(true);
  });

  it("rejects corruption in a complete JSONL record instead of truncating it", async () => {
    const store = new TaskRuntimeStore({ rootDir: await temporaryRoot() });
    store.append(workItemCreatedEvent("awi_corrupt", "sha256:input"));
    appendFileSync(store.eventLogPath, "{not-json}\n", "utf8");

    expect(() => store.inspect()).toThrow(/events\.jsonl line 2 is corrupt/u);
    expect(() => store.recoverTornTail()).toThrow(/events\.jsonl line 2 is corrupt/u);
  });
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-task-runtime-store-"));
  temporaryRoots.push(root);
  return root;
}

function workItemCreatedEvent(
  workItemId: string,
  inputRef: string,
  identity: { eventId?: string; idempotencyKey?: string } = {},
) {
  const workItem = TaskWorkItemSchema.parse({
    id: workItemId,
    status: "queued",
    executor: { backend: "test", operation: "test.echo" },
    createdAt: NOW,
    updatedAt: NOW,
    inputRef,
  });
  return createTaskRuntimeEvent({
    eventType: "work_item_created",
    eventId: identity.eventId,
    timestamp: NOW,
    source: "test",
    idempotencyKey: identity.idempotencyKey ?? `work-item:create:${workItemId}`,
    workItemId,
    payload: { workItem },
  });
}
