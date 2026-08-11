import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { createContextEvent } from "../src/context-engine.js";
import {
  externalizeContextEventPayload,
  JsonlContextEventStore,
  LocalContextArtifactStore,
  SqliteContextEventStore,
} from "../src/context-engine-store.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function temporaryRoot(): string {
  const root = mkdtempSync(path.join(tmpdir(), "swarmx-context-engine-"));
  roots.push(root);
  return root;
}

function event(seq: number, payload = `event ${seq}`) {
  return createContextEvent({
    id: `evt_${seq}`,
    seq,
    sessionId: "session_1",
    taskId: "task_1",
    turnId: `turn_${seq}`,
    timestamp: new Date(Date.parse("2026-08-11T00:00:00.000Z") + seq * 1_000).toISOString(),
    kind: "assistant_message",
    payload,
    causalParents: seq === 1 ? [] : [`evt_${seq - 1}`],
    labels: [],
    metadata: {},
  });
}

describe("JsonlContextEventStore", () => {
  it("appends and replays one deterministic immutable snapshot", () => {
    const rootDir = temporaryRoot();
    const store = new JsonlContextEventStore({ rootDir });
    store.append([event(1), event(2)]);
    store.append(event(2));

    const first = store.snapshot({ sessionId: "session_1" });
    const reopened = new JsonlContextEventStore({ rootDir }).snapshot({ sessionId: "session_1" });

    expect(first.events.map((item) => item.id)).toEqual(["evt_1", "evt_2"]);
    expect(first.snapshotId).toBe(reopened.snapshotId);
    expect(store.get("evt_1")).toEqual(event(1));
    expect(store.scan({ sessionId: "session_1", afterSeq: 1 })).toHaveLength(1);
  });

  it("rejects sequence regression, id collisions, and complete corruption", () => {
    const rootDir = temporaryRoot();
    const store = new JsonlContextEventStore({ rootDir });
    store.append(event(1));

    expect(() => store.append(event(1, "different"))).toThrow(/event id collision/i);
    expect(() =>
      store.append(
        createContextEvent({
          ...event(1),
          id: "evt_other",
          contentHash: undefined,
        }),
      ),
    ).toThrow(/sequence/i);

    writeFileSync(store.eventLogPath, `${readFileSync(store.eventLogPath, "utf8")}not-json\n`);
    expect(() => store.snapshot({ sessionId: "session_1" })).toThrow(/corrupt/i);
  });

  it("fails closed on a JSONL record without its durable newline", () => {
    const store = new JsonlContextEventStore({ rootDir: temporaryRoot() });
    store.append(event(1));
    writeFileSync(store.eventLogPath, readFileSync(store.eventLogPath, "utf8").trimEnd());

    expect(() => store.append(event(2))).toThrow(/torn final record/i);
  });
});

describe("SqliteContextEventStore", () => {
  it("uses WAL and replays an append-only snapshot across reopen", () => {
    const rootDir = temporaryRoot();
    const store = new SqliteContextEventStore({ rootDir });
    expect(store.journalMode).toBe("wal");
    store.append([event(1), event(2)]);
    const first = store.snapshot({ sessionId: "session_1" });
    store.close();

    const reopened = new SqliteContextEventStore({ rootDir });
    expect(reopened.snapshot({ sessionId: "session_1" })).toEqual(first);
    expect(reopened.scan({ sessionId: "session_1", afterSeq: 1 })).toHaveLength(1);
    reopened.close();
  });

  it("rejects id collisions and per-session sequence regression transactionally", () => {
    const store = new SqliteContextEventStore({ rootDir: temporaryRoot() });
    store.append(event(1));

    expect(() => store.append(event(1, "different"))).toThrow(/event id collision/i);
    expect(() =>
      store.append(
        createContextEvent({
          ...event(1),
          id: "evt_other",
          contentHash: undefined,
        }),
      ),
    ).toThrow(/sequence/i);
    expect(store.snapshot({ sessionId: "session_1" }).events).toHaveLength(1);
    store.close();
  });

  it("enforces append-only rows inside SQLite", () => {
    const store = new SqliteContextEventStore({ rootDir: temporaryRoot() });
    store.append(event(1));
    const database = new DatabaseSync(store.databasePath);

    expect(() =>
      database.prepare("UPDATE context_events SET seq = 2 WHERE event_id = 'evt_1'").run(),
    ).toThrow(/append-only/i);
    expect(() =>
      database.prepare("DELETE FROM context_events WHERE event_id = 'evt_1'").run(),
    ).toThrow(/append-only/i);

    database.close();
    store.close();
  });
});

describe("LocalContextArtifactStore", () => {
  it("stores by digest and supports verified previews and byte ranges", () => {
    const store = new LocalContextArtifactStore({ rootDir: temporaryRoot() });
    const ref = store.put(Buffer.from("0123456789", "utf8"), { mediaType: "text/plain" });

    expect(ref.uri).toMatch(/^artifact:\/\/sha256\/[a-f0-9]{64}$/);
    expect(store.readRange(ref, { startByte: 2, endByte: 6 }).toString("utf8")).toBe("2345");
    expect(store.preview(ref, { maxBytes: 4 })).toMatchObject({ text: "0123", truncated: true });
    expect(store.put(Buffer.from("0123456789", "utf8"))).toMatchObject({
      uri: ref.uri,
      contentHash: ref.contentHash,
    });

    writeFileSync(store.pathFor(ref), "tampered");
    expect(() => store.readRange(ref, { startByte: 0 })).toThrow(/checksum mismatch/i);
  });

  it("externalizes oversized tool payloads while retaining a bounded event capsule", () => {
    const store = new LocalContextArtifactStore({ rootDir: temporaryRoot() });
    const original = {
      log: "x".repeat(2_000),
      salient: ["3 tests failed", "context-engine.test.ts"],
    };
    const externalized = externalizeContextEventPayload(
      createContextEvent({
        ...event(1),
        kind: "tool_result",
        toolCallId: "call_1",
        payload: original,
        metadata: { exitCode: 1, errorSignature: "FAIL context engine" },
        contentHash: undefined,
      }),
      store,
      { thresholdBytes: 100 },
    );

    expect(externalized.payload).toMatchObject({
      externalized: true,
      salient: ["3 tests failed", "context-engine.test.ts"],
    });
    expect(externalized.metadata.exitCode).toBe(1);
    const artifactRef = externalized.artifactRef;
    expect(artifactRef).toBeDefined();
    if (!artifactRef) throw new Error("Expected externalized artifact reference.");
    const restored = store.readRange(artifactRef, { startByte: 0 });
    expect(JSON.parse(restored.toString("utf8"))).toEqual(original);
  });
});
