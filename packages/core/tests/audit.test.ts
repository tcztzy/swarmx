import { createHash, randomUUID } from "node:crypto";
import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  AUDIT_METADATA_MAX_ARRAY_ITEMS,
  AUDIT_METADATA_MAX_DEPTH,
  AUDIT_METADATA_MAX_ENTRIES,
  AuditEventSchema,
  AuditInputSchema,
  AuditIntegrityError,
  AuditStore,
  removeStaleLock,
  sanitizeAuditMetadata,
  verifyAuditEventChain,
} from "../src/audit.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

describe("AuditStore", () => {
  it("appends, reopens, verifies, queries, and exports a strict hash chain", () => {
    const fixture = auditFixture();
    let now = new Date("2026-08-05T10:00:00.000Z");
    const store = new AuditStore({ filePath: fixture.filePath, now: () => now });

    const first = store.append({
      category: "session",
      action: "session.created",
      outcome: "completed",
      actor: { kind: "user", id: "user-1" },
      target: { kind: "session", id: "session-1" },
      sessionId: "session-1",
      requestId: "request-1",
      metadata: { messageCount: 0, permissionMode: "default" },
    });
    now = new Date("2026-08-05T10:01:00.000Z");
    const second = store.append({
      category: "permission",
      action: "permission.decided",
      outcome: "denied",
      actor: { kind: "user", id: "user-1" },
      target: { kind: "tool", id: "workspace.delete" },
      sessionId: "session-1",
      requestId: "request-2",
      metadata: { decision: "deny", riskLevel: 3 },
    });
    now = new Date("2026-08-05T10:02:00.000Z");
    const third = store.append({
      category: "task",
      action: "task.cancel_requested",
      outcome: "cancel_requested",
      actor: { kind: "user", id: "user-2" },
      taskId: "task-1",
    });

    expect(first).toMatchObject({ sequence: 1, previousHash: "0".repeat(64) });
    expect(second).toMatchObject({ sequence: 2, previousHash: first.eventHash });
    expect(third).toMatchObject({ sequence: 3, previousHash: second.eventHash });
    expect(fs.statSync(fixture.directory).mode & 0o777).toBe(0o700);
    expect(fs.statSync(fixture.filePath).mode & 0o777).toBe(0o600);
    expect(fs.statSync(store.checkpointPath).mode & 0o777).toBe(0o600);

    const reopened = new AuditStore({ filePath: fixture.filePath });
    expect(reopened.verify()).toMatchObject({
      ok: true,
      eventCount: 3,
      headSequence: 3,
      headHash: third.eventHash,
      checkpointStatus: "matched",
    });
    expect(reopened.query({ sessionId: "session-1" }).map((event) => event.sequence)).toEqual([
      1, 2,
    ]);
    expect(reopened.query({ actorId: "user-1", outcome: "denied", limit: 1 })).toMatchObject([
      { sequence: 2, category: "permission" },
    ]);
    expect(reopened.query({ reverse: true, limit: 2 }).map((event) => event.sequence)).toEqual([
      3, 2,
    ]);
    expect(
      reopened.query({
        from: "2026-08-05T10:01:00.000Z",
        to: "2026-08-05T10:02:00.000Z",
      }),
    ).toHaveLength(2);

    const exported = reopened.exportJsonl();
    const exportedEvents = exported
      .trim()
      .split("\n")
      .map((line) => AuditEventSchema.parse(JSON.parse(line)));
    expect(exportedEvents).toHaveLength(3);
    expect(verifyAuditEventChain(exportedEvents)).toMatchObject({ ok: true, eventCount: 3 });
  });

  it("uses strict input and event schemas", () => {
    expect(() =>
      AuditInputSchema.parse({
        category: "task",
        action: "task.started",
        unexpected: true,
      }),
    ).toThrow();
    expect(() => AuditInputSchema.parse({ category: "task", action: "raw user prompt" })).toThrow();
    expect(() =>
      AuditInputSchema.parse({
        category: "task",
        action: "task.started",
        requestId: "sk-secret-identifier",
      }),
    ).toThrow(/secret-bearing/);
    expect(
      AuditInputSchema.parse({
        category: "provider",
        action: "provider.checked",
        metadata: { apiKey: "sk-secret", prompt: "private question", count: 1 },
      }).metadata,
    ).toEqual({ apiKey: "[redacted]", prompt: "[omitted]", count: 1 });

    const fixture = auditFixture();
    const event = new AuditStore({ filePath: fixture.filePath }).append({
      category: "task",
      action: "task.started",
      outcome: "attempted",
    });
    expect(() => AuditEventSchema.parse({ ...event, extra: true })).toThrow();
    expect(() =>
      AuditEventSchema.parse({
        ...event,
        metadata: { harmlessLabel: "Bearer secret-value" },
      }),
    ).toThrow(/already be sanitized/);
    expect(() =>
      AuditEventSchema.parse({
        ...event,
        metadata: nestedObject(AUDIT_METADATA_MAX_DEPTH + 2),
      }),
    ).toThrow(/depth\/entry limits/);
    expect(() =>
      AuditEventSchema.parse({
        ...event,
        metadata: Object.fromEntries(
          Array.from({ length: AUDIT_METADATA_MAX_ENTRIES + 1 }, (_, index) => [
            `count${index}`,
            index,
          ]),
        ),
      }),
    ).toThrow(/depth\/entry limits/);
  });

  it("recursively redacts secret values and omits raw content within hard bounds", () => {
    const deeplyNested = nestedObject(AUDIT_METADATA_MAX_DEPTH + 3);
    const metadata = sanitizeAuditMetadata({
      apiKey: "sk-top-secret",
      authorization: "Bearer opaque-value",
      nested: {
        password: "hunter2",
        harmlessLabel: "sk-value-secret",
        status: "completed",
      },
      prompt: "raw user prompt that must not persist",
      response: "raw model response that must not persist",
      sourceCode: "const secret = true;",
      terminalOutput: "command output",
      stdout: "process output",
      stackTrace: "Error at secret location",
      disguisedSecret: "password=still-secret",
      multiline: "first line\nsecond line",
      oversized: "x".repeat(500),
      values: Array.from({ length: 40 }, (_, index) => index),
      deeplyNested,
    });
    const serialized = JSON.stringify(metadata);

    expect(metadata).toMatchObject({
      apiKey: "[redacted]",
      authorization: "[redacted]",
      nested: {
        password: "[redacted]",
        harmlessLabel: "[redacted]",
        status: "completed",
      },
      prompt: "[omitted]",
      response: "[omitted]",
      sourceCode: "[omitted]",
      terminalOutput: "[omitted]",
      stdout: "[omitted]",
      stackTrace: "[omitted]",
      disguisedSecret: "[redacted]",
      multiline: "[omitted]",
      oversized: "[omitted]",
    });
    expect((metadata as Record<string, unknown>).values).toHaveLength(
      AUDIT_METADATA_MAX_ARRAY_ITEMS,
    );
    expect(serialized).not.toContain("sk-top-secret");
    expect(serialized).not.toContain("opaque-value");
    expect(serialized).not.toContain("raw user prompt");
    expect(serialized).not.toContain("const secret");
    expect(serialized).not.toContain("process output");
    expect(serialized).not.toContain("secret location");
    expect(serialized).not.toContain("still-secret");
    expect(serialized).not.toContain("second line");
    expect(serialized).not.toContain("x".repeat(100));
    expect(serialized).toContain("[omitted]");

    const fixture = auditFixture();
    const stored = new AuditStore({ filePath: fixture.filePath }).append({
      category: "provider",
      action: "provider.requested",
      metadata: metadata as Record<string, unknown>,
    });
    const persisted = fs.readFileSync(fixture.filePath, "utf8");
    expect(stored.metadata).toEqual(metadata);
    expect(persisted).not.toContain("sk-top-secret");
    expect(persisted).not.toContain("raw user prompt");
  });

  it("detects syntactically valid content changes and refuses later writes", () => {
    const fixture = auditFixture();
    const store = new AuditStore({ filePath: fixture.filePath });
    store.append({ category: "session", action: "session.created" });
    store.append({ category: "session", action: "session.archived" });

    const lines = readLines(fixture.filePath);
    const first = JSON.parse(lines[0] ?? "{}") as Record<string, unknown>;
    first.action = "session.deleted";
    lines[0] = JSON.stringify(first);
    fs.writeFileSync(fixture.filePath, `${lines.join("\n")}\n`, "utf8");

    expect(store.verify()).toMatchObject({
      ok: false,
      issue: { code: "invalid_event_hash", sequence: 1 },
    });
    expect(() => store.query()).toThrow(AuditIntegrityError);
    expect(() => store.append({ category: "session", action: "session.updated" })).toThrow(
      AuditIntegrityError,
    );
  });

  it("detects reordered and middle-deleted records", () => {
    const reordered = populatedStore(3);
    const reorderedLines = readLines(reordered.fixture.filePath);
    [reorderedLines[0], reorderedLines[1]] = [reorderedLines[1] ?? "", reorderedLines[0] ?? ""];
    fs.writeFileSync(reordered.fixture.filePath, `${reorderedLines.join("\n")}\n`, "utf8");
    expect(reordered.store.verify()).toMatchObject({
      ok: false,
      issue: { code: "invalid_sequence" },
    });

    const deleted = populatedStore(3);
    const deletedLines = readLines(deleted.fixture.filePath);
    deletedLines.splice(1, 1);
    fs.writeFileSync(deleted.fixture.filePath, `${deletedLines.join("\n")}\n`, "utf8");
    expect(deleted.store.verify()).toMatchObject({
      ok: false,
      issue: { code: "invalid_sequence" },
    });
  });

  it("uses the atomic head checkpoint to detect tail deletion and a rehashed rewrite", () => {
    const deleted = populatedStore(3);
    const lines = readLines(deleted.fixture.filePath);
    fs.writeFileSync(deleted.fixture.filePath, `${lines.slice(0, -1).join("\n")}\n`, "utf8");
    expect(deleted.store.verify()).toMatchObject({
      ok: false,
      issue: { code: "checkpoint_ahead" },
    });
    expect(() => deleted.store.recoverTornTail()).toThrow(AuditIntegrityError);

    const rewritten = populatedStore(2);
    const rewrittenEvents = readLines(rewritten.fixture.filePath).map(
      (line) => JSON.parse(line) as Record<string, unknown>,
    );
    rewrittenEvents[0] = rehash({ ...rewrittenEvents[0], action: "task.rewritten" });
    rewrittenEvents[1] = rehash({
      ...rewrittenEvents[1],
      previousHash: rewrittenEvents[0]?.eventHash,
    });
    fs.writeFileSync(
      rewritten.fixture.filePath,
      `${rewrittenEvents.map((event) => JSON.stringify(event)).join("\n")}\n`,
      "utf8",
    );
    expect(rewritten.store.verify()).toMatchObject({
      ok: false,
      issue: { code: "checkpoint_hash_mismatch" },
    });
  });

  it("requires explicit recovery before adopting a valid tail ahead of its checkpoint", () => {
    const fixture = auditFixture();
    const store = new AuditStore({ filePath: fixture.filePath });
    store.append({ category: "task", action: "task.first" });
    const firstCheckpoint = fs.readFileSync(store.checkpointPath, "utf8");
    store.append({ category: "task", action: "task.second" });
    fs.writeFileSync(store.checkpointPath, firstCheckpoint, "utf8");

    expect(store.verify()).toMatchObject({
      ok: false,
      checkpointStatus: "lagging",
      issue: { code: "checkpoint_lagging" },
    });
    expect(() => store.query()).toThrow(AuditIntegrityError);
    expect(() => store.append({ category: "task", action: "task.third" })).toThrow(
      AuditIntegrityError,
    );
    expect(store.recoverTornTail()).toMatchObject({
      recovered: true,
      discardedBytes: 0,
      verification: { ok: true, eventCount: 2, checkpointStatus: "matched" },
    });
  });

  it("requires explicit exact recovery for an incomplete final record", () => {
    const fixture = auditFixture();
    const store = new AuditStore({ filePath: fixture.filePath });
    store.append({ category: "task", action: "task.started", outcome: "attempted" });
    store.append({ category: "task", action: "task.completed", outcome: "completed" });
    const validBytes = fs.statSync(fixture.filePath).size;
    fs.appendFileSync(fixture.filePath, '{"schemaVersion":1,"sequence":3');
    const tornBytes = fs.statSync(fixture.filePath).size - validBytes;

    expect(store.verify()).toMatchObject({ ok: false, issue: { code: "torn_tail" } });
    expect(() =>
      store.append({ category: "task", action: "task.retried", outcome: "attempted" }),
    ).toThrow(AuditIntegrityError);

    const recovery = store.recoverTornTail();
    expect(recovery).toMatchObject({
      recovered: true,
      discardedBytes: tornBytes,
      appendedFinalNewline: false,
      verification: { ok: true, eventCount: 2 },
    });
    expect(fs.statSync(fixture.filePath).size).toBe(validBytes);
    expect(
      store.append({ category: "task", action: "task.retried", outcome: "attempted" }),
    ).toMatchObject({ sequence: 3 });
  });

  it("repairs a valid final record missing only its newline without dropping it", () => {
    const fixture = auditFixture();
    const store = new AuditStore({ filePath: fixture.filePath });
    const event = store.append({ category: "system", action: "system.started" });
    const withNewline = fs.readFileSync(fixture.filePath);
    fs.truncateSync(fixture.filePath, withNewline.length - 1);

    expect(store.verify()).toMatchObject({
      ok: false,
      eventCount: 1,
      issue: { code: "missing_final_newline" },
    });
    expect(store.recoverTornTail()).toMatchObject({
      recovered: true,
      discardedBytes: 0,
      appendedFinalNewline: true,
      verification: { ok: true, headHash: event.eventHash },
    });
    expect(fs.readFileSync(fixture.filePath, "utf8").endsWith("\n")).toBe(true);
  });

  it("never treats a newline-terminated corrupt record as a recoverable torn tail", () => {
    const fixture = auditFixture();
    const store = new AuditStore({ filePath: fixture.filePath });
    store.append({ category: "system", action: "system.started" });
    fs.appendFileSync(fixture.filePath, "{bad json}\n", "utf8");

    expect(store.verify()).toMatchObject({
      ok: false,
      issue: { code: "corrupt_record", line: 2 },
    });
    expect(() => store.recoverTornTail()).toThrow(AuditIntegrityError);
    expect(fs.readFileSync(fixture.filePath, "utf8")).toContain("{bad json}");
  });

  it("honors a live writer lock and recovers a dead writer lock", () => {
    const fixture = auditFixture();
    const blocked = new AuditStore({
      filePath: fixture.filePath,
      lockTimeoutMs: 20,
      lockStaleMs: 0,
    });
    fs.mkdirSync(fixture.directory, { recursive: true, mode: 0o700 });
    fs.writeFileSync(
      blocked.lockPath,
      JSON.stringify({
        pid: process.pid,
        token: randomUUID(),
        createdAt: new Date().toISOString(),
      }),
      { encoding: "utf8", mode: 0o600 },
    );
    expect(() => blocked.append({ category: "system", action: "system.blocked" })).toThrow(
      /Timed out waiting for Audit writer lock/,
    );
    fs.unlinkSync(blocked.lockPath);

    fs.writeFileSync(
      blocked.lockPath,
      JSON.stringify({
        pid: 2_147_483_647,
        token: randomUUID(),
        createdAt: "2020-01-01T00:00:00.000Z",
      }),
      { encoding: "utf8", mode: 0o600 },
    );
    expect(blocked.append({ category: "system", action: "system.recovered" })).toMatchObject({
      sequence: 1,
    });
    expect(fs.existsSync(blocked.lockPath)).toBe(false);
  });
});

function auditFixture(): { directory: string; filePath: string } {
  const directory = fs.mkdtempSync(path.join(tmpdir(), "swarmx-audit-"));
  temporaryDirectories.push(directory);
  return { directory, filePath: path.join(directory, "events.jsonl") };
}

function populatedStore(count: number): {
  fixture: ReturnType<typeof auditFixture>;
  store: AuditStore;
} {
  const fixture = auditFixture();
  const store = new AuditStore({ filePath: fixture.filePath });
  for (let index = 0; index < count; index += 1) {
    store.append({
      category: "task",
      action: `task.step_${index + 1}`,
      taskId: "task-1",
      metadata: { step: index + 1 },
    });
  }
  return { fixture, store };
}

function readLines(filePath: string): string[] {
  return fs.readFileSync(filePath, "utf8").trim().split("\n");
}

function nestedObject(depth: number): Record<string, unknown> {
  let value: Record<string, unknown> = { status: "completed" };
  for (let index = 0; index < depth; index += 1) value = { nested: value };
  return value;
}

describe("removeStaleLock", () => {
  it("treats a half-written lock file as not stale instead of crashing", () => {
    const fixture = auditFixture();
    fs.writeFileSync(fixture.filePath, '{"pid":123', "utf8");
    const now = Date.now();
    fs.utimesSync(fixture.filePath, new Date(now - 60_000), new Date(now - 60_000));
    // A garbage lock older than the stale age is removed.
    expect(removeStaleLock(fixture.filePath, 30_000)).toBe(true);
    expect(fs.existsSync(fixture.filePath)).toBe(false);
  });

  it("keeps a fresh half-written lock so a concurrent writer is not clobbered", () => {
    const fixture = auditFixture();
    fs.writeFileSync(fixture.filePath, "", "utf8");
    // A concurrent writer just created the file; it must not be removed.
    expect(removeStaleLock(fixture.filePath, 30_000)).toBe(false);
    expect(fs.existsSync(fixture.filePath)).toBe(true);
  });
});

function rehash(event: Record<string, unknown>): Record<string, unknown> {
  const { eventHash: _eventHash, ...withoutHash } = event;
  return {
    ...withoutHash,
    eventHash: createHash("sha256").update(stableJson(withoutHash), "utf8").digest("hex"),
  };
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${Object.entries(value)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
    .join(",")}}`;
}
