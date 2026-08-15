import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { AuditStore, appendMessages, createSession, saveSession } from "@swarmx/core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  formatSessionTimeline,
  runSessionTimelineCommand,
} from "../src/session-timeline-command.js";

const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;
const directories: string[] = [];

beforeEach(() => {
  process.env.SWARMX_SESSIONS_DIR = temporaryDirectory("swarmx-timeline-sessions-");
});

afterEach(() => {
  if (originalSessionsDir === undefined) Reflect.deleteProperty(process.env, "SWARMX_SESSIONS_DIR");
  else process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
  for (const directory of directories.splice(0))
    fs.rmSync(directory, { recursive: true, force: true });
});

describe("sessions timeline command", () => {
  it("renders concise safe output and strict JSON from read-only evidence", () => {
    const session = createSession("agent", "swarmx", undefined, { projectId: "project-1" });
    saveSession(session);
    appendMessages(session.id, [{ role: "user", kind: "message", content: "raw private prompt" }], {
      requestId: "request-1",
    });
    appendMessages(
      session.id,
      [{ role: "assistant", kind: "message", content: "raw private response" }],
      { requestId: "request-1" },
    );
    const audit = new AuditStore({
      filePath: path.join(temporaryDirectory("swarmx-timeline-audit-"), "events.jsonl"),
    });
    audit.append({
      category: "session",
      action: "session.observed",
      outcome: "completed",
      sessionId: session.id,
      requestId: "request-1",
    });
    const auditBefore = [audit.filePath, audit.checkpointPath].map((filePath) =>
      fs.readFileSync(filePath),
    );

    const human = runSessionTimelineCommand(session.id, {}, audit);
    expect(human).toMatchObject({ exitCode: 0 });
    expect(human.output).toContain("Derived diagnostic timeline");
    expect(human.output).toContain("Turn 1: completed");
    expect(human.output).not.toContain("raw private");

    const json = runSessionTimelineCommand(session.id, { json: true }, audit);
    expect(JSON.parse(json.output)).toMatchObject({
      authority: "derived_diagnostic_projection",
      sessionId: session.id,
    });
    expect(
      [audit.filePath, audit.checkpointPath].map((filePath) => fs.readFileSync(filePath)),
    ).toEqual(auditBefore);
  });

  it("returns an actionable missing-Session result", () => {
    const audit = { queryReadOnly: () => [] } as Pick<AuditStore, "queryReadOnly">;
    expect(runSessionTimelineCommand("missing-session", {}, audit)).toEqual({
      exitCode: 1,
      output: "Session missing-session was not found.\n",
    });
  });

  it("formats an empty diagnostic projection without technical payloads", () => {
    expect(
      formatSessionTimeline({
        schemaVersion: 1,
        sessionId: "session-1",
        fingerprint: "timeline_0000000000000000",
        authority: "derived_diagnostic_projection",
        events: [],
        turns: [],
        steps: [],
        unsettled: [],
        diagnostics: [],
      }),
    ).toContain("No Turn evidence was found.");
  });
});

function temporaryDirectory(prefix: string): string {
  const directory = fs.mkdtempSync(path.join(tmpdir(), prefix));
  directories.push(directory);
  return directory;
}
