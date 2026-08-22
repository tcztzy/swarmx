import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { Context } from "@deepseek-ai/cordis";
import SessionStore, { type SessionId } from "@deepseek-ai/dsh-session";
import LocalSubprocessRuntime from "@deepseek-ai/dsh-subprocess-local";
import { afterEach, describe, expect, it } from "vitest";
import ScienceService, { type ScienceError } from "../src/index.js";

interface Fixture {
  context: Context;
  databasePath: string;
  root: string;
  scienceFiber: Awaited<ReturnType<Context["plugin"]>>;
  sessionA: SessionId;
  sessionB: SessionId;
  workspaceA: string;
  workspaceB: string;
}

const fixtures: Fixture[] = [];

afterEach(async () => {
  const disposed = fixtures.splice(0);
  await Promise.all(disposed.map((fixture) => fixture.context.fiber.dispose()));
  for (const fixture of disposed) {
    rmSync(join(fixture.root, ".."), { recursive: true, force: true });
  }
});

async function createFixture(): Promise<Fixture> {
  const scratch = mkdtempSync(join(tmpdir(), "swarmx-science-test-"));
  const root = join(scratch, "science");
  const workspaceA = join(scratch, "workspace-a");
  const workspaceB = join(scratch, "workspace-b");
  mkdirSync(workspaceA);
  mkdirSync(workspaceB);
  const context = new Context();
  await context.plugin(SessionStore);
  await context.plugin(LocalSubprocessRuntime);
  const sessionA = "science-session-a" as SessionId;
  const sessionB = "science-session-b" as SessionId;
  context.sessions.create(sessionA, { meta: { cwd: workspaceA } });
  context.sessions.create(sessionB, { meta: { cwd: workspaceB } });
  const scienceFiber = await context.plugin(ScienceService, { root });
  const fixture = {
    context,
    databasePath: join(root, "science.sqlite"),
    root,
    scienceFiber,
    sessionA,
    sessionB,
    workspaceA,
    workspaceB,
  };
  fixtures.push(fixture);
  return fixture;
}

describe("T13 Science Journal service", () => {
  it("commits one project fact and projection for one idempotency key", async () => {
    const { context, sessionA } = await createFixture();
    const request = { requestId: randomUUID(), title: "Protein folding baseline" };

    const first = context.science.createProject(sessionA, request);
    const repeated = context.science.createProject(sessionA, request);
    const snapshot = context.science.getWorkspace(sessionA);

    expect(repeated).toEqual(first);
    expect(snapshot.projects).toEqual([first]);
    expect(context.science.journalCount()).toBe(1);
    expect(first.provenance.journalSeq).toBe(1);
  });

  it("rejects reuse of an idempotency key with different input", async () => {
    const { context, sessionA } = await createFixture();
    const requestId = randomUUID();
    context.science.createProject(sessionA, { requestId, title: "Original" });

    expect(() =>
      context.science.createProject(sessionA, { requestId, title: "Conflicting" }),
    ).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "IDEMPOTENCY_CONFLICT" }),
    );
    expect(context.science.journalCount()).toBe(1);
  });

  it("replays projections after disposal and a missing materialized row", async () => {
    const fixture = await createFixture();
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "Replayable project",
    });
    await fixture.scienceFiber.dispose();
    const database = new DatabaseSync(fixture.databasePath);
    database.exec("DELETE FROM science_projects");
    database.close();

    fixture.scienceFiber = await fixture.context.plugin(ScienceService, { root: fixture.root });

    expect(fixture.context.science.getWorkspace(fixture.sessionA).projects).toEqual([project]);
  });

  it("uses a versioned WAL database with owner-only storage", async () => {
    const fixture = await createFixture();
    await fixture.scienceFiber.dispose();
    const database = new DatabaseSync(fixture.databasePath, { readOnly: true });
    const journalMode = database.prepare("PRAGMA journal_mode").get() as { journal_mode: string };
    const migration = database
      .prepare("SELECT MAX(version) AS version FROM science_migrations")
      .get() as { version: number };
    database.close();

    expect(journalMode.journal_mode).toBe("wal");
    expect(migration.version).toBe(5);
    expect(statSync(fixture.root).mode & 0o777).toBe(0o700);
    expect(statSync(fixture.databasePath).mode & 0o777).toBe(0o600);
  });

  it("rejects a pre-aborted mutation before appending", async () => {
    const { context, sessionA } = await createFixture();
    const controller = new AbortController();
    controller.abort();

    expect(() =>
      context.science.createProject(
        sessionA,
        { requestId: randomUUID(), title: "Cancelled" },
        controller.signal,
      ),
    ).toThrowError(expect.objectContaining({ name: "AbortError" }));
    expect(context.science.journalCount()).toBe(0);
  });

  it("isolates project ids by the live session workspace and never returns host paths", async () => {
    const { context, sessionA, sessionB, workspaceA, workspaceB } = await createFixture();
    const project = context.science.createProject(sessionA, {
      requestId: randomUUID(),
      title: "Workspace A",
    });

    expect(() =>
      context.science.createNotebook(sessionB, {
        requestId: randomUUID(),
        projectId: project.id,
        title: "Cross-workspace notebook",
      }),
    ).toThrowError(expect.objectContaining<Partial<ScienceError>>({ code: "PROJECT_NOT_FOUND" }));
    expect(JSON.stringify(context.science.getWorkspace(sessionA))).not.toContain(workspaceA);
    expect(JSON.stringify(context.science.getWorkspace(sessionA))).not.toContain(workspaceB);
  });

  it("closes cleanly and reopens durable state on a replacement mount", async () => {
    const fixture = await createFixture();
    const project = fixture.context.science.createProject(fixture.sessionA, {
      requestId: randomUUID(),
      title: "HMR project",
    });
    const firstService = fixture.context.science;
    await fixture.scienceFiber.dispose();

    expect(() => firstService.getWorkspace(fixture.sessionA)).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "SCIENCE_CLOSED" }),
    );

    fixture.scienceFiber = await fixture.context.plugin(ScienceService, { root: fixture.root });
    expect(fixture.context.science.getWorkspace(fixture.sessionA).projects).toEqual([project]);
  });
});
