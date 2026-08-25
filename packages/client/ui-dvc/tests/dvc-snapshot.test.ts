import { type Context, Context as CordisContext, Service } from "@deepseek-ai/cordis";
import SessionStore, { type SessionId } from "@deepseek-ai/dsh-session";
import { DvcError, type DvcInspection } from "@swarmx/dsh-dvc";
import { describe, expect, it } from "vitest";
import DvcUiService from "../src/index.js";

const PROJECT: DvcInspection = {
  version: "3.67.1",
  root: ".",
  dvcYamlDigest: `sha256:${"a".repeat(64)}`,
  dvcLockDigest: `sha256:${"b".repeat(64)}`,
  data: {
    categories: [{ name: "modified", count: 2 }],
    digest: `sha256:${"c".repeat(64)}`,
    entries: 2,
  },
  pipeline: {
    categories: [{ name: "changed deps", count: 1 }],
    digest: `sha256:${"d".repeat(64)}`,
    entries: 1,
  },
  git: {
    version: "2.50.1",
    objectFormat: "sha1",
    head: "e".repeat(40),
    branch: "main",
    upstream: null,
    ahead: null,
    behind: null,
    clean: true,
    staged: 0,
    unstaged: 0,
    untracked: 0,
    conflicted: 0,
  },
};

class FixtureDvcService extends Service {
  readonly calls: string[] = [];
  error: Error | null = null;

  constructor(ctx: Context) {
    super(ctx, "dvc");
  }

  async inspect(cwd: string, signal?: AbortSignal): Promise<DvcInspection> {
    signal?.throwIfAborted();
    this.calls.push(cwd);
    if (this.error) throw this.error;
    return PROJECT;
  }
}

async function mounted() {
  const context = new CordisContext();
  await context.plugin(SessionStore);
  const fixtureFiber = await context.plugin(FixtureDvcService);
  const sessionId = "dvc-ui-session" as SessionId;
  context.sessions.create(sessionId, { meta: { cwd: "/private/session-workspace" } });
  const fiber = await context.plugin(DvcUiService);
  return {
    context,
    fiber,
    fixture: context.dvc as unknown as FixtureDvcService,
    fixtureFiber,
    sessionId,
  };
}

describe("dsh-ui-dvc Host snapshot", () => {
  it("delegates a live Session workspace and returns a path-free project snapshot", async () => {
    const { context, fiber, fixture, fixtureFiber, sessionId } = await mounted();
    try {
      const snapshot = await context.dvcUi.snapshot(sessionId);
      expect(snapshot).toEqual({ kind: "project", inspection: PROJECT });
      expect(fixture.calls).toEqual(["/private/session-workspace"]);
      expect(JSON.stringify(snapshot)).not.toContain("/private/session-workspace");
    } finally {
      await fiber.dispose();
      await fixtureFiber.dispose();
    }
  });

  it("maps only expected missing project and executable failures to renderable states", async () => {
    const { context, fiber, fixture, fixtureFiber, sessionId } = await mounted();
    try {
      fixture.error = new DvcError("missing project", "NOT_A_DVC_REPOSITORY");
      await expect(context.dvcUi.snapshot(sessionId)).resolves.toEqual({
        kind: "not-project",
        message: "Workspace is not a DVC project",
      });
      fixture.error = new DvcError("missing executable", "DVC_UNAVAILABLE");
      await expect(context.dvcUi.snapshot(sessionId)).resolves.toEqual({
        kind: "unavailable",
        message: "DVC executable is unavailable",
      });
      fixture.error = new DvcError("bad status", "DVC_STATUS_INVALID");
      await expect(context.dvcUi.snapshot(sessionId)).rejects.toThrow("bad status");
    } finally {
      await fiber.dispose();
      await fixtureFiber.dispose();
    }
  });

  it("rejects unknown Sessions and pre-aborted reads", async () => {
    const { context, fiber, fixtureFiber, sessionId } = await mounted();
    try {
      await expect(context.dvcUi.snapshot("missing" as SessionId)).rejects.toThrow(
        "Live session not found",
      );
      const controller = new AbortController();
      controller.abort(new Error("stop"));
      await expect(context.dvcUi.snapshot(sessionId, controller.signal)).rejects.toThrow("stop");
    } finally {
      await fiber.dispose();
      await fixtureFiber.dispose();
    }
  });
});
