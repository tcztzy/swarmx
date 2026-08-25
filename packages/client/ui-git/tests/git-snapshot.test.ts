import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Context } from "@deepseek-ai/cordis";
import SessionStore, { type SessionId } from "@deepseek-ai/dsh-session";
import LocalSubprocessRuntime from "@deepseek-ai/dsh-subprocess-local";
import { afterEach, describe, expect, it } from "vitest";
import GitUiService from "../src/index.js";

const scratch: string[] = [];

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function repository(): string {
  const root = mkdtempSync(join(tmpdir(), "swarmx-ui-git-test-"));
  scratch.push(root);
  git(root, "init", "-b", "main");
  git(root, "config", "user.email", "ui-git-test@swarmx.invalid");
  git(root, "config", "user.name", "SwarmX Git UI Test");
  writeFileSync(join(root, "tracked.txt"), "original\n");
  git(root, "add", "tracked.txt");
  git(root, "commit", "-m", "initial");
  return root;
}

async function mounted(cwd: string, config: Record<string, unknown> = {}) {
  const context = new Context();
  await context.plugin(SessionStore);
  await context.plugin(LocalSubprocessRuntime);
  const sessionId = "git-ui-session" as SessionId;
  context.sessions.create(sessionId, { meta: { cwd } });
  const fiber = await context.plugin(GitUiService, config as never);
  return { context, fiber, sessionId };
}

afterEach(() => {
  for (const path of scratch.splice(0)) rmSync(path, { recursive: true, force: true });
});

describe("dsh-ui-git Host snapshot", () => {
  it("returns repository-relative porcelain-v2 entries without Host paths", async () => {
    const root = repository();
    const { context, fiber, sessionId } = await mounted(root);
    try {
      const clean = await context.gitUi.snapshot(sessionId);
      expect(clean).toMatchObject({ kind: "repository", branch: "main", clean: true, entries: [] });

      writeFileSync(join(root, "tracked.txt"), "staged\n");
      git(root, "add", "tracked.txt");
      writeFileSync(join(root, "tracked.txt"), "staged and unstaged\n");
      writeFileSync(join(root, "untracked name.txt"), "new\n");
      const dirty = await context.gitUi.snapshot(sessionId);
      expect(dirty).toMatchObject({
        kind: "repository",
        clean: false,
        staged: 1,
        unstaged: 1,
        untracked: 1,
        conflicted: 0,
      });
      if (dirty.kind !== "repository") throw new Error("expected repository");
      expect(dirty.entries.map(({ path }) => path).sort()).toEqual([
        "tracked.txt",
        "untracked name.txt",
      ]);
      expect(JSON.stringify(dirty)).not.toContain(root);
    } finally {
      await fiber.dispose();
    }
  });

  it("caps entries and returns typed absent/unavailable states lazily", async () => {
    const root = repository();
    writeFileSync(join(root, "one.txt"), "1");
    writeFileSync(join(root, "two.txt"), "2");
    const capped = await mounted(root, { maxEntries: 1 });
    const snapshot = await capped.context.gitUi.snapshot(capped.sessionId);
    expect(snapshot).toMatchObject({ kind: "repository", truncated: true });
    if (snapshot.kind !== "repository") throw new Error("expected repository");
    expect(snapshot.entries).toHaveLength(1);
    await capped.fiber.dispose();

    const outside = mkdtempSync(join(tmpdir(), "swarmx-ui-git-outside-"));
    scratch.push(outside);
    const absent = await mounted(outside);
    await expect(absent.context.gitUi.snapshot(absent.sessionId)).resolves.toMatchObject({
      kind: "not-repository",
    });
    await absent.fiber.dispose();

    const missing = await mounted(root, { command: "swarmx-git-does-not-exist" });
    await expect(missing.context.gitUi.snapshot(missing.sessionId)).resolves.toMatchObject({
      kind: "unavailable",
    });
    await missing.fiber.dispose();
  });

  it("rejects unknown Sessions and pre-aborted reads", async () => {
    const root = repository();
    const { context, fiber, sessionId } = await mounted(root);
    try {
      await expect(context.gitUi.snapshot("missing" as SessionId)).rejects.toThrow(
        "Live session not found",
      );
      const controller = new AbortController();
      controller.abort(new Error("stop"));
      await expect(context.gitUi.snapshot(sessionId, controller.signal)).rejects.toThrow("stop");
    } finally {
      await fiber.dispose();
    }
  });
});
