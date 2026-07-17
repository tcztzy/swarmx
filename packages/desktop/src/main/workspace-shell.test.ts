import { mkdir, mkdtemp, readFile, rm, symlink } from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { RequestCancelledError, cancelAcpRequest, withAcpRequest } from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import { WorkspaceShell, workspaceShellAgentTool } from "./workspace-shell.js";

const temporaryDirectories = new Set<string>();

afterEach(async () => {
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("WorkspaceShell", () => {
  it.runIf(process.platform === "darwin")(
    "V349 runs from the Project with bounded output and a sanitized environment",
    async () => {
      const root = await temporaryDirectory();
      const previousSecret = process.env.SWARMX_TEST_PROVIDER_API_KEY;
      process.env.SWARMX_TEST_PROVIDER_API_KEY = "must-not-leak";
      try {
        const shell = new WorkspaceShell(root, { maxOutputBytes: 8 });
        const result = await shell.run(
          'printf project > generated.txt; printf 1234567890; printf %s "${SWARMX_TEST_PROVIDER_API_KEY-unset}" >&2',
        );

        expect(result).toMatchObject({
          cwd: await realpathForTest(root),
          exitCode: 0,
          timedOut: false,
          truncated: true,
          stdout: "12345678",
          stderr: "unset",
        });
        expect(result.durationMs).toBeGreaterThanOrEqual(0);
        await expect(readFile(path.join(root, "generated.txt"), "utf8")).resolves.toBe("project");
      } finally {
        if (previousSecret === undefined) {
          Reflect.deleteProperty(process.env, "SWARMX_TEST_PROVIDER_API_KEY");
        } else process.env.SWARMX_TEST_PROVIDER_API_KEY = previousSecret;
      }
    },
    30_000,
  );

  it.runIf(process.platform === "darwin")(
    "V350 denies writes outside the Project and through escaping symlinks",
    async () => {
      const parent = await temporaryDirectory();
      const root = path.join(parent, "workspace");
      const outside = path.join(parent, "outside");
      await mkdir(root);
      await mkdir(outside);
      await symlink(outside, path.join(root, "escape"), "dir");
      const shell = new WorkspaceShell(root);

      const result = await shell.run(
        "printf direct > ../outside/direct.txt; printf linked > escape/linked.txt; printf inside > inside.txt",
      );

      expect(result.exitCode).toBe(0);
      expect(result.stderr).toMatch(/operation not permitted/i);
      await expect(readFile(path.join(root, "inside.txt"), "utf8")).resolves.toBe("inside");
      await expect(readFile(path.join(outside, "direct.txt"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
      await expect(readFile(path.join(outside, "linked.txt"), "utf8")).rejects.toMatchObject({
        code: "ENOENT",
      });
    },
    30_000,
  );

  it.runIf(process.platform === "darwin")(
    "V359 accepts contained Codex workdirs and rejects escaping ones",
    async () => {
      const parent = await temporaryDirectory();
      const root = path.join(parent, "workspace");
      const nested = path.join(root, "nested");
      await mkdir(root);
      await mkdir(nested);
      const shell = new WorkspaceShell(root);
      const resolvedNested = await realpathForTest(nested);

      await expect(shell.run("pwd", { workdir: nested })).resolves.toMatchObject({
        cwd: resolvedNested,
      });
      await expect(shell.run("pwd", { workdir: "../" })).rejects.toThrow(/escapes/i);
    },
  );

  it.runIf(process.platform === "darwin")(
    "V350 denies loopback network connections",
    async () => {
      const root = await temporaryDirectory();
      const server = createServer((_request, response) => response.end("unexpected"));
      await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
      });
      try {
        const address = server.address();
        if (!address || typeof address === "string") {
          throw new Error("Expected TCP server address.");
        }
        const script = [
          'const net=require("node:net")',
          `const socket=net.connect(${address.port},"127.0.0.1")`,
          'socket.on("connect",()=>{console.log("CONNECTED");socket.destroy()})',
          'socket.on("error",error=>console.log(error.code))',
        ].join(";");
        const result = await new WorkspaceShell(root).run(`node -e '${script}'`);

        expect(result.stdout).toContain("EPERM");
        expect(result.stdout).not.toContain("CONNECTED");
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
      }
    },
    30_000,
  );

  it.runIf(process.platform === "darwin")(
    "V349-V383 terminates timed-out and cancelled process groups",
    async () => {
      const root = await temporaryDirectory();
      const shell = new WorkspaceShell(root, { timeoutMs: 50 });
      const timedOut = await shell.run("sleep 5");
      expect(timedOut.timedOut).toBe(true);
      expect(timedOut.durationMs).toBeLessThan(2_000);

      const requestId = `workspace-shell-${Date.now()}`;
      const running = withAcpRequest(requestId, () =>
        new WorkspaceShell(root, { timeoutMs: 5_000 }).run("sleep 5"),
      );
      await delay(50);
      await expect(cancelAcpRequest(requestId)).resolves.toBe(true);
      await expect(running).rejects.toBeInstanceOf(RequestCancelledError);

      const fallbackRequestId = `workspace-shell-fallback-${Date.now()}`;
      const fallback = withAcpRequest(fallbackRequestId, () =>
        new WorkspaceShell(root, { backgroundTimeoutMs: 5_000 }).runWithBackgroundFallback(
          "sleep 5",
          5_000,
        ),
      );
      await delay(50);
      await expect(cancelAcpRequest(fallbackRequestId)).resolves.toBe(true);
      await expect(fallback).rejects.toBeInstanceOf(RequestCancelledError);
    },
  );

  it("fails closed when the platform or sandbox executable is unsupported", async () => {
    const root = await temporaryDirectory();
    await expect(new WorkspaceShell(root, { platform: "linux" }).run("pwd")).rejects.toThrow(
      /refusing unrestricted execution/i,
    );
    if (process.platform === "darwin") {
      await expect(
        new WorkspaceShell(root, { sandboxExecutable: "/missing/sandbox-exec" }).run("pwd"),
      ).rejects.toThrow(/refusing unrestricted shell execution/i);
    }
  });

  it("validates the local tool boundary", async () => {
    const root = await temporaryDirectory();
    const tool = workspaceShellAgentTool(new WorkspaceShell(root, { platform: "linux" }));
    expect(tool.name).toBe("workspace_shell");
    await expect(tool.call({ command: "" })).rejects.toThrow(/non-empty shell command/i);
    await expect(tool.call({ command: "pwd", timeoutMs: 0 })).rejects.toThrow(/positive integer/i);
  });

  it.runIf(process.platform === "darwin")(
    "V365-V367 starts, waits for, and stops background process groups",
    async () => {
      const root = await temporaryDirectory();
      const shell = new WorkspaceShell(root, { backgroundTimeoutMs: 20_000 });
      try {
        const stdout: string[] = [];
        const exits: string[] = [];
        const started = await shell.startBackground("printf start; sleep 0.1; printf end", {
          onStdout: (chunk) => stdout.push(chunk),
          onExit: (snapshot) => exits.push(snapshot.status),
        });
        expect(started).toMatchObject({ status: "running", sessionId: expect.any(Number) });

        const completed = await shell.taskOutput(started.sessionId, {
          block: true,
          timeoutMs: 10_000,
        });
        expect(completed).toMatchObject({
          status: "completed",
          exitCode: 0,
          stdout: "startend",
        });
        expect(stdout.join("")).toBe("startend");
        expect(exits).toEqual(["completed"]);

        const longRunning = await shell.startBackground("sleep 5; printf late > late.txt");
        const stopped = await shell.stop(longRunning.sessionId);
        expect(stopped.status).toBe("stopped");
        await delay(700);
        await expect(readFile(path.join(root, "late.txt"), "utf8")).rejects.toMatchObject({
          code: "ENOENT",
        });

        await shell.startBackground("sleep 0.2; printf leaked > after-close.txt");
        await shell.close();
        await delay(400);
        await expect(readFile(path.join(root, "after-close.txt"), "utf8")).rejects.toMatchObject({
          code: "ENOENT",
        });
      } finally {
        await shell.close();
      }
    },
    30_000,
  );

  it.runIf(process.platform === "darwin")(
    "V366-V393 yields a Codex session and accepts pipe-backed stdin",
    async () => {
      const root = await temporaryDirectory();
      const shell = new WorkspaceShell(root, { backgroundTimeoutMs: 5_000 });
      try {
        const started = await shell.exec('read -r answer; printf "got:%s" "$answer"', {
          yieldTimeMs: 250,
        });
        expect(started).toMatchObject({
          sessionId: expect.any(Number),
          status: "running",
          output: "",
        });

        let completed = await shell.writeStdin(started.sessionId as number, "hello\n", {
          yieldTimeMs: 2_000,
        });
        for (let attempt = 0; completed.status === "running" && attempt < 4; attempt += 1) {
          if (completed.sessionId === undefined) break;
          completed = await shell.writeStdin(completed.sessionId, "", { yieldTimeMs: 1_000 });
        }
        expect(completed).toMatchObject({
          status: "completed",
          exitCode: 0,
          output: "got:hello",
        });
        expect(completed.sessionId).toBeUndefined();
      } finally {
        await shell.close();
      }
    },
    10_000,
  );

  it.runIf(process.platform === "darwin")(
    "V371-V373-V392 allocates a sandboxed real PTY and accepts terminal input and Ctrl-C",
    async () => {
      const parent = await temporaryDirectory();
      const root = path.join(parent, "workspace");
      await mkdir(root);
      const shell = new WorkspaceShell(root, { backgroundTimeoutMs: 10_000 });
      try {
        await expect(shell.run('test -t 0 || printf "pipe:%s" "$TERM"')).resolves.toMatchObject({
          stdout: "pipe:dumb",
          stderr: "",
        });

        let started = await shell.exec(
          'test -t 0 && stty -echo && printf "tty:%s:ready" "$TERM"; IFS= read -r value; printf ":got:%s" "$value"',
          { tty: true, yieldTimeMs: 1_000 },
        );
        for (let attempt = 0; !started.output.includes(":ready") && attempt < 4; attempt += 1) {
          if (started.status !== "running" || started.sessionId === undefined) break;
          started = await shell.writeStdin(started.sessionId, "", { yieldTimeMs: 1_000 });
        }
        expect(started).toMatchObject({
          status: "running",
          sessionId: expect.any(Number),
          output: "tty:xterm-256color:ready",
        });

        const completed = await shell.writeStdin(started.sessionId as number, "hello\r", {
          yieldTimeMs: 2_000,
        });
        expect(completed).toMatchObject({
          status: "completed",
          exitCode: 0,
          output: ":got:hello",
        });

        let interrupted = await shell.exec(
          "trap 'printf ctrl-c; exit 130' INT; printf ready; sleep 10",
          { tty: true, yieldTimeMs: 500 },
        );
        for (let attempt = 0; !interrupted.output.includes("ready") && attempt < 4; attempt += 1) {
          if (interrupted.status !== "running" || interrupted.sessionId === undefined) break;
          interrupted = await shell.writeStdin(interrupted.sessionId, "", { yieldTimeMs: 1_000 });
        }
        expect(interrupted).toMatchObject({
          status: "running",
          sessionId: expect.any(Number),
          output: "ready",
        });
        const afterInterrupt = await shell.writeStdin(interrupted.sessionId as number, "\u0003", {
          yieldTimeMs: 2_000,
        });
        expect(afterInterrupt.status).not.toBe("running");
        expect(afterInterrupt.output).toContain("ctrl-c");

        const timedOut = await shell.run("sleep 5", { tty: true, timeoutMs: 50 });
        expect(timedOut).toMatchObject({ timedOut: true, stderr: "" });
        expect(timedOut.durationMs).toBeLessThan(2_000);

        const denied = await shell.run("printf blocked > ../blocked.txt", { tty: true });
        expect(denied.stdout).toMatch(/operation not permitted/i);
        expect(denied.stderr).toBe("");
        await expect(readFile(path.join(parent, "blocked.txt"), "utf8")).rejects.toMatchObject({
          code: "ENOENT",
        });

        const closing = await shell.exec("sleep 0.5; printf leaked > after-pty-close.txt", {
          tty: true,
          yieldTimeMs: 250,
        });
        expect(closing.status).toBe("running");
        await shell.close();
        await delay(700);
        await expect(
          readFile(path.join(root, "after-pty-close.txt"), "utf8"),
        ).rejects.toMatchObject({ code: "ENOENT" });
      } finally {
        await shell.close();
      }
    },
    20_000,
  );
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), "swarmx-workspace-shell-"));
  temporaryDirectories.add(directory);
  return directory;
}

async function realpathForTest(value: string): Promise<string> {
  const { realpath } = await import("node:fs/promises");
  return realpath(value);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
