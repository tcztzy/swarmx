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
    "V349 terminates timed-out and cancelled process groups",
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
