import { spawn } from "node:child_process";
import { chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it, vi } from "vitest";
import { CodexServerClient } from "../src/codex-server-client.js";
import {
  CODEX_CONTAINER_ENTRY,
  CODEX_MODULE_COMMAND,
  CODEX_SERVER_TRANSPORT,
  codexAcpLauncher,
  codexHarnessPlugin,
  findCodexInPath,
  resolveCodexAcpLaunch,
  resolveCodexContainerAssets,
  resolveDshCodexPermissionHandler,
} from "../src/index.js";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const moduleEntry = fileURLToPath(new URL("../bin/swarmx-codex.js", import.meta.url));

describe("@swarmx/codex", () => {
  it("publishes one owned Codex app-server CLI without an ACP adapter", async () => {
    const manifest = JSON.parse(
      await readFile(new URL("../package.json", import.meta.url), "utf8"),
    ) as {
      name: string;
      version: string;
      bin: Record<string, string>;
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
      peerDependencies: Record<string, string>;
      exports: Record<string, string | Record<string, string>>;
    };

    expect(manifest).toMatchObject({
      name: "@swarmx/codex",
      version: "4.0.0",
      bin: { "swarmx-codex": "./bin/swarmx-codex.js" },
      exports: {
        ".": { types: "./dist/index.d.ts", import: "./dist/index.js" },
        "./cli": "./bin/swarmx-codex.js",
      },
    });
    expect(manifest.dependencies).toEqual({
      "@openai/codex": "0.147.0",
      "@openai/codex-linux-arm64": "npm:@openai/codex@0.147.0-linux-arm64",
      "@openai/codex-linux-x64": "npm:@openai/codex@0.147.0-linux-x64",
    });
    expect(manifest.dependencies).not.toHaveProperty("@agentclientprotocol/codex-acp");
    expect(manifest.devDependencies).not.toHaveProperty("@agentclientprotocol/sdk");
    expect(manifest.peerDependencies["@deepseek-ai/cordis"]).toBe("4.0.1");
    const cli = await readFile(moduleEntry, "utf8");
    expect(cli).not.toContain("npx");
    expect(cli).not.toContain("@agentclientprotocol/codex-acp");
    expect(cli).toContain("app-server");

    const codexRuntime = JSON.parse(
      await readFile(
        new URL("../node_modules/@openai/codex/package.json", import.meta.url),
        "utf8",
      ),
    ) as { optionalDependencies?: Record<string, string> };
    expect(Object.keys(codexRuntime.optionalDependencies ?? {})).toEqual(
      expect.arrayContaining([
        "@openai/codex-linux-x64",
        "@openai/codex-linux-arm64",
        "@openai/codex-darwin-x64",
        "@openai/codex-darwin-arm64",
        "@openai/codex-win32-x64",
        "@openai/codex-win32-arm64",
      ]),
    );
  });

  it("exports a DSH Harness plugin with a direct codex_server transport", async () => {
    expect(codexHarnessPlugin).toMatchObject({
      name: "swarmx-codex-harness",
      inject: ["acpLaunchers", "harnessPermissions", "harnessTransports"],
    });
    expect(codexAcpLauncher.inject).toEqual(["acpLaunchers"]);
    expect(
      resolveCodexAcpLaunch(
        { command: CODEX_MODULE_COMMAND, args: [] },
        {
          nodeExecutable: "/runtime/node",
          envPath: "",
          resolveModule: () => "file:///module/swarmx-codex.js",
        },
      ),
    ).toEqual({ command: "/runtime/node", args: ["/module/swarmx-codex.js"], env: {} });
    expect(
      resolveCodexAcpLaunch(
        { command: CODEX_MODULE_COMMAND, args: [] },
        {
          codexCommand:
            "/Users/tcztzy/.local/share/mise/installs/npm-openai-codex/latest/bin/codex",
        },
      ),
    ).toEqual({
      command: "/Users/tcztzy/.local/share/mise/installs/npm-openai-codex/latest/bin/codex",
      args: ["app-server"],
      env: {},
    });
    const pathRoot = await mkdtemp(join(tmpdir(), "swarmx-codex-path-"));
    try {
      const codexOnPath = join(pathRoot, "codex");
      await writeFile(codexOnPath, "#!/bin/sh\n", "utf8");
      await chmod(codexOnPath, 0o755);
      expect(findCodexInPath(`${pathRoot}:`)).toBe(codexOnPath);
      expect(
        resolveCodexAcpLaunch(
          { command: CODEX_MODULE_COMMAND, args: [] },
          {
            envPath: `${pathRoot}:`,
            resolveModule: () => "file:///module/swarmx-codex.js",
          },
        ),
      ).toEqual({ command: codexOnPath, args: ["app-server"], env: {} });
    } finally {
      await rm(pathRoot, { recursive: true, force: true });
    }

    expect(() =>
      resolveCodexAcpLaunch(
        { command: CODEX_MODULE_COMMAND, args: [] },
        { codexCommand: join(tmpdir(), "missing-swarmx-codex") },
      ),
    ).toThrow(/not an executable file/);

    expect(() =>
      resolveCodexAcpLaunch(
        { command: "custom-codex", args: [] },
        { envPath: "", resolveModule: () => "file:///module/swarmx-codex.js" },
      ),
    ).toThrow(/cannot resolve/);

    const launchers = { register: vi.fn() };
    const permissions = { register: vi.fn(), resolve: () => undefined };
    const transports = { register: vi.fn() };
    codexHarnessPlugin.apply(
      {
        acpLaunchers: launchers,
        harnessPermissions: permissions,
        harnessTransports: transports,
      } as never,
      { nodeExecutable: "/runtime/node", resolveModule: () => "file:///module/swarmx-codex.js" },
    );
    expect(launchers.register).toHaveBeenCalledWith(CODEX_MODULE_COMMAND, expect.any(Function));
    expect(transports.register).toHaveBeenCalledWith(CODEX_SERVER_TRANSPORT, expect.any(Function));
  });

  it("bridges DSH permission policy inside the Codex plugin", async () => {
    expect(resolveDshCodexPermissionHandler({ get: () => undefined })).toBeUndefined();
    const never = resolveDshCodexPermissionHandler({
      get: (name: string) =>
        name === "permissionPresets"
          ? { defaultPreset: "never", resolve: () => ({ approval: "never" }) }
          : undefined,
    });
    await expect(never?.({} as never)).resolves.toEqual({
      outcome: { outcome: "rejected" },
    });
    const ask = resolveDshCodexPermissionHandler({
      get: (name: string) => (name === "approval" ? { config: { policy: "ask" } } : undefined),
    });
    expect(ask).toBeUndefined();
  });

  it("reports the module and Codex runtime versions without starting the server", async () => {
    const result = await runProcess(["--version"]);
    expect(result).toEqual({
      exitCode: 0,
      stdout: "@swarmx/codex 4.0.0 (codex app-server 0.147.0)\n",
      stderr: "",
    });
  });

  it("waits for the managed app-server child exit code", async () => {
    const bin = (await import(
      new URL("../bin/swarmx-codex.js", import.meta.url).href
    )) as unknown as {
      resolveCodexEntry(): string;
      waitForChildExit(child: ReturnType<typeof spawn>): Promise<number | null>;
    };
    expect(bin.resolveCodexEntry().replaceAll("\\", "/")).toMatch(
      /@openai\/codex\/bin\/codex\.js$/,
    );
    const child = spawn(process.execPath, ["-e", "process.exit(7)"], { stdio: "ignore" });
    await expect(bin.waitForChildExit(child)).resolves.toBe(7);
  });

  it("resolves bundled Linux Codex runtimes for protected containers", async () => {
    const assets = resolveCodexContainerAssets();
    expect(assets?.moduleDir).toBe(packageRoot.replace(/[\\/]$/, ""));
    expect(assets?.runtimeDir).toMatch(
      process.arch === "arm64"
        ? /@openai\+codex@0\.147\.0-linux-arm64/
        : /@openai\+codex@0\.147\.0-linux-x64/,
    );

    const containerUrl = new URL("../bin/swarmx-codex-container.js", import.meta.url).href;
    expect(CODEX_CONTAINER_ENTRY).toContain("swarmx-codex-container.js");
    const container = (await import(containerUrl)) as unknown as {
      codexTargetTriple(platform: string, arch: string): string | null;
      codexBinaryPath(runtimeDir: string, platform: string, arch: string): string;
    };
    expect(container.codexTargetTriple("linux", "arm64")).toBe("aarch64-unknown-linux-musl");
    expect(container.codexTargetTriple("darwin", "arm64")).toBeNull();
    expect(container.codexBinaryPath("/opt/runtime", "linux", "x64").replaceAll("\\", "/")).toBe(
      "/opt/runtime/vendor/x86_64-unknown-linux-musl/bin/codex",
    );
  });

  it("keeps the module and Codex runtime outside the packaged Electron asar", async () => {
    const builderConfig = await readFile(
      new URL("../../desktop/electron-builder.yml", import.meta.url),
      "utf8",
    );

    expect(builderConfig).toContain("node_modules/@swarmx/codex/**/*");
    expect(builderConfig).toContain("node_modules/@openai/codex*/**/*");
    expect(builderConfig).not.toContain("@agentclientprotocol/codex-acp");
  });

  it("speaks the Codex app-server JSON-RPC protocol directly", async () => {
    const server = `
import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  const msg = JSON.parse(line);
  if (msg.id === 1 && msg.method === "initialize") {
    process.stdout.write(JSON.stringify({ id: 1, result: { userAgent: "test", codexHome: ".", platformFamily: "unix", platformOs: "linux" } }) + "\\n");
    continue;
  }
  if (msg.method === "initialized") continue;
  if (msg.id === 2 && msg.method === "thread/start") {
    process.stdout.write(JSON.stringify({ id: 2, result: { thread: { id: "thread-1" }, model: "test" } }) + "\\n");
    continue;
  }
  if (msg.method === "thread/list") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { data: [{ id: "thread-1", cwd: process.cwd(), name: "Test thread", updatedAt: 1 }], nextCursor: null, backwardsCursor: null } }) + "\\n");
    continue;
  }
  if (msg.method === "turn/start") {
    const response = JSON.stringify({ id: msg.id, result: { turn: { id: "turn-1" } } });
    const completed = JSON.stringify({ method: "turn/completed", params: { threadId: msg.params.threadId, turn: { id: "turn-1", items: [{ type: "agentMessage", id: "m1", text: "hello from codex server (" + msg.params.approvalPolicy + ")" }] } } });
    process.stdout.write(response + "\\n" + completed + "\\n");
    continue;
  }
}
`;
    const client = new CodexServerClient(
      {
        command: process.execPath,
        args: ["--input-type=module", "-e", server],
        env: {},
      },
      async () => ({ outcome: { outcome: "cancelled" as const } }),
    );

    try {
      await expect(
        client.prompt({ command: "swarmx-codex", args: [], cwd: process.cwd() }, "hello"),
      ).resolves.toEqual({
        messages: [
          { role: "assistant", content: "hello from codex server (on-request)", kind: "message" },
        ],
      });
      await expect(
        client.listSessions({ command: "swarmx-codex", args: [], cwd: process.cwd() }),
      ).resolves.toEqual([
        {
          sessionId: "thread-1",
          cwd: process.cwd(),
          title: "Test thread",
          updatedAt: "1970-01-01T00:00:01.000Z",
        },
      ]);
    } finally {
      client.kill();
    }
  }, 10_000);

  it("rejects the prompt when the app-server exits during a turn", async () => {
    const server = `
import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  const msg = JSON.parse(line);
  if (msg.method === "initialize") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { userAgent: "test" } }) + "\\n");
    continue;
  }
  if (msg.method === "initialized") continue;
  if (msg.method === "thread/start") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { thread: { id: "thread-1" } } }) + "\\n");
    continue;
  }
  if (msg.method === "turn/start") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { turn: { id: "turn-1" } } }) + "\\n");
    process.exit(3);
  }
}
`;
    const client = new CodexServerClient({
      command: process.execPath,
      args: ["--input-type=module", "-e", server],
      env: {},
    });

    try {
      await expect(
        withTimeout(
          client.prompt({ command: "swarmx-codex", args: [], cwd: process.cwd() }, "hello"),
          5_000,
          () => new Error("prompt did not settle after app-server exit"),
        ),
      ).rejects.toThrow(/process exited/);
    } finally {
      client.kill();
    }
  });

  it("rejects an in-flight prompt when the request is aborted", async () => {
    const server = `
import { createInterface } from "node:readline";
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  const msg = JSON.parse(line);
  if (msg.method === "initialize") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { userAgent: "test" } }) + "\\n");
    continue;
  }
  if (msg.method === "initialized") continue;
  if (msg.method === "thread/start") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { thread: { id: "thread-1" } } }) + "\\n");
    continue;
  }
  if (msg.method === "turn/start") {
    process.stdout.write(JSON.stringify({ id: msg.id, result: { turn: { id: "turn-1" } } }) + "\\n");
    continue;
  }
}
`;
    const client = new CodexServerClient({
      command: process.execPath,
      args: ["--input-type=module", "-e", server],
      env: {},
    });
    const controller = new AbortController();

    try {
      const prompt = withTimeout(
        client.prompt(
          { command: "swarmx-codex", args: [], cwd: process.cwd(), signal: controller.signal },
          "hello",
        ),
        5_000,
        () => new Error("prompt did not settle after abort"),
      );
      await new Promise((resolve) => setTimeout(resolve, 50));
      controller.abort();
      await expect(prompt).rejects.toThrow(/stopped|aborted/);
    } finally {
      client.kill();
    }
  });
});

async function withTimeout<T>(
  operation: Promise<T>,
  timeoutMs: number,
  timeoutError: () => Error,
): Promise<T> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(timeoutError()), timeoutMs);
      }),
    ]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function runProcess(args: string[]): Promise<{
  exitCode: number | null;
  stdout: string;
  stderr: string;
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [moduleEntry, ...args], {
      cwd: packageRoot,
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.once("error", reject);
    child.once("exit", (exitCode) => resolve({ exitCode, stdout, stderr }));
  });
}
