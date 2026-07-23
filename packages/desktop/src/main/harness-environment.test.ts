import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  HarnessEnvironmentService,
  configureDesktopHarnessEnvironment,
  containerHostBridgeEnv,
  containerHostBridgeUrl,
} from "./harness-environment.js";

describe("HarnessEnvironmentService", () => {
  it("configures desktop PATH with common user install locations", () => {
    const env = { PATH: "/usr/bin" };

    const merged = configureDesktopHarnessEnvironment(env, "/Users/test");

    expect(merged.split(":")).toContain("/Users/test/.npm-global/bin");
    expect(merged.split(":")).toContain("/Users/test/.local/bin");
    expect(merged.split(":")).not.toContain("/Users/test/.bun/bin");
    expect(merged.split(":")).toContain("/opt/homebrew/bin");
    expect(env.PATH).toBe(merged);
  });

  it("detects built-in and external harness requirements", async () => {
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      now: () => new Date("2026-07-08T00:00:00.000Z"),
      findExecutable: async (command) =>
        command === "node" ||
        command === "claude" ||
        command === "codex" ||
        command === "pi" ||
        command === "kimi" ||
        command === "container" ||
        command === "openclaw"
          ? `/Users/test/bin/${command}`
          : null,
      runCommand: async (program, args) => {
        if (program.endsWith("/node")) return { exitCode: 0, stdout: "v22.17.0\n", stderr: "" };
        if (program.endsWith("/claude")) {
          return { exitCode: 0, stdout: "Claude Code v2.1.0 (stable)\n", stderr: "" };
        }
        if (program.endsWith("/codex")) {
          return { exitCode: 0, stdout: "codex-cli 0.69.0\n", stderr: "" };
        }
        if (program.endsWith("/pi")) {
          return { exitCode: 0, stdout: "0.80.10\n", stderr: "" };
        }
        if (program.endsWith("/kimi")) {
          return { exitCode: 0, stdout: "Kimi Code CLI 1.2.3\n", stderr: "" };
        }
        if (program.endsWith("/openclaw")) {
          return { exitCode: 0, stdout: "2026.6.11\n", stderr: "" };
        }
        if (program.endsWith("/container") && args.includes("--version")) {
          return {
            exitCode: 0,
            stdout: "container CLI version 1.0.0 (build: release, commit: ee848e3)\n",
            stderr: "",
          };
        }
        if (program.endsWith("/container") && args.join(" ") === "system status") {
          return { exitCode: 0, stdout: "status             running\n", stderr: "" };
        }
        return { exitCode: 0, stdout: "", stderr: "" };
      },
    });

    const status = await service.status();

    expect(status.checkedAt).toBe("2026-07-08T00:00:00.000Z");
    expect(status.requirements).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "node",
          status: "ready",
          version: "22.17.0",
          requiredBy: [],
        }),
        expect.objectContaining({
          id: "claude_code",
          status: "ready",
          version: "2.1.0",
          requiredBy: ["claude_code"],
        }),
        expect.objectContaining({
          id: "codex",
          status: "ready",
          version: "0.69.0",
          requiredBy: ["codex"],
        }),
        expect.objectContaining({
          id: "pi",
          status: "ready",
          version: "0.80.10",
          requiredBy: ["pi"],
        }),
        expect.objectContaining({
          id: "kimi",
          status: "ready",
          version: "1.2.3",
          requiredBy: ["kimi"],
        }),
        expect.objectContaining({
          id: "opencode",
          status: "missing",
          installable: true,
        }),
        expect.objectContaining({
          id: "openclaw",
          status: "ready",
          requiredBy: ["openclaw"],
        }),
      ]),
    );
    expect(status.containerRuntimes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "apple_container",
          status: "ready",
          serviceReady: true,
        }),
      ]),
    );
    expect(status.protection).toEqual(
      expect.objectContaining({
        mode: "protected",
        ready: true,
        selectedRuntimeId: "apple_container",
      }),
    );
    expect(status.harnesses).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ harnessId: "swarmx", status: "ready", version: "3.1.4" }),
        expect.objectContaining({ harnessId: "codex", status: "ready", version: "0.69.0" }),
        expect.objectContaining({
          harnessId: "pi",
          status: "ready",
          version: "0.80.10",
          executionMode: "native",
          protectionRequired: false,
          note: expect.stringContaining("owned by Pi"),
        }),
        expect.objectContaining({
          harnessId: "kimi",
          status: "ready",
          version: "1.2.3",
          executionMode: "native",
          protectionRequired: false,
          note: expect.stringContaining("owned by Kimi Code"),
        }),
        expect.objectContaining({
          harnessId: "opencode",
          status: "needs_setup",
          executionMode: "native",
          protectionRequired: false,
        }),
        expect.objectContaining({
          harnessId: "openclaw",
          status: "ready",
          version: "2026.6.11",
          executionMode: "native",
          protectionRequired: false,
        }),
      ]),
    );
    expect(status.ready).toBe(true);
    expect(status.setupAvailable).toBe(false);
  });

  it("V217 V219 treats unavailable OpenClaw as optional and skips unscoped setup", async () => {
    const runCommand = vi.fn(async () => ({ exitCode: 0, stdout: "v22.17.0\n", stderr: "" }));
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      protectionMode: "native",
      homeDir: "/Users/test",
      findExecutable: async (command) => (command === "node" ? "/usr/bin/node" : null),
      runCommand,
    });

    const status = await service.status();
    const result = await service.setup();

    expect(status.ready).toBe(true);
    expect(status.setupAvailable).toBe(false);
    expect(status.harnesses).toContainEqual(
      expect.objectContaining({ harnessId: "openclaw", status: "needs_setup" }),
    );
    expect(result.success).toBe(true);
    expect(result.installedRequirementIds).toEqual([]);
    expect(result.failedRequirementIds).toEqual([]);
    expect(runCommand).toHaveBeenCalled();
    expect(
      runCommand.mock.calls.every(([program, args]) => {
        return program === "/usr/bin/node" && args.join(" ") === "--version";
      }),
    ).toBe(true);
  });

  it("uses each installed harness CLI and keeps only its three-part version", async () => {
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      if (program.endsWith("/opencode") && args.join(" ") === "--version") {
        return { exitCode: 0, stdout: "OpenCode CLI v1.2.3 (build abc123)\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "opencode" ? "/Users/test/bin/opencode" : null,
      runCommand,
    });

    const status = await service.status();

    expect(status.harnesses).toContainEqual(
      expect.objectContaining({ harnessId: "opencode", version: "1.2.3" }),
    );
    expect(runCommand).toHaveBeenCalledWith(
      "/Users/test/bin/opencode",
      ["--version"],
      expect.objectContaining({ timeoutMs: 8_000 }),
    );
  });

  it("keeps only semver from the Hermes banner and generic runtime output", async () => {
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      protectionMode: "native",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "hermes" ? "/Users/test/.hermes/hermes-agent/hermes" : null,
      runCommand: async () => ({
        exitCode: 0,
        stdout: "Hermes Agent v0.18.2 (nousresearch build, Python 3.12)\n",
        stderr: "",
      }),
    });

    const status = await service.status();

    expect(status.requirements.find((item) => item.id === "hermes")?.version).toBe("0.18.2");
    expect(status.harnesses.find((item) => item.harnessId === "hermes")?.version).toBe("0.18.2");
  });

  it("caches a checked harness version until an explicit refresh", async () => {
    let detectedVersion = "OpenCode CLI v1.2.3\n";
    const runCommand = vi.fn(async () => ({
      exitCode: 0,
      stdout: detectedVersion,
      stderr: "",
    }));
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "opencode" ? "/Users/test/bin/opencode" : null,
      runCommand,
    });

    await expect(service.harnessVersion("opencode")).resolves.toEqual({
      harnessId: "opencode",
      version: "1.2.3",
    });
    detectedVersion = "OpenCode CLI v2.0.0\n";
    await expect(service.harnessVersion("opencode")).resolves.toEqual({
      harnessId: "opencode",
      version: "1.2.3",
    });
    await expect(service.harnessVersion("opencode", true)).resolves.toEqual({
      harnessId: "opencode",
      version: "2.0.0",
    });
    expect(runCommand).toHaveBeenCalledTimes(2);
  });

  it("runs one-click Apple Container setup for a protected harness", async () => {
    let containerInstalled = false;
    let serviceStarted = false;
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      const commandLine = args.join(" ");
      if (program === "bash" && commandLine.includes("github.com/apple/container")) {
        containerInstalled = true;
        return { exitCode: 0, stdout: "container installed\n", stderr: "" };
      }
      if (program === "container" && commandLine === "system start") {
        serviceStarted = true;
        return { exitCode: 0, stdout: "service started\n", stderr: "" };
      }
      if (program.endsWith("/container") && args.includes("--version")) {
        return { exitCode: 0, stdout: "container CLI version 1.1.0\n", stderr: "" };
      }
      if (program.endsWith("/container") && commandLine === "system status") {
        return {
          exitCode: 0,
          stdout: serviceStarted ? "status             running\n" : "status             stopped\n",
          stderr: "",
        };
      }
      return { exitCode: 0, stdout: "", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "container" && containerInstalled ? "/usr/local/bin/container" : null,
      runCommand,
    });

    const result = await service.setup({ harnessId: "codex" });

    expect(result.success).toBe(true);
    expect(result.installedContainerRuntimeIds).toEqual(["apple_container"]);
    expect(result.installedRequirementIds).toEqual([]);
    expect(result.failedContainerRuntimeIds).toEqual([]);
    expect(result.log.join("\n")).toContain("container installed");
    expect(result.log.join("\n")).toContain("service started");
    expect(runCommand).toHaveBeenCalledWith(
      "bash",
      ["-lc", expect.stringContaining("github.com/apple/container")],
      expect.objectContaining({ timeoutMs: 1200000 }),
    );
    expect(result.status.protection.ready).toBe(true);
    expect(result.status.harnesses).toEqual(
      expect.arrayContaining([expect.objectContaining({ harnessId: "codex", status: "ready" })]),
    );
  });

  it("runs native one-click setup through npm when protected mode is disabled", async () => {
    let codexInstalled = false;
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      if (program === "bash" && args.join(" ").includes("npm install --global @openai/codex")) {
        codexInstalled = true;
        return { exitCode: 0, stdout: "codex installed\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "codex-cli 0.69.0\n", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      protectionMode: "native",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "codex" && codexInstalled ? "/Users/test/.npm-global/bin/codex" : null,
      runCommand,
    });

    const result = await service.setup({ harnessId: "codex" });

    expect(result.success).toBe(true);
    expect(result.installedRequirementIds).toEqual(["codex"]);
    expect(result.failedRequirementIds).toEqual([]);
    expect(result.log.join("\n")).toContain("codex installed");
    expect(runCommand).toHaveBeenCalledWith(
      "bash",
      ["-lc", "npm install --global @openai/codex"],
      expect.objectContaining({ timeoutMs: 900000 }),
    );
    expect(result.status.harnesses).toEqual(
      expect.arrayContaining([expect.objectContaining({ harnessId: "codex", status: "ready" })]),
    );
  });

  it("V495 installs the official Pi CLI only through confirmed Runtime setup", async () => {
    let piInstalled = false;
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      if (
        program === "bash" &&
        args
          .join(" ")
          .includes("npm install --global --ignore-scripts @earendil-works/pi-coding-agent")
      ) {
        piInstalled = true;
        return { exitCode: 0, stdout: "pi installed\n", stderr: "" };
      }
      if (program.endsWith("/pi")) {
        return { exitCode: 0, stdout: "0.80.10\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "v22.19.0\n", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      protectionMode: "native",
      homeDir: "/Users/test",
      findExecutable: async (command) => {
        if (command === "node") return "/usr/bin/node";
        if (command === "pi" && piInstalled) return "/Users/test/.npm-global/bin/pi";
        return null;
      },
      runCommand,
    });

    const result = await service.setup({ harnessId: "pi" });

    expect(result.success).toBe(true);
    expect(result.installedRequirementIds).toEqual(["pi"]);
    expect(result.failedRequirementIds).toEqual([]);
    expect(runCommand).toHaveBeenCalledWith(
      "bash",
      ["-lc", "npm install --global --ignore-scripts @earendil-works/pi-coding-agent"],
      expect.objectContaining({ timeoutMs: 900000 }),
    );
    expect(result.status.harnesses).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          harnessId: "pi",
          status: "ready",
          version: "0.80.10",
          executionMode: "native",
        }),
      ]),
    );
  });

  it("V503 installs the official Kimi Code CLI only through confirmed Runtime setup", async () => {
    let kimiInstalled = false;
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      if (
        program === "bash" &&
        args.join(" ").includes("https://code.kimi.com/kimi-code/install.sh")
      ) {
        kimiInstalled = true;
        return { exitCode: 0, stdout: "kimi installed\n", stderr: "" };
      }
      if (program.endsWith("/kimi")) {
        return { exitCode: 0, stdout: "Kimi Code CLI 1.2.3\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "v22.19.0\n", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      protectionMode: "native",
      homeDir: "/Users/test",
      findExecutable: async (command) => {
        if (command === "node") return "/usr/bin/node";
        if (command === "kimi" && kimiInstalled) return "/Users/test/.npm-global/bin/kimi";
        return null;
      },
      runCommand,
    });

    const result = await service.setup({ harnessId: "kimi" });

    expect(result.success).toBe(true);
    expect(result.installedRequirementIds).toEqual(["kimi"]);
    expect(result.failedRequirementIds).toEqual([]);
    expect(runCommand).toHaveBeenCalledWith(
      "bash",
      ["-lc", "curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash"],
      expect.objectContaining({ timeoutMs: 900000 }),
    );
    expect(result.status.harnesses).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          harnessId: "kimi",
          status: "ready",
          version: "1.2.3",
          executionMode: "native",
        }),
      ]),
    );
  });

  it("V505 recognizes the native Kimi ACP backend", () => {
    const service = new HarnessEnvironmentService({ protectionMode: "native" });

    expect(
      service.guessProtectedHarnessId({ type: "custom", program: "kimi", args: ["acp"] }),
    ).toBe("kimi");
  });

  it("installs the verified OpenClaw CLI while leaving Gateway setup explicit", async () => {
    let openclawInstalled = false;
    const runCommand = vi.fn(async (program: string, args: string[]) => {
      if (program === "bash" && args.join(" ").includes("openclaw.ai/install.sh")) {
        openclawInstalled = true;
        return { exitCode: 0, stdout: "openclaw installed\n", stderr: "" };
      }
      if (program.endsWith("/openclaw")) {
        return { exitCode: 0, stdout: "2026.6.11\n", stderr: "" };
      }
      return { exitCode: 0, stdout: "", stderr: "" };
    });
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      protectionMode: "protected",
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "openclaw" && openclawInstalled ? "/Users/test/.local/bin/openclaw" : null,
      runCommand,
    });

    const result = await service.setup({ harnessId: "openclaw" });

    expect(result.success).toBe(true);
    expect(result.installedRequirementIds).toEqual(["openclaw"]);
    expect(runCommand).toHaveBeenCalledWith(
      "bash",
      ["-lc", "curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard"],
      expect.objectContaining({ timeoutMs: 900000 }),
    );
    expect(result.status.harnesses).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          harnessId: "openclaw",
          status: "ready",
          executionMode: "native",
          note: expect.stringContaining("Gateway"),
        }),
      ]),
    );
  });

  it("reports setup failure when Apple Container remains unavailable", async () => {
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      findExecutable: async () => null,
      runCommand: async () => ({ exitCode: 1, stdout: "", stderr: "network unavailable\n" }),
    });

    const result = await service.setup({ harnessId: "codex" });

    expect(result.success).toBe(false);
    expect(result.failedContainerRuntimeIds).toEqual(["apple_container"]);
    expect(result.error).toMatch(/runtime tools/);
    expect(result.log.join("\n")).toContain("network unavailable");
  });

  it("wraps protected harness backends in Apple Container", async () => {
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      findExecutable: async (command) =>
        command === "container" ? "/usr/local/bin/container" : null,
      runCommand: async (program, args) => {
        if (program.endsWith("/container") && args.includes("--version")) {
          return { exitCode: 0, stdout: "container CLI version 1.1.0\n", stderr: "" };
        }
        if (program.endsWith("/container") && args.join(" ") === "system status") {
          return { exitCode: 0, stdout: "status             running\n", stderr: "" };
        }
        return { exitCode: 0, stdout: "", stderr: "" };
      },
    });

    const result = await service.protectedBackendForHarness(
      "codex",
      {
        type: "custom",
        program: "npx",
        args: ["--yes", "@agentclientprotocol/codex-acp"],
      },
      { workspaceDir: "/Users/test/project" },
    );

    expect(result.success).toBe(true);
    expect(result.mode).toBe("protected");
    expect(result.backend).toEqual(
      expect.objectContaining({
        type: "custom",
        program: "container",
        args: expect.arrayContaining([
          "run",
          "--rm",
          "-i",
          "node:22-slim",
          "npx",
          "--yes",
          "@agentclientprotocol/codex-acp@1.1.2",
        ]),
      }),
    );
    expect(result.backend?.type === "custom" ? result.backend.args : []).toEqual(
      expect.arrayContaining(["type=bind,source=/Users/test/project,target=/Users/test/project"]),
    );
    expect(result.backend?.type === "custom" ? result.backend.args : []).toEqual(
      expect.arrayContaining([
        "CLAUDE_MODEL_CONFIG",
        "CODEX_CONFIG",
        "OPENCODE_CONFIG_CONTENT",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_EFFORT_LEVEL",
        "YALLM_DEFAULT_PROVIDER",
      ]),
    );
  });

  it("V336 mounts a private official Codex auth file read-only", async () => {
    const homeDir = await mkdtemp(join(tmpdir(), "swarmx-protected-codex-"));
    try {
      const codexHome = join(homeDir, "managed-codex");
      await mkdir(codexHome, { recursive: true });
      const authPath = join(codexHome, "auth.json");
      await writeFile(authPath, JSON.stringify({ tokens: { access_token: "secret" } }), {
        mode: 0o600,
      });
      const service = new HarnessEnvironmentService({
        env: { PATH: "/usr/bin", CODEX_HOME: codexHome, CODEX_ACCESS_TOKEN: "must-not-forward" },
        platform: "darwin",
        arch: "arm64",
        macosVersion: "26.5.2",
        homeDir,
        findExecutable: async (command) =>
          command === "container" ? "/usr/local/bin/container" : null,
        runCommand: async (program, args) => {
          if (program.endsWith("/container") && args.includes("--version")) {
            return { exitCode: 0, stdout: "container CLI version 1.1.0\n", stderr: "" };
          }
          if (program.endsWith("/container") && args.join(" ") === "system status") {
            return { exitCode: 0, stdout: "status             running\n", stderr: "" };
          }
          return { exitCode: 0, stdout: "", stderr: "" };
        },
      });

      const result = await service.protectedBackendForHarness("codex", {
        type: "custom",
        program: "npx",
        args: ["--yes", "@agentclientprotocol/codex-acp"],
      });
      const args = result.backend?.type === "custom" ? result.backend.args : [];

      expect(result.success).toBe(true);
      expect(args).toEqual(
        expect.arrayContaining([`${authPath}:/tmp/auth.json:ro`, "CODEX_HOME=/tmp"]),
      );
      expect(args).not.toContain("CODEX_ACCESS_TOKEN");
      expect(args.join(" ")).not.toContain("must-not-forward");
    } finally {
      await rm(homeDir, { recursive: true, force: true });
    }
  });

  it("translates host-loopback yallm URLs without changing other routes", () => {
    expect(containerHostBridgeUrl("http://127.0.0.1:4000/v1")).toBe(
      "http://host.docker.internal:4000/v1",
    );
    expect(containerHostBridgeUrl("http://localhost:11434")).toBe(
      "http://host.docker.internal:11434",
    );
    expect(containerHostBridgeUrl("https://api.openai.com/v1")).toBe("https://api.openai.com/v1");
    expect(
      containerHostBridgeEnv({
        YALLM_DEFAULT_PROVIDER: "anthropic",
        OPENAI_BASE_URL: "http://127.0.0.1:4000/v1",
        OPENAI_MODEL: "anthropic:claude-sonnet",
      }),
    ).toEqual({
      YALLM_DEFAULT_PROVIDER: "anthropic",
      OPENAI_BASE_URL: "http://host.docker.internal:4000/v1",
      OPENAI_MODEL: "anthropic:claude-sonnet",
    });
  });

  it("blocks protected harness backends when Apple Container is not ready", async () => {
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "darwin",
      arch: "arm64",
      macosVersion: "26.5.2",
      homeDir: "/Users/test",
      findExecutable: async () => null,
      runCommand: async () => ({ exitCode: 0, stdout: "", stderr: "" }),
    });

    const result = await service.protectedBackendForHarness("codex", {
      type: "custom",
      program: "npx",
      args: ["--yes", "@agentclientprotocol/codex-acp"],
    });

    expect(result.success).toBe(false);
    expect(result.mode).toBe("protected");
    expect(result.error).toMatch(/Apple Container/);
  });
});
