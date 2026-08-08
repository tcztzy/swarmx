import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  computePythonEnvironmentDigest,
  type PythonEnvironmentCommandResult,
  type PythonWorkerEnvironmentConfig,
  type PythonWorkerEnvironmentHost,
  PythonWorkerEnvironmentService,
} from "./python-environment.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("PythonWorkerEnvironmentService", () => {
  it("discovers uv-managed Python without downloads or environment mutation", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture);
    const service = new PythonWorkerEnvironmentService(fixture.config, host);

    const status = await service.status();

    expect(status.state).toBe("environment_missing");
    expect(status.ready).toBe(false);
    expect(status.setupAvailable).toBe(true);
    expect(status.environment?.digest).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(status.environment?.path).toBe(
      path.join(fixture.config.environmentRoot, status.environment?.digest.replace(":", "-") ?? ""),
    );
    const discoveryCall = host.runCommand.mock.calls.find(
      ([program, args]) => program === fixture.uvPath && args[0] === "python",
    );
    expect(discoveryCall?.[1]).toEqual(
      expect.arrayContaining([
        "find",
        "--managed-python",
        "--system",
        "--no-python-downloads",
        "--offline",
        "--no-project",
        "--no-cache",
        "--no-config",
      ]),
    );
    expect(host.runCommand.mock.calls.some(([, args]) => args.includes("install"))).toBe(false);
    expect(host.runCommand.mock.calls.some(([, args]) => args[0] === "sync")).toBe(false);

    const callsBeforePlan = host.runCommand.mock.calls.length;
    const plan = service.plan(status);
    expect(plan).toMatchObject({ requiresConfirmation: true, requiresUserAction: false });
    expect(plan.actions).toEqual([
      expect.objectContaining({
        kind: "sync_environment",
        risk: "install",
        command: expect.objectContaining({
          program: fixture.uvPath,
          env: expect.objectContaining({
            UV_PROJECT_ENVIRONMENT: status.environment?.path,
            UV_PYTHON_DOWNLOADS: "never",
          }),
        }),
      }),
    ]);
    expect(host.runCommand).toHaveBeenCalledTimes(callsBeforePlan);
  });

  it("plans an explicit managed-Python install when read-only discovery finds none", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture, { managedPythonMissing: true });
    const service = new PythonWorkerEnvironmentService(fixture.config, host);

    const status = await service.status();
    const callsBeforePlan = host.runCommand.mock.calls.length;
    const plan = service.plan(status);

    expect(status.state).toBe("managed_python_missing");
    expect(status.environment).toBeUndefined();
    expect(plan.requiresConfirmation).toBe(true);
    expect(plan.actions).toEqual([
      expect.objectContaining({
        kind: "install_managed_python",
        command: {
          program: fixture.uvPath,
          args: ["python", "install", ">=3.11", "--managed-python"],
          cwd: fixture.projectDirectory,
          env: {},
        },
      }),
    ]);
    expect(host.runCommand).toHaveBeenCalledTimes(callsBeforePlan);
    expect(host.runCommand.mock.calls.some(([, args]) => args.includes("install"))).toBe(false);
  });

  it("emits a secret-minimal direct launch spec only for a checked environment", async () => {
    const fixture = await createFixture();
    const host = fakeHost(fixture);
    const service = new PythonWorkerEnvironmentService(fixture.config, host);
    const missing = await service.status();
    await createEnvironment(missing);

    const status = await service.status();
    const launch = await service.launchSpec(status);

    expect(status.state).toBe("ready");
    expect(status.environment?.status).toBe("ready");
    expect(launch).toEqual(
      expect.objectContaining({
        program: status.environment?.pythonPath,
        args: expect.arrayContaining([
          "-I",
          "-B",
          "-u",
          "-c",
          "--environment-digest",
          status.environment?.digest,
        ]),
        cwd: fixture.projectDirectory,
        environmentDigest: status.environment?.digest,
      }),
    );
    expect(launch.args[4]).toBe("# worker\n");
    expect(launch.env).toMatchObject({
      PATH: path.dirname(status.environment?.pythonPath ?? ""),
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONUTF8: "1",
    });
    expect(launch.env).not.toHaveProperty("OPENAI_API_KEY");
    expect(launch.env).not.toHaveProperty("ANTHROPIC_API_KEY");
    expect(launch.env).not.toHaveProperty("PYTHONPATH");

    const syncCheck = host.runCommand.mock.calls.find(([, args]) => args[0] === "sync");
    expect(syncCheck?.[1]).toEqual(
      expect.arrayContaining([
        "--locked",
        "--check",
        "--no-default-groups",
        "--managed-python",
        "--no-python-downloads",
        "--offline",
        "--no-cache",
      ]),
    );
    expect(syncCheck?.[2].env).toMatchObject({
      UV_PROJECT_ENVIRONMENT: status.environment?.path,
      UV_PYTHON_DOWNLOADS: "never",
    });
    expect(syncCheck?.[2].env).not.toHaveProperty("OPENAI_API_KEY");
    const environment = status.environment;
    if (!environment) throw new Error("Expected a ready environment.");
    await expect(
      service.launchSpec({
        ...status,
        environment: { ...environment, path: "/tmp/forged-python-environment" },
      }),
    ).rejects.toThrow("does not match the configured environment");

    await writeFile(fixture.config.workerPath, "print('mutated worker')\n", "utf8");
    await expect(service.launchSpec(status)).rejects.toThrow(
      /changed after the supplied health check/,
    );
  });

  it("changes the digest with runtime assets and plans stale-environment repair purely", async () => {
    const fixture = await createFixture();
    const healthyHost = fakeHost(fixture);
    const healthyService = new PythonWorkerEnvironmentService(fixture.config, healthyHost);
    const missing = await healthyService.status();
    await createEnvironment(missing);

    const staleHost = fakeHost(fixture, { syncFailure: "environment is not synchronized" });
    const staleService = new PythonWorkerEnvironmentService(fixture.config, staleHost);
    const stale = await staleService.status();
    const callsBeforePlan = staleHost.runCommand.mock.calls.length;
    const plan = staleService.plan(stale);

    expect(stale.state).toBe("environment_stale");
    expect(plan.actions).toEqual([
      expect.objectContaining({
        kind: "sync_environment",
        risk: "repair",
        command: expect.objectContaining({
          args: expect.arrayContaining([
            "--locked",
            "--no-default-groups",
            "--no-python-downloads",
          ]),
        }),
      }),
    ]);
    expect(staleHost.runCommand).toHaveBeenCalledTimes(callsBeforePlan);

    const originalDigest = stale.environment?.digest;
    await writeFile(fixture.config.workerPath, "# changed worker\n", "utf8");
    const changed = await healthyService.status();
    expect(changed.environment?.digest).not.toBe(originalDigest);
    expect(changed.state).toBe("environment_missing");
  });

  it("produces stable canonical environment digests", () => {
    const input = {
      projectSha256: "1".repeat(64),
      lockSha256: "2".repeat(64),
      workerSha256: "3".repeat(64),
      additionalSourceSha256s: [],
      dependencyGroups: [],
      uvVersion: "0.11.21",
      pythonRequest: ">=3.11",
      pythonImplementation: "cpython",
      pythonVersion: "3.13.14",
      platform: "darwin" as const,
      architecture: "arm64",
    };
    expect(computePythonEnvironmentDigest(input)).toBe(computePythonEnvironmentDigest(input));
    expect(computePythonEnvironmentDigest({ ...input, workerSha256: "4".repeat(64) })).not.toBe(
      computePythonEnvironmentDigest(input),
    );
    expect(
      computePythonEnvironmentDigest({
        ...input,
        additionalSourceSha256s: ["5".repeat(64)],
      }),
    ).not.toBe(computePythonEnvironmentDigest(input));
  });
});

interface Fixture {
  root: string;
  projectDirectory: string;
  uvPath: string;
  managedPythonPath: string;
  config: PythonWorkerEnvironmentConfig;
}

async function createFixture(): Promise<Fixture> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-python-environment-"));
  temporaryRoots.push(root);
  const projectDirectory = path.join(root, "python");
  const toolsDirectory = path.join(root, "tools");
  const uvPath = path.join(toolsDirectory, "uv");
  const managedPythonPath = path.join(root, "managed", "bin", "python3.13");
  await Promise.all([
    mkdir(projectDirectory, { recursive: true }),
    mkdir(toolsDirectory, { recursive: true }),
    mkdir(path.dirname(managedPythonPath), { recursive: true }),
  ]);
  const config: PythonWorkerEnvironmentConfig = {
    projectPath: path.join(projectDirectory, "pyproject.toml"),
    lockPath: path.join(projectDirectory, "uv.lock"),
    workerPath: path.join(projectDirectory, "worker.py"),
    environmentRoot: path.join(root, "environments"),
    workingDirectory: projectDirectory,
  };
  await Promise.all([
    writeFile(config.projectPath, "[project]\nname='swarmx'\n", "utf8"),
    writeFile(config.lockPath, "version = 1\n", "utf8"),
    writeFile(config.workerPath, "# worker\n", "utf8"),
    writeFile(uvPath, "", "utf8"),
    writeFile(managedPythonPath, "", "utf8"),
  ]);
  await Promise.all([chmod(uvPath, 0o755), chmod(managedPythonPath, 0o755)]);
  return { root, projectDirectory, uvPath, managedPythonPath, config };
}

function fakeHost(
  fixture: Fixture,
  options: { managedPythonMissing?: boolean; syncFailure?: string } = {},
) {
  const runCommand = vi.fn(
    async (program: string, args: string[]): Promise<PythonEnvironmentCommandResult> => {
      if (program === fixture.uvPath && args[0] === "--version") {
        return success("uv 0.11.21\n");
      }
      if (program === fixture.uvPath && args[0] === "python") {
        return options.managedPythonMissing
          ? { exitCode: 2, stdout: "", stderr: "managed Python missing\n" }
          : success(`${fixture.managedPythonPath}\n`);
      }
      if (program === fixture.uvPath && args[0] === "sync") {
        return options.syncFailure
          ? { exitCode: 1, stdout: "", stderr: `${options.syncFailure}\n` }
          : success("environment is synchronized\n");
      }
      if (program === fixture.managedPythonPath || program.endsWith(path.join("bin", "python"))) {
        return success(
          JSON.stringify({
            implementation: "cpython",
            version: "3.13.14",
            architecture: "arm64",
            basePrefix: path.dirname(path.dirname(fixture.managedPythonPath)),
          }),
        );
      }
      return { exitCode: 1, stdout: "", stderr: `unexpected command: ${program}\n` };
    },
  );
  return {
    env: {
      PATH: path.dirname(fixture.uvPath),
      HOME: fixture.root,
      OPENAI_API_KEY: "must-not-forward",
      ANTHROPIC_API_KEY: "must-not-forward",
      PYTHONPATH: "/untrusted/python/path",
    },
    platform: "darwin",
    homeDir: fixture.root,
    now: () => new Date("2026-08-05T00:00:00.000Z"),
    findExecutable: vi.fn(async (command: string) => (command === "uv" ? fixture.uvPath : null)),
    runCommand,
  } satisfies PythonWorkerEnvironmentHost & { runCommand: typeof runCommand };
}

async function createEnvironment(
  status: Awaited<ReturnType<PythonWorkerEnvironmentService["status"]>>,
) {
  const pythonPath = status.environment?.pythonPath;
  if (!pythonPath) throw new Error("Expected an environment path.");
  await mkdir(path.dirname(pythonPath), { recursive: true });
  await writeFile(pythonPath, "", "utf8");
  await chmod(pythonPath, 0o755);
}

function success(stdout: string): PythonEnvironmentCommandResult {
  return { exitCode: 0, stdout, stderr: "" };
}
