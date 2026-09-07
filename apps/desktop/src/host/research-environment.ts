import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { readFile, realpath, stat } from "node:fs/promises";
import { isAbsolute, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import type {
  ScienceProcessHandle,
  ScienceProcessRuntime,
  ScienceProcessSpec,
} from "@swarmx/science";
import { z } from "zod";
import { type ResearchEnvironment as EnvironmentRecord, EnvironmentSchema } from "../settings.js";
import { spawnProcess } from "./process-runner.js";
import type { SettingsStore } from "./workspace-settings.js";

const exec = promisify(execFile);
const RECIPE = fileURLToPath(new URL("../../resources/python/", import.meta.url));
const MANIFEST_SOURCE =
  "import importlib.metadata,json,platform; print(json.dumps({'pythonVersion':platform.python_version(),'packages':sorted(d.metadata['Name']+'=='+d.version for d in importlib.metadata.distributions())}))";
const Manifest = EnvironmentSchema.pick({ pythonVersion: true, packages: true });

export class ResearchEnvironment implements ScienceProcessRuntime {
  private readonly active = new Set<ScienceProcessHandle>();
  private build: AbortController | undefined;
  private buildDone?: Promise<EnvironmentRecord>;
  private log = "";
  private failed = false;

  constructor(
    private readonly settings: SettingsStore,
    private readonly workspace: string,
    private readonly staging: string,
  ) {}

  get busy(): boolean {
    return this.build !== undefined || this.active.size > 0;
  }

  status() {
    const environment = this.settings.read().environment;
    return {
      state: this.build
        ? ("building" as const)
        : this.failed
          ? ("failed" as const)
          : environment
            ? ("ready" as const)
            : ("missing" as const),
      log: this.log,
      environment,
      activeProcesses: this.active.size,
    };
  }

  setup(signal?: AbortSignal): Promise<EnvironmentRecord> {
    if (this.busy)
      throw new Error("The research environment is busy. Stop execution before setup.");
    this.build = new AbortController();
    this.log = "Building the bundled Python environment…\n";
    this.failed = false;
    this.buildDone = this.buildImage(
      AbortSignal.any([this.build.signal, ...(signal ? [signal] : [])]),
    )
      .catch((error: unknown) => {
        this.failed = true;
        this.log += `\n${error instanceof Error ? error.message : String(error)}`;
        throw error;
      })
      .finally(() => {
        this.build = undefined;
      });
    return this.buildDone;
  }

  cancelSetup(): void {
    this.build?.abort(new Error("Environment setup cancelled."));
  }

  async inspect(): Promise<EnvironmentRecord> {
    const environment = this.settings.read().environment;
    if (!environment) throw new Error("Set up the research environment in Settings first.");
    const { stdout } = await exec(
      "docker",
      ["image", "inspect", "--format", "{{.Id}}", environment.imageId],
      { timeout: 15000 },
    );
    if (stdout.trim() !== environment.imageId)
      throw new Error("The saved research image is unavailable.");
    return environment;
  }

  async resolveExecutable(
    command: string,
    _env?: Readonly<Record<string, string>>,
    signal?: AbortSignal,
  ): Promise<string> {
    signal?.throwIfAborted();
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/u.test(command))
      throw new Error(
        "The research executable must be a command inside the configured container image.",
      );
    await this.inspect();
    return command;
  }

  async spawn(spec: ScienceProcessSpec): Promise<ScienceProcessHandle> {
    spec.signal?.throwIfAborted();
    const { environment, policy } = this.settings.read();
    if (!environment) throw new Error("Set up the research environment in Settings first.");
    const workspace = await realpath(this.workspace);
    const cwd = await realpath(spec.cwd);
    if (!contains(workspace, cwd))
      throw new Error("Research working directory is outside the workspace.");
    const mounts = [mount(workspace, policy.filesystem === "read-only")];
    const variables = [
      "HOME=/tmp",
      "MPLCONFIGDIR=/tmp/matplotlib",
      "MPLBACKEND=Agg",
      "PYTHONDONTWRITEBYTECODE=1",
      `SWARMX_RUNTIME_IMAGE=${environment.imageId}`,
      `SWARMX_RUNTIME_PLATFORM=${environment.platform}`,
      `SWARMX_RUNTIME_POLICY=${policy.filesystem};network=none;cpus=${policy.cpus};memoryMb=${policy.memoryMb};timeoutSeconds=${policy.timeoutSeconds}`,
    ];
    for (const [key, value] of Object.entries(spec.env ?? {})) {
      if (!/^SWARMX_SCIENCE_INPUT_\d+$/u.test(key) || value === undefined) continue;
      const path = await realpath(value);
      const staging = await realpath(this.staging);
      if (!contains(staging, path) || !(await stat(path)).isFile())
        throw new Error("Research input is not a staged artifact.");
      mounts.push(mount(path, true));
      variables.push(`${key}=${path}`);
    }
    const name = `swarmx-research-${randomUUID()}`;
    const args = [
      "create",
      "--name",
      name,
      "--pull",
      "never",
      "--platform",
      environment.platform,
      "--interactive",
      "--init",
      "--network",
      "none",
      "--read-only",
      "--cap-drop",
      "ALL",
      "--security-opt",
      "no-new-privileges",
      "--pids-limit",
      "128",
      "--cpus",
      String(policy.cpus),
      "--memory",
      `${policy.memoryMb}m`,
      "--memory-swap",
      `${policy.memoryMb}m`,
      "--tmpfs",
      "/tmp:rw,nosuid,nodev,size=256m",
      "--user",
      `${process.getuid?.() ?? 1000}:${process.getgid?.() ?? 1000}`,
      "--workdir",
      cwd,
      ...mounts.flatMap((value) => ["--mount", value]),
      ...variables.flatMap((value) => ["--env", value]),
      "--entrypoint",
      "/usr/bin/timeout",
      environment.imageId,
      "--signal=KILL",
      `${policy.timeoutSeconds}s`,
      ...spec.argv,
    ];
    await exec("docker", args, { timeout: 30000 });
    try {
      spec.signal?.throwIfAborted();
      const spawned = spawnProcess({
        argv: ["docker", "start", "--attach", "--interactive", name],
        cwd: workspace,
        graceMs: spec.graceMs,
        ...(spec.signal ? { signal: spec.signal } : {}),
        stdin: spec.stdio.stdin,
        stdout: spec.stdio.stdout,
        stderr: spec.stdio.stderr,
      });
      const done = spawned.done.finally(async () => {
        await exec("docker", ["rm", "--force", name], { timeout: 15000 });
        this.active.delete(handle);
      });
      const handle: ScienceProcessHandle = {
        pid: spawned.child.pid ?? 0,
        stdin: spawned.child.stdin ?? undefined,
        stdout: spawned.child.stdout ?? undefined,
        stderr: spawned.child.stderr ?? undefined,
        collected: {
          ...(spawned.stdout ? { stdout: spawned.stdout } : {}),
          ...(spawned.stderr ? { stderr: spawned.stderr } : {}),
        },
        done,
        terminate: spawned.terminate,
        async waitForExit() {
          await done;
          return true;
        },
      };
      this.active.add(handle);
      return handle;
    } catch (error) {
      await exec("docker", ["rm", "--force", name], { timeout: 15000 });
      throw error;
    }
  }

  async close(): Promise<void> {
    const building = this.build;
    this.cancelSetup();
    for (const handle of this.active) handle.terminate();
    const results = await Promise.allSettled([...this.active].map((handle) => handle.done));
    if (building && this.buildDone)
      await this.buildDone.catch((error: unknown) => {
        if (!building.signal.aborted) throw error;
      });
    const failure = results.find((result) => result.status === "rejected");
    if (failure?.status === "rejected") throw failure.reason;
  }

  private async buildImage(signal: AbortSignal): Promise<EnvironmentRecord> {
    const recipe = await readFile(join(RECIPE, "Dockerfile"));
    const recipeDigest = `sha256:${createHash("sha256").update(recipe).digest("hex")}`;
    const tag = `swarmx-research:${recipeDigest.slice(7, 23)}`;
    const build = spawnProcess({
      argv: ["docker", "build", "--tag", tag, RECIPE],
      graceMs: 2000,
      signal,
      stdin: "ignore",
      stdout: { maxBytes: 64000 },
      stderr: { maxBytes: 64000 },
    });
    const append = (chunk: Buffer) => {
      this.log = (this.log + chunk.toString()).slice(-64000);
    };
    build.child.stdout?.on("data", append);
    build.child.stderr?.on("data", append);
    const outcome = await build.done;
    signal.throwIfAborted();
    if (outcome.exitCode !== 0) throw new Error("Environment build failed. See the setup log.");
    const { stdout } = await exec("docker", ["image", "inspect", "--format", "{{json .}}", tag], {
      timeout: 15000,
      signal,
    });
    const image = z
      .object({ Id: z.string(), Os: z.string(), Architecture: z.string() })
      .parse(JSON.parse(stdout));
    const { imageId, platform } = EnvironmentSchema.pick({ imageId: true, platform: true }).parse({
      imageId: image.Id,
      platform: `${image.Os}/${image.Architecture}`,
    });
    const probeName = `swarmx-environment-${randomUUID()}`;
    await exec(
      "docker",
      [
        "create",
        "--name",
        probeName,
        "--platform",
        platform,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--entrypoint",
        "python3",
        imageId,
        "-B",
        "-c",
        MANIFEST_SOURCE,
      ],
      { timeout: 30000 },
    );
    let manifest: z.infer<typeof Manifest>;
    try {
      const result = await exec("docker", ["start", "--attach", probeName], {
        timeout: 30000,
        signal,
      });
      manifest = Manifest.parse(JSON.parse(result.stdout));
    } finally {
      await exec("docker", ["rm", "--force", probeName], { timeout: 15000 });
    }
    const environment = EnvironmentSchema.parse({
      ...manifest,
      imageId,
      recipeDigest,
      platform,
      createdAt: new Date().toISOString(),
    });
    this.settings.write({ ...this.settings.read(), environment });
    this.log += `\nReady: ${imageId}\n`;
    return environment;
  }
}

function contains(root: string, path: string): boolean {
  const part = relative(root, path);
  return !isAbsolute(part) && part !== ".." && !part.startsWith(`..${sep}`);
}

function mount(path: string, readonly: boolean): string {
  if (/[",\n\r]/u.test(path))
    throw new Error("Docker workspace paths cannot contain commas, quotes or newlines.");
  return `type=bind,source=${path},target=${path}${readonly ? ",readonly" : ""}`;
}
