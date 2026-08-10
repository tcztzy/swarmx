import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { z } from "zod";

const SHA256_PATTERN = /^sha256:[a-f0-9]{64}$/;
const VERSION_PATTERN = /^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/;
const VERSION_TIMEOUT_MS = 5_000;
const OUTPUT_LIMIT_BYTES = 16_384;

const MemoryRuntimeTargetSchema = z
  .object({
    platform: z.enum(["darwin", "linux", "win32"]),
    architecture: z.enum(["arm64", "x64"]),
    path: z.string().min(1).max(4_096),
    sha256: z.string().regex(SHA256_PATTERN),
  })
  .strict();

export const MemoryRuntimeManifestSchema = z
  .object({
    schemaVersion: z.literal(1),
    runtimeVersion: z.string().regex(VERSION_PATTERN),
    protocolVersion: z.literal(1),
    targets: z.array(MemoryRuntimeTargetSchema).min(1).max(8),
  })
  .strict()
  .superRefine((manifest, context) => {
    const targets = new Set<string>();
    for (const [index, target] of manifest.targets.entries()) {
      const key = `${target.platform}:${target.architecture}`;
      if (targets.has(key)) {
        context.addIssue({
          code: "custom",
          message: `Duplicate Memory runtime target: ${key}`,
          path: ["targets", index],
        });
      }
      targets.add(key);
    }
  });

const MemoryRuntimeHandshakeSchema = z
  .object({
    name: z.literal("swarmx-mem"),
    version: z.string().regex(VERSION_PATTERN),
    protocolVersion: z.literal(1),
  })
  .strict();

const MemoryRuntimeReadyStatusSchema = z
  .object({
    state: z.literal("ready"),
    ready: z.literal(true),
    repairAvailable: z.literal(false),
    version: z.string().regex(VERSION_PATTERN),
    protocolVersion: z.literal(1),
    binaryPath: z.string().min(1).max(4_096),
    binaryDigest: z.string().regex(SHA256_PATTERN),
  })
  .strict();

const MemoryRuntimeUnavailableStatusSchema = z
  .object({
    state: z.enum(["missing", "invalid", "unsupported"]),
    ready: z.literal(false),
    repairAvailable: z.boolean(),
    reason: z
      .enum([
        "binary_missing",
        "not_executable",
        "digest_mismatch",
        "version_check_failed",
        "incompatible_version",
        "incompatible_protocol",
        "unsupported_target",
      ])
      .optional(),
  })
  .strict();

export const MemoryRuntimeStatusSchema = z.discriminatedUnion("ready", [
  MemoryRuntimeReadyStatusSchema,
  MemoryRuntimeUnavailableStatusSchema,
]);

export const MemoryRuntimeLaunchSpecSchema = z
  .object({
    program: z.string().min(1).max(4_096),
    args: z.array(z.string().max(4_096)).max(16),
    cwd: z.string().min(1).max(4_096),
    env: z.record(z.string(), z.string().max(32_768)),
    binaryDigest: z.string().regex(SHA256_PATTERN),
    protocolVersion: z.literal(1),
    runtimeVersion: z.string().regex(VERSION_PATTERN),
    memoryRoot: z.string().min(1).max(4_096),
  })
  .strict();

export type MemoryRuntimeManifest = z.infer<typeof MemoryRuntimeManifestSchema>;
export type MemoryRuntimeStatus = z.infer<typeof MemoryRuntimeStatusSchema>;
export type MemoryRuntimeLaunchSpec = z.infer<typeof MemoryRuntimeLaunchSpecSchema>;

export interface MemoryRuntimeCommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export interface MemoryRuntimeEnvironmentHost {
  platform: NodeJS.Platform;
  architecture: string;
  homeDir: string;
  env: NodeJS.ProcessEnv;
  runCommand(
    program: string,
    args: readonly string[],
    options: { cwd: string; env: Record<string, string>; timeoutMs: number },
  ): Promise<MemoryRuntimeCommandResult>;
}

export interface MemoryRuntimeRepairPlan {
  requiresConfirmation: true;
  actions: Array<{
    kind: "restore_packaged_runtime";
    risk: "repair";
    description: string;
  }>;
}

export interface MemoryRuntimeEnvironmentOptions {
  manifestRoot?: string;
}

export class MemoryRuntimeEnvironmentService {
  readonly manifest: MemoryRuntimeManifest;
  readonly host: MemoryRuntimeEnvironmentHost;
  readonly manifestRoot: string;

  constructor(
    manifest: MemoryRuntimeManifest,
    host: MemoryRuntimeEnvironmentHost = defaultMemoryRuntimeHost(),
    options: MemoryRuntimeEnvironmentOptions = {},
  ) {
    this.manifest = MemoryRuntimeManifestSchema.parse(manifest);
    this.host = host;
    this.manifestRoot = path.resolve(options.manifestRoot ?? process.cwd());
  }

  async status(): Promise<MemoryRuntimeStatus> {
    const target = this.manifest.targets.find(
      (candidate) =>
        candidate.platform === this.host.platform &&
        candidate.architecture === this.host.architecture,
    );
    if (!target) {
      return MemoryRuntimeStatusSchema.parse({
        state: "unsupported",
        ready: false,
        repairAvailable: false,
        reason: "unsupported_target",
      });
    }

    const binaryPath = path.isAbsolute(target.path)
      ? path.normalize(target.path)
      : path.resolve(this.manifestRoot, target.path);
    let bytes: Buffer;
    try {
      bytes = await readFile(binaryPath);
    } catch (error) {
      if (isNodeError(error, "ENOENT")) {
        return MemoryRuntimeStatusSchema.parse({
          state: "missing",
          ready: false,
          repairAvailable: true,
          reason: "binary_missing",
        });
      }
      return MemoryRuntimeStatusSchema.parse({
        state: "invalid",
        ready: false,
        repairAvailable: true,
        reason: "not_executable",
      });
    }

    if (this.host.platform !== "win32") {
      try {
        await access(binaryPath, constants.X_OK);
      } catch {
        return MemoryRuntimeStatusSchema.parse({
          state: "invalid",
          ready: false,
          repairAvailable: true,
          reason: "not_executable",
        });
      }
    }

    const binaryDigest = sha256(bytes);
    if (binaryDigest !== target.sha256) {
      return MemoryRuntimeStatusSchema.parse({
        state: "invalid",
        ready: false,
        repairAvailable: true,
        reason: "digest_mismatch",
      });
    }

    const environment = this.commandEnvironment(binaryPath);
    const version = await this.host.runCommand(binaryPath, ["--version-json"], {
      cwd: path.dirname(binaryPath),
      env: environment,
      timeoutMs: VERSION_TIMEOUT_MS,
    });
    if (version.exitCode !== 0) {
      return MemoryRuntimeStatusSchema.parse({
        state: "invalid",
        ready: false,
        repairAvailable: true,
        reason: "version_check_failed",
      });
    }

    const parsed = parseHandshake(version.stdout);
    if (!parsed) {
      return MemoryRuntimeStatusSchema.parse({
        state: "invalid",
        ready: false,
        repairAvailable: true,
        reason: "incompatible_protocol",
      });
    }
    if (parsed.version !== this.manifest.runtimeVersion) {
      return MemoryRuntimeStatusSchema.parse({
        state: "invalid",
        ready: false,
        repairAvailable: true,
        reason: "incompatible_version",
      });
    }

    return MemoryRuntimeStatusSchema.parse({
      state: "ready",
      ready: true,
      repairAvailable: false,
      version: parsed.version,
      protocolVersion: parsed.protocolVersion,
      binaryPath,
      binaryDigest,
    });
  }

  plan(status: MemoryRuntimeStatus): MemoryRuntimeRepairPlan {
    const parsed = MemoryRuntimeStatusSchema.parse(status);
    if (parsed.ready || !parsed.repairAvailable) {
      throw new Error("Memory runtime status does not have an available repair action.");
    }
    return {
      requiresConfirmation: true,
      actions: [
        {
          kind: "restore_packaged_runtime",
          risk: "repair",
          description: "Restore the version-pinned Memory runtime from the SwarmX application.",
        },
      ],
    };
  }

  async launchSpec(
    checkedStatus: MemoryRuntimeStatus,
    options: { memoryRoot: string },
  ): Promise<MemoryRuntimeLaunchSpec> {
    const checked = MemoryRuntimeStatusSchema.parse(checkedStatus);
    if (!checked.ready) throw new Error("Memory runtime is not ready to launch.");
    const current = await this.status();
    if (
      !current.ready ||
      current.binaryPath !== checked.binaryPath ||
      current.binaryDigest !== checked.binaryDigest ||
      current.version !== checked.version ||
      current.protocolVersion !== checked.protocolVersion
    ) {
      throw new Error("Memory runtime changed after the supplied health check.");
    }

    const memoryRoot = path.resolve(z.string().min(1).max(4_096).parse(options.memoryRoot));
    return MemoryRuntimeLaunchSpecSchema.parse({
      program: current.binaryPath,
      args: ["serve", "--root", memoryRoot, "--stdio"],
      cwd: path.dirname(memoryRoot),
      env: this.commandEnvironment(current.binaryPath),
      binaryDigest: current.binaryDigest,
      protocolVersion: current.protocolVersion,
      runtimeVersion: current.version,
      memoryRoot,
    });
  }

  private commandEnvironment(binaryPath: string): Record<string, string> {
    return {
      HOME: this.host.homeDir,
      PATH: path.dirname(binaryPath),
      RUST_BACKTRACE: "0",
    };
  }
}

function parseHandshake(stdout: string): z.infer<typeof MemoryRuntimeHandshakeSchema> | undefined {
  try {
    return MemoryRuntimeHandshakeSchema.parse(JSON.parse(stdout));
  } catch {
    return undefined;
  }
}

function sha256(bytes: Uint8Array): string {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

function isNodeError(error: unknown, code: string): boolean {
  return error instanceof Error && "code" in error && error.code === code;
}

function defaultMemoryRuntimeHost(): MemoryRuntimeEnvironmentHost {
  return {
    platform: process.platform,
    architecture: process.arch,
    homeDir: homedir(),
    env: process.env,
    runCommand: runMemoryRuntimeCommand,
  };
}

async function runMemoryRuntimeCommand(
  program: string,
  args: readonly string[],
  options: { cwd: string; env: Record<string, string>; timeoutMs: number },
): Promise<MemoryRuntimeCommandResult> {
  return await new Promise((resolve) => {
    const child = spawn(program, [...args], {
      cwd: options.cwd,
      env: { ...options.env },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let stderr: Buffer<ArrayBufferLike> = Buffer.alloc(0);
    let settled = false;
    const finish = (result: MemoryRuntimeCommandResult): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish({ exitCode: null, stdout: "", stderr: "version check timed out" });
    }, options.timeoutMs);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout = appendBounded(stdout, chunk);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = appendBounded(stderr, chunk);
    });
    child.once("error", () => finish({ exitCode: null, stdout: "", stderr: "spawn failed" }));
    child.once("close", (exitCode) =>
      finish({ exitCode, stdout: stdout.toString("utf8"), stderr: stderr.toString("utf8") }),
    );
  });
}

function appendBounded(
  current: Buffer<ArrayBufferLike>,
  chunk: Buffer<ArrayBufferLike>,
): Buffer<ArrayBufferLike> {
  if (current.byteLength >= OUTPUT_LIMIT_BYTES) return current;
  return Buffer.concat([current, chunk.subarray(0, OUTPUT_LIMIT_BYTES - current.byteLength)]);
}
