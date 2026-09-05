import { type ChildProcess, spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { delimiter, dirname, isAbsolute, join, resolve } from "node:path";
import type {
  ProcessHandle,
  ProcessOutcome,
  ProcessRunner,
  ProcessSpawnOptions,
} from "@swarmx/dvc";
import type {
  ScienceProcessHandle,
  ScienceProcessOutcome,
  ScienceProcessOutputRead,
  ScienceProcessRuntime,
  ScienceProcessSpec,
} from "@swarmx/science";

interface SpawnSpec {
  readonly argv: readonly string[];
  readonly cwd?: string;
  readonly env?: NodeJS.ProcessEnv;
  readonly graceMs: number;
  readonly signal?: AbortSignal;
  readonly stdin: "ignore" | "pipe" | { readonly data: string };
  readonly stdout: "inherit" | "pipe" | { readonly maxBytes: number };
  readonly stderr: "inherit" | "pipe" | { readonly maxBytes: number };
}

class OutputBuffer {
  private content = Buffer.alloc(0);
  private totalBytes = 0;

  constructor(private readonly maxBytes: number) {}

  append(chunk: Buffer | string): void {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    this.totalBytes += bytes.length;
    this.content = Buffer.concat([this.content, bytes]).subarray(-this.maxBytes);
  }

  readFrom(offset: number): ScienceProcessOutputRead {
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error("Output offset is invalid.");
    const retainedFrom = this.totalBytes - this.content.length;
    return {
      text: this.content.subarray(Math.max(0, offset - retainedFrom)).toString("utf8"),
      nextOffset: this.totalBytes,
      lossy: offset < retainedFrom,
    };
  }
}

interface SpawnedProcess {
  readonly child: ChildProcess;
  readonly done: Promise<ProcessOutcome>;
  readonly stdout?: OutputBuffer;
  readonly stderr?: OutputBuffer;
  terminate(): void;
}

function spawnProcess(spec: SpawnSpec): SpawnedProcess {
  const command = spec.argv[0];
  if (command === undefined) throw new Error("Process command is missing.");
  spec.signal?.throwIfAborted();
  const stdout =
    typeof spec.stdout === "object" ? new OutputBuffer(spec.stdout.maxBytes) : undefined;
  const stderr =
    typeof spec.stderr === "object" ? new OutputBuffer(spec.stderr.maxBytes) : undefined;
  const child = spawn(command, spec.argv.slice(1), {
    ...(spec.cwd === undefined ? {} : { cwd: spec.cwd }),
    ...(spec.env === undefined ? {} : { env: spec.env }),
    stdio: [
      spec.stdin === "ignore" ? "ignore" : "pipe",
      spec.stdout === "inherit" ? "inherit" : "pipe",
      spec.stderr === "inherit" ? "inherit" : "pipe",
    ],
  });
  child.stdout?.on("data", (chunk: Buffer) => stdout?.append(chunk));
  child.stderr?.on("data", (chunk: Buffer) => stderr?.append(chunk));
  if (typeof spec.stdin === "object") child.stdin?.end(spec.stdin.data);

  let killTimer: NodeJS.Timeout | undefined;
  const terminate = () => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    child.kill("SIGTERM");
    killTimer ??= setTimeout(() => child.kill("SIGKILL"), spec.graceMs);
    killTimer.unref();
  };
  const aborted = () => terminate();
  spec.signal?.addEventListener("abort", aborted, { once: true });
  const done = new Promise<ProcessOutcome>((resolveDone, reject) => {
    child.once("error", reject);
    child.once("exit", (exitCode, signal) => resolveDone({ exitCode, signal }));
  }).finally(() => {
    if (killTimer !== undefined) clearTimeout(killTimer);
    spec.signal?.removeEventListener("abort", aborted);
  });
  return {
    child,
    done,
    ...(stdout === undefined ? {} : { stdout }),
    ...(stderr === undefined ? {} : { stderr }),
    terminate,
  };
}

async function resolveExecutable(
  command: string,
  environment: Readonly<Record<string, string | undefined>>,
  signal?: AbortSignal,
): Promise<string> {
  signal?.throwIfAborted();
  if (command.includes("/") || command.includes("\\")) {
    const candidate = isAbsolute(command) ? command : resolve(command);
    await access(candidate);
    return candidate;
  }
  const path = environment.PATH ?? process.env.PATH ?? "";
  const extensions =
    process.platform === "win32"
      ? (environment.PATHEXT ?? process.env.PATHEXT ?? ".EXE;.CMD;.BAT").split(";")
      : [""];
  for (const directory of path.split(delimiter)) {
    for (const extension of extensions) {
      signal?.throwIfAborted();
      const candidate = join(directory || dirname(process.execPath), `${command}${extension}`);
      try {
        await access(candidate);
        return candidate;
      } catch {
        // Continue through PATH; absence is expected.
      }
    }
  }
  throw new Error(`Executable "${command}" was not found on PATH.`);
}

function collected(process: SpawnedProcess) {
  return {
    ...(process.stdout === undefined ? {} : { stdout: process.stdout }),
    ...(process.stderr === undefined ? {} : { stderr: process.stderr }),
  };
}

export class NodeScienceProcessRuntime implements ScienceProcessRuntime {
  resolveExecutable(
    command: string,
    environment: Readonly<Record<string, string>> = {},
    signal?: AbortSignal,
  ): Promise<string> {
    return resolveExecutable(command, { ...process.env, ...environment }, signal);
  }

  spawn(spec: ScienceProcessSpec): ScienceProcessHandle {
    const spawned = spawnProcess({
      argv: spec.argv,
      cwd: spec.cwd,
      env: { ...process.env, ...spec.env },
      graceMs: spec.graceMs,
      ...(spec.signal === undefined ? {} : { signal: spec.signal }),
      stdin: spec.stdio.stdin,
      stdout: spec.stdio.stdout,
      stderr: spec.stdio.stderr,
    });
    return {
      pid: spawned.child.pid ?? 0,
      stdin: spawned.child.stdin ?? undefined,
      stdout: spawned.child.stdout ?? undefined,
      stderr: spawned.child.stderr ?? undefined,
      collected: collected(spawned),
      done: spawned.done as Promise<ScienceProcessOutcome>,
      terminate: spawned.terminate,
      async waitForExit(signal?: AbortSignal) {
        if (signal === undefined) {
          await spawned.done;
          return true;
        }
        if (signal.aborted) return false;
        return Promise.race([
          spawned.done.then(() => true),
          new Promise<false>((resolveWait) =>
            signal.addEventListener("abort", () => resolveWait(false), { once: true }),
          ),
        ]);
      },
    };
  }
}

export class NodeProcessRunner implements ProcessRunner {
  resolveExecutable(
    command: string,
    _options: Readonly<Record<string, unknown>> = {},
    signal?: AbortSignal,
  ): Promise<string> {
    return resolveExecutable(command, process.env, signal);
  }

  spawn(spec: ProcessSpawnOptions): ProcessHandle {
    const spawned = spawnProcess({
      argv: spec.argv,
      ...(spec.cwd === undefined ? {} : { cwd: spec.cwd }),
      env: { ...process.env, ...spec.env },
      graceMs: spec.graceMs ?? 2_000,
      ...(spec.signal === undefined ? {} : { signal: spec.signal }),
      stdin: "ignore",
      stdout: spec.stdio.stdout,
      stderr: spec.stdio.stderr,
    });
    return {
      done: spawned.done,
      collected: collected(spawned),
      terminate: spawned.terminate,
      waitForExit: () => spawned.done,
    };
  }
}
