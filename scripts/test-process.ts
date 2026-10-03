import { type ChildProcess, spawn } from "node:child_process";

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

  readFrom(offset: number): { text: string; nextOffset: number; lossy: boolean } {
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
  readonly done: Promise<{ exitCode: number | null; signal: NodeJS.Signals | null }>;
  readonly stdout?: OutputBuffer;
  readonly stderr?: OutputBuffer;
  terminate(): void;
}

export function spawnProcess(spec: SpawnSpec): SpawnedProcess {
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
  const done = new Promise<{ exitCode: number | null; signal: NodeJS.Signals | null }>(
    (resolveDone, reject) => {
      child.once("error", reject);
      child.once("close", (exitCode, signal) => resolveDone({ exitCode, signal }));
    },
  ).finally(() => {
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
