export interface ProcessOutcome {
  exitCode: number | null;
  signal: NodeJS.Signals | null;
}

export interface ProcessOutputSnapshot {
  text: string;
  lossy: boolean;
}

export interface ProcessHandle {
  done: Promise<ProcessOutcome>;
  collected: {
    stdout?: { readFrom(offset: number): ProcessOutputSnapshot };
    stderr?: { readFrom(offset: number): ProcessOutputSnapshot };
  };
  terminate(): void;
  waitForExit(): Promise<ProcessOutcome>;
}

export interface ProcessSpawnOptions {
  argv: readonly string[];
  cwd?: string;
  env?: Readonly<Record<string, string>>;
  graceMs?: number;
  signal?: AbortSignal;
  stdio: {
    stdin: "ignore";
    stdout: { maxBytes: number };
    stderr: { maxBytes: number };
  };
}

export interface ProcessRunner {
  resolveExecutable(
    command: string,
    options?: Readonly<Record<string, unknown>>,
    signal?: AbortSignal,
  ): Promise<string>;
  spawn(options: ProcessSpawnOptions): ProcessHandle;
}
