import type { Readable, Writable } from "node:stream";

export interface ScienceProcessOutcome {
  readonly exitCode: number | null;
  readonly signal: NodeJS.Signals | null;
}

export interface ScienceProcessOutputRead {
  readonly text: string;
  readonly nextOffset: number;
  readonly lossy: boolean;
  readonly spillPath?: string;
}

export interface ScienceProcessOutputReader {
  readFrom(fromByte: number): ScienceProcessOutputRead;
}

export interface ScienceProcessHandle {
  readonly pid: number;
  readonly stdin: Writable | undefined;
  readonly stdout: Readable | undefined;
  readonly stderr: Readable | undefined;
  readonly collected: {
    readonly stdout?: ScienceProcessOutputReader;
    readonly stderr?: ScienceProcessOutputReader;
  };
  readonly done: Promise<ScienceProcessOutcome>;
  terminate(): void;
  waitForExit(signal?: AbortSignal): Promise<boolean>;
}

export interface ScienceProcessSpec {
  readonly argv: readonly string[];
  readonly cwd: string;
  readonly stdio: {
    readonly stdin: "ignore" | "pipe" | { readonly data: string };
    readonly stdout: "pipe" | "inherit" | { readonly maxBytes: number };
    readonly stderr: "pipe" | "inherit" | { readonly maxBytes: number };
  };
  readonly graceMs: number;
  readonly signal?: AbortSignal | undefined;
  readonly env?: NodeJS.ProcessEnv | undefined;
}

export interface ScienceProcessRuntime {
  resolveExecutable(
    command: string,
    env?: Readonly<Record<string, string>>,
    signal?: AbortSignal,
  ): Promise<string>;
  spawn(spec: ScienceProcessSpec): ScienceProcessHandle;
}

const SENSITIVE_ENVIRONMENT_NAME = /KEY|PASSWORD|SECRET|TOKEN/iu;

export function scrubbedScienceEnvironment(): Record<string, string> {
  const environment: Record<string, string> = {};
  for (const [key, value] of Object.entries(process.env)) {
    if (
      value !== undefined &&
      !SENSITIVE_ENVIRONMENT_NAME.test(key) &&
      !key.toUpperCase().startsWith("DSH_")
    ) {
      environment[key] = value;
    }
  }
  return environment;
}
