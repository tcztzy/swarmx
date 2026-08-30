import { ScienceError } from "./errors.js";
import type {
  ScienceProcessHandle,
  ScienceProcessOutcome,
  ScienceProcessRuntime,
} from "./subprocess.js";

const PROBE_SOURCE = [
  "import hashlib, importlib.metadata, json, platform",
  'packages = sorted((distribution.metadata.get("Name", ""), distribution.version) for distribution in importlib.metadata.distributions())',
  'package_bytes = json.dumps(packages, ensure_ascii=True, separators=(",", ":")).encode("utf-8")',
  'print(json.dumps({"packageSetHash": "sha256:" + hashlib.sha256(package_bytes).hexdigest(), "pythonImplementation": platform.python_implementation(), "pythonVersion": platform.python_version()}, sort_keys=True))',
].join("\n");
const CELL_RUNNER_SOURCE = [
  "import sys",
  "source = sys.stdin.read()",
  'namespace = {"__name__": "__main__"}',
  'exec(compile(source, "<science-cell>", "exec"), namespace, namespace)',
].join("\n");
const CELL_ENVIRONMENT: NodeJS.ProcessEnv = {
  ALL_PROXY: undefined,
  HTTP_PROXY: undefined,
  HTTPS_PROXY: undefined,
  NO_PROXY: "*",
  PYTHONHOME: undefined,
  PYTHONINSPECT: undefined,
  PYTHONPATH: undefined,
  PYTHONSTARTUP: undefined,
  all_proxy: undefined,
  http_proxy: undefined,
  https_proxy: undefined,
  no_proxy: "*",
};

export interface PythonRuntimeConfig {
  readonly command: string;
  readonly graceMs: number;
  readonly maxOutputBytes: number;
}

export interface PythonExecutionResult {
  readonly durationMs: number;
  readonly environment: Record<string, string>;
  readonly outcome: ScienceProcessOutcome;
  readonly stderr: { readonly text: string; readonly truncated: boolean };
  readonly stdout: { readonly text: string; readonly truncated: boolean };
}

interface CollectedProcess {
  readonly outcome: ScienceProcessOutcome;
  readonly stderr: { readonly text: string; readonly truncated: boolean };
  readonly stdout: { readonly text: string; readonly truncated: boolean };
}

/** Stateless Python cell runner over the DSH managed subprocess seam. */
export class PythonRuntime {
  private readonly active = new Set<ScienceProcessHandle>();
  private open = true;

  constructor(
    private readonly subprocess: ScienceProcessRuntime,
    private readonly config: PythonRuntimeConfig,
  ) {}

  async execute(
    cwd: string,
    source: string,
    signal?: AbortSignal,
    inputEnvironment: Readonly<Record<string, string>> = {},
  ): Promise<PythonExecutionResult> {
    this.ensureOpen();
    signal?.throwIfAborted();
    let executable: string;
    try {
      executable = await this.subprocess.resolveExecutable(this.config.command, {}, signal);
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new ScienceError("Configured Python executable is unavailable", "PYTHON_UNAVAILABLE", {
        cause: error,
      });
    }
    this.ensureOpen();

    const probe = await this.collect(
      {
        argv: [executable, "-B", "-c", PROBE_SOURCE],
        cwd,
        stdin: "ignore",
        maxOutputBytes: 4_096,
      },
      signal,
    );
    if (probe.outcome.exitCode !== 0 || probe.outcome.signal !== null) {
      throw new ScienceError("Python environment probe failed", "PYTHON_UNAVAILABLE");
    }
    let environment: Record<string, string>;
    try {
      const value = JSON.parse(probe.stdout.text) as Record<string, unknown>;
      if (
        typeof value.pythonImplementation !== "string" ||
        typeof value.pythonVersion !== "string" ||
        typeof value.packageSetHash !== "string"
      ) {
        throw new Error("Python probe returned an invalid environment object");
      }
      environment = {
        packageSetHash: value.packageSetHash,
        pythonImplementation: value.pythonImplementation,
        pythonVersion: value.pythonVersion,
      };
    } catch (error) {
      throw new ScienceError(
        "Python environment probe returned invalid output",
        "PYTHON_UNAVAILABLE",
        {
          cause: error,
        },
      );
    }

    const startedAt = Date.now();
    const process = await this.collect(
      {
        argv: [executable, "-B", "-c", CELL_RUNNER_SOURCE],
        cwd,
        stdin: { data: source },
        maxOutputBytes: this.config.maxOutputBytes,
        env: inputEnvironment,
      },
      signal,
    );
    return {
      ...process,
      durationMs: Date.now() - startedAt,
      environment,
    };
  }

  async close(): Promise<void> {
    if (!this.open) return;
    this.open = false;
    const active = [...this.active];
    for (const handle of active) handle.terminate();
    await Promise.all(active.map((handle) => handle.waitForExit()));
  }

  private async collect(
    spec: {
      readonly argv: readonly string[];
      readonly cwd: string;
      readonly stdin: "ignore" | { readonly data: string };
      readonly maxOutputBytes: number;
      readonly env?: Readonly<Record<string, string>>;
    },
    signal?: AbortSignal,
  ): Promise<CollectedProcess> {
    this.ensureOpen();
    signal?.throwIfAborted();
    let handle: ScienceProcessHandle;
    try {
      handle = this.subprocess.spawn({
        argv: spec.argv,
        cwd: spec.cwd,
        stdio: {
          stdin: spec.stdin,
          stdout: { maxBytes: spec.maxOutputBytes },
          stderr: { maxBytes: spec.maxOutputBytes },
        },
        graceMs: this.config.graceMs,
        signal,
        env: { ...CELL_ENVIRONMENT, ...spec.env },
      });
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      throw new ScienceError("Python process could not be started", "PYTHON_UNAVAILABLE", {
        cause: error,
      });
    }
    this.active.add(handle);
    try {
      const outcome = await handle.done;
      signal?.throwIfAborted();
      this.ensureOpen();
      const stdout = handle.collected.stdout?.readFrom(0);
      const stderr = handle.collected.stderr?.readFrom(0);
      if (!stdout || !stderr) {
        throw new ScienceError("Python output collection is unavailable", "PYTHON_UNAVAILABLE");
      }
      return {
        outcome,
        stdout: { text: stdout.text, truncated: stdout.lossy },
        stderr: { text: stderr.text, truncated: stderr.lossy },
      };
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      if (error instanceof ScienceError) throw error;
      throw new ScienceError("Python process failed to settle", "PYTHON_UNAVAILABLE", {
        cause: error,
      });
    } finally {
      this.active.delete(handle);
    }
  }

  private ensureOpen(): void {
    if (!this.open) throw new ScienceError("Python runtime is closed", "SCIENCE_CLOSED");
  }
}
