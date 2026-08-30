import { randomUUID } from "node:crypto";
import type { Writable } from "node:stream";
import { z } from "zod";
import { ScienceError } from "./errors.js";
import type { ScienceProcessHandle, ScienceProcessRuntime } from "./subprocess.js";

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/u);
const engineSchema = z.enum(["typst"]);
const runtimeTargetSchema = z.strictObject({
  relativePath: z.string().min(1).max(4_096),
  sourceRevision: digestSchema,
  offset: z.number().int().nonnegative(),
});
const compiledEventSchema = z.strictObject({
  type: z.literal("compiled"),
  engine: engineSchema,
  pdfFile: z.string().min(1).max(8_192),
  pdfRevision: digestSchema,
  sourceRevision: digestSchema,
  pdfSize: z.number().int().nonnegative(),
  compiledAt: z.number().int().nonnegative(),
  diagnostics: z.array(z.string().max(4_096)).max(100),
});
const diagnosticEventSchema = z.discriminatedUnion("type", [
  z.strictObject({
    type: z.literal("compile-error"),
    engine: engineSchema,
    diagnostics: z.array(z.string().max(4_096)).max(100),
  }),
  z.strictObject({
    type: z.literal("fatal"),
    engine: engineSchema,
    message: z.string().min(1).max(4_096),
  }),
  z.strictObject({
    type: z.literal("protocol-error"),
    engine: engineSchema,
    message: z.string().min(1).max(4_096),
  }),
]);
const resolvedEventSchema = z.strictObject({
  type: z.literal("resolved"),
  engine: engineSchema,
  id: z.string().uuid(),
  pdfRevision: digestSchema,
  target: runtimeTargetSchema.nullable(),
});
const runtimeEventSchema = z.union([
  compiledEventSchema,
  diagnosticEventSchema,
  resolvedEventSchema,
]);

export type WritingPreviewCompiledEvent = z.infer<typeof compiledEventSchema>;
export type WritingPreviewTarget = z.infer<typeof runtimeTargetSchema>;

interface PendingResolution {
  readonly pdfRevision: string;
  readonly resolve: (target: WritingPreviewTarget | null) => void;
  readonly reject: (error: Error) => void;
  readonly cleanup: () => void;
}

interface WritingPreviewProcessConfig {
  readonly command: string;
  readonly cwd: string;
  readonly input: string;
  readonly outputDirectory: string;
  readonly maxDiagnosticsBytes: number;
  readonly maxPdfBytes: number;
  readonly graceMs: number;
  readonly signal?: AbortSignal;
  readonly onCompiled: (event: WritingPreviewCompiledEvent) => void;
  readonly onDiagnostics: (diagnostics: readonly string[], fatal: boolean) => void;
}

const MAX_PROTOCOL_LINE_BYTES = 1024 * 1024;
const RESOLVE_TIMEOUT_MS = 5_000;

export class WritingPreviewRuntimeProcess {
  readonly handle: ScienceProcessHandle;
  private buffer = "";
  private closed = false;
  private readonly pending = new Map<string, PendingResolution>();

  private constructor(
    handle: ScienceProcessHandle,
    private readonly stdin: Writable,
    private readonly onCompiled: WritingPreviewProcessConfig["onCompiled"],
    private readonly onDiagnostics: WritingPreviewProcessConfig["onDiagnostics"],
  ) {
    this.handle = handle;
    handle.stdout?.setEncoding("utf8");
    handle.stdout?.on("data", (chunk: string) => this.consume(chunk));
    handle.stdout?.on("end", () => this.consume("\n"));
    void handle.done.then(
      (outcome) => {
        this.failPending(
          new ScienceError(
            `Writing preview runtime stopped (exit ${String(outcome.exitCode ?? outcome.signal ?? "unknown")})`,
            "WORKSPACE_UNAVAILABLE",
          ),
        );
      },
      () => {
        this.failPending(
          new ScienceError("Writing preview runtime failed to start", "WORKSPACE_UNAVAILABLE"),
        );
      },
    );
  }

  static async start(
    subprocess: ScienceProcessRuntime,
    config: WritingPreviewProcessConfig,
  ): Promise<WritingPreviewRuntimeProcess> {
    const executable = await subprocess.resolveExecutable(config.command, {}, config.signal);
    config.signal?.throwIfAborted();
    const handle = subprocess.spawn({
      argv: [
        executable,
        "--root",
        config.cwd,
        "--input",
        config.input,
        "--output-directory",
        config.outputDirectory,
        "--max-pdf-bytes",
        String(config.maxPdfBytes),
      ],
      cwd: config.cwd,
      stdio: {
        stdin: "pipe",
        stdout: "pipe",
        stderr: { maxBytes: config.maxDiagnosticsBytes },
      },
      graceMs: config.graceMs,
    });
    if (handle.stdin === undefined || handle.stdout === undefined) {
      handle.terminate();
      throw new ScienceError(
        "Writing preview runtime pipes are unavailable",
        "WORKSPACE_UNAVAILABLE",
      );
    }
    return new WritingPreviewRuntimeProcess(
      handle,
      handle.stdin,
      config.onCompiled,
      config.onDiagnostics,
    );
  }

  resolve(
    request: {
      readonly pdfRevision: string;
      readonly page: number;
      readonly x: number;
      readonly y: number;
    },
    signal?: AbortSignal,
  ): Promise<WritingPreviewTarget | null> {
    if (this.closed) {
      return Promise.reject(
        new ScienceError("Writing preview runtime is unavailable", "WORKSPACE_UNAVAILABLE"),
      );
    }
    signal?.throwIfAborted();
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new ScienceError("Typst source lookup timed out", "WORKSPACE_UNAVAILABLE"));
      }, RESOLVE_TIMEOUT_MS);
      const abort = () => {
        this.pending.delete(id);
        clearTimeout(timeout);
        reject(
          signal?.reason instanceof Error
            ? signal.reason
            : new DOMException("Aborted", "AbortError"),
        );
      };
      signal?.addEventListener("abort", abort, { once: true });
      const cleanup = () => {
        clearTimeout(timeout);
        signal?.removeEventListener("abort", abort);
      };
      this.pending.set(id, {
        pdfRevision: request.pdfRevision,
        resolve,
        reject,
        cleanup,
      });
      this.stdin.write(`${JSON.stringify({ type: "resolve", id, ...request })}\n`, (error) => {
        if (error === null || error === undefined) return;
        const pending = this.pending.get(id);
        if (pending === undefined) return;
        this.pending.delete(id);
        pending.cleanup();
        pending.reject(
          new ScienceError("Typst source lookup could not be sent", "WORKSPACE_UNAVAILABLE", {
            cause: error,
          }),
        );
      });
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.stdin.end();
    this.handle.terminate();
    this.failPending(new ScienceError("Typst semantic watcher was closed", "SCIENCE_CLOSED"));
  }

  private consume(chunk: string): void {
    this.buffer += chunk;
    if (Buffer.byteLength(this.buffer) > MAX_PROTOCOL_LINE_BYTES) {
      this.onDiagnostics(["Writing preview runtime sent an oversized response."], true);
      this.close();
      return;
    }
    while (true) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) return;
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line.length === 0) continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line);
      } catch {
        this.onDiagnostics(["Writing preview runtime sent invalid JSON."], true);
        this.close();
        return;
      }
      const event = runtimeEventSchema.safeParse(parsed);
      if (!event.success) {
        this.onDiagnostics(["Writing preview runtime sent an invalid response."], true);
        this.close();
        return;
      }
      if (event.data.type === "compiled") this.onCompiled(event.data);
      else if (event.data.type === "compile-error") {
        this.onDiagnostics(event.data.diagnostics, false);
      } else if (event.data.type === "resolved") {
        const pending = this.pending.get(event.data.id);
        if (pending === undefined) continue;
        this.pending.delete(event.data.id);
        pending.cleanup();
        if (pending.pdfRevision !== event.data.pdfRevision) {
          pending.reject(
            new ScienceError("Typst PDF revision changed during lookup", "REVISION_CONFLICT"),
          );
        } else {
          pending.resolve(event.data.target);
        }
      } else {
        this.onDiagnostics([event.data.message], true);
      }
    }
  }

  private failPending(error: Error): void {
    this.closed = true;
    for (const pending of this.pending.values()) {
      pending.cleanup();
      pending.reject(error);
    }
    this.pending.clear();
  }
}
