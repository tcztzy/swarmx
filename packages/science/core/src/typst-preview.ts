import { createHash, randomUUID } from "node:crypto";
import {
  chmod,
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  realpath,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve, sep } from "node:path";
import type {
  ResolveTypstSourceAtPointRequest,
  TypstDocumentPreview,
  TypstSourceTarget,
  TypstSourceUpdate,
  UpdateTypstSourceRequest,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import type { ScienceProcessHandle, ScienceProcessRuntime } from "./subprocess.js";
import {
  type WritingPreviewCompiledEvent,
  WritingPreviewRuntimeProcess,
} from "./writing-preview-runtime.js";

const PDF_HEADER = Buffer.from("%PDF-");
const WATCH_ENVIRONMENT: NodeJS.ProcessEnv = {
  ALL_PROXY: "http://127.0.0.1:9",
  HTTP_PROXY: "http://127.0.0.1:9",
  HTTPS_PROXY: "http://127.0.0.1:9",
  NO_PROXY: "",
  all_proxy: "http://127.0.0.1:9",
  http_proxy: "http://127.0.0.1:9",
  https_proxy: "http://127.0.0.1:9",
  no_proxy: "",
};

export interface TypstPreviewRuntimeConfig {
  readonly command: string;
  readonly runtimeCommand?: string;
  readonly graceMs: number;
  readonly initialCompileTimeoutMs: number;
  readonly maxDiagnosticsBytes: number;
  readonly maxPdfBytes: number;
  readonly maxSourceBytes: number;
}

interface PreviewRequest {
  readonly workspaceKey: string;
  readonly workspaceRoot: string;
  readonly relativePath: string;
  readonly signal?: AbortSignal;
}

interface UpdateRequest extends Omit<PreviewRequest, "relativePath">, UpdateTypstSourceRequest {}

interface ResolveRequest
  extends Omit<PreviewRequest, "relativePath">,
    ResolveTypstSourceAtPointRequest {}

interface AuthorizedPaper {
  readonly path: string;
  readonly source: string;
  readonly sourceRevision: `sha256:${string}`;
  readonly sourceMtimeMs: number;
  readonly title: string;
}

interface LastPdf {
  readonly base64: string;
  readonly compiledAt: number;
  readonly revision: `sha256:${string}`;
  readonly size: number;
  readonly sourceRevision: `sha256:${string}`;
}

interface Controller {
  readonly key: string;
  readonly outputDirectory: string;
  readonly outputPath: string;
  readonly paperPath: string;
  readonly relativePath: string;
  readonly workspaceRoot: string;
  diagnostics: string[];
  diagnosticsOffset: number;
  events: Promise<void>;
  handle: ScienceProcessHandle | null;
  lastOutputMtimeMs: number;
  lastPdf: LastPdf | null;
  lastUsedAt: number;
  previewProcess: WritingPreviewRuntimeProcess | null;
  previewProcessCurrent: boolean;
  unavailable: string | null;
}

const MAX_WATCH_CONTROLLERS = 16;
const IDLE_CONTROLLER_MS = 5 * 60 * 1_000;
const TYPST_PROGRESS_LINE =
  /^(?:watching |writing to |\[\d{2}:\d{2}:\d{2}\] compiling\b|\[\d{2}:\d{2}:\d{2}\] compiled successfully\b)/u;

function sha256(bytes: Uint8Array | string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

function isInside(root: string, candidate: string): boolean {
  const fromRoot = relative(root, candidate);
  return fromRoot === "" || (!fromRoot.startsWith(`..${sep}`) && fromRoot !== "..");
}

function validRelativePath(path: string): boolean {
  return (
    path.length > 0 &&
    path.length <= 4_096 &&
    !path.startsWith("/") &&
    !/^[a-z]:[\\/]/iu.test(path) &&
    !path.includes("\\") &&
    !path.includes("\0") &&
    path.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..") &&
    /\.(?:typ|typst)$/iu.test(path)
  );
}

function boundedDiagnostics(text: string, maxBytes: number): string[] {
  const bytes = Buffer.from(text);
  const bounded = bytes.byteLength > maxBytes ? bytes.subarray(bytes.byteLength - maxBytes) : bytes;
  return bounded
    .toString("utf8")
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !TYPST_PROGRESS_LINE.test(line))
    .slice(-100)
    .map((line) => line.slice(0, 4_096));
}

/** Managed, workspace-authorized Typst watch controllers for the live paper workbench. */
export class TypstPreviewRuntime {
  private readonly controllers = new Map<string, Controller>();
  private open = true;

  constructor(
    private readonly subprocess: ScienceProcessRuntime,
    private readonly config: TypstPreviewRuntimeConfig,
  ) {}

  controllerCount(): number {
    return this.controllers.size;
  }

  async preview(request: PreviewRequest): Promise<TypstDocumentPreview> {
    this.ensureOpen();
    request.signal?.throwIfAborted();
    const paper = await this.authorize(request.workspaceRoot, request.relativePath);
    const controller = await this.controller(request, paper);
    await this.refresh(controller, paper);
    if (controller.lastPdf === null && controller.unavailable === null) {
      await this.waitForFirstResult(controller, request.signal);
      const refreshed = await this.authorize(request.workspaceRoot, request.relativePath);
      await this.refresh(controller, refreshed);
      return this.snapshot(controller, refreshed);
    }
    return this.snapshot(controller, paper);
  }

  async updateSource(request: UpdateRequest): Promise<TypstSourceUpdate> {
    this.ensureOpen();
    request.signal?.throwIfAborted();
    const sourceBytes = Buffer.from(request.source, "utf8");
    if (sourceBytes.byteLength > this.config.maxSourceBytes) {
      throw new ScienceError("Typst source exceeds the configured limit", "INVALID_REQUEST");
    }
    const paper = await this.authorize(request.workspaceRoot, request.relativePath);
    if (paper.sourceRevision !== request.expectedSourceRevision) {
      throw new ScienceError("Typst source changed since it was opened", "REVISION_CONFLICT");
    }
    const sourceDirectory = dirname(paper.path);
    const temporaryPath = join(sourceDirectory, `.dsh-science-${randomUUID()}.typ.tmp`);
    let temporaryCreated = false;
    try {
      const existing = await lstat(paper.path);
      if (!existing.isFile() || existing.isSymbolicLink()) {
        throw new ScienceError("Typst source is no longer a regular file", "REVISION_CONFLICT");
      }
      const temporary = await open(temporaryPath, "wx", existing.mode & 0o777);
      temporaryCreated = true;
      try {
        await temporary.writeFile(sourceBytes);
        await temporary.sync();
      } finally {
        await temporary.close();
      }
      request.signal?.throwIfAborted();
      const stillAuthorized = await this.authorize(request.workspaceRoot, request.relativePath);
      if (
        stillAuthorized.path !== paper.path ||
        stillAuthorized.sourceRevision !== request.expectedSourceRevision
      ) {
        throw new ScienceError("Typst source changed during save", "REVISION_CONFLICT");
      }
      await rename(temporaryPath, paper.path);
      temporaryCreated = false;
      const directory = await open(sourceDirectory, "r");
      try {
        await directory.sync();
      } finally {
        await directory.close();
      }
      const updated = await this.authorize(request.workspaceRoot, request.relativePath);
      return {
        relativePath: request.relativePath,
        title: updated.title,
        source: updated.source,
        sourceRevision: updated.sourceRevision,
      };
    } catch (error) {
      if (request.signal?.aborted) request.signal.throwIfAborted();
      if (error instanceof ScienceError) throw error;
      throw new ScienceError("Typst source could not be saved", "WORKSPACE_UNAVAILABLE", {
        cause: error,
      });
    } finally {
      if (temporaryCreated) await rm(temporaryPath, { force: true });
    }
  }

  async resolveSourceAtPoint(request: ResolveRequest): Promise<TypstSourceTarget | null> {
    this.ensureOpen();
    request.signal?.throwIfAborted();
    const paper = await this.authorize(request.workspaceRoot, request.relativePath);
    const controller = await this.controller(request, paper);
    await controller.events;
    if (controller.lastPdf?.revision !== request.pdfRevision) {
      throw new ScienceError(
        "The PDF changed before its source position could be resolved",
        "REVISION_CONFLICT",
      );
    }
    if (controller.previewProcess === null) {
      throw new ScienceError(
        "This Typst watcher cannot resolve PDF positions",
        "WORKSPACE_UNAVAILABLE",
      );
    }
    const target = await controller.previewProcess.resolve(
      {
        pdfRevision: request.pdfRevision,
        page: request.page,
        x: request.x,
        y: request.y,
      },
      request.signal,
    );
    if (target === null) return null;
    const authorized = await this.authorize(request.workspaceRoot, target.relativePath);
    if (authorized.sourceRevision !== target.sourceRevision) {
      throw new ScienceError(
        "The Typst source changed before its position could be opened",
        "REVISION_CONFLICT",
      );
    }
    if (target.offset > authorized.source.length) {
      throw new ScienceError("Typst returned an invalid source position", "WORKSPACE_UNAVAILABLE");
    }
    return {
      relativePath: target.relativePath,
      title: authorized.title,
      source: authorized.source,
      sourceRevision: authorized.sourceRevision,
      offset: target.offset,
    };
  }

  async close(): Promise<void> {
    if (!this.open) return;
    this.open = false;
    const controllers = [...this.controllers.values()];
    this.controllers.clear();
    for (const controller of controllers) {
      if (controller.previewProcess !== null) controller.previewProcess.close();
      else controller.handle?.terminate();
    }
    await Promise.all(
      controllers.map(async (controller) => {
        await controller.handle?.waitForExit();
        await rm(controller.outputDirectory, { recursive: true, force: true });
      }),
    );
  }

  private async authorize(workspaceRoot: string, relativePath: string): Promise<AuthorizedPaper> {
    if (!validRelativePath(relativePath)) {
      throw new ScienceError("Invalid Typst paper path", "INVALID_REQUEST");
    }
    try {
      const canonicalRoot = await realpath(workspaceRoot);
      const unresolved = resolve(canonicalRoot, relativePath);
      const sourceStat = await lstat(unresolved);
      if (!sourceStat.isFile() || sourceStat.isSymbolicLink()) {
        throw new ScienceError("Typst paper must be a regular workspace file", "INVALID_REQUEST");
      }
      const canonicalPaper = await realpath(unresolved);
      if (!isInside(canonicalRoot, canonicalPaper)) {
        throw new ScienceError("Typst paper escapes the workspace", "INVALID_REQUEST");
      }
      if (sourceStat.size > this.config.maxSourceBytes) {
        throw new ScienceError("Typst source exceeds the configured limit", "INVALID_REQUEST");
      }
      const bytes = await readFile(canonicalPaper);
      if (bytes.byteLength > this.config.maxSourceBytes) {
        throw new ScienceError("Typst source exceeds the configured limit", "INVALID_REQUEST");
      }
      return {
        path: canonicalPaper,
        source: bytes.toString("utf8"),
        sourceRevision: sha256(bytes),
        sourceMtimeMs: sourceStat.mtimeMs,
        title: basename(relativePath),
      };
    } catch (error) {
      if (error instanceof ScienceError) throw error;
      throw new ScienceError("Typst paper is unavailable in this workspace", "INVALID_REQUEST", {
        cause: error,
      });
    }
  }

  private async controller(request: PreviewRequest, paper: AuthorizedPaper): Promise<Controller> {
    const key = `${request.workspaceKey}\0${request.relativePath}`;
    const existing = this.controllers.get(key);
    if (
      existing !== undefined &&
      existing.paperPath === paper.path &&
      existing.workspaceRoot === request.workspaceRoot
    ) {
      existing.lastUsedAt = Date.now();
      return existing;
    }
    if (existing !== undefined) await this.disposeController(existing);
    await this.pruneControllers(key);
    const outputDirectory = await mkdtemp(join(tmpdir(), "dsh-science-typst-"));
    await chmod(outputDirectory, 0o700);
    await mkdir(join(outputDirectory, "packages"), { mode: 0o700 });
    await mkdir(join(outputDirectory, "package-cache"), { mode: 0o700 });
    const controller: Controller = {
      key,
      outputDirectory,
      outputPath: join(outputDirectory, "preview.pdf"),
      paperPath: paper.path,
      relativePath: request.relativePath,
      workspaceRoot: request.workspaceRoot,
      diagnostics: [],
      diagnosticsOffset: 0,
      events: Promise.resolve(),
      handle: null,
      lastOutputMtimeMs: 0,
      lastPdf: null,
      lastUsedAt: Date.now(),
      previewProcess: null,
      previewProcessCurrent: false,
      unavailable: null,
    };
    this.controllers.set(key, controller);
    try {
      if (this.config.runtimeCommand !== undefined) {
        controller.previewProcess = await WritingPreviewRuntimeProcess.start(this.subprocess, {
          command: this.config.runtimeCommand,
          cwd: request.workspaceRoot,
          input: paper.path,
          outputDirectory,
          maxDiagnosticsBytes: this.config.maxDiagnosticsBytes,
          maxPdfBytes: this.config.maxPdfBytes,
          graceMs: this.config.graceMs,
          ...(request.signal === undefined ? {} : { signal: request.signal }),
          onCompiled: (event) => {
            controller.events = controller.events.then(() =>
              this.publishFromRuntime(controller, event),
            );
          },
          onDiagnostics: (diagnostics, fatal) => {
            controller.diagnostics = diagnostics.slice(0, 100);
            controller.previewProcessCurrent = false;
            if (fatal) {
              controller.unavailable = diagnostics[0] ?? "Writing preview runtime is unavailable.";
            }
          },
        });
        controller.handle = controller.previewProcess.handle;
        void controller.handle.done.then(
          (outcome) => {
            if (this.controllers.get(key) !== controller || !this.open) return;
            controller.unavailable = `Writing preview runtime stopped (exit ${String(outcome.exitCode ?? outcome.signal ?? "unknown")}).`;
          },
          () => {
            if (this.controllers.get(key) === controller && this.open) {
              controller.unavailable = "Writing preview runtime failed to start.";
            }
          },
        );
        return controller;
      }
      const executable = await this.subprocess.resolveExecutable(
        this.config.command,
        {},
        request.signal,
      );
      request.signal?.throwIfAborted();
      controller.handle = this.subprocess.spawn({
        argv: [
          executable,
          "watch",
          "--root",
          request.workspaceRoot,
          "--diagnostic-format",
          "short",
          "--package-path",
          join(outputDirectory, "packages"),
          "--package-cache-path",
          join(outputDirectory, "package-cache"),
          paper.path,
          controller.outputPath,
        ],
        cwd: request.workspaceRoot,
        stdio: {
          stdin: "ignore",
          stdout: { maxBytes: 4_096 },
          stderr: { maxBytes: this.config.maxDiagnosticsBytes },
        },
        graceMs: this.config.graceMs,
        env: WATCH_ENVIRONMENT,
      });
      void controller.handle.done.then(
        (outcome) => {
          if (this.controllers.get(key) !== controller || !this.open) return;
          controller.unavailable = `Typst watcher stopped (exit ${String(outcome.exitCode ?? outcome.signal ?? "unknown")}).`;
        },
        () => {
          if (this.controllers.get(key) === controller && this.open) {
            controller.unavailable = "Typst watcher failed to start.";
          }
        },
      );
    } catch {
      if (request.signal?.aborted) {
        await this.disposeController(controller);
        request.signal.throwIfAborted();
      }
      controller.unavailable = "The configured Typst compiler is unavailable.";
    }
    return controller;
  }

  private async refresh(controller: Controller, paper: AuthorizedPaper): Promise<void> {
    if (controller.previewProcess !== null) {
      await controller.events;
      return;
    }
    const diagnostics = controller.handle?.collected.stderr?.readFrom(controller.diagnosticsOffset);
    if (diagnostics !== undefined) {
      controller.diagnosticsOffset = diagnostics.nextOffset;
      if (diagnostics.text.length > 0) {
        controller.diagnostics = boundedDiagnostics(
          `${controller.diagnostics.join("\n")}\n${diagnostics.text}`,
          this.config.maxDiagnosticsBytes,
        );
      }
    }
    try {
      const outputStat = await stat(controller.outputPath);
      if (outputStat.size > this.config.maxPdfBytes) {
        controller.unavailable = "Compiled PDF exceeds the configured preview limit.";
        return;
      }
      if (outputStat.mtimeMs <= controller.lastOutputMtimeMs && controller.lastPdf !== null) return;
      const bytes = await readFile(controller.outputPath);
      const verifiedStat = await stat(controller.outputPath);
      if (
        verifiedStat.size !== outputStat.size ||
        verifiedStat.mtimeMs !== outputStat.mtimeMs ||
        outputStat.mtimeMs + 1 < paper.sourceMtimeMs
      ) {
        return;
      }
      if (bytes.byteLength === 0 || bytes.byteLength > this.config.maxPdfBytes) return;
      if (!bytes.subarray(0, PDF_HEADER.byteLength).equals(PDF_HEADER)) {
        controller.diagnostics = ["Typst produced an invalid PDF preview."];
        return;
      }
      controller.lastOutputMtimeMs = outputStat.mtimeMs;
      controller.lastPdf = {
        base64: bytes.toString("base64"),
        compiledAt: Math.max(0, Math.trunc(outputStat.mtimeMs)),
        revision: sha256(bytes),
        size: bytes.byteLength,
        sourceRevision: paper.sourceRevision,
      };
      controller.diagnostics = [];
      controller.unavailable = null;
    } catch {
      // A watcher may not have produced its first output yet.
    }
  }

  private async publishFromRuntime(
    controller: Controller,
    event: WritingPreviewCompiledEvent,
  ): Promise<void> {
    try {
      const canonicalOutputDirectory = await realpath(controller.outputDirectory);
      const outputStat = await lstat(event.pdfFile);
      const canonicalOutput = await realpath(event.pdfFile);
      if (
        !outputStat.isFile() ||
        outputStat.isSymbolicLink() ||
        !isInside(canonicalOutputDirectory, canonicalOutput) ||
        outputStat.size !== event.pdfSize ||
        outputStat.size > this.config.maxPdfBytes
      ) {
        throw new Error("invalid PDF publication");
      }
      const bytes = await readFile(canonicalOutput);
      if (
        bytes.byteLength !== event.pdfSize ||
        !bytes.subarray(0, PDF_HEADER.byteLength).equals(PDF_HEADER) ||
        sha256(bytes) !== event.pdfRevision
      ) {
        throw new Error("invalid PDF contents");
      }
      controller.lastPdf = {
        base64: bytes.toString("base64"),
        compiledAt: event.compiledAt,
        revision: event.pdfRevision,
        size: event.pdfSize,
        sourceRevision: event.sourceRevision as `sha256:${string}`,
      };
      controller.diagnostics = event.diagnostics;
      controller.previewProcessCurrent = true;
      controller.unavailable = null;
      await rm(canonicalOutput, { force: true });
    } catch {
      controller.unavailable = "The writing preview runtime produced an invalid PDF preview.";
    }
  }

  private snapshot(controller: Controller, paper: AuthorizedPaper): TypstDocumentPreview {
    const pdf = controller.lastPdf;
    const current =
      pdf?.sourceRevision === paper.sourceRevision &&
      (controller.previewProcess === null || controller.previewProcessCurrent);
    const status: TypstDocumentPreview["status"] =
      controller.unavailable !== null
        ? "unavailable"
        : pdf === null
          ? controller.diagnostics.length > 0
            ? "error"
            : "compiling"
          : current
            ? "ready"
            : controller.diagnostics.length > 0
              ? "stale"
              : "compiling";
    return {
      relativePath: controller.relativePath,
      title: paper.title,
      source: paper.source,
      sourceRevision: paper.sourceRevision,
      status,
      diagnostics:
        controller.unavailable === null
          ? controller.diagnostics
          : [controller.unavailable, ...controller.diagnostics].slice(0, 100),
      pdfBase64: pdf?.base64 ?? null,
      pdfRevision: pdf?.revision ?? null,
      pdfSourceRevision: pdf?.sourceRevision ?? null,
      pdfSize: pdf?.size ?? null,
      compiledAt: pdf?.compiledAt ?? null,
    };
  }

  private async waitForFirstResult(controller: Controller, signal?: AbortSignal): Promise<void> {
    const deadline = Date.now() + this.config.initialCompileTimeoutMs;
    while (
      Date.now() < deadline &&
      controller.lastPdf === null &&
      controller.diagnostics.length === 0 &&
      controller.unavailable === null
    ) {
      signal?.throwIfAborted();
      await new Promise<void>((resolvePromise) => setTimeout(resolvePromise, 25));
      const paper = await this.authorize(controller.workspaceRoot, controller.relativePath);
      await this.refresh(controller, paper);
    }
  }

  private async disposeController(controller: Controller): Promise<void> {
    if (this.controllers.get(controller.key) === controller)
      this.controllers.delete(controller.key);
    if (controller.previewProcess !== null) controller.previewProcess.close();
    else controller.handle?.terminate();
    await controller.handle?.waitForExit();
    await rm(controller.outputDirectory, { recursive: true, force: true });
  }

  private async pruneControllers(preserveKey: string): Promise<void> {
    const now = Date.now();
    const candidates = [...this.controllers.values()]
      .filter((controller) => controller.key !== preserveKey)
      .sort((left, right) => left.lastUsedAt - right.lastUsedAt);
    for (const controller of candidates) {
      if (
        controller.lastUsedAt > now - IDLE_CONTROLLER_MS &&
        this.controllers.size < MAX_WATCH_CONTROLLERS
      ) {
        break;
      }
      await this.disposeController(controller);
    }
  }

  private ensureOpen(): void {
    if (!this.open) throw new ScienceError("Typst preview runtime is closed", "SCIENCE_CLOSED");
  }
}
