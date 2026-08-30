import { createHash } from "node:crypto";
import { realpathSync } from "node:fs";
import type { Context } from "@deepseek-ai/cordis";
import { SessionId } from "@deepseek-ai/dsh-session";
import type {} from "@deepseek-ai/dsh-subprocess";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import s from "@deepseek-ai/schemastery";
import { MAX_TYPST_PDF_BYTES, MAX_TYPST_SOURCE_BYTES } from "./contracts.js";
import {
  type Config,
  DEFAULT_JUPYMCP_COMMAND,
  DEFAULT_JUPYMCP_REQUEST_TIMEOUT_MS,
  DEFAULT_MAX_ARTIFACT_BYTES,
  DEFAULT_MAX_CELL_OUTPUT_BYTES,
  DEFAULT_MAX_EXPORT_BYTES,
  DEFAULT_MAX_NOTEBOOK_DOCUMENT_BYTES,
  DEFAULT_PROCESS_GRACE_MS,
  DEFAULT_PYTHON_COMMAND,
  DEFAULT_TYPST_COMMAND,
  DEFAULT_TYPST_INITIAL_COMPILE_TIMEOUT_MS,
  DEFAULT_TYPST_MAX_DIAGNOSTICS_BYTES,
  DEFAULT_TYPST_MAX_PDF_BYTES,
  DEFAULT_TYPST_MAX_SOURCE_BYTES,
  DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND,
  ScienceCore,
  type ScienceWorkspaceResolver,
} from "./core.js";
import { ScienceError } from "./errors.js";

export * from "./core.js";
export { runScienceDemo, type ScienceDemoResult } from "./demo.js";

class ScienceServiceHost extends TypertRemoteService {
  static inject = ["sessions", "subprocess"];
  static Config = s.object({
    embedArtifactMetadata: s.boolean().default(true),
    root: s.string().required(),
    maxArtifactBytes: s.natural().min(1).default(DEFAULT_MAX_ARTIFACT_BYTES),
    maxCellOutputBytes: s.natural().min(1).max(1_000_000).default(DEFAULT_MAX_CELL_OUTPUT_BYTES),
    maxExportBytes: s.natural().min(1).max(10_000_000).default(DEFAULT_MAX_EXPORT_BYTES),
    maxNotebookDocumentBytes: s
      .natural()
      .min(1)
      .max(10_000_000)
      .default(DEFAULT_MAX_NOTEBOOK_DOCUMENT_BYTES),
    notebookRuntime: s.union(["jupymcp", "isolated"]).default("jupymcp"),
    processGraceMs: s.natural().min(1).max(60_000).default(DEFAULT_PROCESS_GRACE_MS),
    jupymcpArgs: s.array(s.string()).default([]),
    jupymcpCommand: s.string().default(DEFAULT_JUPYMCP_COMMAND),
    jupymcpRequestTimeoutMs: s
      .natural()
      .min(1_000)
      .max(3_600_000)
      .default(DEFAULT_JUPYMCP_REQUEST_TIMEOUT_MS),
    pythonCommand: s.string().default(DEFAULT_PYTHON_COMMAND),
    typstCommand: s.string().default(DEFAULT_TYPST_COMMAND),
    writingPreviewRuntimeCommand: s.string().default(DEFAULT_WRITING_PREVIEW_RUNTIME_COMMAND),
    typstInitialCompileTimeoutMs: s
      .natural()
      .min(100)
      .max(60_000)
      .default(DEFAULT_TYPST_INITIAL_COMPILE_TIMEOUT_MS),
    typstMaxDiagnosticsBytes: s
      .natural()
      .min(1_024)
      .max(1024 * 1024)
      .default(DEFAULT_TYPST_MAX_DIAGNOSTICS_BYTES),
    typstMaxPdfBytes: s
      .natural()
      .min(1_024)
      .max(MAX_TYPST_PDF_BYTES)
      .default(DEFAULT_TYPST_MAX_PDF_BYTES),
    typstMaxSourceBytes: s
      .natural()
      .min(1_024)
      .max(MAX_TYPST_SOURCE_BYTES)
      .default(DEFAULT_TYPST_MAX_SOURCE_BYTES),
  });

  readonly core: ScienceCore;

  constructor(ctx: Context, config: Config, resolveWorkspace?: ScienceWorkspaceResolver) {
    super(ctx, "science");
    const sessions = ctx.sessions;
    this.core = new ScienceCore(
      {
        subprocess: ctx.subprocess,
        onDispose: (dispose) => {
          ctx.effect(() => dispose, "dsh-science: close runtime and journal");
        },
      },
      config,
      resolveWorkspace ??
        ((actorId) => {
          const session = sessions.get(SessionId(actorId));
          if (session === undefined) {
            throw new ScienceError("Live session not found", "SESSION_NOT_FOUND");
          }
          const cwd = session.header.cwd;
          if (cwd === undefined) {
            throw new ScienceError(
              "The live session has no workspace directory",
              "WORKSPACE_UNAVAILABLE",
            );
          }
          try {
            const root = realpathSync.native(cwd);
            return { key: createHash("sha256").update(root).digest("hex"), root };
          } catch (error) {
            throw new ScienceError(
              "The live session workspace cannot be resolved",
              "WORKSPACE_UNAVAILABLE",
              { cause: error },
            );
          }
        }),
    );
  }
}

const SCIENCE_METHODS = [
  "createProject",
  "createQuestion",
  "createHypothesis",
  "recordClaim",
  "linkEvidence",
  "defineExperiment",
  "startRun",
  "finishRun",
  "compareRuns",
  "exportProject",
  "createDocument",
  "createFigure",
  "createNotebook",
  "executeNotebookCell",
  "registerArtifact",
  "importArtifact",
  "searchLiterature",
  "getWorkspace",
  "headResource",
  "batchHeadResources",
  "getResource",
  "selectResource",
  "getResourceNeighbors",
  "getResearchObject",
  "previewArtifact",
  "previewTypstDocument",
  "updateTypstSource",
  "resolveTypstSourceAtPoint",
  "modifyDocument",
  "modifyFigureCode",
  "journalCount",
] as const satisfies readonly (keyof ScienceCore)[];

for (const name of SCIENCE_METHODS) {
  Object.defineProperty(ScienceServiceHost.prototype, name, {
    configurable: false,
    value(this: ScienceServiceHost, ...args: unknown[]) {
      const method = Reflect.get(ScienceCore.prototype, name) as (...values: unknown[]) => unknown;
      return Reflect.apply(method, this.core, args);
    },
    writable: false,
  });
}

type ScienceServiceConstructor = typeof ScienceServiceHost & {
  new (
    ctx: Context,
    config: Config,
    resolveWorkspace?: ScienceWorkspaceResolver,
  ): ScienceServiceHost & ScienceCore;
};

export const ScienceService = ScienceServiceHost as ScienceServiceConstructor;
export type ScienceService = InstanceType<typeof ScienceService>;

declare module "@deepseek-ai/cordis" {
  interface Context {
    science: ScienceService;
  }
}

export default ScienceService;
