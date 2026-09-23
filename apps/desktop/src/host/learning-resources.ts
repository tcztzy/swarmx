import { execFileSync, spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { chmod, lstat, open, realpath, rename, unlink } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";
import { evaluationSchema } from "@swarmx/memory";
import { z } from "zod";

const revision = (content: string) =>
  `sha256:${createHash("sha256").update(content).digest("hex")}`;
const RevisionSchema = z.string().regex(/^sha256:[a-f0-9]{64}$/u);
const ContentSchema = z
  .string()
  .min(1)
  .refine((value) => Buffer.byteLength(value) <= 131_072);
const CommandSchema = z.array(z.string().min(1).max(4096)).min(1).max(32);
const ResourceSchema = z.strictObject({
  id: z.string().min(1).max(128),
  kind: z.enum(["agent", "skill"]),
  path: z.string().min(1).max(2048).regex(/\.md$/iu),
  validate: CommandSchema,
  evaluate: CommandSchema.optional(),
});
const MeasurementSchema = z
  .strictObject({
    revision: RevisionSchema,
    passedCases: z.number().int().nonnegative(),
    totalCases: z.number().int().positive(),
    cost: z
      .strictObject({ amount: z.number().nonnegative(), unit: z.string().min(1).max(32) })
      .optional(),
  })
  .refine(({ passedCases, totalCases }) => passedCases <= totalCases);
const EvaluationReportSchema = z.strictObject({
  evaluatorVersion: z.string().min(1).max(128),
  passed: z.boolean(),
  baseline: MeasurementSchema,
  candidate: MeasurementSchema,
  summary: z.string().max(2000).optional(),
});
type EvaluationReport = z.infer<typeof EvaluationReportSchema>;
type ResourceStage = {
  stage: "candidate" | "validate" | "evaluate" | "adopt";
  status: "prepared" | "passed" | "failed" | "not-configured" | "applied" | "already-present";
  durationMs?: number;
};

export class ResourceEvaluationError extends Error {
  readonly reportHash: string;
  constructor(
    message: string,
    readonly report: EvaluationReport,
    readonly baselineRevision: string,
    readonly candidateRevision: string,
    readonly configurationRevision: string,
    readonly stages: ResourceStage[],
  ) {
    super(message);
    this.reportHash = revision(JSON.stringify(report));
  }
}
const ConfigSchema = z.strictObject({ resources: z.array(ResourceSchema).max(10) });
export const ResourceUpdateSchema = z.strictObject({
  action: z.literal("update_resource"),
  request: z.strictObject({
    id: ResourceSchema.shape.id,
    expectedRevision: RevisionSchema,
    content: ContentSchema,
    evaluation: evaluationSchema.optional(),
  }),
});
export const ResourceSnapshotSchema = ResourceSchema.extend({
  configurationRevision: RevisionSchema,
  expectedRevision: RevisionSchema,
  content: ContentSchema,
});
export type ResourceSnapshot = z.infer<typeof ResourceSnapshotSchema>;

export class LearningResources {
  private readonly applying = new Set<string>();
  constructor(private readonly cwd: string) {}

  private async run(argv: string[], paths: string[], signal: AbortSignal, evaluate = false) {
    const executionSignal = AbortSignal.any([
      signal,
      AbortSignal.timeout(evaluate ? 300_000 : 30_000),
    ]);
    executionSignal.throwIfAborted();
    const [command, ...args] = argv;
    if (!command) throw new Error("Missing learning resource validator.");
    const started = Date.now();
    const output = await new Promise<string>((resolve, reject) => {
      const child = spawn(command, [...args, ...paths], {
        cwd: this.cwd,
        stdio: ["ignore", evaluate ? "pipe" : "ignore", "ignore"],
        detached: process.platform !== "win32",
        shell: false,
      });
      let failure: Error | undefined;
      const terminate = (reason: Error) => {
        if (failure) return;
        failure = reason;
        if (child.pid === undefined) return;
        try {
          if (process.platform === "win32")
            execFileSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
              stdio: "ignore",
              timeout: 10_000,
            });
          else process.kill(-child.pid, "SIGKILL");
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code !== "ESRCH") reject(error);
        }
      };
      const abort = () =>
        terminate(new Error("Learning resource check aborted.", { cause: executionSignal.reason }));
      executionSignal.addEventListener("abort", abort, { once: true });
      if (executionSignal.aborted) abort();
      let bytes = 0;
      const chunks: Buffer[] = [];
      child.stdout?.on("data", (chunk: Buffer) => {
        bytes += chunk.length;
        if (bytes > 16_384) {
          terminate(new Error("Learning resource behavioral evaluation report exceeds 16 KiB."));
        } else chunks.push(chunk);
      });
      child.on("error", (error) => {
        failure = error;
      });
      child.on("close", (code) => {
        executionSignal.removeEventListener("abort", abort);
        if (failure) reject(failure);
        else if (code !== 0)
          reject(
            new Error(
              `Learning resource ${evaluate ? "behavioral evaluation" : "validation"} failed (exit ${code}).`,
            ),
          );
        else resolve(Buffer.concat(chunks).toString("utf8"));
      });
    });
    return { output, durationMs: Date.now() - started };
  }

  private async read(path: string, signal: AbortSignal) {
    signal.throwIfAborted();
    if (
      isAbsolute(path) ||
      /\\/u.test(path) ||
      path.split("/").some((part) => !part || part === "." || part === "..")
    )
      throw new Error("Learning resource path must stay inside the project.");
    let target = await realpath(this.cwd);
    for (const part of path.split("/")) {
      target = join(target, part);
      if ((await lstat(target)).isSymbolicLink())
        throw new Error("Learning resources cannot use symbolic links.");
    }
    const file = await open(target, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const status = await file.stat();
      if (!status.isFile() || status.size > 131_072)
        throw new Error("Learning resource must be a regular file of at most 128 KiB.");
      const content = new TextDecoder("utf-8", { fatal: true }).decode(
        await file.readFile({ signal }),
      );
      return { target, content, mode: status.mode & 0o777 };
    } finally {
      await file.close();
    }
  }

  private async configuration(signal: AbortSignal) {
    let content: string;
    try {
      ({ content } = await this.read(".swarmx/learning.json", signal));
    } catch (error) {
      if (error instanceof Error && "code" in error && error.code === "ENOENT") return undefined;
      throw error;
    }
    const config = ConfigSchema.parse(JSON.parse(content));
    if (
      new Set(config.resources.map(({ id }) => id)).size !== config.resources.length ||
      new Set(config.resources.map(({ path }) => path)).size !== config.resources.length
    )
      throw new Error("Learning resource IDs and paths must be unique.");
    return { ...config, revision: revision(content) };
  }

  async snapshot(signal: AbortSignal): Promise<ResourceSnapshot[]> {
    const config = await this.configuration(signal);
    if (!config) return [];
    const snapshots: ResourceSnapshot[] = [];
    for (const resource of config.resources) {
      const { content } = await this.read(resource.path, signal);
      ContentSchema.parse(content);
      snapshots.push({
        ...resource,
        configurationRevision: config.revision,
        expectedRevision: revision(content),
        content,
      });
      if (JSON.stringify(snapshots).length > 32_000)
        throw new Error("Learning resource snapshots exceed 32,000 characters.");
    }
    return snapshots;
  }

  async apply(
    request: z.infer<typeof ResourceUpdateSchema>["request"],
    snapshot: ResourceSnapshot,
    signal: AbortSignal,
  ) {
    const input = ResourceUpdateSchema.shape.request.parse(request);
    if (
      input.id !== snapshot.id ||
      input.expectedRevision !== snapshot.expectedRevision ||
      revision(snapshot.content) !== snapshot.expectedRevision
    )
      throw new Error("Resource update does not match the supplied snapshot revision.");
    if (this.applying.has(input.id)) throw new Error("Resource update is already running.");
    this.applying.add(input.id);
    try {
      const check = async () => {
        signal.throwIfAborted();
        const config = await this.configuration(signal);
        const resource = config?.resources.find(({ id }) => id === snapshot.id);
        if (
          config?.revision !== snapshot.configurationRevision ||
          !resource ||
          resource.path !== snapshot.path ||
          resource.kind !== snapshot.kind ||
          JSON.stringify(resource.validate) !== JSON.stringify(snapshot.validate) ||
          JSON.stringify(resource.evaluate) !== JSON.stringify(snapshot.evaluate)
        )
          throw new Error("Learning resource registration changed.");
        const current = await this.read(resource.path, signal);
        if (
          revision(current.content) !== input.expectedRevision &&
          current.content !== input.content
        )
          throw new Error("Learning resource revision changed.");
        return current;
      };
      const original = await check();
      const candidate = join(
        dirname(original.target),
        `.${basename(original.target)}.${randomUUID()}.candidate.md`,
      );
      const baseline = snapshot.evaluate ? `${candidate}.baseline.md` : undefined;
      const stages: ResourceStage[] = [];
      const candidateRevision = revision(input.content);
      const write = async (path: string, content: string) => {
        const file = await open(path, "wx", 0o600);
        try {
          await file.writeFile(content);
          await file.sync();
        } finally {
          await file.close();
        }
      };
      try {
        await write(candidate, input.content);
        stages.push({ stage: "candidate", status: "prepared" });
        const validation = await this.run(snapshot.validate, [candidate], signal);
        stages.push({ stage: "validate", status: "passed", durationMs: validation.durationMs });
        const unchanged = async (path: string, content: string) => {
          const checked = await this.read(join(dirname(snapshot.path), basename(path)), signal);
          if (checked.content !== content)
            throw new Error(
              `Learning resource validation/evaluation changed the ${path === candidate ? "candidate" : "baseline"}.`,
            );
        };
        await unchanged(candidate, input.content);
        let report: EvaluationReport | undefined;
        if (snapshot.evaluate && baseline) {
          await write(baseline, snapshot.content);
          const evaluation = await this.run(snapshot.evaluate, [baseline, candidate], signal, true);
          report = EvaluationReportSchema.parse(JSON.parse(evaluation.output));
          await unchanged(baseline, snapshot.content);
          await unchanged(candidate, input.content);
          const passed =
            report.passed &&
            report.baseline.revision === snapshot.expectedRevision &&
            report.candidate.revision === candidateRevision &&
            report.baseline.totalCases === report.candidate.totalCases &&
            report.candidate.passedCases >= report.baseline.passedCases;
          stages.push({
            stage: "evaluate",
            status: passed ? "passed" : "failed",
            durationMs: evaluation.durationMs,
          });
          if (!passed)
            throw new ResourceEvaluationError(
              "Learning resource behavioral evaluation failed or reported mismatched revisions/cases.",
              report,
              snapshot.expectedRevision,
              candidateRevision,
              snapshot.configurationRevision,
              stages,
            );
        } else stages.push({ stage: "evaluate", status: "not-configured" });
        await chmod(candidate, original.mode);
        const current = await check();
        await unchanged(candidate, input.content);
        const changed = current.content !== input.content;
        if (changed) {
          signal.throwIfAborted();
          await rename(candidate, current.target);
        }
        stages.push({ stage: "adopt", status: changed ? "applied" : "already-present" });
        return {
          id: input.id,
          path: snapshot.path,
          revision: candidateRevision,
          baselineRevision: snapshot.expectedRevision,
          configurationRevision: snapshot.configurationRevision,
          changed,
          validation: "passed" as const,
          assessment: report ? ("behavior-tested" as const) : ("structural-only" as const),
          ...(report ? { report, reportHash: revision(JSON.stringify(report)) } : {}),
          stages,
        };
      } finally {
        for (const path of baseline ? [baseline, candidate] : [candidate])
          await unlink(path).catch((error: unknown) => {
            if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
              throw error;
          });
      }
    } finally {
      this.applying.delete(input.id);
    }
  }
}
