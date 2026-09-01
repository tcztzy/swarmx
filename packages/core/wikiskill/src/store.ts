import { createHash, randomUUID } from "node:crypto";
import { access, chmod, mkdir, readFile, rename, rm, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { withFileLock, writeFileAtomic } from "@deepseek-ai/dsh-atomic-write";
import { isSkillName } from "@deepseek-ai/dsh-skill";
import { parse } from "yaml";
import { z } from "zod";
import {
  type SkillImpactDraft,
  sha256Schema,
  skillImpactDraftSchema,
  type WikiSkillActiveSkill,
  type WikiSkillEvaluation,
  type WikiSkillOutcome,
  type WikiSkillProposal,
  type WikiSkillProposalRecord,
  type WikiSkillStageRequest,
  type WikiSkillTarget,
  wikiSkillEvaluationSchema,
  wikiSkillOutcomeSchema,
  wikiSkillProposalRecordSchema,
  wikiSkillStageRequestSchema,
  wikiSkillTargetSchema,
} from "./contracts.js";
import { WikiSkillError } from "./errors.js";

const FILE_MODE = 0o600;
const DIRECTORY_MODE = 0o700;
const SKILL_MAX_BYTES = 128 * 1_024;
const PURPOSE_MAX_BYTES = 128 * 1_024;
const STATE_MAX_BYTES = 256 * 1_024;
const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/u;

const activeMetadataSchema = z.strictObject({
  candidateRevision: sha256Schema,
  name: wikiSkillStageRequestSchema.shape.name,
  proposalId: wikiSkillStageRequestSchema.shape.proposalId,
  schemaVersion: z.literal(1),
  target: wikiSkillTargetSchema,
});

type ActiveMetadata = z.infer<typeof activeMetadataSchema>;

function wikiSkillError(
  message: string,
  code: ConstructorParameters<typeof WikiSkillError>[1],
  cause?: unknown,
): WikiSkillError {
  return new WikiSkillError(message, code, cause === undefined ? undefined : { cause });
}

function parsed<T>(schema: { parse(value: unknown): T }, value: unknown, message: string): T {
  try {
    return schema.parse(value);
  } catch (cause) {
    throw wikiSkillError(message, "WIKISKILL_INVALID_REQUEST", cause);
  }
}

function hash(value: unknown): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(JSON.stringify(value)).digest("hex")}`;
}

function json(value: unknown): string {
  return `${JSON.stringify(value, undefined, 2)}\n`;
}

function sameTarget(left: WikiSkillTarget, right: WikiSkillTarget): boolean {
  return left.model === right.model && left.preset === right.preset;
}

function activeRevision(
  target: WikiSkillTarget,
  name: string,
  skillMarkdown: string,
  purposeMarkdown: string,
): `sha256:${string}` {
  return hash({ name, purposeMarkdown, skillMarkdown, target });
}

function validateSkill(name: string, skillMarkdown: string): void {
  if (!isSkillName(name) || Buffer.byteLength(skillMarkdown, "utf8") > SKILL_MAX_BYTES) {
    throw wikiSkillError("Invalid WikiSkill SKILL.md", "WIKISKILL_INVALID_SKILL");
  }
  const frontmatter = FRONTMATTER.exec(skillMarkdown);
  if (frontmatter === null) {
    throw wikiSkillError("WikiSkill SKILL.md requires YAML frontmatter", "WIKISKILL_INVALID_SKILL");
  }
  let data: unknown;
  try {
    data = parse(frontmatter[1] as string);
  } catch (cause) {
    throw wikiSkillError(
      "WikiSkill SKILL.md frontmatter is invalid",
      "WIKISKILL_INVALID_SKILL",
      cause,
    );
  }
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    throw wikiSkillError(
      "WikiSkill SKILL.md frontmatter must be an object",
      "WIKISKILL_INVALID_SKILL",
    );
  }
  const metadata = data as Record<string, unknown>;
  if (metadata.name !== name) {
    throw wikiSkillError(
      "WikiSkill SKILL.md name must match the proposal name",
      "WIKISKILL_INVALID_SKILL",
    );
  }
  if (
    typeof metadata.description !== "string" ||
    metadata.description.trim().length === 0 ||
    metadata.description.length > 500
  ) {
    throw wikiSkillError(
      "WikiSkill SKILL.md requires a bounded description",
      "WIKISKILL_INVALID_SKILL",
    );
  }
  if (metadata["disable-model-invocation"] === true) {
    throw wikiSkillError(
      "An active WikiSkill must permit model invocation",
      "WIKISKILL_INVALID_SKILL",
    );
  }
  if (skillMarkdown.slice(frontmatter[0].length).trim().length === 0) {
    throw wikiSkillError("WikiSkill SKILL.md body is empty", "WIKISKILL_INVALID_SKILL");
  }
}

function renderPurpose(request: WikiSkillStageRequest): string {
  const lines = [
    request.purposeMarkdown.trimEnd(),
    "",
    "## SwarmX provenance",
    "",
    `Target: \`${request.target.preset}\` + \`${request.target.model}\``,
    "",
    ...request.sources.patterns.map(
      ({ id, revision }) => `- PKB pattern: \`${id}\` at \`${revision}\``,
    ),
    ...request.sources.traces.map(
      ({ endSeq, sessionId, startSeq }) =>
        `- DSH Raw trace: \`${sessionId}#${String(startSeq)}-${String(endSeq)}\``,
    ),
    "",
  ];
  const purpose = lines.join("\n");
  if (Buffer.byteLength(purpose, "utf8") > PURPOSE_MAX_BYTES) {
    throw wikiSkillError("WikiSkill PURPOSE.md is too large", "WIKISKILL_INVALID_REQUEST");
  }
  return purpose;
}

function proposalPayload(
  request: WikiSkillStageRequest,
  purposeMarkdown: string,
): Record<string, unknown> {
  return {
    expectedActiveRevision: request.expectedActiveRevision,
    name: request.name,
    operation: request.operation,
    proposalId: request.proposalId,
    purposeMarkdown,
    skillMarkdown: request.skillMarkdown,
    sources: request.sources,
    target: request.target,
  };
}

function impactDraft(
  proposal: WikiSkillProposal,
  evaluation: WikiSkillEvaluation,
  verdict: "accepted" | "rejected",
): SkillImpactDraft {
  const baselineRevision = evaluation.baseline.skillRevision ?? "none";
  return skillImpactDraftSchema.parse({
    body: [
      `# WikiSkill impact: ${proposal.name}`,
      "",
      `- Verdict: **${verdict}**`,
      `- Operation: \`${proposal.operation}\``,
      `- Target preset: \`${proposal.target.preset}\``,
      `- Target model: \`${proposal.target.model}\``,
      `- Benchmark: \`${evaluation.baseline.benchmarkId}\``,
      `- Task set: \`${evaluation.baseline.taskSetRevision}\``,
      `- Baseline revision: \`${baselineRevision}\``,
      `- Candidate revision: \`${proposal.candidateRevision}\``,
      `- Baseline score: ${String(evaluation.baseline.score)} (${String(evaluation.baseline.runs)} runs)`,
      `- Candidate score: ${String(evaluation.candidate.score)} (${String(evaluation.candidate.runs)} runs)`,
      "",
      `PURPOSE.md traces this proposal to ${String(proposal.sources.patterns.length)} PKB pattern(s) and ${String(proposal.sources.traces.length)} DSH Raw trace(s).`,
    ].join("\n"),
    description: `WikiSkill proposal ${proposal.name} was ${verdict}.`,
    sources: [
      ...proposal.sources.traces.map(({ endSeq, sessionId, startSeq }) => ({
        resource: `dsh-session:${sessionId}#seq=${String(startSeq)}-${String(endSeq)}`,
        title: "DSH Raw execution trace",
      })),
      ...proposal.sources.patterns.map(({ id, revision }) => ({
        resource: `pkb:${id}@${revision}`,
        title: "PKB AgentPattern",
      })),
    ],
    status: "draft",
    tags: ["wikiskill", "skill-impact", verdict],
    title: `WikiSkill ${proposal.name}: ${verdict}`,
    type: "SkillImpact",
  });
}

async function privateDirectory(path: string): Promise<void> {
  await mkdir(path, { mode: DIRECTORY_MODE, recursive: true });
  await chmod(path, DIRECTORY_MODE);
}

async function privateFile(path: string, content: string): Promise<void> {
  await writeFileAtomic(path, content, { dirMode: DIRECTORY_MODE, mode: FILE_MODE });
}

async function readBoundedText(path: string, maxBytes: number): Promise<string> {
  const info = await stat(path);
  if (!info.isFile() || info.size > maxBytes) {
    throw wikiSkillError("WikiSkill state file is invalid or too large", "WIKISKILL_IO_ERROR");
  }
  return readFile(path, "utf8");
}

async function readJson(path: string): Promise<unknown> {
  try {
    return JSON.parse(await readBoundedText(path, STATE_MAX_BYTES));
  } catch (cause) {
    throw wikiSkillError(
      `Cannot read WikiSkill state ${path.split("/").at(-1)}`,
      "WIKISKILL_IO_ERROR",
      cause,
    );
  }
}

async function pathMissing(path: string): Promise<boolean> {
  try {
    await access(path);
    return false;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return true;
    throw error;
  }
}

export function wikiSkillTargetKey(rawTarget: WikiSkillTarget): string {
  const target = parsed(wikiSkillTargetSchema, rawTarget, "Invalid WikiSkill target");
  return createHash("sha256")
    .update(JSON.stringify([target.preset, target.model]))
    .digest("hex");
}

export class WikiSkillStore {
  readonly root: string;

  constructor(root: string) {
    if (root.trim().length === 0) {
      throw wikiSkillError("WikiSkill root is empty", "WIKISKILL_INVALID_REQUEST");
    }
    this.root = resolve(root);
  }

  private get lockPath(): string {
    return join(this.root, ".swarmx", "write");
  }

  async initialize(): Promise<void> {
    await Promise.all([
      privateDirectory(this.root),
      privateDirectory(join(this.root, ".swarmx")),
      privateDirectory(join(this.root, "active")),
      privateDirectory(join(this.root, "staging")),
    ]);
  }

  activeRoot(target: WikiSkillTarget): string {
    return join(this.root, "active", wikiSkillTargetKey(target));
  }

  async stage(rawRequest: WikiSkillStageRequest, signal?: AbortSignal): Promise<WikiSkillProposal> {
    const request = parsed(wikiSkillStageRequestSchema, rawRequest, "Invalid WikiSkill proposal");
    validateSkill(request.name, request.skillMarkdown);
    const purposeMarkdown = renderPurpose(request);
    const candidateRevision = activeRevision(
      request.target,
      request.name,
      request.skillMarkdown,
      purposeMarkdown,
    );
    const proposalRevision = hash(proposalPayload(request, purposeMarkdown));
    await this.initialize();
    return withFileLock(this.lockPath, async () => {
      signal?.throwIfAborted();
      const proposalDirectory = join(this.root, "staging", request.proposalId);
      const proposalFile = join(proposalDirectory, "proposal.json");
      if (!(await pathMissing(proposalFile))) {
        const existing = await this.readProposal(request.proposalId);
        if (existing.proposalRevision !== proposalRevision) {
          throw wikiSkillError(
            "WikiSkill proposal id was reused for different content",
            "WIKISKILL_REVISION_CONFLICT",
          );
        }
        return existing;
      }
      await this.assertExpectedActive(request);
      signal?.throwIfAborted();
      const stagedAt = new Date().toISOString();
      const record: WikiSkillProposalRecord = {
        candidateRevision,
        ...(request.expectedActiveRevision === undefined
          ? {}
          : { expectedActiveRevision: request.expectedActiveRevision }),
        name: request.name,
        operation: request.operation,
        proposalId: request.proposalId,
        proposalRevision,
        schemaVersion: 1,
        sources: request.sources,
        stagedAt,
        target: request.target,
      };
      const temporary = join(this.root, "staging", `.${request.proposalId}-${randomUUID()}`);
      try {
        const skillDirectory = join(temporary, request.name);
        await privateDirectory(skillDirectory);
        await Promise.all([
          privateFile(join(skillDirectory, "SKILL.md"), request.skillMarkdown),
          privateFile(join(skillDirectory, "PURPOSE.md"), purposeMarkdown),
          privateFile(join(temporary, "proposal.json"), json(record)),
        ]);
        signal?.throwIfAborted();
        await rename(temporary, proposalDirectory);
        await chmod(proposalDirectory, DIRECTORY_MODE);
      } catch (cause) {
        await rm(temporary, { force: true, recursive: true });
        throw cause;
      }
      return { ...record, purposeMarkdown, skillMarkdown: request.skillMarkdown };
    });
  }

  async readProposal(proposalId: string): Promise<WikiSkillProposal> {
    const id = parsed(
      wikiSkillStageRequestSchema.shape.proposalId,
      proposalId,
      "Invalid WikiSkill proposal id",
    );
    const directory = join(this.root, "staging", id);
    let record: WikiSkillProposalRecord;
    try {
      record = wikiSkillProposalRecordSchema.parse(
        await readJson(join(directory, "proposal.json")),
      );
    } catch (cause) {
      if (
        cause instanceof WikiSkillError &&
        (cause.cause as NodeJS.ErrnoException | undefined)?.code === "ENOENT"
      ) {
        throw wikiSkillError(
          "WikiSkill proposal was not found",
          "WIKISKILL_PROPOSAL_NOT_FOUND",
          cause,
        );
      }
      throw cause;
    }
    if (record.proposalId !== id) {
      throw wikiSkillError("WikiSkill proposal identity changed", "WIKISKILL_IO_ERROR");
    }
    const skillDirectory = join(directory, record.name);
    const [skillMarkdown, purposeMarkdown] = await Promise.all([
      readBoundedText(join(skillDirectory, "SKILL.md"), SKILL_MAX_BYTES),
      readBoundedText(join(skillDirectory, "PURPOSE.md"), PURPOSE_MAX_BYTES),
    ]);
    validateSkill(record.name, skillMarkdown);
    const candidateRevision = activeRevision(
      record.target,
      record.name,
      skillMarkdown,
      purposeMarkdown,
    );
    if (
      record.candidateRevision !== candidateRevision ||
      record.proposalRevision !==
        hash(
          proposalPayload(
            {
              expectedActiveRevision: record.expectedActiveRevision,
              name: record.name,
              operation: record.operation,
              proposalId: record.proposalId,
              purposeMarkdown,
              skillMarkdown,
              sources: record.sources,
              target: record.target,
            },
            purposeMarkdown,
          ),
        )
    ) {
      throw wikiSkillError(
        "WikiSkill proposal content changed after staging",
        "WIKISKILL_REVISION_CONFLICT",
      );
    }
    return { ...record, purposeMarkdown, skillMarkdown };
  }

  async readActive(
    rawTarget: WikiSkillTarget,
    name: string,
  ): Promise<WikiSkillActiveSkill | undefined> {
    const target = parsed(wikiSkillTargetSchema, rawTarget, "Invalid WikiSkill target");
    if (!isSkillName(name)) {
      throw wikiSkillError("Invalid WikiSkill name", "WIKISKILL_INVALID_REQUEST");
    }
    const directory = join(this.activeRoot(target), name);
    const skillPath = join(directory, "SKILL.md");
    if (await pathMissing(skillPath)) return undefined;
    let skillMarkdown: string;
    let purposeMarkdown: string;
    let metadata: ActiveMetadata;
    try {
      [skillMarkdown, purposeMarkdown, metadata] = await Promise.all([
        readBoundedText(skillPath, SKILL_MAX_BYTES),
        readBoundedText(join(directory, "PURPOSE.md"), PURPOSE_MAX_BYTES),
        readJson(join(directory, ".wikiskill.json")).then((value) =>
          activeMetadataSchema.parse(value),
        ),
      ]);
    } catch (cause) {
      throw wikiSkillError("WikiSkill active skill is incomplete", "WIKISKILL_IO_ERROR", cause);
    }
    validateSkill(name, skillMarkdown);
    if (
      metadata.schemaVersion !== 1 ||
      metadata.name !== name ||
      !sameTarget(metadata.target, target)
    ) {
      throw wikiSkillError("WikiSkill active metadata is invalid", "WIKISKILL_IO_ERROR");
    }
    const revision = activeRevision(target, name, skillMarkdown, purposeMarkdown);
    if (metadata.candidateRevision !== revision) {
      throw wikiSkillError(
        "WikiSkill active content changed outside validation",
        "WIKISKILL_REVISION_CONFLICT",
      );
    }
    return {
      name,
      purposeMarkdown,
      revision,
      skillMarkdown,
      target,
    };
  }

  async resolve(
    proposalId: string,
    rawEvaluation: WikiSkillEvaluation,
    signal?: AbortSignal,
  ): Promise<WikiSkillOutcome> {
    const evaluation = parsed(
      wikiSkillEvaluationSchema,
      rawEvaluation,
      "Invalid WikiSkill evaluation",
    );
    const id = parsed(
      wikiSkillStageRequestSchema.shape.proposalId,
      proposalId,
      "Invalid WikiSkill proposal id",
    );
    await this.initialize();
    return withFileLock(this.lockPath, async () => {
      signal?.throwIfAborted();
      const proposal = await this.readProposal(id);
      this.assertEvaluation(proposal, evaluation);
      const evaluationRevision = hash(evaluation);
      const outcomePath = join(this.root, "staging", id, "outcome.json");
      if (!(await pathMissing(outcomePath))) {
        const existing = wikiSkillOutcomeSchema.parse(await readJson(outcomePath));
        if (
          existing.proposalRevision !== proposal.proposalRevision ||
          existing.evaluationRevision !== evaluationRevision
        ) {
          throw wikiSkillError(
            "WikiSkill proposal was already resolved with another evaluation",
            "WIKISKILL_REVISION_CONFLICT",
          );
        }
        return existing;
      }
      const verdict =
        evaluation.candidate.score > evaluation.baseline.score ? "accepted" : "rejected";
      const impact = impactDraft(proposal, evaluation, verdict);
      const current = await this.assertExpectedActive(proposal, proposal.candidateRevision);
      signal?.throwIfAborted();
      if (verdict === "accepted" && current?.revision !== proposal.candidateRevision) {
        await this.promote(proposal);
      }
      const outcome: WikiSkillOutcome = {
        ...(verdict === "accepted" ? { activeRevision: proposal.candidateRevision } : {}),
        candidateRevision: proposal.candidateRevision,
        evaluation,
        evaluationRevision,
        impact,
        name: proposal.name,
        proposalId: proposal.proposalId,
        proposalRevision: proposal.proposalRevision,
        resolvedAt: new Date().toISOString(),
        schemaVersion: 1,
        target: proposal.target,
        verdict,
      };
      await privateFile(outcomePath, json(outcome));
      return outcome;
    });
  }

  private async assertExpectedActive(
    request: Pick<
      WikiSkillStageRequest | WikiSkillProposal,
      "expectedActiveRevision" | "name" | "operation" | "target"
    >,
    committedCandidateRevision?: `sha256:${string}`,
  ): Promise<WikiSkillActiveSkill | undefined> {
    const active = await this.readActive(request.target, request.name);
    if (
      committedCandidateRevision !== undefined &&
      active?.revision === committedCandidateRevision
    ) {
      return active;
    }
    if (request.operation === "create") {
      if (active !== undefined) {
        throw wikiSkillError(
          "WikiSkill create target already exists",
          "WIKISKILL_REVISION_CONFLICT",
        );
      }
      return undefined;
    }
    if (active === undefined || active.revision !== request.expectedActiveRevision) {
      throw wikiSkillError("WikiSkill active revision changed", "WIKISKILL_REVISION_CONFLICT");
    }
    return active;
  }

  private assertEvaluation(proposal: WikiSkillProposal, evaluation: WikiSkillEvaluation): void {
    const { baseline, candidate } = evaluation;
    if (
      !sameTarget(baseline.target, proposal.target) ||
      !sameTarget(candidate.target, proposal.target) ||
      baseline.benchmarkId !== candidate.benchmarkId ||
      baseline.taskSetRevision !== candidate.taskSetRevision ||
      baseline.runs !== candidate.runs ||
      candidate.skillRevision !== proposal.candidateRevision ||
      (proposal.operation === "create"
        ? baseline.skillRevision !== undefined
        : baseline.skillRevision !== proposal.expectedActiveRevision)
    ) {
      throw wikiSkillError(
        "WikiSkill evaluation does not match the proposal target and revisions",
        "WIKISKILL_INVALID_REQUEST",
      );
    }
  }

  private async promote(proposal: WikiSkillProposal): Promise<void> {
    const targetRoot = this.activeRoot(proposal.target);
    const directory = join(targetRoot, proposal.name);
    await privateDirectory(targetRoot);
    await privateDirectory(directory);
    await privateFile(
      join(targetRoot, ".target.json"),
      json({ schemaVersion: 1, target: proposal.target }),
    );
    await privateFile(join(directory, "PURPOSE.md"), proposal.purposeMarkdown);
    const metadata: ActiveMetadata = {
      candidateRevision: proposal.candidateRevision,
      name: proposal.name,
      proposalId: proposal.proposalId,
      schemaVersion: 1,
      target: proposal.target,
    };
    await privateFile(join(directory, ".wikiskill.json"), json(metadata));
    await privateFile(join(directory, "SKILL.md"), proposal.skillMarkdown);
  }
}
