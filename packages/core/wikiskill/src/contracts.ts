import { isAbsolute, posix } from "node:path";
import { z } from "zod";

export const sha256Schema = z
  .string()
  .regex(/^sha256:[a-f0-9]{64}$/u)
  .transform((value): `sha256:${string}` => value as `sha256:${string}`);

const identitySchema = z
  .string()
  .trim()
  .min(1)
  .max(200)
  .refine((value) => !/[\0\r\n]/u.test(value), "Identity contains a control character");

export const wikiSkillTargetSchema = z.strictObject({
  model: identitySchema,
  preset: identitySchema,
});

export const dshRawTraceLocatorSchema = z
  .strictObject({
    endSeq: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
    sessionId: z
      .string()
      .trim()
      .min(1)
      .max(1_024)
      .refine((value) => !/[\0\r\n`]/u.test(value), "Session id is not provenance-safe"),
    startSeq: z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER),
  })
  .refine(({ endSeq, startSeq }) => endSeq >= startSeq, {
    message: "Raw trace endSeq must not precede startSeq",
  });

export const pkbConceptReferenceSchema = z.strictObject({
  id: z
    .string()
    .trim()
    .min(1)
    .max(1_024)
    .refine(
      (value) =>
        !/[\0\r\n`]/u.test(value) &&
        !value.includes("\\") &&
        !isAbsolute(value) &&
        posix.normalize(value) === value &&
        !value.startsWith("../") &&
        value.endsWith(".md"),
      "PKB concept id must be a portable relative Markdown id",
    ),
  revision: sha256Schema,
});

export const wikiSkillSourcesSchema = z.strictObject({
  patterns: z.array(pkbConceptReferenceSchema).min(1).max(32),
  traces: z.array(dshRawTraceLocatorSchema).min(1).max(32),
});

export const wikiSkillStageRequestSchema = z
  .strictObject({
    expectedActiveRevision: sha256Schema.optional(),
    name: z
      .string()
      .regex(/^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/u)
      .max(100),
    operation: z.enum(["create", "patch"]),
    proposalId: z.string().uuid(),
    purposeMarkdown: z.string().trim().min(1).max(131_072),
    skillMarkdown: z.string().min(1).max(131_072),
    sources: wikiSkillSourcesSchema,
    target: wikiSkillTargetSchema,
  })
  .superRefine((request, context) => {
    if (request.operation === "patch" && request.expectedActiveRevision === undefined) {
      context.addIssue({
        code: "custom",
        message: "Patch proposals require expectedActiveRevision",
        path: ["expectedActiveRevision"],
      });
    }
    if (request.operation === "create" && request.expectedActiveRevision !== undefined) {
      context.addIssue({
        code: "custom",
        message: "Create proposals cannot carry expectedActiveRevision",
        path: ["expectedActiveRevision"],
      });
    }
  });

export const wikiSkillMeasurementSchema = z.strictObject({
  benchmarkId: identitySchema,
  runs: z.number().int().positive().max(1_000),
  score: z.number().finite(),
  skillRevision: sha256Schema.optional(),
  target: wikiSkillTargetSchema,
  taskSetRevision: sha256Schema,
});

export const wikiSkillEvaluationSchema = z.strictObject({
  baseline: wikiSkillMeasurementSchema,
  candidate: wikiSkillMeasurementSchema,
});

export const wikiSkillProposalRecordSchema = z.strictObject({
  candidateRevision: sha256Schema,
  expectedActiveRevision: sha256Schema.optional(),
  name: wikiSkillStageRequestSchema.shape.name,
  operation: z.enum(["create", "patch"]),
  proposalId: z.string().uuid(),
  proposalRevision: sha256Schema,
  schemaVersion: z.literal(1),
  sources: wikiSkillSourcesSchema,
  stagedAt: z.string().datetime(),
  target: wikiSkillTargetSchema,
});

const impactSourceSchema = z.strictObject({
  resource: z.string().min(1).max(2_048),
  title: z.string().min(1).max(500),
});

export const skillImpactDraftSchema = z.strictObject({
  body: z.string().min(1).max(65_536),
  description: z.string().min(1).max(500),
  sources: z.array(impactSourceSchema).min(1).max(64),
  status: z.literal("draft"),
  tags: z.array(z.string().min(1).max(80)).max(32),
  title: z.string().min(1).max(500),
  type: z.literal("SkillImpact"),
});

export const wikiSkillOutcomeSchema = z.strictObject({
  activeRevision: sha256Schema.optional(),
  candidateRevision: sha256Schema,
  evaluation: wikiSkillEvaluationSchema,
  evaluationRevision: sha256Schema,
  impact: skillImpactDraftSchema,
  name: wikiSkillStageRequestSchema.shape.name,
  proposalId: z.string().uuid(),
  proposalRevision: sha256Schema,
  resolvedAt: z.string().datetime(),
  schemaVersion: z.literal(1),
  target: wikiSkillTargetSchema,
  verdict: z.enum(["accepted", "rejected"]),
});

export type DshRawTraceLocator = z.infer<typeof dshRawTraceLocatorSchema>;
export type PkbConceptReference = z.infer<typeof pkbConceptReferenceSchema>;
export type SkillImpactDraft = z.infer<typeof skillImpactDraftSchema>;
export type WikiSkillEvaluation = z.infer<typeof wikiSkillEvaluationSchema>;
export type WikiSkillMeasurement = z.infer<typeof wikiSkillMeasurementSchema>;
export type WikiSkillOutcome = z.infer<typeof wikiSkillOutcomeSchema>;
export type WikiSkillProposalRecord = z.infer<typeof wikiSkillProposalRecordSchema>;
export type WikiSkillSources = z.infer<typeof wikiSkillSourcesSchema>;
export type WikiSkillStageRequest = z.infer<typeof wikiSkillStageRequestSchema>;
export type WikiSkillTarget = z.infer<typeof wikiSkillTargetSchema>;

export interface WikiSkillProposal extends WikiSkillProposalRecord {
  readonly purposeMarkdown: string;
  readonly skillMarkdown: string;
}

export interface WikiSkillActiveSkill {
  readonly name: string;
  readonly purposeMarkdown: string;
  readonly revision: `sha256:${string}`;
  readonly skillMarkdown: string;
  readonly target: WikiSkillTarget;
}
