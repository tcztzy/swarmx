import { createHash } from "node:crypto";
import { EventType } from "@ag-ui/core";
import { evaluationSchema, type MemoryConcept } from "@swarmx/memory";
import {
  RO_CRATE_CONTEXT,
  RO_CRATE_FILENAME,
  RO_CRATE_PROFILE,
  type RoCrateEntity,
} from "@swarmx/science/types";
import { z } from "zod";
import { EvaluationCrateSchema } from "../evaluation-crate.js";
import type { ExecutionRecord } from "../execution-record.js";
import { executionStatistics } from "./execution-evidence.js";
import type { ExecutionJournal } from "./execution-journal.js";
import { ResourceSnapshotSchema, ResourceUpdateSchema } from "./learning-resources.js";

const ref = (id: string) => ({ "@id": id });
const urn = (id: string) => `urn:swarmx:execution:${id}`;
const recordPath = (id: string) => `records/${id}.json`;
const digest = (text: string) => createHash("sha256").update(text).digest("hex");
const MAX_BYTES = 8 * 1024 * 1024;
const ReviewerSchema = z.object({
  harness: z.string(),
  requestedModel: z.string().nullable().optional(),
  model: z.string().nullable().optional(),
  provider: z.string().nullable().optional(),
  version: z.string().nullable().optional(),
});
const StartedSchema = z.object({
  jobId: z.string().optional(),
  prompt: z.string().optional(),
  promptRevision: z.string().optional(),
  reviewer: ReviewerSchema.optional(),
  snapshot: z
    .object({
      resources: z.array(ResourceSnapshotSchema).optional(),
      concepts: z
        .array(z.object({ id: z.string(), revision: z.string(), body: z.string() }))
        .optional(),
    })
    .passthrough(),
});
const PlanSchema = z.object({
  summary: z.string(),
  operations: z.array(z.object({ action: z.string(), request: z.record(z.string(), z.unknown()) })),
  reviewer: ReviewerSchema.optional(),
});

/** One selected evaluation/review, projected from existing stores into an Attached RO-Crate. */
export function createEvaluationCrate(
  journal: ExecutionJournal,
  selection: { concept: MemoryConcept; source: string } | { reviewSource: string },
) {
  const selected = "concept" in selection ? selection.concept : undefined;
  const evaluation = selected?.metadata.swarmx_evaluation;
  if (selected && !evaluation)
    throw new Error("This concept has no structured evaluation to export.");
  const citations = evaluation ? [...evaluation.evidence, ...evaluation.counterEvidence] : [];
  const reviewSource = "reviewSource" in selection ? selection.reviewSource : evaluation?.review;
  const review = reviewSource ? journal.resolveSource(reviewSource) : undefined;
  if (
    review &&
    (review.event.type !== EventType.CUSTOM || review.event.name !== "swarmx.memory.review.started")
  )
    throw new Error("Evaluation review source must identify a review.started snapshot.");
  const started =
    review?.event.type === EventType.CUSTOM ? StartedSchema.parse(review.event.value) : undefined;
  const evidence = journal.evidence(reviewSource ? [reviewSource] : citations);
  const records = new Map(evidence.records.map((record) => [record.id, record]));
  const citedRunIds = new Set<string>();
  for (const citation of citations) {
    const record = journal.resolveSource(citation);
    if (record.event.type === EventType.CUSTOM && record.event.name.startsWith("swarmx.memory."))
      throw new Error("An evaluation must cite original execution evidence.");
    if (review && !records.has(record.id))
      throw new Error("Evaluation evidence was not included in its attributed review.");
    records.set(record.id, record);
    if (record.runId) citedRunIds.add(record.runId);
  }
  const runs = selected ? evidence.runs.filter((run) => citedRunIds.has(run.runId)) : evidence.runs;
  for (const run of runs)
    for (const source of run.sources) {
      const record = journal.resolveSource(source);
      records.set(record.id, record);
    }
  const attempt: ExecutionRecord[] = [];
  if (review && started?.jobId) {
    const job = journal.resolveSource(urn(started.jobId));
    records.set(job.id, job);
    for (const record of journal.memoryJobEvents(started.jobId)) {
      if (record.seq <= review.seq || record.event.type !== EventType.CUSTOM) continue;
      if (record.event.name === "swarmx.memory.review.started") break;
      if (
        record.event.name === "swarmx.memory.review.response" &&
        z.object({ source: z.string() }).parse(record.event.value).source !== reviewSource
      )
        continue;
      attempt.push(record);
      records.set(record.id, record);
    }
  }
  const proposals = attempt.filter(
    (record) =>
      record.event.type === EventType.CUSTOM && record.event.name === "swarmx.memory.proposed",
  );
  const decisions = journal.memoryDecisions(proposals.map(({ id }) => id));
  for (const record of decisions) records.set(record.id, record);
  const planRecord = attempt.find(
    (record) =>
      record.event.type === EventType.CUSTOM &&
      record.event.name === "swarmx.memory.review.planned",
  );
  const plan =
    planRecord?.event.type === EventType.CUSTOM
      ? PlanSchema.parse(planRecord.event.value)
      : undefined;
  if (
    evaluation?.review &&
    !plan?.operations.some((operation) => {
      if (!selected || !["create_memory", "update_memory"].includes(operation.action)) return false;
      const request = operation.request;
      const updating = operation.action === "update_memory";
      if (
        typeof request.requestId !== "string" ||
        request.requestId !==
          selected.metadata[updating ? "swarmx_update_request_id" : "swarmx_request_id"] ||
        (updating && request.id !== selected.id)
      )
        return false;
      const body =
        typeof request.body === "string"
          ? request.body
          : updating
            ? started?.snapshot.concepts?.find(
                (concept) =>
                  concept.id === request.id && concept.revision === request.expectedRevision,
              )?.body
            : undefined;
      if (body?.trim() !== selected.body.trim()) return false;
      const parsed = evaluationSchema.safeParse(operation.request.evaluation);
      return parsed.success && JSON.stringify(parsed.data) === JSON.stringify(evaluation);
    })
  )
    throw new Error("The attributed review did not generate this evaluation.");

  const files: { path: string; content: string }[] = [];
  const entities: RoCrateEntity[] = [];
  let bytes = 0;
  const addFile = (path: string, content: string, name: string, mime: string) => {
    bytes += Buffer.byteLength(content);
    if (bytes > MAX_BYTES) throw new Error("Evaluation evidence package exceeds 8 MiB.");
    files.push({ path, content });
    entities.push({
      "@id": path,
      "@type": "File",
      name,
      encodingFormat: mime,
      contentSize: String(Buffer.byteLength(content)),
      sha256: digest(content),
    });
    return ref(path);
  };
  const ordered = [...records.values()].sort((a, b) => a.seq - b.seq);
  for (const record of ordered)
    addFile(
      recordPath(record.id),
      journal.sourceText(urn(record.id)),
      `Original execution record ${record.seq}`,
      "application/json",
    );
  const statistics = executionStatistics(runs);
  addFile(
    "statistics.json",
    `${JSON.stringify({ runs, statistics }, null, 2)}\n`,
    "Computed execution statistics and source scope",
    "application/json",
  );
  entities.push({
    "@id": "#statistics-software",
    "@type": "SoftwareApplication",
    name: "SwarmX execution statistics",
    version: statistics.recipe,
  });
  entities.push({
    "@id": "#statistics",
    "@type": "CreateAction",
    name: "Compute cited execution statistics",
    instrument: ref("#statistics-software"),
    object: [...new Set(runs.flatMap((run) => run.sources))].map((source) =>
      ref(recordPath(journal.resolveSource(source).id)),
    ),
    result: [ref("statistics.json")],
    actionStatus: ref("https://schema.org/CompletedActionStatus"),
  });
  const allRuns = evidence.runs;
  for (const run of allRuns) {
    entities.push({
      "@id": `#software-${run.runId}`,
      "@type": "SoftwareApplication",
      name: run.harness ?? "Unknown harness",
      ...(run.harnessVersion ? { version: run.harnessVersion } : {}),
      description: JSON.stringify({
        requestedModel: run.requestedModel,
        requestedEffort: run.requestedEffort,
        provider: run.provider,
        modelVersion: run.modelVersion,
        profile: run.profile,
        usageBasis: run.usageBasis,
      }),
    });
    const inputs = ordered.filter(
      (record) => record.runId === run.runId && record.event.type === EventType.RUN_STARTED,
    );
    const outputs = ordered.filter(
      (record) =>
        record.runId === run.runId &&
        record.event.type !== EventType.RUN_STARTED &&
        record.event.type !== EventType.CUSTOM,
    );
    entities.push({
      "@id": `#run-${run.runId}`,
      "@type": "CreateAction",
      name: "Observed agent execution",
      description: `Observed outcome: ${run.outcome}; normal completion does not establish correctness.`,
      instrument: ref(`#software-${run.runId}`),
      object: inputs.map(({ id }) => ref(recordPath(id))),
      result: outputs.map(({ id }) => ref(recordPath(id))),
      ...(run.startedAt ? { startTime: run.startedAt } : {}),
      ...(run.finishedAt ? { endTime: run.finishedAt } : {}),
      actionStatus: ref(
        `https://schema.org/${run.outcome === "completed" ? "CompletedActionStatus" : run.outcome === "incomplete" ? "ActiveActionStatus" : "FailedActionStatus"}`,
      ),
    });
  }
  entities.push({
    "@id": "#evaluated-routes",
    "@type": "Collection",
    name: "Runtime routes in the selected evidence",
    hasPart: runs.map((run) => ref(`#software-${run.runId}`)),
  });
  const addAssessment = (
    id: string,
    name: string,
    body: string,
    value: z.infer<typeof evaluationSchema>,
    target: string,
  ) => {
    for (const citation of [...value.evidence, ...value.counterEvidence]) {
      const record = journal.resolveSource(citation);
      if (
        !records.has(record.id) ||
        (record.event.type === EventType.CUSTOM && record.event.name.startsWith("swarmx.memory."))
      )
        throw new Error("Assessment evidence is not original evidence in this export.");
    }
    const sources = (references: string[]) =>
      references.map((source) => ref(recordPath(journal.resolveSource(source).id)));
    entities.push({
      "@id": id,
      "@type": "Review",
      name,
      reviewBody: body,
      reviewAspect: value.criteria,
      description: `${value.kind}. Task: ${value.task}. Limitations: ${value.limitations}`,
      itemReviewed: ref(target),
      citation: sources([...value.evidence, ...value.counterEvidence]),
    });
    for (const [suffix, references, direction] of [
      ["support", value.evidence, 1],
      ["counter", value.counterEvidence, -1],
    ] as const) {
      if (!references.length) continue;
      entities.push({
        "@id": `${id}-${suffix}`,
        "@type": "Review",
        name: `${suffix} evidence supplied for this assessment`,
        itemReviewed: ref(id),
        citation: sources(references),
        reviewRating: ref(`${id}-${suffix}-direction`),
      });
      entities.push({
        "@id": `${id}-${suffix}-direction`,
        "@type": "Rating",
        name: "Evidence direction, not a quality score",
        ratingValue: direction,
        bestRating: 1,
        worstRating: -1,
      });
    }
  };
  if ("concept" in selection && evaluation) {
    addFile(
      "evaluation.md",
      selection.source,
      selected?.metadata.title ?? "Evaluation",
      "text/markdown",
    );
    addFile(
      "evaluation.json",
      `${JSON.stringify(selection.concept, null, 2)}\n`,
      "Parsed evaluation and its revision",
      "application/json",
    );
    addAssessment(
      "#evaluation",
      selection.concept.metadata.title,
      selection.concept.body,
      evaluation,
      "#evaluated-routes",
    );
    const entity = entities.find((entity) => entity["@id"] === "#evaluation");
    if (entity) {
      entity.isBasedOn = [ref("evaluation.md")];
      entity.version = selection.concept.revision;
    }
  }
  if (review && started) {
    const responseRecord = attempt.find(
      (record) =>
        record.event.type === EventType.CUSTOM &&
        record.event.name === "swarmx.memory.review.response",
    );
    const response =
      responseRecord?.event.type === EventType.CUSTOM
        ? z.object({ text: z.string(), reviewer: ReviewerSchema }).parse(responseRecord.event.value)
        : undefined;
    const reviewer = response?.reviewer ?? plan?.reviewer ?? started.reviewer;
    entities.push({
      "@id": "#reviewer",
      "@type": "SoftwareApplication",
      name: reviewer?.harness ?? "Unknown review runtime",
      ...(reviewer?.version ? { version: reviewer.version } : {}),
      description: JSON.stringify(reviewer ?? { identity: null }),
    });
    const inputs = [ref(recordPath(review.id))];
    const outputs = planRecord ? [ref(recordPath(planRecord.id))] : [];
    if (selected) outputs.push(ref("#evaluation"));
    if (started.prompt !== undefined) {
      if (started.promptRevision !== `sha256:${digest(started.prompt)}`)
        throw new Error("Review prompt does not match its recorded revision.");
      inputs.push(
        addFile("review/prompt.txt", started.prompt, "Exact review prompt", "text/plain"),
      );
    }
    if (response)
      outputs.push(
        addFile(
          "review/response.txt",
          response.text,
          "Exact model response before parsing or Host changes",
          "text/plain",
        ),
      );
    const finishedRecord = attempt.findLast(
      (record) =>
        record.event.type === EventType.CUSTOM &&
        record.event.name === "swarmx.memory.review.finished",
    );
    const finished =
      finishedRecord?.event.type === EventType.CUSTOM
        ? z.object({ state: z.enum(["completed", "failed"]) }).parse(finishedRecord.event.value)
        : undefined;
    const reviewAction: RoCrateEntity = {
      "@id": "#review",
      "@type": "CreateAction",
      name: "Generate an evidence-based evaluation",
      description: `${plan?.summary ?? "No validated plan recorded."} Original model response: ${response ? "included" : "not recorded"}.`,
      instrument: ref("#reviewer"),
      object: inputs,
      result: outputs,
      startTime: review.observedAt,
      ...(finishedRecord ? { endTime: finishedRecord.observedAt } : {}),
      actionStatus: ref(
        `https://schema.org/${finished?.state === "completed" ? "CompletedActionStatus" : finished?.state === "failed" ? "FailedActionStatus" : "ActiveActionStatus"}`,
      ),
    };
    entities.push(reviewAction);
    for (const [index, operation] of (plan?.operations ?? []).entries()) {
      if (operation.request.evaluation === undefined) continue;
      const assessment = evaluationSchema.parse(operation.request.evaluation);
      let target = "#evaluated-routes";
      if (operation.action === "update_resource") {
        const update = ResourceUpdateSchema.parse(operation).request;
        const resource = started.snapshot.resources?.find((item) => item.id === update.id);
        if (
          !resource ||
          resource.expectedRevision !== update.expectedRevision ||
          `sha256:${digest(resource.content)}` !== resource.expectedRevision
        )
          throw new Error("Resource candidate does not match its review input revision.");
        const original = addFile(
          `resources/${index}-original.md`,
          resource.content,
          `Original ${resource.kind}: ${resource.id}`,
          "text/markdown",
        );
        const candidate = addFile(
          `resources/${index}-candidate.md`,
          update.content,
          `Proposed ${resource.kind}: ${resource.id}`,
          "text/markdown",
        );
        outputs.push(candidate);
        target = original["@id"];
        const proposed = proposals.find(
          (record) =>
            record.event.type === EventType.CUSTOM &&
            z.object({ operationIndex: z.number() }).parse(record.event.value).operationIndex ===
              index,
        );
        const decision =
          proposed &&
          decisions.find(
            (record) =>
              record.event.type === EventType.CUSTOM &&
              z.object({ proposalId: z.string() }).parse(record.event.value).proposalId ===
                proposed.id,
          );
        const saved = attempt.find(
          (record) =>
            record.event.type === EventType.CUSTOM &&
            record.event.name === "swarmx.memory.saved" &&
            z.object({ operationIndex: z.number() }).parse(record.event.value).operationIndex ===
              index,
        );
        const applied = Boolean(
          saved ||
            (decision?.event.type === EventType.CUSTOM &&
              decision.event.name === "swarmx.memory.accepted"),
        );
        const rejected =
          decision?.event.type === EventType.CUSTOM &&
          decision.event.name === "swarmx.memory.rejected";
        const receipt = saved ?? decision;
        entities.push({
          "@id": `#update-${index}`,
          "@type": "UpdateAction",
          name: `Proposed ${resource.kind} improvement: ${resource.id}`,
          description: `${applied ? "Applied after its configured validator passed" : rejected ? "Rejected; not applied" : "Proposed; no successful application recorded"}. The validator is not proof of improved task performance.`,
          object: [original],
          result: [candidate],
          instrument: ref("#swarmx"),
          actionStatus: ref(
            `https://schema.org/${applied ? "CompletedActionStatus" : rejected ? "FailedActionStatus" : "PotentialActionStatus"}`,
          ),
          ...(receipt ? { subjectOf: ref(recordPath(receipt.id)) } : {}),
        });
      }
      addAssessment(
        `#assessment-${index}`,
        `Assessment for ${operation.action}`,
        typeof operation.request.body === "string" ? operation.request.body : assessment.criteria,
        assessment,
        target,
      );
      outputs.push(ref(`#assessment-${index}`));
    }
  }
  entities.push({ "@id": "#swarmx", "@type": "SoftwareApplication", name: "SwarmX Host" });
  const root: RoCrateEntity = {
    "@id": "./",
    "@type": "Dataset",
    name: selected?.metadata.title ?? "Execution evaluation research object",
    description:
      "Private selected execution evidence and evaluation provenance. Includes original content; not redacted or automatically published. Hashes establish content identity, not truth.",
    datePublished:
      "concept" in selection
        ? selection.concept.metadata.generated.at
        : journal.resolveSource(selection.reviewSource).observedAt,
    license: "All rights reserved",
    hasPart: files.map(({ path }) => ref(path)),
    mentions: entities
      .filter((entity) => entity["@id"].startsWith("#"))
      .map((entity) => ref(entity["@id"])),
  };
  const metadata = {
    "@context": RO_CRATE_CONTEXT,
    "@graph": [
      {
        "@id": RO_CRATE_FILENAME,
        "@type": "CreativeWork",
        about: ref("./"),
        conformsTo: ref(RO_CRATE_PROFILE),
      },
      root,
      ...entities,
    ],
  };
  bytes += Buffer.byteLength(`${JSON.stringify(metadata, null, 2)}\n`);
  if (bytes > MAX_BYTES) throw new Error("Evaluation evidence package exceeds 8 MiB.");
  return EvaluationCrateSchema.parse({ metadata, files });
}
