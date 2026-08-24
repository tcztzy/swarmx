import {
  type ProvenanceReceipt,
  RO_CRATE_CONTEXT,
  RO_CRATE_FILENAME,
  RO_CRATE_PROFILE,
  type RoCrateEntity,
  type RoCrateMetadataDocument,
  roCrateEntityId,
  roCrateMetadataDocumentSchema,
  type ScienceArtifact,
  type ScienceResearchRecord,
  type ScienceWorkspaceSnapshot,
} from "./contracts.js";
import { ScienceError } from "./errors.js";

const SWARMX_SOFTWARE_ID = "https://github.com/tcztzy/swarmx";

type Reference = { readonly "@id": string };

function reference(id: string): Reference {
  return { "@id": id };
}

function references(ids: readonly string[]): Reference[] {
  return [...new Set(ids)].map(reference);
}

function instant(timestamp: number): string {
  return new Date(timestamp).toISOString();
}

function sourceReferences(entityIds: readonly string[]): Reference[] {
  return references(entityIds.map(roCrateEntityId));
}

function artifactTypes(artifact: ScienceArtifact): string[] {
  const types = ["MediaObject"];
  if (artifact.kind === "figure" || artifact.mime.startsWith("image/")) types.push("ImageObject");
  if (artifact.kind === "pdf") types.push("DigitalDocument");
  if (artifact.kind === "dataset") types.push("Dataset");
  if (artifact.kind === "code" || artifact.kind === "notebook") {
    types.push("SoftwareSourceCode");
  }
  return types;
}

function recordType(record: ScienceResearchRecord): string {
  if (record.kind === "question" || record.kind === "open-question") return "Question";
  if (record.kind === "claim") return "Claim";
  if (record.kind === "evidence" || record.kind === "review") return "Review";
  return "CreativeWork";
}

function recordAdditionalType(record: ScienceResearchRecord): string | undefined {
  if (record.kind === "hypothesis") return "Hypothesis";
  if (record.kind === "decision") return "Decision";
  if (record.kind === "open-question") return "OpenQuestion";
  return undefined;
}

function actionStatus(status: "running" | "succeeded" | "failed" | "cancelled"): Reference {
  const name =
    status === "running"
      ? "ActiveActionStatus"
      : status === "succeeded"
        ? "CompletedActionStatus"
        : "FailedActionStatus";
  return reference(`https://schema.org/${name}`);
}

function referenceIds(value: RoCrateEntity["object"]): string[] {
  if (!value) return [];
  return (Array.isArray(value) ? value : [value]).map((item) => item["@id"]);
}

interface ActionInput {
  readonly name: string;
  readonly objectIds?: readonly string[];
  readonly occurredAt: number;
  readonly provenance: ProvenanceReceipt;
  readonly resultIds: readonly string[];
  readonly type: "CreateAction" | "UpdateAction";
}

function addAction(actions: Map<string, RoCrateEntity>, input: ActionInput): void {
  const id = `#action-${input.provenance.eventId}`;
  const previous = actions.get(id);
  const objectIds = [
    ...referenceIds(previous?.object),
    ...(input.objectIds ?? []).map(roCrateEntityId),
  ];
  const resultIds = [...referenceIds(previous?.result), ...input.resultIds.map(roCrateEntityId)];
  actions.set(id, {
    "@id": id,
    "@type": previous?.["@type"] ?? input.type,
    name: previous?.name ?? input.name,
    identifier: `science-journal:${input.provenance.journalSeq}`,
    instrument: reference(SWARMX_SOFTWARE_ID),
    ...(objectIds.length > 0 ? { object: references(objectIds) } : {}),
    result: references(resultIds),
    endTime: instant(input.occurredAt),
    actionStatus: reference("https://schema.org/CompletedActionStatus"),
  });
}

/** Project client-safe Journal projections → one flat RO-Crate 1.3 Metadata Document. */
export function createResearchObject(
  snapshot: ScienceWorkspaceSnapshot,
  projectId: string,
): RoCrateMetadataDocument {
  const project = snapshot.projects.find((candidate) => candidate.id === projectId);
  if (!project) throw new ScienceError("Project not found in this workspace", "PROJECT_NOT_FOUND");

  const rootId = roCrateEntityId(project.id);
  const entities: RoCrateEntity[] = [];
  const actions = new Map<string, RoCrateEntity>();
  const contentIds: string[] = [];
  const contextualIds: string[] = [];
  const addContent = (entity: RoCrateEntity): void => {
    entities.push(entity);
    contentIds.push(entity["@id"]);
  };
  const addContext = (entity: RoCrateEntity): void => {
    entities.push(entity);
    contextualIds.push(entity["@id"]);
  };

  addAction(actions, {
    name: "Create Science project",
    occurredAt: project.createdAt,
    provenance: project.provenance,
    resultIds: [project.id],
    type: "CreateAction",
  });

  for (const notebook of snapshot.notebooks.filter((item) => item.projectId === project.id)) {
    const inputIds = notebook.cells.flatMap((cell) => cell.inputArtifactIds ?? []);
    addContent({
      "@id": roCrateEntityId(notebook.id),
      "@type": "SoftwareSourceCode",
      name: notebook.title,
      description: `Science notebook with ${notebook.cells.length} recorded cells`,
      isPartOf: reference(rootId),
      ...(inputIds.length > 0 ? { isBasedOn: sourceReferences(inputIds) } : {}),
      dateCreated: instant(notebook.createdAt),
      dateModified: instant(notebook.updatedAt),
      version: notebook.revision,
    });
    addAction(actions, {
      name: notebook.revision === 1 ? "Create notebook" : "Execute notebook cell",
      occurredAt: notebook.updatedAt,
      objectIds: inputIds,
      provenance: notebook.provenance,
      resultIds: [notebook.id],
      type: notebook.revision === 1 ? "CreateAction" : "UpdateAction",
    });
  }

  for (const artifact of snapshot.artifacts.filter((item) => item.projectId === project.id)) {
    addContent({
      "@id": roCrateEntityId(artifact.id),
      "@type": artifactTypes(artifact),
      name: artifact.title,
      encodingFormat: artifact.mime,
      contentSize: String(artifact.size),
      sha256: artifact.digest.slice("sha256:".length),
      isPartOf: reference(rootId),
      ...(artifact.license ? { license: artifact.license } : {}),
      ...(artifact.sourceEntityIds.length > 0
        ? { isBasedOn: sourceReferences(artifact.sourceEntityIds) }
        : {}),
      dateCreated: instant(artifact.createdAt),
      dateModified: instant(artifact.updatedAt),
      version: artifact.revision,
    });
    addAction(actions, {
      name: "Register research artifact",
      occurredAt: artifact.updatedAt,
      objectIds: artifact.sourceEntityIds,
      provenance: artifact.provenance,
      resultIds: [artifact.id],
      type: "CreateAction",
    });
  }

  for (const document of snapshot.documents.filter((item) => item.projectId === project.id)) {
    const revision = document.revisions.at(-1);
    addContent({
      "@id": roCrateEntityId(document.id),
      "@type": "DigitalDocument",
      name: document.name,
      description: `${document.format} writing source`,
      encodingFormat:
        document.format === "typst"
          ? "application/x-typst"
          : document.format === "latex"
            ? "application/x-latex"
            : document.format === "markdown"
              ? "text/markdown"
              : "application/x-bibtex",
      isPartOf: reference(rootId),
      ...(revision ? { sha256: revision.sourceHash.slice("sha256:".length) } : {}),
      dateCreated: instant(document.createdAt),
      dateModified: instant(document.updatedAt),
      version: document.contentRevision,
    });
    addAction(actions, {
      name: document.revision === 1 ? "Create writing document" : "Update writing document",
      occurredAt: document.updatedAt,
      provenance: document.provenance,
      resultIds: [document.id],
      type: document.revision === 1 ? "CreateAction" : "UpdateAction",
    });
  }

  for (const figure of snapshot.figures.filter((item) => item.projectId === project.id)) {
    const revision = figure.revisions.at(-1);
    const sourceIds = figure.artifactId ? [figure.artifactId] : [];
    addContent({
      "@id": roCrateEntityId(figure.id),
      "@type": "SoftwareSourceCode",
      name: figure.title,
      description: `${figure.library} figure source`,
      programmingLanguage: figure.library === "ggplot2" ? "R" : "Python",
      isPartOf: reference(rootId),
      ...(sourceIds.length > 0 ? { isBasedOn: sourceReferences(sourceIds) } : {}),
      ...(revision ? { sha256: revision.codeHash.slice("sha256:".length) } : {}),
      dateCreated: instant(figure.createdAt),
      dateModified: instant(figure.updatedAt),
      version: figure.codeRevision,
    });
    addAction(actions, {
      name: figure.revision === 1 ? "Create figure source" : "Update figure source",
      occurredAt: figure.updatedAt,
      objectIds: sourceIds,
      provenance: figure.provenance,
      resultIds: [figure.id],
      type: figure.revision === 1 ? "CreateAction" : "UpdateAction",
    });
  }

  const projectRelations = snapshot.relations.filter((item) => item.projectId === project.id);
  for (const record of snapshot.records.filter((item) => item.projectId === project.id)) {
    const additionalType = recordAdditionalType(record);
    const evidenceRelation = projectRelations.find(
      (relation) =>
        relation.fromId === record.id &&
        (relation.type === "supports" || relation.type === "refutes"),
    );
    const ratingId = evidenceRelation ? `#rating-${record.id}` : undefined;
    addContent({
      "@id": roCrateEntityId(record.id),
      "@type": recordType(record),
      name: record.title,
      description: record.summary,
      text: record.summary,
      creativeWorkStatus: record.status,
      keywords: record.tags,
      isPartOf: reference(rootId),
      ...(additionalType ? { additionalType } : {}),
      ...(record.sourceEntityIds.length > 0
        ? { isBasedOn: sourceReferences(record.sourceEntityIds) }
        : {}),
      ...(evidenceRelation
        ? {
            itemReviewed: reference(roCrateEntityId(evidenceRelation.toId)),
            reviewRating: reference(ratingId ?? ""),
          }
        : {}),
      dateCreated: instant(record.createdAt),
      dateModified: instant(record.updatedAt),
      version: record.revision,
    });
    if (evidenceRelation && ratingId) {
      addContext({
        "@id": ratingId,
        "@type": "Rating",
        name: evidenceRelation.type,
        ratingValue: evidenceRelation.type === "supports" ? 1 : -1,
        bestRating: 1,
        worstRating: -1,
      });
    }
    addAction(actions, {
      name: `Record research ${record.kind}`,
      occurredAt: record.updatedAt,
      objectIds: [...record.sourceEntityIds, ...(evidenceRelation ? [evidenceRelation.toId] : [])],
      provenance: record.provenance,
      resultIds: [record.id],
      type: "CreateAction",
    });
  }

  for (const experiment of snapshot.experiments.filter((item) => item.projectId === project.id)) {
    addContent({
      "@id": roCrateEntityId(experiment.id),
      "@type": "HowTo",
      name: experiment.title,
      description: experiment.summary,
      text: experiment.protocol,
      creativeWorkStatus: experiment.status,
      keywords: experiment.tags,
      isPartOf: reference(rootId),
      ...(experiment.hypothesisIds.length > 0
        ? { isBasedOn: sourceReferences(experiment.hypothesisIds) }
        : {}),
      dateCreated: instant(experiment.createdAt),
      dateModified: instant(experiment.updatedAt),
      version: experiment.revision,
    });
    addAction(actions, {
      name: experiment.revision === 1 ? "Define experiment" : "Update experiment",
      occurredAt: experiment.updatedAt,
      objectIds: experiment.hypothesisIds,
      provenance: experiment.provenance,
      resultIds: [experiment.id],
      type: experiment.revision === 1 ? "CreateAction" : "UpdateAction",
    });
  }

  for (const run of snapshot.runs.filter((item) => item.projectId === project.id)) {
    const id = roCrateEntityId(run.id);
    addContext({
      "@id": id,
      "@type": "CreateAction",
      name: `Experiment run ${run.id.slice(0, 8)}`,
      description: run.notes || `Run of experiment ${run.experimentId}`,
      identifier: `science-journal:${run.provenance.journalSeq}`,
      object: reference(roCrateEntityId(run.experimentId)),
      instrument: reference(SWARMX_SOFTWARE_ID),
      ...(run.artifactIds.length > 0 ? { result: sourceReferences(run.artifactIds) } : {}),
      actionStatus: actionStatus(run.status),
      startTime: instant(run.startedAt),
      ...(run.finishedAt === null ? {} : { endTime: instant(run.finishedAt) }),
    });
  }

  const actionEntities = [...actions.values()];
  const root: RoCrateEntity = {
    "@id": rootId,
    "@type": "Dataset",
    name: project.title,
    description: `Research Object for ${project.title}`,
    datePublished: instant(project.createdAt),
    license: "All rights reserved",
    hasPart: references(contentIds),
    mentions: references([...contextualIds, ...actionEntities.map((entity) => entity["@id"])]),
    version: project.revision,
  };
  const descriptor: RoCrateEntity = {
    "@id": RO_CRATE_FILENAME,
    "@type": "CreativeWork",
    about: reference(rootId),
    conformsTo: reference(RO_CRATE_PROFILE),
  };
  const software: RoCrateEntity = {
    "@id": SWARMX_SOFTWARE_ID,
    "@type": "SoftwareApplication",
    name: "SwarmX",
  };

  try {
    return roCrateMetadataDocumentSchema.parse({
      "@context": RO_CRATE_CONTEXT,
      "@graph": [descriptor, root, ...entities, software, ...actionEntities],
    });
  } catch (error) {
    throw new ScienceError(
      "Research Object exceeds its bounds or is invalid",
      "RESEARCH_OBJECT_INVALID",
      {
        cause: error,
      },
    );
  }
}
