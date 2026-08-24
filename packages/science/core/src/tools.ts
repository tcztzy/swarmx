import type { Context } from "@deepseek-ai/cordis";
import type { AttachmentStore, ImageAttachmentRef } from "@deepseek-ai/dsh-attachment";
import type { SessionId } from "@deepseek-ai/dsh-session";
import type {
  JsonSchemaNode,
  JsonValue,
  ToolDefinition,
  ToolRunContext,
} from "@deepseek-ai/dsh-tools";
import { commentAnnotationSchema } from "@swarmx/annotation";
import { z } from "zod";
import { literatureSearchRequestSchema } from "./contracts.js";
import { ScienceError } from "./errors.js";
import type { ScienceService } from "./index.js";

export const SCIENCE_TOOL_NAMES = [
  "science_notebook",
  "science_write",
  "science_figure",
  "science_experiment",
  "science_record",
  "science_query",
  "literature_search",
  "science_export",
] as const;

const aggregateInputSchema = z.strictObject({
  action: z.string().min(1).max(80),
  request: z.unknown(),
});

const outputSchema: JsonSchemaNode = {
  type: "object",
  additionalProperties: false,
  properties: {
    classification: { type: "string", enum: ["fact", "inference", "proposal"] },
    summary: { type: "string" },
    locator: {
      oneOf: [
        { type: "null" },
        {
          type: "object",
          additionalProperties: false,
          properties: {
            sessionId: { type: "string" },
            toolCallId: { type: "string" },
            entityKind: { type: "string" },
            entityId: { type: "string" },
            journalSeq: { type: "integer" },
          },
          required: ["sessionId", "toolCallId", "entityKind", "entityId", "journalSeq"],
        },
      ],
    },
    data: {},
  },
  required: ["classification", "summary", "locator", "data"],
};

const scienceQueryParameters: ToolDefinition["parameters"] = {
  type: "object",
  additionalProperties: false,
  properties: {
    action: { type: "string", enum: ["research_object", "inspect_annotation"] },
    request: {
      description:
        "For inspect_annotation, copy the complete comment annotation from the annotation reference. Never reduce it to artifact_id.",
      oneOf: [
        {
          type: "object",
          additionalProperties: false,
          properties: { projectId: { type: "string" } },
          required: ["projectId"],
          title: "RO-Crate Research Object request",
        },
        {
          type: "object",
          additionalProperties: false,
          properties: {
            type: { type: "string", const: "comment" },
            id: { type: "string" },
            comment: { type: "string" },
            created_at: { type: "integer" },
            target: {
              type: "object",
              additionalProperties: false,
              properties: {
                type: { type: "string", const: "image_point" },
                artifact_id: { type: "string" },
                project_id: { type: "string" },
                title: { type: "string" },
                digest: { type: "string" },
                mime: {
                  type: "string",
                  enum: ["image/png", "image/jpeg", "image/gif", "image/webp"],
                },
                point: {
                  type: "object",
                  additionalProperties: false,
                  properties: { x: { type: "number" }, y: { type: "number" } },
                  required: ["x", "y"],
                },
              },
              required: ["type", "artifact_id", "project_id", "title", "digest", "mime", "point"],
            },
          },
          required: ["type", "id", "comment", "created_at", "target"],
          title: "Image annotation request",
        },
      ],
    },
  },
  required: ["action", "request"],
};

const literatureSearchParameters: ToolDefinition["parameters"] = {
  type: "object",
  additionalProperties: false,
  properties: {
    query: {
      type: "string",
      description:
        "Concise title, author, DOI, or topic terms to search in the running local Zotero library.",
    },
    limit: { type: "integer", minimum: 1, maximum: 20 },
    filters: {
      type: "object",
      additionalProperties: false,
      properties: {
        years: {
          type: "object",
          additionalProperties: false,
          properties: {
            from: { type: "integer", minimum: 1000, maximum: 3000 },
            to: { type: "integer", minimum: 1000, maximum: 3000 },
          },
        },
        entryTypes: {
          type: "array",
          items: {
            type: "string",
            enum: [
              "article",
              "book",
              "incollection",
              "inproceedings",
              "techreport",
              "phdthesis",
              "unpublished",
              "misc",
            ],
          },
          minItems: 1,
          maxItems: 16,
        },
      },
    },
  },
  required: ["query"],
};

interface ScienceLocator {
  readonly sessionId: string;
  readonly toolCallId: string;
  readonly entityKind: string;
  readonly entityId: string;
  readonly journalSeq: number;
}

interface ScienceToolResult {
  readonly classification: "fact" | "inference" | "proposal";
  readonly summary: string;
  readonly locator: ScienceLocator | null;
  readonly data: JsonValue;
}

interface ScienceToolContext {
  readonly attachments: Pick<AttachmentStore, "saveImage">;
  readonly science: ScienceService;
  readonly tools: { register(definition: ToolDefinition): () => void };
}

function parseInput(args: unknown, actions: readonly string[]) {
  try {
    const input = aggregateInputSchema.parse(args);
    if (!actions.includes(input.action)) {
      throw new Error(`Unsupported action '${input.action}'`);
    }
    return input;
  } catch (error) {
    throw new ScienceError("Invalid aggregate science tool request", "INVALID_REQUEST", {
      cause: error,
    });
  }
}

function session(exec: ToolRunContext): SessionId {
  if (!exec.agent) {
    throw new ScienceError("Science tools require an owning agent session", "SESSION_NOT_FOUND");
  }
  return exec.agent.id;
}

function result(
  exec: ToolRunContext,
  classification: ScienceToolResult["classification"],
  summary: string,
  entity: {
    readonly id: string;
    readonly kind: string;
    readonly provenance: { readonly journalSeq: number };
  } | null,
  data: unknown,
): ScienceToolResult {
  return {
    classification,
    summary,
    locator: entity
      ? {
          sessionId: session(exec),
          toolCallId: String(exec.callId),
          entityKind: entity.kind,
          entityId: entity.id,
          journalSeq: entity.provenance.journalSeq,
        }
      : null,
    data: data as JsonValue,
  };
}

function definition(
  name: (typeof SCIENCE_TOOL_NAMES)[number],
  description: string,
  actions: readonly string[],
  execute: (
    science: ScienceService,
    input: { readonly action: string; readonly request: unknown },
    exec: ToolRunContext,
  ) => Promise<ScienceToolResult>,
  science: ScienceService,
  render: ToolDefinition["output"]["render"] = (_args, value) => [
    { type: "text", text: JSON.stringify(value) },
  ],
  parameters?: ToolDefinition["parameters"],
): ToolDefinition {
  return {
    name,
    description,
    parameters: parameters ?? {
      type: "object",
      additionalProperties: false,
      properties: {
        action: { type: "string", enum: [...actions] },
        request: { description: "Strict request object for the selected action." },
      },
      required: ["action", "request"],
    },
    output: {
      schema: outputSchema,
      render,
      presentationMeta: (_args, value) => {
        const candidate = value as unknown as ScienceToolResult;
        return {
          classification: candidate.classification,
          locator: candidate.locator,
        } as unknown as JsonValue;
      },
    },
    async execute(args, exec) {
      const input = parseInput(args, actions);
      exec.signal.throwIfAborted();
      return execute(science, input, exec);
    },
  };
}

function literatureDefinition(science: ScienceService): ToolDefinition {
  return {
    name: "literature_search",
    description:
      "Search the running local Zotero library for scientific literature. Results are read-only, ranked locally, normalized through a sanitized BibTeX file, and include citation-ready BibTeX. This never searches the web or reads attachment paths.",
    parameters: literatureSearchParameters,
    output: {
      schema: outputSchema,
      render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
      presentationMeta: (_args, value) => {
        const candidate = value as unknown as ScienceToolResult;
        return {
          classification: candidate.classification,
          locator: candidate.locator,
        } as unknown as JsonValue;
      },
    },
    async execute(args, exec) {
      let request: z.infer<typeof literatureSearchRequestSchema>;
      try {
        request = literatureSearchRequestSchema.parse(args);
      } catch (error) {
        throw new ScienceError("Invalid literature search request", "INVALID_REQUEST", {
          cause: error,
        });
      }
      exec.signal.throwIfAborted();
      const data = await science.searchLiterature(session(exec), request, exec.signal);
      return result(exec, "inference", "Searched local Zotero literature", null, data);
    },
  };
}

function annotationImage(value: JsonValue): ImageAttachmentRef | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const data = value.data;
  if (typeof data !== "object" || data === null || Array.isArray(data)) return null;
  const attachment = data.attachment;
  if (typeof attachment !== "object" || attachment === null || Array.isArray(attachment)) {
    return null;
  }
  return attachment as unknown as ImageAttachmentRef;
}

function scienceQueryRender(args: unknown, value: JsonValue) {
  const text = { type: "text" as const, text: JSON.stringify(value) };
  if (
    typeof args !== "object" ||
    args === null ||
    Array.isArray(args) ||
    !("action" in args) ||
    args.action !== "inspect_annotation"
  ) {
    return [text];
  }
  const attachment = annotationImage(value);
  return attachment === null ? [text] : [text, { type: "image" as const, attachment }];
}

function imageBytes(dataUrl: string, mime: string): Uint8Array {
  const prefix = `data:${mime};base64,`;
  if (!dataUrl.startsWith(prefix)) {
    throw new ScienceError("Artifact preview image encoding is invalid", "ARTIFACT_IO_FAILED");
  }
  const encoded = dataUrl.slice(prefix.length);
  const bytes = Buffer.from(encoded, "base64");
  if (bytes.toString("base64") !== encoded) {
    throw new ScienceError("Artifact preview image encoding is invalid", "ARTIFACT_IO_FAILED");
  }
  return bytes;
}

function parseImageAnnotation(request: unknown) {
  try {
    const annotation = commentAnnotationSchema.parse(request);
    const target = annotation.target;
    if (target.type !== "image_point") {
      throw new Error("annotation target must be image_point");
    }
    return { ...annotation, target };
  } catch (error) {
    throw new ScienceError("Invalid science image annotation", "INVALID_REQUEST", {
      cause: error,
    });
  }
}

export function createScienceToolDefinitions(
  science: ScienceService,
  attachments: Pick<AttachmentStore, "saveImage">,
): readonly ToolDefinition[] {
  return [
    definition(
      "science_notebook",
      "Create a Science Notebook or execute one Python cell with provenance and bounded output. Figure PNG/SVG/PDF output embeds exact code and input identities by default; outputArtifact.reproducibilityMetadata=false opts out.",
      ["create", "execute"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        if (input.action === "create") {
          const notebook = service.createNotebook(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Created science notebook", notebook, notebook);
        }
        const execution = await service.executeNotebookCell(
          sessionId,
          input.request as never,
          exec.signal,
        );
        return result(
          exec,
          "fact",
          "Executed science notebook cell",
          execution.notebook,
          execution,
        );
      },
      science,
    ),
    definition(
      "science_write",
      "Create scientific source or propose, accept, or reject a source-linked revision.",
      ["create", "modify"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        const document =
          input.action === "create"
            ? service.createDocument(sessionId, input.request as never, exec.signal)
            : service.modifyDocument(sessionId, input.request as never, exec.signal);
        const proposal =
          input.action === "modify" &&
          typeof input.request === "object" &&
          input.request !== null &&
          "action" in input.request &&
          input.request.action === "propose";
        return result(
          exec,
          proposal ? "proposal" : "fact",
          proposal ? "Proposed scientific writing patch" : "Updated scientific writing",
          document,
          document,
        );
      },
      science,
    ),
    definition(
      "science_figure",
      "Create a semantic scientific figure or propose, accept, or reject linked plotting code.",
      ["create", "modify"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        const figure =
          input.action === "create"
            ? service.createFigure(sessionId, input.request as never, exec.signal)
            : service.modifyFigureCode(sessionId, input.request as never, exec.signal);
        const proposal =
          input.action === "modify" &&
          typeof input.request === "object" &&
          input.request !== null &&
          "action" in input.request &&
          input.request.action === "propose";
        return result(
          exec,
          proposal ? "proposal" : "fact",
          proposal ? "Proposed scientific figure patch" : "Updated scientific figure",
          figure,
          figure,
        );
      },
      science,
    ),
    definition(
      "science_experiment",
      "Define an experiment, start or finish a Run, or compare completed Runs.",
      ["define", "start", "finish", "compare"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        if (input.action === "define") {
          const experiment = service.defineExperiment(
            sessionId,
            input.request as never,
            exec.signal,
          );
          return result(exec, "fact", "Defined science experiment", experiment, experiment);
        }
        if (input.action === "start") {
          const mutation = service.startRun(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Started experiment run", mutation.run, mutation);
        }
        if (input.action === "finish") {
          const run = service.finishRun(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Finished experiment run", run, run);
        }
        const comparison = service.compareRuns(sessionId, input.request as never, exec.signal);
        const workspace = service.getWorkspace(sessionId, exec.signal);
        const baseline = workspace.runs.find((run) => run.id === comparison.baselineRunId);
        if (!baseline) throw new ScienceError("Comparison baseline disappeared", "RUN_NOT_FOUND");
        return result(
          exec,
          "inference",
          "Compared completed experiment runs",
          baseline,
          comparison,
        );
      },
      science,
    ),
    definition(
      "science_record",
      "Record a research question, hypothesis, claim, evidence link, or artifact fact. Figure PNG/SVG/PDF registration may pass reproducibilityMetadata with library, exact code, and workspace/artifact/S3 sources; false opts out.",
      [
        "create_question",
        "create_hypothesis",
        "record_claim",
        "link_evidence",
        "register_artifact",
      ],
      async (service, input, exec) => {
        const sessionId = session(exec);
        if (input.action === "create_question") {
          const record = service.createQuestion(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Created research question", record, record);
        }
        if (input.action === "create_hypothesis") {
          const record = service.createHypothesis(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Created research hypothesis", record, record);
        }
        if (input.action === "record_claim") {
          const record = service.recordClaim(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Recorded scientific claim", record, record);
        }
        if (input.action === "link_evidence") {
          const linked = service.linkEvidence(sessionId, input.request as never, exec.signal);
          return result(exec, "fact", "Linked scientific evidence", linked.evidence, linked);
        }
        const artifact = await service.registerArtifact(
          sessionId,
          input.request as never,
          exec.signal,
        );
        return result(exec, "fact", "Registered science artifact", artifact, artifact);
      },
      science,
    ),
    definition(
      "science_query",
      "Read one project as an RO-Crate 1.3 Research Object or inspect one image annotation. This differs from file search: it returns structured research entities and provenance Actions. For inspect_annotation, copy the complete comment object from the annotation reference; never submit artifact_id alone.",
      ["research_object", "inspect_annotation"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        if (input.action === "research_object") {
          const request = input.request as { readonly projectId: string };
          const project = service
            .getWorkspace(sessionId, exec.signal)
            .projects.find((candidate) => candidate.id === request.projectId);
          if (!project) {
            throw new ScienceError("Project not found in this workspace", "PROJECT_NOT_FOUND");
          }
          const researchObject = service.getResearchObject(sessionId, request, exec.signal);
          return result(exec, "fact", "Read project Research Object", project, researchObject);
        }
        if (input.action === "inspect_annotation") {
          const annotation = parseImageAnnotation(input.request);
          const target = annotation.target;
          const artifact = service
            .getWorkspace(sessionId, exec.signal)
            .artifacts.find((candidate) => candidate.id === target.artifact_id);
          if (!artifact) {
            throw new ScienceError("Artifact not found in this workspace", "ARTIFACT_NOT_FOUND");
          }
          if (
            artifact.projectId !== target.project_id ||
            artifact.title !== target.title ||
            artifact.digest !== target.digest ||
            artifact.mime !== target.mime
          ) {
            throw new ScienceError(
              "Image annotation no longer matches the registered artifact",
              "ARTIFACT_SOURCE_CHANGED",
            );
          }
          const preview = service.previewArtifact(
            sessionId,
            { artifactId: target.artifact_id },
            exec.signal,
          );
          if (preview.kind !== "image") {
            throw new ScienceError(
              preview.kind === "unavailable" && preview.reason === "too-large"
                ? "Annotated image is too large to inspect"
                : "Annotated artifact is not a supported image",
              preview.kind === "unavailable" && preview.reason === "too-large"
                ? "ARTIFACT_TOO_LARGE"
                : "INVALID_REQUEST",
            );
          }
          const attachment = await attachments.saveImage({
            data: imageBytes(preview.dataUrl, target.mime),
            mediaType: target.mime,
            name: target.title,
          });
          exec.signal.throwIfAborted();
          return result(exec, "fact", "Inspected science image annotation", artifact, {
            annotation,
            attachment,
          });
        }
        throw new ScienceError("Unsupported science query action", "INVALID_REQUEST");
      },
      science,
      scienceQueryRender,
      scienceQueryParameters,
    ),
    literatureDefinition(science),
    definition(
      "science_export",
      "Export one local project as a deterministic RO-Crate 1.3 Metadata Document named ro-crate-metadata.json; oversized text uses the profile spill policy.",
      ["project"],
      async (service, input, exec) => {
        const exported = service.exportProject(session(exec), input.request as never, exec.signal);
        return result(exec, "fact", "Exported science project", exported, exported);
      },
      science,
    ),
  ];
}

export function registerScienceTools(ctx: ScienceToolContext): () => void {
  const disposers = createScienceToolDefinitions(ctx.science, ctx.attachments).map((tool) =>
    ctx.tools.register(tool),
  );
  return () => {
    for (const dispose of disposers.reverse()) dispose();
  };
}

export const name = "swarmx-science-tools";
export const inject = ["attachments", "science", "tools"];

export function apply(ctx: Context): void {
  ctx.effect(() => registerScienceTools(ctx), "dsh-science: register aggregate tools");
}
