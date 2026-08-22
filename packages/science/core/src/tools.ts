import type { Context } from "@deepseek-ai/cordis";
import type { SessionId } from "@deepseek-ai/dsh-session";
import type {
  JsonSchemaNode,
  JsonValue,
  ToolDefinition,
  ToolRunContext,
} from "@deepseek-ai/dsh-tools";
import { z } from "zod";
import { ScienceError } from "./errors.js";
import type { ScienceService } from "./index.js";

export const SCIENCE_TOOL_NAMES = [
  "science_notebook",
  "science_write",
  "science_figure",
  "science_experiment",
  "science_record",
  "science_query",
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
): ToolDefinition {
  return {
    name,
    description,
    parameters: {
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
      const input = parseInput(args, actions);
      exec.signal.throwIfAborted();
      return execute(science, input, exec);
    },
  };
}

export function createScienceToolDefinitions(science: ScienceService): readonly ToolDefinition[] {
  return [
    definition(
      "science_notebook",
      "Create a Science Notebook or execute one Python cell with provenance and bounded output.",
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
      "Record a research question, hypothesis, claim, evidence link, or artifact fact.",
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
        const artifact = service.registerArtifact(sessionId, input.request as never, exec.signal);
        return result(exec, "fact", "Registered science artifact", artifact, artifact);
      },
      science,
    ),
    definition(
      "science_query",
      "Read a bounded Science Workspace snapshot or trace one entity's provenance.",
      ["workspace", "trace"],
      async (service, input, exec) => {
        const sessionId = session(exec);
        if (input.action === "workspace") {
          const workspace = service.getWorkspace(sessionId, exec.signal);
          return result(exec, "fact", "Read science workspace", null, workspace);
        }
        const trace = service.traceProvenance(sessionId, input.request as never, exec.signal);
        const root = trace.entities.find((entity) => entity.id === trace.rootId);
        const event = [...trace.events]
          .reverse()
          .find((candidate) => candidate.entityId === trace.rootId);
        const locatorEntity =
          root && event
            ? {
                id: root.id,
                kind: root.kind,
                provenance: { journalSeq: event.journalSeq },
              }
            : null;
        return result(exec, "fact", "Traced science provenance", locatorEntity, trace);
      },
      science,
    ),
    definition(
      "science_export",
      "Export one local project as deterministic JSON; oversized text uses the profile spill policy.",
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
  const disposers = createScienceToolDefinitions(ctx.science).map((tool) =>
    ctx.tools.register(tool),
  );
  return () => {
    for (const dispose of disposers.reverse()) dispose();
  };
}

export const name = "swarmx-science-tools";
export const inject = ["science", "tools"];

export function apply(ctx: Context): void {
  ctx.effect(() => registerScienceTools(ctx), "dsh-science: register aggregate tools");
}
