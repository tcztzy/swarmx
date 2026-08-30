import type { Agent } from "@deepseek-ai/dsh-agent";
import type { JsonSchemaNode, ToolDefinition, ToolRunContext } from "@deepseek-ai/dsh-tools";
import { z } from "zod";
import type {
  AddSwarmMemberRequest,
  AdmitKnowledgeRequest,
  CreateSwarmRequest,
  CreateSwarmTaskRequest,
  EscalateSwarmTaskRequest,
  InterruptSwarmMemberRequest,
  ReassignSwarmTaskRequest,
  RecordSemanticFindingRequest,
  RecordSwarmVerdictRequest,
  ResolveSwarmEffectRequest,
  SendSwarmMessageRequest,
  StartSwarmVerificationRequest,
  SubmitSwarmTaskRequest,
  SwarmSnapshot,
  UpdateSwarmTaskRequest,
  WaitForSwarmChangeRequest,
} from "./contracts.js";
import type { SwarmActor } from "./coordinator.js";
import { SwarmError } from "./errors.js";

export const SWARM_ACTIONS = [
  "create",
  "status",
  "add_member",
  "send_message",
  "create_task",
  "update_task",
  "submit_task",
  "start_verification",
  "record_verdict",
  "record_monitor_finding",
  "escalate_task",
  "reassign_task",
  "interrupt_member",
  "admit_knowledge",
  "resolve_effect",
  "wait",
  "archive",
] as const;

type SwarmAction = (typeof SWARM_ACTIONS)[number];

const ACTION_TITLES: Record<SwarmAction, string> = {
  create: "Create Team",
  status: "Read Team status",
  add_member: "Add Team member",
  send_message: "Send Team message",
  create_task: "Create Team task",
  update_task: "Update Team task",
  submit_task: "Submit Team task",
  start_verification: "Start task verification",
  record_verdict: "Record verification verdict",
  record_monitor_finding: "Record semantic monitor finding",
  escalate_task: "Escalate Team task",
  reassign_task: "Reassign Team task",
  interrupt_member: "Interrupt Team member",
  admit_knowledge: "Admit verified knowledge",
  resolve_effect: "Resolve uncertain effect",
  wait: "Wait for Team change",
  archive: "Archive Team",
};

const inputSchema = z.strictObject({
  action: z.enum(SWARM_ACTIONS),
  request: z.unknown(),
});

const outputSchema: JsonSchemaNode = {
  type: "object",
  additionalProperties: false,
  properties: {
    action: { type: "string", enum: [...SWARM_ACTIONS] },
    data: {},
  },
  required: ["action", "data"],
};

export interface SwarmToolService<Actor extends SwarmActor = SwarmActor> {
  create(actor: Actor, request: CreateSwarmRequest, signal?: AbortSignal): Promise<SwarmSnapshot>;
  snapshot(actor: Actor): Promise<SwarmSnapshot>;
  addMember(
    actor: Actor,
    request: AddSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  sendMessage(
    actor: Actor,
    request: SendSwarmMessageRequest,
    signal?: AbortSignal,
  ): Promise<{ id: string; status: "delivered" | "queued" | "uncertain" }>;
  createTask(
    actor: Actor,
    request: CreateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  updateTask(
    actor: Actor,
    request: UpdateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  submitTask(
    actor: Actor,
    request: SubmitSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  startVerification(
    actor: Actor,
    request: StartSwarmVerificationRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  recordVerdict(
    actor: Actor,
    request: RecordSwarmVerdictRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  recordMonitorFinding(
    actor: Actor,
    request: RecordSemanticFindingRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  escalateTask(
    actor: Actor,
    request: EscalateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  reassignTask(
    actor: Actor,
    request: ReassignSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  interruptMember(
    actor: Actor,
    request: InterruptSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  admitKnowledge(
    actor: Actor,
    request: AdmitKnowledgeRequest,
    callId: string,
    signal?: AbortSignal,
  ): Promise<unknown>;
  resolveEffect(
    actor: Actor,
    request: ResolveSwarmEffectRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  waitForChange(
    actor: Actor,
    request: WaitForSwarmChangeRequest,
    signal: AbortSignal,
  ): Promise<SwarmSnapshot>;
  archive(actor: Actor, signal?: AbortSignal): Promise<SwarmSnapshot>;
}

export interface SwarmToolExecution<Actor extends SwarmActor = SwarmActor> {
  readonly actor: Actor;
  readonly callId: string;
  readonly signal: AbortSignal;
}

export interface SwarmToolDefinition<Actor extends SwarmActor = SwarmActor> extends ToolDefinition {
  invoke(args: unknown, execution: SwarmToolExecution<Actor>): Promise<unknown>;
}

function dshActor<Actor extends SwarmActor>(execution: ToolRunContext): Actor {
  if (!execution.agent) {
    throw new SwarmError("Swarm tools require an exact Agent carrier", "SWARM_UNAUTHORIZED");
  }
  return execution.agent as unknown as Actor;
}

async function dispatch<Actor extends SwarmActor>(
  service: SwarmToolService<Actor>,
  actor: Actor,
  action: SwarmAction,
  request: unknown,
  signal: AbortSignal,
  callId: string,
): Promise<unknown> {
  signal.throwIfAborted();
  switch (action) {
    case "create":
      return service.create(actor, request as CreateSwarmRequest, signal);
    case "status":
      return service.snapshot(actor);
    case "add_member":
      return service.addMember(actor, request as AddSwarmMemberRequest, signal);
    case "send_message":
      return service.sendMessage(actor, request as SendSwarmMessageRequest, signal);
    case "create_task":
      return service.createTask(actor, request as CreateSwarmTaskRequest, signal);
    case "update_task":
      return service.updateTask(actor, request as UpdateSwarmTaskRequest, signal);
    case "submit_task":
      return service.submitTask(actor, request as SubmitSwarmTaskRequest, signal);
    case "start_verification":
      return service.startVerification(actor, request as StartSwarmVerificationRequest, signal);
    case "record_verdict":
      return service.recordVerdict(actor, request as RecordSwarmVerdictRequest, signal);
    case "record_monitor_finding":
      return service.recordMonitorFinding(actor, request as RecordSemanticFindingRequest, signal);
    case "escalate_task":
      return service.escalateTask(actor, request as EscalateSwarmTaskRequest, signal);
    case "reassign_task":
      return service.reassignTask(actor, request as ReassignSwarmTaskRequest, signal);
    case "interrupt_member":
      return service.interruptMember(actor, request as InterruptSwarmMemberRequest, signal);
    case "admit_knowledge":
      return service.admitKnowledge(actor, request as AdmitKnowledgeRequest, callId, signal);
    case "resolve_effect":
      return service.resolveEffect(actor, request as ResolveSwarmEffectRequest, signal);
    case "wait":
      return service.waitForChange(actor, request as WaitForSwarmChangeRequest, signal);
    case "archive":
      return service.archive(actor, signal);
  }
}

export function createSwarmToolDefinition<Actor extends SwarmActor>(
  service: SwarmToolService<Actor>,
): SwarmToolDefinition<Actor> {
  const invoke = async (args: unknown, execution: SwarmToolExecution<Actor>): Promise<unknown> => {
    let input: z.infer<typeof inputSchema>;
    try {
      input = inputSchema.parse(args);
    } catch (cause) {
      throw new SwarmError("Invalid aggregate swarm tool request", "SWARM_INVALID_REQUEST", {
        cause,
      });
    }
    const data = await dispatch(
      service,
      execution.actor,
      input.action,
      input.request,
      execution.signal,
      execution.callId,
    );
    return { action: input.action, data };
  };
  return {
    name: "swarm",
    description:
      "Create and coordinate one durable Team. Implementers submit bounded evidence; only an exact authorized verifier or lead records acceptance. Task actions require exact revisions, attempts, and submissions.",
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        action: { type: "string", enum: [...SWARM_ACTIONS] },
        request: { description: "Strict request object for the selected swarm action." },
      },
      required: ["action", "request"],
    },
    output: {
      schema: outputSchema,
      render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
    },
    presentCall: (args) => {
      const input = inputSchema.safeParse(args);
      return {
        card: "generic",
        kind: input.success && input.data.action === "status" ? "read" : "other",
        title: input.success ? ACTION_TITLES[input.data.action] : "Team action",
      };
    },
    presentResult: (args, result) => {
      const input = inputSchema.safeParse(args);
      return {
        card: "generic",
        title: input.success ? ACTION_TITLES[input.data.action] : "Team action",
        content: [
          {
            type: "text",
            text: result.isError ? "Team action failed." : "Team activity updated.",
          },
        ],
      };
    },
    invoke,
    async execute(args, execution) {
      return invoke(args, {
        actor: dshActor<Actor>(execution),
        callId: String(execution.callId),
        signal: execution.signal,
      });
    },
  };
}

interface SwarmToolContext {
  readonly swarm: SwarmToolService<Agent>;
  readonly systemPrompt: {
    section(input: { readonly name: string; readonly order: number; readonly text: string }): void;
  };
  readonly tools: { register(definition: ToolDefinition): () => void };
}

export function registerSwarmTool(ctx: SwarmToolContext): () => void {
  return ctx.tools.register(createSwarmToolDefinition(ctx.swarm));
}

export const name = "swarmx-swarm-tools";
export const inject = ["swarm", "systemPrompt", "tools"];

export function apply(ctx: SwarmToolContext): void {
  const effectContext = ctx as SwarmToolContext & {
    effect(effect: () => () => void, label: string): void;
  };
  ctx.systemPrompt.section({
    name: "swarmx:team-mode",
    order: 194,
    text: "Use the swarm tool as the only Team coordination surface. Classify tasks as read (R), write (W), or knowledge (K). The lead creates and administers the Team. Members act only within their durable role and current task attempt; they must not delegate or access PKB. A write task grants one effect-fenced mutation lane but does not replace ordinary approvals or sandbox policy. Implementers submit bounded artifacts and evidence; submission revokes mutation authority and is not completion. Only the exact assigned verifier or lead may start verification and record a verdict, with self-verification explicitly degraded. A semantic monitor is optional and event-triggered; it may only record a strict read-only monitor finding and never a command or task verdict. Tool timeout/error may mean an uncertain effect and must be resolved before retry. Task completion and messages are candidates, not knowledge; only lead-only admit_knowledge with verified sources can commit a K task through Science Journal or approved PKB ownership.",
  });
  effectContext.effect(() => registerSwarmTool(ctx), "dsh-swarm: register aggregate tool");
}
