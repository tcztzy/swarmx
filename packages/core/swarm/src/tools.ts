import type { Agent } from "@deepseek-ai/dsh-agent";
import type { JsonSchemaNode, ToolDefinition, ToolRunContext } from "@deepseek-ai/dsh-tools";
import { z } from "zod";
import type {
  AddSwarmMemberRequest,
  CreateSwarmRequest,
  CreateSwarmTaskRequest,
  InterruptSwarmMemberRequest,
  ReassignSwarmTaskRequest,
  SendSwarmMessageRequest,
  SwarmSnapshot,
  UpdateSwarmTaskRequest,
  WaitForSwarmChangeRequest,
} from "./contracts.js";
import { SwarmError } from "./errors.js";

export const SWARM_ACTIONS = [
  "create",
  "status",
  "add_member",
  "send_message",
  "create_task",
  "update_task",
  "reassign_task",
  "interrupt_member",
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
  reassign_task: "Reassign Team task",
  interrupt_member: "Interrupt Team member",
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

export interface SwarmToolService {
  create(agent: Agent, request: CreateSwarmRequest, signal?: AbortSignal): Promise<SwarmSnapshot>;
  snapshot(agent: Agent): Promise<SwarmSnapshot>;
  addMember(
    agent: Agent,
    request: AddSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  sendMessage(
    agent: Agent,
    request: SendSwarmMessageRequest,
    signal?: AbortSignal,
  ): Promise<{ id: string; status: "delivered" | "queued" }>;
  createTask(
    agent: Agent,
    request: CreateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  updateTask(
    agent: Agent,
    request: UpdateSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  reassignTask(
    agent: Agent,
    request: ReassignSwarmTaskRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  interruptMember(
    agent: Agent,
    request: InterruptSwarmMemberRequest,
    signal?: AbortSignal,
  ): Promise<SwarmSnapshot>;
  waitForChange(
    agent: Agent,
    request: WaitForSwarmChangeRequest,
    signal: AbortSignal,
  ): Promise<SwarmSnapshot>;
  archive(agent: Agent, signal?: AbortSignal): Promise<SwarmSnapshot>;
}

function requireAgent(execution: ToolRunContext): Agent {
  if (!execution.agent) {
    throw new SwarmError("Swarm tools require an exact Agent carrier", "SWARM_UNAUTHORIZED");
  }
  return execution.agent;
}

async function dispatch(
  service: SwarmToolService,
  agent: Agent,
  action: SwarmAction,
  request: unknown,
  signal: AbortSignal,
): Promise<unknown> {
  signal.throwIfAborted();
  switch (action) {
    case "create":
      return service.create(agent, request as CreateSwarmRequest, signal);
    case "status":
      return service.snapshot(agent);
    case "add_member":
      return service.addMember(agent, request as AddSwarmMemberRequest, signal);
    case "send_message":
      return service.sendMessage(agent, request as SendSwarmMessageRequest, signal);
    case "create_task":
      return service.createTask(agent, request as CreateSwarmTaskRequest, signal);
    case "update_task":
      return service.updateTask(agent, request as UpdateSwarmTaskRequest, signal);
    case "reassign_task":
      return service.reassignTask(agent, request as ReassignSwarmTaskRequest, signal);
    case "interrupt_member":
      return service.interruptMember(agent, request as InterruptSwarmMemberRequest, signal);
    case "wait":
      return service.waitForChange(agent, request as WaitForSwarmChangeRequest, signal);
    case "archive":
      return service.archive(agent, signal);
  }
}

export function createSwarmToolDefinition(service: SwarmToolService): ToolDefinition {
  return {
    name: "swarm",
    description:
      "Create and coordinate one durable DSH-native Team. Task updates require exact revisions and attempts; archive retires history without deleting it.",
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
    async execute(args, execution) {
      const agent = requireAgent(execution);
      let input: z.infer<typeof inputSchema>;
      try {
        input = inputSchema.parse(args);
      } catch (cause) {
        throw new SwarmError("Invalid aggregate swarm tool request", "SWARM_INVALID_REQUEST", {
          cause,
        });
      }
      const data = await dispatch(service, agent, input.action, input.request, execution.signal);
      return { action: input.action, data };
    },
  };
}

interface SwarmToolContext {
  readonly swarm: SwarmToolService;
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
    text: "Use the swarm tool as the only Team coordination surface. The lead creates and administers the Team. Members settle only their current task attempt and must not delegate or access PKB. A write task grants coordination authority for workspace mutation but does not replace ordinary tool approvals or sandbox policy.",
  });
  effectContext.effect(() => registerSwarmTool(ctx), "dsh-swarm: register aggregate tool");
}
