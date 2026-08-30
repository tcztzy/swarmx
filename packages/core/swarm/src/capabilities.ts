import type { SwarmRole } from "./contracts.js";
import type { SwarmActor, SwarmCoordinator } from "./coordinator.js";

const FORBIDDEN_MEMBER_TOOLS = new Set([
  "interrupt_agent",
  "list_agents",
  "pkb",
  "ralph",
  "report",
  "send_message",
  "subagent",
  "subagent_claude_code",
  "subagent_codex",
  "subagent_fork",
  "workflow",
]);

const MUTATING_MEMBER_TOOLS = new Set([
  "bash",
  "edit",
  "pwsh",
  "run_code",
  "science_experiment",
  "science_export",
  "science_figure",
  "science_notebook",
  "science_record",
  "science_write",
  "write",
]);

const PKB_READ_ONLY_ACTIONS = new Set([
  "read_conversation",
  "read_knowledge",
  "search_conversations",
  "search_knowledge",
]);

export function isMutatingMemberTool(name: string, arguments_?: unknown): boolean {
  if (MUTATING_MEMBER_TOOLS.has(name)) return true;
  if (name !== "pkb") return false;
  if (typeof arguments_ !== "object" || arguments_ === null) return true;
  const action = (arguments_ as { readonly action?: unknown }).action;
  return typeof action !== "string" || !PKB_READ_ONLY_ACTIONS.has(action);
}

export interface MemberToolExecution {
  readonly agent?: SwarmActor;
  readonly arguments?: unknown;
  readonly mutating?: boolean;
  readonly name: string;
}

export function leadToolGuard(
  lead: SwarmActor,
  coordinator: Pick<SwarmCoordinator, "hasActiveWriteAttempt">,
  execution: MemberToolExecution,
): string | undefined {
  if (execution.agent !== lead) return "Swarm lead authority is not exact.";
  const mutating = execution.mutating ?? isMutatingMemberTool(execution.name, execution.arguments);
  if (mutating && !coordinator.hasActiveWriteAttempt(lead)) {
    return "Team workspace mutation requires this exact lead to own an active write task attempt.";
  }
  return undefined;
}

export function memberToolGuard(
  member: SwarmActor,
  coordinator: Pick<SwarmCoordinator, "hasActiveWriteAttempt">,
  execution: MemberToolExecution,
  role: Exclude<SwarmRole, "lead"> = "legacy",
): string | undefined {
  if (execution.agent !== member) return "Swarm member authority is not exact.";
  if (FORBIDDEN_MEMBER_TOOLS.has(execution.name)) {
    return "Swarm members must use the swarm tool for coordination and cannot delegate or access PKB.";
  }
  const mutating = execution.mutating ?? isMutatingMemberTool(execution.name, execution.arguments);
  if ((role === "monitor" || role === "verifier") && mutating) {
    return `Swarm ${role} role is read-only and cannot mutate the workspace.`;
  }
  if (mutating && !coordinator.hasActiveWriteAttempt(member)) {
    return "Workspace mutation requires this exact Swarm member to own an active write task attempt.";
  }
  return undefined;
}
