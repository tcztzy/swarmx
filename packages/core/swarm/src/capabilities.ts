import type { Agent } from "@deepseek-ai/dsh-agent";
import type { SwarmRole } from "./contracts.js";
import type { SwarmCoordinator } from "./coordinator.js";

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

export function isMutatingMemberTool(name: string): boolean {
  return MUTATING_MEMBER_TOOLS.has(name);
}

export interface MemberToolExecution {
  readonly agent?: Agent;
  readonly name: string;
}

export function leadToolGuard(
  lead: Agent,
  coordinator: Pick<SwarmCoordinator, "hasActiveWriteAttempt">,
  execution: MemberToolExecution,
): string | undefined {
  if (execution.agent !== lead) return "Swarm lead authority is not exact.";
  if (isMutatingMemberTool(execution.name) && !coordinator.hasActiveWriteAttempt(lead)) {
    return "Team workspace mutation requires this exact lead to own an active write task attempt.";
  }
  return undefined;
}

export function memberToolGuard(
  member: Agent,
  coordinator: Pick<SwarmCoordinator, "hasActiveWriteAttempt">,
  execution: MemberToolExecution,
  role: Exclude<SwarmRole, "lead"> = "legacy",
): string | undefined {
  if (execution.agent !== member) return "Swarm member authority is not exact.";
  if (FORBIDDEN_MEMBER_TOOLS.has(execution.name)) {
    return "Swarm members must use the swarm tool for coordination and cannot delegate or access PKB.";
  }
  if ((role === "monitor" || role === "verifier") && isMutatingMemberTool(execution.name)) {
    return `Swarm ${role} role is read-only and cannot mutate the workspace.`;
  }
  if (isMutatingMemberTool(execution.name) && !coordinator.hasActiveWriteAttempt(member)) {
    return "Workspace mutation requires this exact Swarm member to own an active write task attempt.";
  }
  return undefined;
}
