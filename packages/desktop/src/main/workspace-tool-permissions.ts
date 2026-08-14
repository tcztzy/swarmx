import {
  type HarnessPermissionPolicy,
  HarnessPermissionPolicySchema,
  type HarnessToolAccess,
  type ResolvedHarnessPermissionPolicy,
  ResolvedHarnessPermissionPolicySchema,
  resolveHarnessToolPermission,
} from "@swarmx/core";
import type { LocalTool } from "@swarmx/core/local-tool-contracts";
import type {
  ClaudeInteractionRequest,
  ClaudeInteractionResponse,
  ToolApprovalOption,
} from "./agent-interactions.js";

export interface WorkspacePermissionReviewRequest {
  source: "direct";
  toolName: string;
  toolKind: HarnessToolAccess;
  summary: string;
  toolInput: Record<string, unknown> | string;
  options: ToolApprovalOption[];
  policySourceIds: string[];
}

export interface WorkspaceToolPermissionOptions {
  interact?: (request: ClaudeInteractionRequest) => Promise<ClaudeInteractionResponse>;
  reviewPermission?: (request: WorkspacePermissionReviewRequest) => Promise<boolean>;
  permissionPolicy?: HarnessPermissionPolicy | ResolvedHarnessPermissionPolicy;
}

const READ_ONLY_PERMISSION_TOOLS = new Set([
  "AskUserQuestion",
  "CronList",
  "EnterPlanMode",
  "ExitPlanMode",
  "Glob",
  "Grep",
  "LSP",
  "Read",
  "ReportFindings",
  "TaskCreate",
  "TaskGet",
  "TaskList",
  "TaskOutput",
  "TaskUpdate",
  "TodoList",
  "TodoWrite",
]);

const WRITE_PERMISSION_TOOLS = new Set(["Edit", "NotebookEdit", "Write", "apply_patch"]);

export function applyWorkspaceToolPolicy(
  tools: LocalTool[],
  options: WorkspaceToolPermissionOptions,
) {
  if (!options.permissionPolicy) return tools;
  const layered = ResolvedHarnessPermissionPolicySchema.safeParse(options.permissionPolicy);
  const policy = layered.success
    ? layered.data
    : HarnessPermissionPolicySchema.parse(options.permissionPolicy);
  return tools.map((tool) =>
    permissionGuardedTool(tool, policy, options.reviewPermission, options.interact),
  );
}

export function workspaceToolAccess(toolName: string): HarnessToolAccess {
  if (READ_ONLY_PERMISSION_TOOLS.has(toolName)) return "read";
  if (WRITE_PERMISSION_TOOLS.has(toolName)) return "write";
  return "execute";
}

function permissionGuardedTool(
  tool: LocalTool,
  policy: HarnessPermissionPolicy | ResolvedHarnessPermissionPolicy,
  reviewPermission: WorkspaceToolPermissionOptions["reviewPermission"],
  interact: WorkspaceToolPermissionOptions["interact"],
): LocalTool {
  const authorize = (input: Record<string, unknown> | string) =>
    authorizeWorkspaceTool(
      tool.name,
      workspaceToolAccess(tool.name),
      input,
      policy,
      reviewPermission,
      interact,
    );
  if (tool.kind === "text") {
    return {
      ...tool,
      call: async (input: string, context) => {
        await authorize(input);
        return tool.call(input, context);
      },
    };
  }
  return {
    ...tool,
    call: async (input: Record<string, unknown>, context) => {
      await authorize(input);
      return tool.call(input, context);
    },
  };
}

async function authorizeWorkspaceTool(
  toolName: string,
  access: HarnessToolAccess,
  input: Record<string, unknown> | string,
  policy: HarnessPermissionPolicy | ResolvedHarnessPermissionPolicy,
  reviewPermission: WorkspaceToolPermissionOptions["reviewPermission"],
  interact: WorkspaceToolPermissionOptions["interact"],
): Promise<void> {
  const resolved = resolveHarnessToolPermission(policy, { toolName, access });
  if (resolved.decision === "allow") return;
  if (resolved.decision === "deny") {
    throw new Error(
      `Tool "${toolName}" is denied by Harness permission policy (${resolved.reason}).`,
    );
  }
  const approvalOptions: ToolApprovalOption[] = [
    { optionId: "reject_once", name: "Reject", kind: "reject_once" },
    { optionId: "allow_once", name: "Allow once", kind: "allow_once" },
  ];
  const summary = workspaceToolApprovalSummary(toolName, input);
  if (resolved.reason === "auto" && reviewPermission) {
    try {
      if (
        await reviewPermission({
          source: "direct",
          toolName,
          toolKind: access,
          summary,
          toolInput: input,
          options: approvalOptions,
          policySourceIds: resolved.sourceIds,
        })
      ) {
        return;
      }
    } catch {
      // The human bridge remains the fail-closed fallback for reviewer errors.
    }
  }
  if (!interact) {
    throw new Error(
      `Tool "${toolName}" requires approval, but no interaction bridge is available.`,
    );
  }
  const response = await interact({
    kind: "tool_approval",
    title: `Allow ${toolName}?`,
    toolKind: access,
    source: "direct",
    policySourceIds: resolved.sourceIds,
    summary,
    options: approvalOptions,
  });
  if (response.kind !== "tool_approval" || response.optionId !== "allow_once") {
    throw new Error(`Tool "${toolName}" was rejected by the user.`);
  }
}

function workspaceToolApprovalSummary(toolName: string, input: Record<string, unknown> | string) {
  if (typeof input === "string") return `${toolName} requested a bounded Project patch.`;
  const safeFields = ["file_path", "path", "workdir", "name", "action", "description", "cron"];
  const details = safeFields.flatMap((field) => {
    const value = input[field];
    return typeof value === "string" && value.trim()
      ? [`${field}: ${boundedApprovalText(value)}`]
      : [];
  });
  if ("command" in input || "cmd" in input) {
    details.push("command: Project-sandboxed shell command");
  }
  return details.length > 0
    ? `${toolName}\n${details.join("\n")}`
    : `${toolName} requested a ${workspaceToolAccess(toolName)} operation in the active Project.`;
}

function boundedApprovalText(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length <= 240 ? compact : `${compact.slice(0, 239)}…`;
}
