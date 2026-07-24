import type { ExtensionCapabilityInventory } from "../../shared/desktop-api.js";

type ExtensionBundle = ExtensionCapabilityInventory["bundles"][number];
type ExtensionAgent = ExtensionCapabilityInventory["agents"][number];
type ExtensionAgentPlan = NonNullable<ExtensionCapabilityInventory["agentPlans"]>[number];
type ExtensionAgentPlanRequirement = NonNullable<ExtensionAgentPlan["requirements"]>[number];
type ExtensionSkill = ExtensionCapabilityInventory["skills"][number];
type ExtensionUiContribution = ExtensionCapabilityInventory["uiContributions"][number];

export interface AgentCompositionPayload {
  id: string;
  agentProfileId?: string;
  harnessId?: string;
  modelId?: string;
  effort?: string;
  host?: "local" | "server";
}

export interface ExtensionComponentRow {
  id: string;
  kind: string;
  title: string;
  detail?: string;
  chips: string[];
}

export function nativeAgentHostLabel(
  host: "claude_code" | "codex" | "swarmx" | "custom" | undefined,
): string {
  if (host === "codex") return "Codex";
  if (host === "claude_code") return "Claude Code";
  return "Native Agent";
}

export function uniqueById<T extends { id: string }>(items: T[]): T[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

export function capabilityCount(
  bundle: ExtensionBundle,
  key: keyof NonNullable<ExtensionBundle["capabilities"]>,
): number {
  return bundle.capabilities?.[key]?.length ?? 0;
}

export function formatComponentCounts(counts: Record<string, number> | undefined): string[] {
  if (!counts) return [];
  return Object.entries(counts)
    .filter(([, count]) => Number.isFinite(count) && count > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, count]) => `${count} ${key}`);
}

export function formatSoftwareSummary(
  software: ExtensionCapabilityInventory["harnesses"][number]["software"],
): string {
  if (!software?.name) return "software";
  return software.version ? `${software.name}@${software.version}` : software.name;
}

export function modelApiLabel(api: string): string {
  switch (api) {
    case "openai_chat":
      return "OpenAI Chat API";
    case "openai_responses":
      return "OpenAI Responses API";
    case "anthropic":
      return "Anthropic API";
    case "ollama":
      return "Ollama API";
    default:
      return api;
  }
}

export function extensionComponentRows(
  inventory: ExtensionCapabilityInventory | undefined,
): ExtensionComponentRow[] {
  if (!inventory) return [];
  return [
    ...(inventory.commands ?? []).map((item) => ({
      id: item.id,
      kind: "command",
      title: item.name ?? item.id,
      detail: item.scope,
      chips: item.command ? [item.command.join(" ")] : [],
    })),
    ...(inventory.lspServers ?? []).map((item) => ({
      id: item.id,
      kind: "LSP",
      title: item.name ?? item.id,
      detail: item.scope,
      chips: [
        ...(item.languages ?? []),
        ...(item.languageIds ?? []),
        ...formatExtensionCommand(item.command, item.args),
      ],
    })),
    ...(inventory.hooks ?? []).map((item) => ({
      id: item.id,
      kind: "hook",
      title: item.name ?? item.id,
      detail: item.event,
      chips: [],
    })),
    ...(inventory.monitors ?? []).map((item) => ({
      id: item.id,
      kind: "monitor",
      title: item.name ?? item.id,
      detail: item.trigger,
      chips: item.schedule ? [item.schedule] : [],
    })),
    ...(inventory.outputStyles ?? []).map((item) => ({
      id: item.id,
      kind: "output style",
      title: item.name ?? item.id,
      detail: item.path,
      chips: [],
    })),
    ...(inventory.settings ?? []).map((item) => ({
      id: item.id,
      kind: "setting",
      title: item.name ?? item.id,
      detail: item.valueType,
      chips: item.required ? ["required"] : [],
    })),
    ...(inventory.assets ?? []).map((item) => ({
      id: item.id,
      kind: "asset",
      title: item.name ?? item.id,
      detail: item.kind,
      chips: [item.path, item.url].filter((value): value is string => Boolean(value)),
    })),
    ...(inventory.permissions ?? []).map((item) => ({
      id: item.id,
      kind: "permission",
      title: item.kind,
      detail: item.access,
      chips: [item.target, item.required ? "required" : undefined].filter(
        (value): value is string => Boolean(value),
      ),
    })),
    ...(inventory.authPolicies ?? []).map((item) => ({
      id: item.id,
      kind: "auth policy",
      title: item.kind ?? item.id,
      detail: item.required ? "required" : "optional",
      chips: item.secretRefs?.length ? [`${item.secretRefs.length} secret refs`] : [],
    })),
  ];
}

export function extensionUiContributionChips(contribution: ExtensionUiContribution): string[] {
  const chips: Array<string | undefined> = [
    contribution.sourcePluginId ? `via ${contribution.sourcePluginId}` : undefined,
    contribution.commandId ? `command ${contribution.commandId}` : undefined,
    contribution.assetRef ? `asset ${contribution.assetRef}` : undefined,
    contribution.target ? `target ${contribution.target}` : undefined,
    contribution.provenance,
  ];
  chips.push(...(contribution.settingIds ?? []).map((id) => `setting ${id}`));
  chips.push(...(contribution.permissionIds ?? []).map((id) => `permission ${id}`));
  chips.push(...(contribution.authPolicyIds ?? []).map((id) => `auth ${id}`));
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

export function extensionSkillChips(skill: ExtensionSkill): string[] {
  const chips: Array<string | undefined> = [
    skill.canonicalPath ? `canonical ${skill.canonicalPath}` : undefined,
    skill.governanceRef ? `governance ${skill.governanceRef}` : undefined,
    skill.readOnly ? "read-only" : undefined,
  ];
  chips.push(...(skill.requiresGateSkillIds ?? []).map((id) => `gate ${id}`));
  chips.push(
    ...(skill.hostExposures ?? []).map((exposure) =>
      [exposure.host, exposure.status ?? "plugin"].filter(Boolean).join(" "),
    ),
  );
  chips.push(
    ...(skill.hostExposures ?? []).flatMap((exposure) =>
      [
        exposure.manifestPath ? `manifest ${exposure.manifestPath}` : undefined,
        exposure.rulesPath ? `rules ${exposure.rulesPath}` : undefined,
        exposure.marketplaceSourceId ? `source ${exposure.marketplaceSourceId}` : undefined,
      ].filter((value): value is string => Boolean(value)),
    ),
  );
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

export function agentPlanTone(plan: ExtensionAgentPlan): string {
  if (plan.status === "ready") return "active";
  if (plan.status === "draft" || plan.status === "stale") return "neutral";
  return "danger";
}

export function planBlockedTitle(plan: ExtensionAgentPlan | undefined): string {
  if (!plan) return "Agent profile is not ready.";
  const blocked = blockedPlanRequirements(plan);
  return blocked.length > 0
    ? blocked.map((requirement) => requirement.message).join("; ")
    : plan.status;
}

export function agentPlanChips(
  plan: ExtensionAgentPlan | undefined,
  agent: ExtensionAgent,
): string[] {
  const chips: Array<string | undefined> = [];
  if (plan) {
    chips.push(`${plan.pluginIds?.length ?? 0} plugins`);
    chips.push(...(plan.skills ?? []).map((skill) => skill.id));
    chips.push(...(plan.mcpServers ?? []).map((server) => server.id));
    chips.push(...capabilitySourceChips(plan));
    if (plan.context) chips.push(`context ${plan.context.mode}/${plan.context.strategy}`);
    if (plan.permissions?.summary) chips.push(`permissions ${plan.permissions.summary}`);
    chips.push(...(plan.requirements ?? []).filter(showPlanRequirement).map(planRequirementLabel));
  } else {
    chips.push(...(agent.skills ?? []));
    chips.push(...(agent.mcpServers ?? []));
  }
  chips.push(...(agent.tools ?? []).map((tool) => `tool ${tool}`));
  chips.push(...(agent.disallowedTools ?? []).map((tool) => `blocked ${tool}`));
  if (agent.maxTurns) chips.push(`${agent.maxTurns} turns`);
  if (agent.effort) chips.push(`effort ${agent.effort}`);
  if (agent.definition?.host) chips.push(nativeAgentHostLabel(agent.definition.host));
  if (!agent.modelId && agent.nativeModel) chips.push(`native model ${agent.nativeModel}`);
  if (agent.sandboxMode) chips.push(`sandbox ${agent.sandboxMode}`);
  if (agent.isolation) chips.push(`isolation ${agent.isolation}`);
  return uniqueStrings(chips.filter((chip): chip is string => Boolean(chip)));
}

export function extensionAgentComposition(agent: ExtensionAgent): AgentCompositionPayload {
  return {
    id: `desktop-${agent.id}`,
    agentProfileId: agent.id,
    host: "local",
  };
}

function formatExtensionCommand(
  command: string[] | string | undefined,
  args: string[] | undefined,
): string[] {
  if (Array.isArray(command)) return [command.join(" ")];
  if (command) return [[command, ...(args ?? [])].join(" ")];
  return [];
}

function capabilitySourceChips(plan: ExtensionAgentPlan): string[] {
  const sourceIds = [
    ...(plan.skills ?? []).flatMap((skill) => (skill.sourcePluginId ? [skill.sourcePluginId] : [])),
    ...(plan.mcpServers ?? []).flatMap((server) =>
      server.sourcePluginId ? [server.sourcePluginId] : [],
    ),
  ];
  return uniqueStrings(sourceIds).map((sourceId) => `via ${sourceId}`);
}

function blockedPlanRequirements(plan: ExtensionAgentPlan): ExtensionAgentPlanRequirement[] {
  return (plan.requirements ?? []).filter(
    (requirement) => requirement.status !== "ok" && requirement.status !== "unknown",
  );
}

function showPlanRequirement(requirement: ExtensionAgentPlanRequirement): boolean {
  return requirement.status !== "ok";
}

function planRequirementLabel(requirement: ExtensionAgentPlanRequirement): string {
  if (requirement.kind === "secret") return "secret required";
  const id = requirement.id ? ` ${requirement.id}` : "";
  return `${requirement.status} ${requirement.kind}${id}`;
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}
