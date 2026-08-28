export interface TeamPlanTask {
  readonly id: string;
  readonly kind: "read" | "write" | "knowledge";
  readonly blockedBy: readonly string[];
  readonly writeScopes: readonly string[];
  readonly toolCalls: number;
  readonly effectfulToolCalls: number;
}

export interface TeamPlanAssessment {
  readonly cognitiveParallelWidth: number;
  readonly writeConflictRate: number;
  readonly effectfulToolDensity: number;
  readonly recommendedMembers: number;
  readonly recommendation: "stay_solo" | "create_team";
  readonly serializationPressure: "low" | "high";
}

function scopesConflict(left: readonly string[], right: readonly string[]): boolean {
  return left.some((a) =>
    right.some((b) => a === b || a.startsWith(`${b}/`) || b.startsWith(`${a}/`)),
  );
}

export function assessTeamPlan(tasks: readonly TeamPlanTask[]): TeamPlanAssessment {
  const byId = new Map(tasks.map((task) => [task.id, task]));
  if (byId.size !== tasks.length) throw new Error("Team plan task ids must be unique");
  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (task: TeamPlanTask): number => {
    const known = depths.get(task.id);
    if (known !== undefined) return known;
    if (visiting.has(task.id)) throw new Error("Team plan task graph must be acyclic");
    visiting.add(task.id);
    const dependencies = task.blockedBy.map((id) => {
      const dependency = byId.get(id);
      if (!dependency) throw new Error(`Team plan dependency not found: ${id}`);
      return depth(dependency);
    });
    visiting.delete(task.id);
    const value = dependencies.length === 0 ? 0 : Math.max(...dependencies) + 1;
    depths.set(task.id, value);
    return value;
  };

  const cognitiveByDepth = new Map<number, number>();
  for (const task of tasks) {
    if (!Number.isSafeInteger(task.toolCalls) || task.toolCalls < 0) {
      throw new Error("Team plan toolCalls must be a non-negative integer");
    }
    if (
      !Number.isSafeInteger(task.effectfulToolCalls) ||
      task.effectfulToolCalls < 0 ||
      task.effectfulToolCalls > task.toolCalls
    ) {
      throw new Error("Team plan effectfulToolCalls must be bounded by toolCalls");
    }
    const layer = depth(task);
    if (task.kind !== "write") {
      cognitiveByDepth.set(layer, (cognitiveByDepth.get(layer) ?? 0) + 1);
    }
  }

  const writes = tasks.filter((task) => task.kind === "write");
  let writePairs = 0;
  let conflicts = 0;
  for (let left = 0; left < writes.length; left += 1) {
    for (let right = left + 1; right < writes.length; right += 1) {
      writePairs += 1;
      if (scopesConflict(writes[left]?.writeScopes ?? [], writes[right]?.writeScopes ?? [])) {
        conflicts += 1;
      }
    }
  }
  const toolCalls = tasks.reduce((total, task) => total + task.toolCalls, 0);
  const effectfulCalls = tasks.reduce((total, task) => total + task.effectfulToolCalls, 0);
  const cognitiveParallelWidth = Math.max(0, ...cognitiveByDepth.values());
  const writeConflictRate = writePairs === 0 ? 0 : conflicts / writePairs;
  const effectfulToolDensity = toolCalls === 0 ? 0 : effectfulCalls / toolCalls;
  const recommendation = cognitiveParallelWidth >= 2 ? "create_team" : "stay_solo";
  return {
    cognitiveParallelWidth,
    writeConflictRate,
    effectfulToolDensity,
    recommendedMembers: recommendation === "create_team" ? Math.min(8, cognitiveParallelWidth) : 1,
    recommendation,
    serializationPressure: writeConflictRate >= 0.5 || effectfulToolDensity >= 0.5 ? "high" : "low",
  };
}
