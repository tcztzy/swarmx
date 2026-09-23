import { EventType } from "@ag-ui/core";
import type {
  ExecutionRecord,
  ExecutionRunSummary,
  ExecutionStatistics,
} from "../execution-record.js";

/** Only lifecycle records count as executions; tool ancestry is context, not another sample. */
export function executionRuns(
  records: readonly ExecutionRecord[],
  rootRunIds?: readonly string[],
  recordsComplete = true,
): ExecutionRunSummary[] {
  const grouped = new Map<string, ExecutionRecord[]>();
  for (const record of records) {
    if (!record.runId) continue;
    const run = grouped.get(record.runId) ?? [];
    run.push(record);
    grouped.set(record.runId, run);
  }
  const runs = [...grouped].flatMap(([runId, observed]): ExecutionRunSummary[] => {
    const records = observed.filter(
      ({ event }) =>
        event.type === EventType.RUN_STARTED ||
        event.type === EventType.RUN_FINISHED ||
        event.type === EventType.RUN_ERROR,
    );
    records.sort((a, b) => a.seq - b.seq);
    const started = records.find(({ event }) => event.type === EventType.RUN_STARTED);
    if (!started) return [];
    const terminal = records.findLast(
      ({ event }) => event.type === EventType.RUN_FINISHED || event.type === EventType.RUN_ERROR,
    );
    const attributes = Object.assign(
      {},
      ...records.map((record) => record.attributes),
    ) as ExecutionRecord["attributes"];
    const value = (key: string) => (typeof attributes[key] === "string" ? attributes[key] : null);
    const quantity = (key: string, integer = false) => {
      const number = terminal?.attributes[key];
      return typeof number === "number" &&
        Number.isFinite(number) &&
        number >= 0 &&
        (!integer || Number.isInteger(number))
        ? number
        : null;
    };
    const input = started?.event.type === EventType.RUN_STARTED ? started.event.input : undefined;
    const task =
      input?.messages
        .filter((message) => message.role === "user")
        .map((message) => (typeof message.content === "string" ? message.content : ""))
        .join("\n") || null;
    const profile = input?.forwardedProps?.profile;
    const reason: unknown =
      terminal?.event.type === EventType.RUN_FINISHED
        ? terminal.event.result?.stopReason
        : undefined;
    const outcome = !terminal
      ? "incomplete"
      : terminal.event.type === EventType.RUN_ERROR
        ? "error"
        : reason === "end_turn"
          ? "completed"
          : reason === "cancelled"
            ? "cancelled"
            : "other";
    const elapsed =
      started && terminal ? Date.parse(terminal.observedAt) - Date.parse(started.observedAt) : null;
    return [
      {
        runId,
        parentRunId: value("swarmx.execution.parent_run_id"),
        purpose: value("swarmx.execution.purpose"),
        sessionId: started?.sessionId ?? terminal?.sessionId ?? null,
        task,
        harness: value("swarmx.harness.name"),
        requestedModel:
          typeof started?.attributes["gen_ai.request.model"] === "string"
            ? started.attributes["gen_ai.request.model"]
            : null,
        requestedEffort:
          typeof started.attributes["gen_ai.request.reasoning.level"] === "string"
            ? started.attributes["gen_ai.request.reasoning.level"]
            : null,
        provider: value("gen_ai.provider.name"),
        harnessVersion: value("swarmx.harness.version") ?? value("swarmx.agent.version"),
        modelVersion: value("swarmx.model.version"),
        profile: typeof profile === "string" ? profile : null,
        startedAt: started?.observedAt ?? null,
        finishedAt: terminal?.observedAt ?? null,
        outcome,
        elapsedMs: elapsed !== null && Number.isFinite(elapsed) && elapsed >= 0 ? elapsed : null,
        inputTokens: quantity("gen_ai.usage.input_tokens", true),
        outputTokens: quantity("gen_ai.usage.output_tokens", true),
        cachedInputTokens: quantity("swarmx.usage.cached_input_tokens", true),
        reasoningOutputTokens: quantity("swarmx.usage.reasoning_output_tokens", true),
        costUsd: quantity("swarmx.usage.cost_usd"),
        tools: executionToolCost(observed),
        totalCostUsd: null,
        totalCostComplete: false,
        costSource:
          quantity("swarmx.usage.cost_usd") === null
            ? "unknown"
            : value("swarmx.usage.cost_source") === "native-estimate"
              ? "native-estimate"
              : value("swarmx.usage.cost_source") === "invoice"
                ? "invoice"
                : value("swarmx.usage.cost_source") === "price-snapshot"
                  ? "price-snapshot"
                  : "unknown",
        usageCoverage:
          quantity("gen_ai.usage.input_tokens", true) === null ||
          quantity("gen_ai.usage.output_tokens", true) === null
            ? "unknown"
            : value("swarmx.usage.coverage") === "complete"
              ? "complete"
              : "partial",
        usageBasis:
          typeof terminal?.attributes["swarmx.usage.basis"] === "string"
            ? terminal.attributes["swarmx.usage.basis"]
            : null,
        sources: records.map(({ id }) => `urn:swarmx:execution:${id}`),
      },
    ];
  });
  return descendantRuns(runs, rootRunIds).map((run) => {
    const cost = executionCost(runs, [run.runId]);
    const complete = recordsComplete && cost.complete;
    return {
      ...run,
      totalCostUsd: complete ? cost.usd : null,
      totalCostComplete: complete,
    };
  });
}

/** Sum independent charges once, optionally including all descendants of the selected runs. */
export function executionCost(
  runs: readonly ExecutionRunSummary[],
  rootRunIds?: readonly string[],
) {
  const charged = descendantRuns(runs, rootRunIds);
  const known = charged.filter(({ costUsd }) => costUsd !== null);
  const toolUsd = charged.reduce((sum, { tools }) => sum + tools.usd, 0);
  return {
    sampleCount: known.length,
    usd:
      known.length || toolUsd
        ? known.reduce((sum, run) => sum + (run.costUsd ?? 0), toolUsd)
        : null,
    complete: charged.length > 0 && known.length === charged.length,
  };
}

function descendantRuns(runs: readonly ExecutionRunSummary[], rootRunIds?: readonly string[]) {
  const unique = new Map(runs.map((run) => [run.runId, run]));
  const selected = new Set(rootRunIds ?? unique.keys());
  for (const runId of selected)
    for (const run of unique.values()) if (run.parentRunId === runId) selected.add(run.runId);
  return [...unique.values()].filter(({ runId }) => selected.has(runId));
}

/** Unpriced observed tools currently cost zero; no second resource ledger is stored. */
export function executionToolCost(records: readonly ExecutionRecord[]) {
  const calls = new Map<string, number | null>();
  for (const { runId, event, attributes } of records) {
    if (!("toolCallId" in event) || typeof event.toolCallId !== "string") continue;
    const key = `${runId}:${event.toolCallId}`;
    const cost = attributes["swarmx.tool.cost_usd"];
    if (typeof cost === "number" && Number.isFinite(cost) && cost >= 0) calls.set(key, cost);
    else if (!calls.has(key)) calls.set(key, null);
  }
  return {
    callCount: calls.size,
    usd: [...calls.values()].reduce<number>((sum, cost) => sum + (cost ?? 0), 0),
    unpricedCalls: [...calls.values()].filter((cost) => cost === null).length,
  };
}

export function executionStatistics(runs: readonly ExecutionRunSummary[]): ExecutionStatistics {
  const elapsed = runs
    .flatMap((run) => (run.elapsedMs === null ? [] : [run.elapsedMs]))
    .sort((a, b) => a - b);
  const usage = runs.flatMap(({ inputTokens, outputTokens }) =>
    inputTokens === null || outputTokens === null ? [] : [{ inputTokens, outputTokens }],
  );
  const starts = runs.flatMap((run) => (run.startedAt === null ? [] : [run.startedAt])).sort();
  const finishes = runs.flatMap((run) => (run.finishedAt === null ? [] : [run.finishedAt])).sort();
  const lower = elapsed[Math.floor((elapsed.length - 1) / 2)];
  const upper = elapsed[Math.floor(elapsed.length / 2)];
  return {
    recipe: "swarmx.execution.v1",
    scope: "cited executions",
    runIds: runs.map(({ runId }) => runId),
    window: { startedAt: starts[0] ?? null, finishedAt: finishes.at(-1) ?? null },
    sampleCount: runs.length,
    completed: runs.filter(({ outcome }) => outcome === "completed").length,
    error: runs.filter(({ outcome }) => outcome === "error").length,
    cancelled: runs.filter(({ outcome }) => outcome === "cancelled").length,
    incomplete: runs.filter(({ outcome }) => outcome === "incomplete").length,
    other: runs.filter(({ outcome }) => outcome === "other").length,
    elapsed: {
      sampleCount: elapsed.length,
      medianMs: lower === undefined || upper === undefined ? null : (lower + upper) / 2,
    },
    usage: {
      sampleCount: usage.length,
      inputTokens: usage.length ? usage.reduce((sum, run) => sum + run.inputTokens, 0) : null,
      outputTokens: usage.length ? usage.reduce((sum, run) => sum + run.outputTokens, 0) : null,
    },
    cost: {
      ...executionCost(runs),
      complete: runs.length > 0 && runs.every((run) => run.totalCostComplete),
    },
    tools: runs.reduce(
      (sum, { tools }) => ({
        callCount: sum.callCount + tools.callCount,
        usd: sum.usd + tools.usd,
        unpricedCalls: sum.unpricedCalls + tools.unpricedCalls,
      }),
      { callCount: 0, usd: 0, unpricedCalls: 0 },
    ),
  };
}
