import { EventType } from "@ag-ui/core";
import type {
  ExecutionRecord,
  ExecutionRunSummary,
  ExecutionStatistics,
} from "../execution-record.js";

/** Only lifecycle records count as executions; tool ancestry is context, not another sample. */
export function executionRuns(records: readonly ExecutionRecord[]): ExecutionRunSummary[] {
  const grouped = new Map<string, ExecutionRecord[]>();
  for (const record of records) {
    if (
      !record.runId ||
      (record.event.type !== EventType.RUN_STARTED &&
        record.event.type !== EventType.RUN_FINISHED &&
        record.event.type !== EventType.RUN_ERROR)
    )
      continue;
    const run = grouped.get(record.runId) ?? [];
    run.push(record);
    grouped.set(record.runId, run);
  }
  return [...grouped].flatMap(([runId, records]): ExecutionRunSummary[] => {
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
        sessionId: started?.sessionId ?? terminal?.sessionId ?? null,
        task,
        harness: value("swarmx.harness.name"),
        requestedModel:
          typeof started?.attributes["gen_ai.request.model"] === "string"
            ? started.attributes["gen_ai.request.model"]
            : null,
        reportedModel:
          "gen_ai.response.model" in attributes
            ? value("gen_ai.response.model")
            : value("swarmx.agent.model"),
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
        costUsd: quantity("swarmx.usage.cost_usd"),
        usageBasis:
          typeof terminal?.attributes["swarmx.usage.basis"] === "string"
            ? terminal.attributes["swarmx.usage.basis"]
            : null,
        sources: records.map(({ id }) => `urn:swarmx:execution:${id}`),
      },
    ];
  });
}

export function executionStatistics(runs: readonly ExecutionRunSummary[]): ExecutionStatistics {
  const elapsed = runs
    .flatMap((run) => (run.elapsedMs === null ? [] : [run.elapsedMs]))
    .sort((a, b) => a - b);
  const usage = runs.flatMap(({ inputTokens, outputTokens }) =>
    inputTokens === null || outputTokens === null ? [] : [{ inputTokens, outputTokens }],
  );
  const cost = runs.flatMap((run) => (run.costUsd === null ? [] : [run.costUsd]));
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
    cost: { sampleCount: cost.length, usd: cost.length ? cost.reduce((a, b) => a + b, 0) : null },
  };
}
