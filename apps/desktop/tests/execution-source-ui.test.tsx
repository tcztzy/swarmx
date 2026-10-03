// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { i18n } from "../src/renderer/i18n.js";
import { ObservePanel } from "../src/renderer/observe.js";
import { readConceptResult, SavedConcept } from "../src/renderer/saved-concept.js";
import { SourceInspection } from "../src/renderer/source-inspection.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

const source = {
  resource: "urn:swarmx:execution:11111111-1111-4111-8111-111111111111",
  title: "Selection observation",
};
const startedAt = "2026-09-20T01:00:00.000Z";
const finishedAt = "2026-09-20T01:00:02.000Z";
const record = {
  schemaVersion: 1,
  seq: 1,
  id: source.resource.split(":").at(-1),
  observedAt: startedAt,
  workspaceId: "workspace",
  sessionId: "codex:parent",
  runId: "run",
  causedBy: null,
  attributes: {},
  event: {
    type: "RUN_STARTED",
    threadId: "codex:parent",
    runId: "run",
    input: {
      threadId: "codex:parent",
      runId: "run",
      tools: [],
      context: [],
      messages: [{ id: "input", role: "user", content: "Check the original data." }],
    },
  },
};
const evidence = {
  records: [
    record,
    {
      ...record,
      seq: 2,
      id: "22222222-2222-4222-8222-222222222222",
      event: {
        type: "TEXT_MESSAGE_CHUNK",
        messageId: "answer",
        role: "assistant",
        delta: "The original answer is preserved.",
      },
    },
  ],
  runs: [
    {
      runId: "run",
      parentRunId: null,
      purpose: null,
      sessionId: "codex:parent",
      task: "Check the original data.",
      harness: "codex",
      requestedModel: "requested-route",
      requestedEffort: "low",
      provider: null,
      harnessVersion: "native-1",
      modelVersion: null,
      profile: null,
      startedAt,
      finishedAt,
      outcome: "cancelled",
      elapsedMs: 2000,
      inputTokens: null,
      outputTokens: null,
      cachedInputTokens: null,
      reasoningOutputTokens: null,
      costUsd: null,
      tools: { callCount: 0, usd: 0, unpricedCalls: 0 },
      totalCostUsd: null,
      totalCostComplete: false,
      costSource: "unknown",
      usageCoverage: "unknown",
      usageBasis: "Native usage; cost estimated from configured model pricing.",
      sources: [source.resource],
    },
  ],
  statistics: {
    recipe: "swarmx.execution.v1",
    scope: "cited executions",
    runIds: ["run"],
    window: { startedAt, finishedAt },
    sampleCount: 1,
    completed: 0,
    error: 0,
    cancelled: 1,
    incomplete: 0,
    other: 0,
    elapsed: { sampleCount: 1, medianMs: 2000 },
    usage: { sampleCount: 0, inputTokens: null, outputTokens: null },
    cost: { sampleCount: 0, usd: null, complete: false },
    tools: { callCount: 0, usd: 0, unpricedCalls: 0 },
  },
};
let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("en");
  gateway = installBridge();
  gateway.logsEvidence.mockResolvedValue(evidence);
});
afterEach(() => cleanup());

it("opens cited execution evidence without a domain workspace and preserves original events", async () => {
  render(<ObservePanel source={source} onClose={() => {}} />);
  await screen.findByText("Cited executions");
  expect(gateway.logsEvidence).toHaveBeenCalledExactlyOnceWith({ sources: [source.resource] });
  expect(
    screen.getByText("0 completed · 0 errors · 1 cancelled · 0 incomplete · 0 other"),
  ).toBeTruthy();
  expect(screen.getByText("2.00 s")).toBeTruthy();
  expect(screen.getByText("requested-route")).toBeTruthy();
  expect(screen.getByText("Requested reasoning effort").nextElementSibling?.textContent).toBe(
    "low",
  );
  expect(screen.queryByText("Runtime-reported reasoning effort")).toBeNull();
  expect(
    screen.getByText("Native usage; cost estimated from configured model pricing."),
  ).toBeTruthy();
  expect(screen.getAllByText("Not recorded").length).toBeGreaterThan(0);
  expect(screen.queryByText("Succeeded")).toBeNull();
  expect(screen.queryByText(/The original answer is preserved/)).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: /Original input/ }));
  expect(screen.getByText(/"content": "Check the original data\."/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: /Original output/ }));
  expect(screen.getByText(/The original answer is preserved/)).toBeTruthy();
});

it("shows missing requested effort as not recorded", async () => {
  gateway.logsEvidence.mockResolvedValue({
    ...evidence,
    runs: evidence.runs.map((run) => ({ ...run, requestedEffort: null })),
  });
  render(<SourceInspection source={source} onClose={() => {}} />);
  expect(
    (await screen.findByText("Requested reasoning effort")).nextElementSibling?.textContent,
  ).toBe("Not recorded");
});

it("surfaces unavailable and foreign execution references", async () => {
  gateway.logsEvidence.mockRejectedValue(
    new Error("Execution source is unavailable in this directory."),
  );
  render(<SourceInspection source={source} onClose={() => {}} />);
  expect((await screen.findByRole("alert")).textContent).toBe(
    "Execution source is unavailable in this directory.",
  );
});

it("opens the frozen review snapshot with its omissions and original cited records", async () => {
  const snapshot = {
    ...record,
    seq: 3,
    id: "33333333-3333-4333-8333-333333333333",
    runId: null,
    event: {
      type: "CUSTOM",
      name: "swarmx.memory.review.started",
      value: { snapshot: { evidence: { ...evidence, omittedCount: 1, omitted: ["omitted-id"] } } },
    },
  };
  const planned = {
    ...snapshot,
    seq: 4,
    id: "44444444-4444-4444-8444-444444444444",
    event: {
      type: "CUSTOM",
      name: "swarmx.memory.review.planned",
      value: {
        reviewer: { harness: "pi", model: "reported-reviewer", version: "native-reviewer-version" },
        summary: "Keep this observation limited to the cited task.",
        operations: [],
      },
    },
  };
  gateway.logsEvidence.mockResolvedValue({
    ...evidence,
    records: [...evidence.records, snapshot, planned],
  });
  render(
    <SourceInspection
      source={{ resource: `urn:swarmx:execution:${snapshot.id}` }}
      snapshot={undefined}
      executions={[]}
      onClose={() => {}}
    />,
  );
  fireEvent.click(await screen.findByRole("button", { name: /Review snapshot/ }));
  expect(screen.getByText(/"omittedCount": 1/)).toBeTruthy();
  expect(screen.getByText(/"omitted-id"/)).toBeTruthy();
  expect(screen.getByRole("button", { name: /Original input/ })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: /Review conclusion and reviewer/ }));
  expect(screen.getByText(/"model": "reported-reviewer"/)).toBeTruthy();
  expect(screen.getByText(/"version": "native-reviewer-version"/)).toBeTruthy();
  expect(gateway.logsEvidence).toHaveBeenCalledExactlyOnceWith({
    sources: [`urn:swarmx:execution:${snapshot.id}`],
  });
});

it("shows evaluation scope without claiming verification and dispatches exact source references", () => {
  const concept = readConceptResult({
    action: "read_memory",
    data: {
      id: "selection.md",
      revision: `sha256:${"a".repeat(64)}`,
      metadata: {
        title: "Selection",
        status: "draft",
        tags: ["agent-selection"],
        sources: [source],
        swarmx_evaluation: {
          kind: "judgment",
          task: "Research writing",
          criteria: "Follow the requested scope",
          limitations: "One observed task",
          evidence: [source.resource],
          counterEvidence: [],
          extra: "ignored",
        },
        unrelatedFutureMetadata: true,
      },
    },
  });
  expect(concept).toBeDefined();
  if (!concept) throw new Error("Missing concept");
  const opened = vi.fn();
  window.addEventListener("swarmx:open-source", opened);
  try {
    render(<SavedConcept concept={concept} />);
    expect(screen.getByText("AI judgment")).toBeTruthy();
    expect(screen.getByText("Research writing")).toBeTruthy();
    expect(screen.getByText("Follow the requested scope")).toBeTruthy();
    expect(screen.getByText("One observed task")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /Selection observation/ }));
    expect(opened).toHaveBeenCalledWith(expect.objectContaining({ detail: { source } }));
  } finally {
    window.removeEventListener("swarmx:open-source", opened);
  }
});

it("marks only legacy selection concepts without evaluations as unverified", () => {
  const base = {
    id: "legacy.md",
    revision: "old",
    metadata: { title: "Legacy", status: "draft", sources: [], tags: ["agent-selection"] },
  };
  const concept = readConceptResult({ action: "read_memory", data: base });
  if (!concept) throw new Error("Missing concept");
  const view = render(<SavedConcept concept={concept} />);
  expect(screen.getByText("Unverified evaluation")).toBeTruthy();
  const ordinary = readConceptResult({
    action: "read_memory",
    data: { ...base, metadata: { ...base.metadata, tags: ["research"] } },
  });
  if (!ordinary) throw new Error("Missing concept");
  view.rerender(<SavedConcept concept={ordinary} />);
  expect(screen.queryByText("Unverified evaluation")).toBeNull();
});

it.each(["read_memory", "load_memory"])(
  "preserves an unresolved structured evaluation's unverified status in %s",
  (action) => {
    const stored = {
      id: "foreign-selection.md",
      revision: "saved-revision",
      metadata: {
        title: "Foreign selection evidence",
        status: "stable",
        tags: ["agent-selection"],
        sources: [source],
        swarmx_evaluation: {
          kind: "judgment",
          task: "Research writing",
          criteria: "Follow the requested scope",
          limitations: "One observed task",
        },
      },
      evaluation: {
        status: "unverified",
        reason: "Execution source is unavailable in this directory.",
      },
    };
    const concept = readConceptResult({
      action,
      data: action === "read_memory" ? stored : { concepts: [stored] },
    });
    if (!concept) throw new Error("Missing concept");
    render(<SavedConcept concept={concept} />);
    expect(screen.getByText("AI judgment")).toBeTruthy();
    expect(screen.getByText(/Unverified evaluation/)).toBeTruthy();
    expect(screen.getByText(/Execution source is unavailable in this directory\./)).toBeTruthy();
  },
);
