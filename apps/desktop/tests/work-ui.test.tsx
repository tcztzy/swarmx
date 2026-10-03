// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "../src/renderer/app.js";
import { i18n } from "../src/renderer/i18n.js";
import { WorkPanel } from "../src/renderer/work.js";
import {
  WorkAttemptSchema,
  WorkCycleSchema,
  WorkItemSchema,
  type WorkRead,
  WorkReadSchema,
  type WorkSnapshot,
} from "../src/work.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

vi.mock("../src/renderer/chat.js", () => ({
  ConversationSurface: ({ sidePanel }: { sidePanel?: ReactNode }) => (
    <>
      <input aria-label="draft fixture" defaultValue="保留我的草稿" />
      {sidePanel}
    </>
  ),
}));
vi.mock("../src/renderer/trace.js", () => ({ TracePanel: () => null }));

const cycle = WorkCycleSchema.parse({
  id: "cycle",
  project: "分析计划",
  budgetUsd: 10,
  configurations: [{ id: "config", harness: "codex", model: "test-model" }],
});
const item = WorkItemSchema.parse({
  id: "item",
  cycleId: cycle.id,
  goal: "分析样本",
  criteria: "报告样本数量并引用数据",
  criteriaVersion: "v1",
  taskClass: "analysis",
  state: "queued",
  blockedReason: null,
  firstAcceptedAt: null,
  createdAt: "2026-09-21T00:00:00.000Z",
});
const attempt = WorkAttemptSchema.parse({
  id: "attempt",
  workId: item.id,
  cycleId: cycle.id,
  runtimeId: "attempt",
  configuration: cycle.configurations[0],
  purpose: "execution",
  reservedUsd: 2,
  costUsd: null,
  costSource: "unknown",
  coverage: "unknown",
  state: "settled",
  outcome: "completed",
  createdAt: item.createdAt,
  finishedAt: item.createdAt,
  runIds: [],
  artifacts: [{ id: "artifact", revision: "sha256:abc" }],
  criteriaVersion: "v1",
  report: null,
});
function state(): WorkRead & { snapshot: WorkSnapshot } {
  const result = WorkReadSchema.parse({
    cycles: [cycle],
    snapshot: {
      cycle,
      items: [item],
      reservations: [],
      feedback: [],
      executions: [],
      decisions: [],
      tools: { callCount: 0, usd: 0, unpricedCalls: 0 },
      outcomes: {
        acceptedItems: 0,
        acceptedValue: 0,
        partialValue: 0,
        interventions: 0,
        valueBasis: "configured",
      },
      balance: {
        budgetUsd: 10,
        spentUsd: 0,
        heldUsd: 0,
        remainingUsd: 10,
        costCoverage: { reported: 0, total: 0, completeExecutions: 0 },
        enforcement: "admission",
      },
      coverageGaps: [],
    },
    activeWorkIds: [],
    interactions: [],
  });
  if (!result.snapshot) throw new Error("Fixture snapshot is required.");
  return { ...result, snapshot: result.snapshot };
}
let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("zh");
  gateway = installBridge();
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
function fill(label: string, value: string, root: Pick<typeof screen, "getByLabelText"> = screen) {
  fireEvent.change(root.getByLabelText(label), { target: { value } });
}

it("creates an explicitly configured cycle and multiple goals without starting paid work", async () => {
  let data: WorkRead = { cycles: [], snapshot: null, activeWorkIds: [], interactions: [] };
  gateway.workRead.mockImplementation(async () => structuredClone(data));
  gateway.workCommand.mockImplementation(async ({ action, request }) => {
    if (action === "createCycle") {
      data = state();
      data.cycles = [request];
      if (!data.snapshot) throw new Error("Fixture snapshot is required.");
      data.snapshot.cycle = request;
      data.snapshot.items = [];
    } else if (action === "createItem") data.snapshot?.items.push({ ...item, ...request });
    return structuredClone(data);
  });
  render(<WorkPanel harnesses={["codex", "claude"]} onClose={() => {}} />);
  await screen.findByText("先创建工作周期，再添加目标。每次执行都需要你明确开始。");
  fireEvent.click(screen.getByText("新建工作周期", { selector: "summary" }));
  const create = within(screen.getByRole("form", { name: "新建工作周期" }));
  fill("周期名称", "药物筛选", create);
  fill("周期预算（美元）", "10", create);
  fill("模型名称", "my-model", create);
  fireEvent.click(create.getByRole("button", { name: "创建周期" }));
  const add = within(await screen.findByRole("form", { name: "添加工作目标" }));
  fill("工作目标", "分析第一批数据", add);
  fill("验收标准", "包含样本数量", add);
  fill("任务类型", "analysis", add);
  fireEvent.click(add.getByRole("button", { name: "添加目标" }));
  await screen.findByRole("heading", { name: "分析第一批数据" });
  fill("工作目标", "分析第二批数据", add);
  fill("验收标准", "包含样本数量", add);
  fill("任务类型", "analysis", add);
  fireEvent.click(add.getByRole("button", { name: "添加目标" }));
  await screen.findByRole("heading", { name: "分析第二批数据" });
  expect(gateway.workCommand.mock.calls.map(([command]) => command.action)).toEqual([
    "createCycle",
    "createItem",
    "createItem",
  ]);
  expect(gateway.workCommand.mock.calls[0]?.[0].request).toMatchObject({
    budgetUsd: 10,
    configurations: [{ harness: "codex", model: "my-model" }],
  });
  expect(gateway.workCommand.mock.calls[1]?.[0].request).toMatchObject({
    mode: "manual",
    configuration: { harness: "codex", model: "my-model" },
    runtime: {},
  });
});

it("saves an explicitly chosen managed supervisor and runtime limits without starting execution", async () => {
  const data = state();
  data.snapshot.items = [];
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockResolvedValue(data);
  render(<WorkPanel harnesses={["codex", "claude"]} onClose={() => {}} />);
  const add = within(await screen.findByRole("form", { name: "添加工作目标" }));
  fill("工作目标", "连续分析更新的数据", add);
  fill("验收标准", "每批报告数量", add);
  fill("任务类型", "analysis", add);
  fill("运行模式", "managed", add);
  fill("监督 Agent", "", add);
  fill("Harness", "claude", add);
  fill("模型名称", "supervisor-model", add);
  fill("推理强度（可选）", "high", add);
  fill("本次预算（美元，可选）", "4", add);
  fill("运行时限（分钟，可选）", "5", add);
  fireEvent.click(add.getByRole("button", { name: "添加目标" }));
  await waitFor(() => expect(gateway.workCommand).toHaveBeenCalledTimes(1));
  expect(gateway.workCommand).toHaveBeenCalledWith({
    action: "createItem",
    request: expect.objectContaining({
      mode: "managed",
      supervisor: { id: "temporary", harness: "claude", model: "supervisor-model", effort: "high" },
      runtime: { budgetUsd: 4, timeoutMs: 300_000 },
    }),
  });
  expect(gateway.workCommand.mock.calls[0]?.[0].request).not.toHaveProperty("configuration");
});

it("allows a cycle with no presets and a managed supervisor selected from an existing preset", async () => {
  const data = state();
  data.snapshot.items = [];
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockResolvedValue(data);
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  const create = within(await screen.findByRole("form", { name: "新建工作周期" }));
  fill("周期名称", "不保存预设", create);
  fill("周期预算（美元）", "10", create);
  fireEvent.click(create.getByRole("button", { name: "移除此配置" }));
  fireEvent.click(create.getByRole("button", { name: "创建周期" }));
  await waitFor(() => expect(gateway.workCommand).toHaveBeenCalledTimes(1));
  expect(gateway.workCommand.mock.calls[0]?.[0].request.configurations).toEqual([]);
  const add = within(screen.getByRole("form", { name: "添加工作目标" }));
  fill("工作目标", "使用常用配置", add);
  fill("验收标准", "报告数量", add);
  fill("任务类型", "analysis", add);
  fill("运行模式", "managed", add);
  fireEvent.click(add.getByRole("button", { name: "添加目标" }));
  await waitFor(() => expect(gateway.workCommand).toHaveBeenCalledTimes(2));
  expect(gateway.workCommand.mock.calls[1]?.[0].request.supervisor).toEqual(
    cycle.configurations[0],
  );
});

it("keeps stopping pending until the Host reports completion, then records and corrects artifact-pinned acceptance", async () => {
  const data = state();
  const native = Promise.withResolvers<void>();
  gateway.workRead.mockImplementation(async () => structuredClone(data));
  gateway.workCommand.mockImplementation(async ({ action, request }) => {
    if (action === "start") {
      data.activeWorkIds = [item.id];
      data.snapshot.items = [{ ...item, state: "running" }];
      data.interactions = [
        {
          workId: item.id,
          id: "question",
          title: "确认数据范围",
          schema: {
            type: "object",
            properties: { scope: { type: "string", title: "样本范围" } },
            required: ["scope"],
          },
        },
      ];
      await native.promise;
    }
    if (action === "respond") data.interactions = [];
    if (action === "accept") {
      data.snapshot.feedback.push({
        ...request,
        source: "user",
        layer: "user",
        evaluator: "desktop-user",
        evaluatorVersion: "v1",
        recordedAt: item.createdAt,
      });
      data.snapshot.items = [{ ...item, state: request.accepted ? "accepted" : "queued" }];
    }
    return structuredClone(data);
  });
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  fireEvent.click(await screen.findByRole("button", { name: "开始执行" }));
  await screen.findByRole("heading", { name: "确认数据范围" });
  fill("样本范围", "全体样本");
  fireEvent.click(screen.getByRole("button", { name: "继续", exact: true }));
  await waitFor(() =>
    expect(gateway.workCommand).toHaveBeenLastCalledWith({
      action: "respond",
      workId: item.id,
      interactionId: "question",
      answer: { scope: "全体样本" },
    }),
  );
  fireEvent.click(await screen.findByRole("button", { name: "请求停止" }));
  expect(await screen.findByText("已请求停止，等待执行结束")).toBeTruthy();
  expect(screen.queryByText("待验收")).toBeNull();
  data.activeWorkIds = [];
  data.snapshot.items = [{ ...item, state: "awaiting-acceptance" }];
  data.snapshot.reservations = [attempt];
  data.snapshot.balance.heldUsd = 2;
  data.snapshot.balance.remainingUsd = 8;
  data.snapshot.balance.costCoverage.total = 1;
  data.snapshot.coverageGaps = ["unreported cost"];
  await act(async () => {
    native.resolve();
    await native.promise;
  });
  await screen.findByText("待验收");
  expect(screen.getByText("存在未报告的费用；未知费用不等于零。")).toBeTruthy();
  expect(screen.getByText("未知", { exact: true })).toBeTruthy();
  const review = within(screen.getByRole("form", { name: "记录用户验收" }));
  fill("验收报告", "样本数与原始数据一致", review);
  fireEvent.click(review.getByLabelText("接受本次结果"));
  fireEvent.click(review.getByRole("button", { name: "保存验收" }));
  await screen.findByText("已验收");
  const original = gateway.workCommand.mock.calls.find(
    ([command]) => command.action === "accept",
  )?.[0].request;
  expect(original).toMatchObject({
    attemptId: attempt.id,
    artifacts: attempt.artifacts,
    accepted: true,
    fraction: 1,
    criteriaVersion: "v1",
  });
  const correction = within(screen.getByRole("form", { name: "记录用户验收" }));
  fill("验收结论", "failed", correction);
  fill("验收报告", "发现重复样本，撤回先前验收", correction);
  fireEvent.click(correction.getByRole("button", { name: "更正验收" }));
  await waitFor(() =>
    expect(gateway.workCommand).toHaveBeenLastCalledWith({
      action: "accept",
      request: expect.objectContaining({
        supersedes: original.id,
        verdict: "failed",
        accepted: false,
        fraction: 0,
        report: "发现重复样本，撤回先前验收",
      }),
    }),
  );
});

it("preserves budget input on a conflict and uses the displayed expected budget", async () => {
  gateway.workRead.mockResolvedValue(state());
  gateway.workCommand.mockRejectedValue(new Error("Budget changed in another window."));
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  const budget = await screen.findByLabelText("新预算（美元）");
  fireEvent.change(budget, { target: { value: "4" } });
  fireEvent.click(screen.getByRole("button", { name: "调整预算" }));
  expect((await screen.findByRole("alert")).textContent).toContain(
    "Budget changed in another window.",
  );
  expect((budget as HTMLInputElement).value).toBe("4");
  expect(gateway.workCommand).toHaveBeenCalledWith({
    action: "setBudget",
    request: { cycleId: cycle.id, expectedBudgetUsd: 10, budgetUsd: 4 },
  });
});

it("starts the next eligible item through the Host and leaves Stop available while it runs", async () => {
  const data = state();
  const running = Promise.withResolvers<void>();
  gateway.workRead.mockImplementation(async () => structuredClone(data));
  gateway.workCommand.mockImplementation(async ({ action }) => {
    if (action === "startNext") {
      data.activeWorkIds = [item.id];
      await running.promise;
    }
    return structuredClone(data);
  });
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  fireEvent.click(await screen.findByRole("button", { name: "运行下一项" }));
  await screen.findByRole("button", { name: "请求停止" });
  expect(gateway.workCommand).toHaveBeenCalledWith({ action: "startNext", cycleId: cycle.id });
  expect((screen.getByRole("button", { name: "运行下一项" }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  fireEvent.click(screen.getByRole("button", { name: "请求停止" }));
  await waitFor(() =>
    expect(gateway.workCommand).toHaveBeenLastCalledWith({ action: "stop", workId: item.id }),
  );
  await act(async () => {
    running.resolve();
    await running.promise;
  });
});

it.each([
  { missing: false, invoice: null },
  { missing: true, invoice: null },
  { missing: false, invoice: 8 },
])("shows recursive cost once and prefers reconciled bills: %j", async ({ missing, invoice }) => {
  const data = state();
  data.snapshot.items = [
    {
      ...item,
      mode: "managed",
      supervisor: cycle.configurations[0],
      runtime: { timeoutMs: 300_000 },
    },
  ];
  data.snapshot.reservations = [
    {
      ...attempt,
      costUsd: invoice ?? 1,
      costSource: invoice === null ? "native-estimate" : "invoice",
      runIds: ["A"],
    },
    { ...attempt, id: "B", purpose: "delegation", costUsd: 2, runIds: ["B"] },
    { ...attempt, id: "C", purpose: "delegation", costUsd: missing ? null : 3, runIds: ["C"] },
  ];
  data.snapshot.tools = { callCount: 2, usd: 0, unpricedCalls: 2 };
  data.snapshot.executions = ["A", "B", "C"].map((runId, index, ids) => ({
    runId,
    parentRunId: ids[index - 1] ?? null,
    purpose: "execution",
    sessionId: `codex:${runId}`,
    task: item.goal,
    harness: "codex",
    requestedModel: "test-model",
    requestedEffort: null,
    provider: null,
    harnessVersion: null,
    modelVersion: null,
    profile: null,
    startedAt: item.createdAt,
    finishedAt: item.createdAt,
    outcome: "completed",
    elapsedMs: 0,
    inputTokens: null,
    outputTokens: null,
    cachedInputTokens: null,
    reasoningOutputTokens: null,
    costUsd: missing && index === 2 ? null : index + 1,
    tools: { callCount: 0, usd: 0, unpricedCalls: 0 },
    totalCostUsd: missing ? null : index === 0 ? 6 : index === 1 ? 5 : 3,
    totalCostComplete: !missing,
    costSource: "native-estimate",
    usageCoverage: "partial",
    usageBasis: "Local fixture",
    sources: [],
  }));
  gateway.workRead.mockResolvedValue(data);
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  const cost = within(await screen.findByRole("region", { name: "执行与验收" }));
  expect(cost.getByText(missing ? "未知" : `$${(invoice ?? 1) + 5}`, { exact: true })).toBeTruthy();
  expect(Boolean(cost.queryByText("费用不完整"))).toBe(missing);
  expect(screen.getByText(/时限 5 分钟/)).toBeTruthy();
  expect(screen.getByText("2 次工具调用尚未定价，暂按零计费。")).toBeTruthy();
});

it("only offers native tool profiles for DSH and removes the choice when switching harness", async () => {
  const data = state();
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockResolvedValue(data);
  render(<WorkPanel harnesses={["dsh", "codex"]} onClose={() => {}} />);
  const create = within(await screen.findByRole("form", { name: "新建工作周期" }));
  fill("周期名称", "常用配置", create);
  fill("周期预算（美元）", "10", create);
  fill("模型名称", "test-model", create);
  fill("工具配置（可选）", "sdk-minimal", create);
  fill("Harness", "codex", create);
  expect(create.queryByLabelText("工具配置（可选）")).toBeNull();
  fireEvent.click(create.getByRole("button", { name: "创建周期" }));
  await waitFor(() => expect(gateway.workCommand).toHaveBeenCalledTimes(1));
  expect(gateway.workCommand.mock.calls[0]?.[0].request.configurations[0]).toEqual({
    id: expect.any(String),
    harness: "codex",
    model: "test-model",
  });
});

it("keeps the conversation draft mounted and never cancels managed work when closing its side panel", async () => {
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  gateway.bootstrap.mockResolvedValue({
    agents: ["swarm", "codex"],
    defaultHarness: "codex",
    language: "zh",
    sessions: [{ sessionId: "codex:one", title: "研究" }],
    cwd: "/research",
  });
  gateway.workRead.mockResolvedValue({ ...state(), activeWorkIds: [item.id] });
  render(<App />);
  const draft = await screen.findByLabelText("draft fixture");
  fireEvent.click(screen.getByRole("button", { name: "长期工作", exact: true }));
  await screen.findByRole("complementary", { name: "长期工作侧栏" });
  expect(screen.getByLabelText("draft fixture")).toBe(draft);
  fireEvent.click(screen.getByRole("button", { name: "关闭侧栏" }));
  expect(screen.getByLabelText("draft fixture")).toBe(draft);
  expect((draft as HTMLInputElement).value).toBe("保留我的草稿");
  expect(gateway.workCommand).not.toHaveBeenCalled();
  expect(gateway.aguiCancel).not.toHaveBeenCalled();
});

it("retains a revised goal on conflict and saves the new criteria under the same work identity", async () => {
  const data = state();
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockRejectedValueOnce(new Error("Criteria changed in another window."));
  gateway.workCommand.mockImplementationOnce(async ({ request }) => {
    data.snapshot.items = [
      {
        ...item,
        goal: request.goal,
        criteria: request.criteria,
        criteriaVersion: request.criteriaVersion,
      },
    ];
    return structuredClone(data);
  });
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  await screen.findByRole("heading", { name: item.goal });
  fireEvent.click(screen.getByText("修改目标与验收标准", { selector: "summary" }));
  const revision = within(screen.getByRole("form", { name: "修改目标与验收标准" }));
  fill("修改后的目标", "分析更新后的样本", revision);
  fill("修改后的验收标准", "排除重复样本并报告数量", revision);
  fill("新标准版本", "v2", revision);
  fireEvent.click(revision.getByRole("button", { name: "保存目标修改" }));
  expect((await screen.findByRole("alert")).textContent).toContain("Criteria changed");
  expect((revision.getByLabelText("修改后的验收标准") as HTMLTextAreaElement).value).toBe(
    "排除重复样本并报告数量",
  );
  fireEvent.click(revision.getByRole("button", { name: "保存目标修改" }));
  await screen.findByRole("heading", { name: "分析更新后的样本" });
  expect(gateway.workCommand).toHaveBeenLastCalledWith({
    action: "revise",
    request: {
      id: item.id,
      expectedCriteriaVersion: "v1",
      criteriaVersion: "v2",
      goal: "分析更新后的样本",
      criteria: "排除重复样本并报告数量",
    },
  });
});

it("requires explicit invoice and outcome evidence, keeps unknown costs, and preserves opaque pinned artifact references", async () => {
  const data = state();
  data.snapshot.items = [{ ...item, state: "blocked" }];
  data.snapshot.reservations = [
    { ...attempt, finishedAt: null, state: "uncertain", outcome: "dispatch-unconfirmed" },
  ];
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockResolvedValue(data);
  const copy = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  expect(await screen.findByText(/成果版本: artifact/)).toBeTruthy();
  expect(screen.getByText("未验证的旧版记录：未引用执行证据。")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "复制成果引用" }));
  await waitFor(() =>
    expect(copy).toHaveBeenCalledWith(JSON.stringify({ id: "artifact", revision: "sha256:abc" })),
  );
  fireEvent.click(screen.getByText("核对费用与外部执行状态", { selector: "summary" }));
  const charge = within(screen.getByRole("form", { name: "登记核对后的费用" }));
  expect((charge.getByLabelText("账单费用（美元）") as HTMLInputElement).value).toBe("");
  fireEvent.click(charge.getByRole("button", { name: "登记费用" }));
  expect(gateway.workCommand).not.toHaveBeenCalled();
  fill("账单费用（美元）", "0.35", charge);
  fill("费用凭据", "账单 INV-42", charge);
  fireEvent.click(charge.getByRole("button", { name: "登记费用" }));
  await waitFor(() =>
    expect(gateway.workCommand).toHaveBeenCalledWith({
      action: "reconcileCharge",
      request: {
        id: expect.any(String),
        reservationId: attempt.id,
        costUsd: 0.35,
        source: "invoice",
        reference: "账单 INV-42",
      },
    }),
  );
  expect(screen.queryByRole("form", { name: "记录用户验收" })).toBeNull();
  const outcome = within(screen.getByRole("form", { name: "确认外部执行状态" }));
  fill("已核实的执行结果", "cancelled", outcome);
  fill("执行状态依据", "已在原生会话核实取消", outcome);
  fireEvent.click(outcome.getByRole("button", { name: "确认执行已结束" }));
  await waitFor(() =>
    expect(gateway.workCommand).toHaveBeenLastCalledWith({
      action: "reconcileOutcome",
      request: {
        reservationId: attempt.id,
        outcome: "cancelled",
        reference: "已在原生会话核实取消",
      },
    }),
  );
  expect(gateway.workCommand).toHaveBeenCalledTimes(2);
});

it("shows every cited execution without certifying the artifact and preserves evidence in copying and acceptance", async () => {
  await i18n.changeLanguage("en");
  const artifact = {
    id: "external:dataset/opaque",
    revision: "revision:unchanged",
    evidence: [
      "urn:swarmx:execution:11111111-1111-4111-8111-111111111111",
      "urn:swarmx:execution:22222222-2222-4222-8222-222222222222",
    ],
  };
  const data = state();
  data.snapshot.items = [{ ...item, state: "awaiting-acceptance" }];
  data.snapshot.reservations = [WorkAttemptSchema.parse({ ...attempt, artifacts: [artifact] })];
  gateway.workRead.mockResolvedValue(data);
  gateway.workCommand.mockResolvedValue(data);
  const copy = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
  const opened = vi.fn();
  window.addEventListener("swarmx:open-source", opened);
  try {
    render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
    await screen.findByText(`Artifact revision: ${artifact.id} · ${artifact.revision}`);
    expect(
      screen.getByText(
        "Recorded provenance does not verify the domain claim. Acceptance is separate.",
      ),
    ).toBeTruthy();
    expect(screen.queryByText("Unverified legacy record: no execution evidence.")).toBeNull();
    for (const resource of artifact.evidence) {
      fireEvent.click(screen.getByRole("button", { name: resource }));
      expect(opened.mock.calls.at(-1)?.[0].detail).toEqual({
        source: { resource, title: "Cited execution evidence" },
      });
    }
    expect(gateway.workCommand).not.toHaveBeenCalled();
    expect(gateway.aguiStart).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Copy artifact reference" }));
    await waitFor(() => expect(copy).toHaveBeenCalledWith(JSON.stringify(artifact)));
    const review = within(screen.getByRole("form", { name: "Record user acceptance" }));
    fill("Acceptance report", "Reviewed the external result independently", review);
    fireEvent.click(review.getByRole("button", { name: "Save acceptance" }));
    await waitFor(() =>
      expect(gateway.workCommand).toHaveBeenCalledWith({
        action: "accept",
        request: expect.objectContaining({ artifacts: [artifact], accepted: false }),
      }),
    );
  } finally {
    window.removeEventListener("swarmx:open-source", opened);
  }
});

it("labels explicitly empty artifact evidence as an unverified legacy record", async () => {
  await i18n.changeLanguage("en");
  const data = state();
  data.snapshot.reservations = [
    WorkAttemptSchema.parse({
      ...attempt,
      artifacts: [{ ...attempt.artifacts[0], evidence: [] }],
    }),
  ];
  gateway.workRead.mockResolvedValue(data);
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  expect(await screen.findByText("Unverified legacy record: no execution evidence.")).toBeTruthy();
  expect(screen.queryByText("Cited execution evidence")).toBeNull();
  expect(gateway.workCommand).not.toHaveBeenCalled();
});

it("opens artifact provenance through the existing inspector and surfaces unavailable evidence without rerunning work", async () => {
  const resource = "urn:swarmx:execution:11111111-1111-4111-8111-111111111111";
  const data = state();
  data.snapshot.reservations = [
    WorkAttemptSchema.parse({
      ...attempt,
      artifacts: [{ ...attempt.artifacts[0], evidence: [resource] }],
    }),
  ];
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  gateway.bootstrap.mockResolvedValue({
    agents: ["swarm", "codex"],
    defaultHarness: "codex",
    language: "zh",
    sessions: [{ sessionId: "codex:one", title: "研究" }],
    cwd: "/research",
  });
  gateway.workRead.mockResolvedValue(data);
  gateway.logsEvidence.mockRejectedValue(
    new Error("Execution source is unavailable in this directory."),
  );
  render(<App />);
  const draft = await screen.findByLabelText("draft fixture");
  fireEvent.click(screen.getByRole("button", { name: "长期工作", exact: true }));
  fireEvent.click(await screen.findByRole("button", { name: resource }));
  await screen.findByRole("complementary", { name: "来源检查" });
  expect(gateway.logsEvidence).toHaveBeenCalledExactlyOnceWith({ sources: [resource] });
  expect((await screen.findByRole("alert")).textContent).toBe(
    "Execution source is unavailable in this directory.",
  );
  expect(screen.getByLabelText("draft fixture")).toBe(draft);
  expect(gateway.workCommand).not.toHaveBeenCalled();
  expect(gateway.aguiStart).not.toHaveBeenCalled();
});

it("polls and switches cycles while start remains pending without overwriting the selection when it finishes", async () => {
  const first = state();
  const second = state();
  second.snapshot.cycle = { ...cycle, id: "second", project: "另一周期" };
  second.snapshot.items = [{ ...item, id: "second-item", cycleId: "second", goal: "另一个目标" }];
  first.cycles = second.cycles = [cycle, second.snapshot.cycle];
  const native = Promise.withResolvers<void>();
  gateway.workRead.mockImplementation(async ({ cycleId }) =>
    structuredClone(cycleId === "second" ? second : first),
  );
  gateway.workCommand.mockImplementation(async ({ action }) => {
    if (action === "start") {
      first.activeWorkIds = second.activeWorkIds = [item.id];
      await native.promise;
    }
    return structuredClone(first);
  });
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  fireEvent.click(await screen.findByRole("button", { name: "开始执行" }));
  fill("工作周期", "second");
  await screen.findByRole("heading", { name: "另一个目标" });
  second.snapshot.items = [
    { ...item, id: "second-item", cycleId: "second", goal: "其他窗口更新的目标" },
  ];
  await screen.findByRole("heading", { name: "其他窗口更新的目标" }, { timeout: 3500 });
  await act(async () => {
    native.resolve();
    await native.promise;
  });
  expect((screen.getByLabelText("工作周期") as HTMLSelectElement).value).toBe("second");
  expect(screen.getByRole("heading", { name: "其他窗口更新的目标" })).toBeTruthy();
  expect(gateway.workCommand.mock.calls.map(([command]) => command.action)).toEqual(["start"]);
});

it("exposes English labels for explicit budget and execution configuration", async () => {
  await i18n.changeLanguage("en");
  render(<WorkPanel harnesses={["codex"]} onClose={() => {}} />);
  await screen.findByRole("complementary", { name: "Long-term work side panel" });
  fireEvent.click(await screen.findByText("New work cycle", { selector: "summary" }));
  expect(screen.getByRole("textbox", { name: "Cycle name" })).toBeTruthy();
  expect(screen.getByRole("spinbutton", { name: "Cycle budget (USD)" })).toBeTruthy();
  expect(screen.queryByRole("spinbutton", { name: "Reservation per execution (USD)" })).toBeNull();
  expect(screen.queryByRole("textbox", { name: "Configuration version" })).toBeNull();
  expect(screen.getByRole("button", { name: "Create cycle" })).toBeTruthy();
});
