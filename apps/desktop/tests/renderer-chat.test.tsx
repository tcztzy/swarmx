// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ThreadItem } from "../src/agents/generated/v2/ThreadItem.js";
import type { ExecutionRecord } from "../src/execution-record.js";
import { ConversationSurface } from "../src/renderer/chat.js";
import { i18n, t } from "../src/renderer/i18n.js";
import { TracePanel } from "../src/renderer/trace.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

let gateway: BridgeHarness;
const catalog = {
  current: { model: "model-a", effort: "low" },
  models: [
    { id: "model-a", name: "Model A", efforts: [{ id: "low", name: "Low" }], defaultEffort: "low" },
    {
      id: "model-b",
      name: "Model B",
      efforts: [
        { id: "low", name: "Low" },
        { id: "medium", name: "Medium" },
        { id: "high", name: "High" },
        { id: "xhigh", name: "XHigh" },
        { id: "max", name: "Max" },
      ],
      defaultEffort: "medium",
    },
    { id: "model-c", name: "Model C", efforts: [] },
  ],
};
const copy = vi.fn().mockResolvedValue(undefined);
const props = {
  agentId: "swarm",
  harness: "codex",
  harnesses: ["codex", "claude"],
  onHarnessChange: vi.fn(),
  harnessDisabled: false,
  threadId: "codex:session",
};
const started = { type: "RUN_STARTED", threadId: props.threadId, runId: "run" };
const finished = { type: "RUN_FINISHED", threadId: props.threadId, runId: "run" };

/** Answers the next AG-UI run with the given events. */
function respond(...items: object[]) {
  gateway.aguiStart.mockImplementationOnce(async ({ input }: { input: { threadId: string } }) => {
    gateway.emit(input.threadId, ...items);
    return {};
  });
}

async function emit(...items: object[]) {
  await act(async () => gateway.emit(props.threadId, ...items));
}

beforeEach(async () => {
  await i18n.changeLanguage("zh");
  gateway = installBridge();
  gateway.modelsRead.mockResolvedValue(catalog);
  gateway.sessionsHistory.mockResolvedValue({ supported: true, messages: [] });
  gateway.logsRead.mockResolvedValue({ events: [], nextAfter: 0, activeRunIds: [] });
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    },
  );
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
  Element.prototype.scrollTo = vi.fn();
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

async function send(text: string) {
  const input = await screen.findByRole("textbox", { name: t("发送消息") });
  await waitFor(() => expect(input.hasAttribute("disabled")).toBe(false));
  fireEvent.change(input, { target: { value: text } });
  const dispatched = gateway.aguiStart.mock.calls.length;
  fireEvent.click(screen.getByRole("button", { name: t("发送消息") }));
  await waitFor(() => expect(gateway.aguiStart).toHaveBeenCalledTimes(dispatched + 1));
}

it("shows stored DSH output and disables another execution in the same task", async () => {
  gateway.sessionsHistory.mockResolvedValue({
    supported: true,
    messages: [
      { id: "input", role: "user", content: "Old task" },
      { id: "answer", role: "assistant", content: "Stored DSH output" },
    ],
  });
  gateway.modelsRead.mockResolvedValue({ models: [], current: {} });
  render(<ConversationSurface {...props} harness="dsh" threadId="dsh:stored" />);
  expect(await screen.findByText("Stored DSH output")).toBeTruthy();
  expect(screen.getByRole("textbox", { name: "发送消息" }).hasAttribute("disabled")).toBe(true);
  expect(screen.getByPlaceholderText("DSH 每个任务独立执行；请新建任务继续")).toBeTruthy();
  expect(gateway.aguiStart).not.toHaveBeenCalled();
});

async function agentCard(name: string) {
  const title = await screen.findByText(name);
  const card = title.closest("details");
  if (!card) throw new Error("Missing child card");
  return card;
}

async function workPanel(name: string) {
  const trigger = await screen.findByRole("button", { name });
  const panel = trigger.closest<HTMLDivElement>('[data-slot="reasoning-root"]');
  if (!panel) throw new Error("Missing work disclosure");
  return panel;
}

function delegationLog() {
  const records: ExecutionRecord[] = [];
  const starts = new Map<string, string>();
  function append(
    name: string,
    event: ExecutionRecord["event"],
    attributes: ExecutionRecord["attributes"] = {},
    causedBy = starts.get(name) ?? null,
  ) {
    const record: ExecutionRecord = {
      schemaVersion: 1,
      seq: records.length + 1,
      id: `record-${records.length + 1}`,
      observedAt: "2026-09-05T12:00:00Z",
      workspaceId: "research",
      sessionId: name === "parent" ? props.threadId : `claude:${name}`,
      runId: `run-${name}`,
      causedBy,
      attributes: {
        "swarmx.harness.name": "claude",
        "gen_ai.request.model": "requested-model",
        "gen_ai.request.reasoning.level": "high",
        ...attributes,
      },
      event,
    };
    records.push(record);
    return record.id;
  }
  function child(name: string, parent = "parent") {
    const cause = append(parent, {
      type: "TOOL_CALL_START",
      toolCallId: `call-${name}`,
      toolCallName: "swarm",
    });
    append(
      parent,
      {
        type: "TOOL_CALL_ARGS",
        toolCallId: `call-${name}`,
        delta: JSON.stringify({ action: "send_message", agentId: name, text: `任务 ${name}` }),
      },
      {},
      cause,
    );
    const id = append(
      name,
      {
        type: "RUN_STARTED",
        threadId: `claude:${name}`,
        runId: `run-${name}`,
        input: {
          threadId: `claude:${name}`,
          runId: `run-${name}`,
          tools: [],
          context: [],
          state: {},
          forwardedProps: {},
          messages: [{ id: `prompt-${name}`, role: "user" as const, content: `任务 ${name}` }],
        },
      },
      {},
      cause,
    );
    starts.set(name, id);
  }
  function finish(name: string, interruptionRequested = false) {
    append(name, {
      type: "RUN_FINISHED",
      threadId: `claude:${name}`,
      runId: `run-${name}`,
      result: { interruptionRequested },
    });
  }
  starts.set(
    "parent",
    append("parent", {
      type: "RUN_STARTED",
      threadId: props.threadId,
      runId: "run-parent",
    }),
  );
  child("worker-a");
  child("worker-b");
  return { records, append, child, finish };
}

describe("assistant-ui conversation", () => {
  it("does not count an ACP token limit as a completed child", async () => {
    const log = delegationLog();
    log.append("worker-a", {
      type: "RUN_FINISHED",
      threadId: "claude:worker-a",
      runId: "run-worker-a",
      result: { stopReason: "max_tokens" },
    });
    log.finish("worker-b");
    gateway.logsRead.mockResolvedValue({
      events: log.records,
      nextAfter: log.records.length,
      activeRunIds: [],
    });
    render(<ConversationSurface {...props} />);
    expect(within(await agentCard("worker-a")).getByText("已停止")).toBeTruthy();
    expect(within(await agentCard("worker-b")).getByText("已完成")).toBeTruthy();
  });
  it("follows journal pagination and refreshes a running child from the last cursor", async () => {
    const log = delegationLog();
    while (log.records.length < 200)
      log.append("worker-a", {
        type: "RAW",
        source: "claude",
        event: { chunk: log.records.length },
      });
    log.finish("worker-b");
    let activeRunIds = ["run-worker-a"];
    gateway.logsRead.mockImplementation(async (query: { after: number }) => {
      const events = log.records.filter((record) => record.seq > query.after).slice(0, 200);
      return { events, nextAfter: events.at(-1)?.seq ?? query.after, activeRunIds };
    });
    render(<ConversationSurface {...props} />);
    expect(within(await agentCard("worker-b")).getByText("已完成")).toBeTruthy();
    expect(gateway.logsRead.mock.calls[1]?.[0]).toMatchObject({ after: 200 });
    const stop = log.append("worker-a", {
      type: "CUSTOM",
      name: "swarmx.run.interrupt_requested",
      value: {},
    });
    log.append(
      "worker-a",
      {
        type: "CUSTOM",
        name: "swarmx.control.failed",
        value: { message: "native stop rejected" },
      },
      {},
      stop,
    );
    const first = await agentCard("worker-a");
    fireEvent.click(within(first).getByText("worker-a"));
    await within(first).findByRole("textbox", { name: "给 worker-a 补充指令" });
    await waitFor(() => expect(within(first).getByText("native stop rejected")).toBeTruthy(), {
      timeout: 2500,
    });
    expect(within(first).getByText("运行中")).toBeTruthy();
    expect(
      within(first).getByRole("button", { name: "停止 worker-a" }).hasAttribute("disabled"),
    ).toBe(false);
    expect(gateway.logsRead.mock.calls[2]?.[0]).toMatchObject({ after: 201 });
    log.finish("worker-a", true);
    activeRunIds = [];
    await waitFor(() => expect(within(first).getByText("已结束 · 请求过停止")).toBeTruthy(), {
      timeout: 2500,
    });
  });

  it("shows independently completed children and nested delegation without replacing the parent draft", async () => {
    const log = delegationLog();
    log.append(
      "worker-b",
      { type: "RAW", source: "claude", event: { original: "native payload" } },
      { "gen_ai.response.model": "actual-model" },
    );
    log.append("worker-b", {
      type: "TEXT_MESSAGE_CHUNK",
      messageId: "answer",
      role: "assistant",
      delta: "**审查完成**",
    });
    log.finish("worker-b");
    log.child("nested-reviewer", "worker-a");
    gateway.logsRead.mockResolvedValue({
      events: log.records,
      nextAfter: log.records.length,
      activeRunIds: ["run-worker-a", "run-nested-reviewer"],
    });
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [{ id: "parent", role: "assistant", content: "父会话内容" }],
    });
    render(<ConversationSurface {...props} />);
    await screen.findByText("父会话内容");
    fireEvent.change(screen.getByRole("textbox", { name: "发送消息" }), {
      target: { value: "保留父会话草稿" },
    });
    const first = await agentCard("worker-a");
    const second = await agentCard("worker-b");
    expect(within(first).getByText("运行中")).toBeTruthy();
    expect(within(second).getByText("已完成")).toBeTruthy();
    expect(screen.getByText("1 / 3 已完成")).toBeTruthy();
    expect(screen.getByText("由 worker-a 委派")).toBeTruthy();
    fireEvent.click(within(second).getByText("worker-b"));
    await within(second).findByLabelText("worker-b 本轮对话");
    expect(within(second).getByText("审查完成").tagName).toBe("STRONG");
    expect(within(second).getByText("已报告模型：actual-model")).toBeTruthy();
    expect(within(second).getByText("requested-model")).toBeTruthy();
    expect(within(second).queryByRole("button", { name: "停止 worker-b" })).toBeNull();
    fireEvent.click(within(second).getByText(/执行日志 ·/));
    fireEvent.click(within(second).getByText(/RAW/, { selector: "summary" }));
    expect(await within(second).findByText(/native payload/)).toBeTruthy();
    expect((screen.getByRole("textbox", { name: "发送消息" }) as HTMLTextAreaElement).value).toBe(
      "保留父会话草稿",
    );
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("sends child steering and cancellation to an execution ID and preserves rejected input", async () => {
    const log = delegationLog();
    log.finish("worker-b");
    gateway.logsRead.mockResolvedValue({
      events: log.records,
      nextAfter: log.records.length,
      activeRunIds: ["run-worker-a"],
    });
    render(<ConversationSurface {...props} />);
    const card = await agentCard("worker-a");
    fireEvent.click(within(card).getByText("worker-a"));
    const draft = await within(card).findByRole("textbox", { name: "给 worker-a 补充指令" });
    fireEvent.change(draft, { target: { value: "补充检查误差" } });
    gateway.runsControl.mockRejectedValueOnce(new Error("Native steering rejected"));
    fireEvent.click(within(card).getByRole("button", { name: "发送指令" }));
    await within(card).findByRole("alert");
    expect((draft as HTMLTextAreaElement).value).toBe("补充检查误差");
    gateway.runsControl.mockResolvedValueOnce({ runId: "run-worker-a" });
    fireEvent.click(within(card).getByRole("button", { name: "发送指令" }));
    await within(card).findByText("补充指令已发送");
    expect(gateway.runsControl).toHaveBeenLastCalledWith({
      runId: "run-worker-a",
      command: { action: "steer", text: "补充检查误差" },
    });
    expect((draft as HTMLTextAreaElement).value).toBe("");
    gateway.runsControl.mockResolvedValueOnce({ runId: "run-worker-a" });
    fireEvent.click(within(card).getByRole("button", { name: "停止 worker-a" }));
    await within(card).findByText("已请求停止");
    expect(gateway.runsControl).toHaveBeenLastCalledWith({
      runId: "run-worker-a",
      command: { action: "cancel" },
    });
  });

  it("restores unknown, failed and stopped runs without fabricating successful completion", async () => {
    const log = delegationLog();
    log.append("worker-b", { type: "RUN_ERROR", message: "child failed" });
    log.child("stopped");
    log.finish("stopped", true);
    gateway.logsRead.mockResolvedValue({
      events: log.records,
      nextAfter: log.records.length,
      activeRunIds: [],
    });
    render(<ConversationSurface {...props} />);
    const card = await agentCard("worker-a");
    expect(within(card).getByText("状态未知")).toBeTruthy();
    expect(within(await agentCard("worker-b")).getByText("失败")).toBeTruthy();
    expect(within(await agentCard("stopped")).getByText("已结束 · 请求过停止")).toBeTruthy();
    expect(screen.getByText("0 / 3 已完成")).toBeTruthy();
    fireEvent.click(within(card).getByText("worker-a"));
    await within(card).findByText(/没有结束记录/);
    expect(within(card).queryByRole("button", { name: "停止 worker-a" })).toBeNull();
  });

  it("shows child confirmation in the parent and disables conflicting controls", async () => {
    const log = delegationLog();
    log.append("worker-a", {
      type: "CUSTOM",
      name: "swarmx.interaction.requested",
      value: { id: "approve", title: "Permission", schema: {} },
    });
    log.finish("worker-b");
    gateway.logsRead.mockResolvedValue({
      events: log.records,
      nextAfter: log.records.length,
      activeRunIds: ["run-worker-a"],
    });
    render(<ConversationSurface {...props} />);
    const card = await agentCard("worker-a");
    expect(within(card).getByText("等待确认")).toBeTruthy();
    fireEvent.click(within(card).getByText("worker-a"));
    await within(card).findByText(/请在主对话中回答或取消/);
    expect(
      within(card).getByRole("button", { name: "停止 worker-a" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(within(card).getByRole("button", { name: "发送指令" }).hasAttribute("disabled")).toBe(
      true,
    );
  });

  it("keeps the draft and history while native mode, model and effort choices reach the next send", async () => {
    gateway.modelsRead.mockResolvedValueOnce({
      ...catalog,
      modes: [
        { id: "plan", name: "Plan" },
        { id: "full", name: "Full access" },
      ],
      current: { ...catalog.current, mode: "plan" },
    });
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [{ id: "old", role: "assistant", content: "已有历史" }],
    });
    render(<ConversationSurface {...props} />);
    await screen.findByText("已有历史");
    await screen.findByText("Model A");
    fireEvent.change(screen.getByRole("combobox", { name: "原生模式" }), {
      target: { value: "full" },
    });
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "保留草稿" } });
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    expect(
      (await screen.findByRole("option", { name: "Model A" })).getAttribute("aria-selected"),
    ).toBe("true");
    fireEvent.click(screen.getByRole("option", { name: "Model B" }));
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole("combobox", { name: "选择模型" })),
    );
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("保留草稿");
    expect(screen.getByText("已有历史")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "选择模型" }).textContent).toContain("Low");
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    await screen.findByRole("radiogroup", { name: "推理强度" });
    expect(screen.getAllByRole("radio").map((item) => item.textContent)).toEqual([
      "Low",
      "Med",
      "High",
      "XHigh",
      "Max",
    ]);
    expect(screen.queryByRole("slider")).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "XHigh" }));
    expect(screen.getByRole("combobox", { name: "选择模型" }).textContent).toContain("XHigh");
    fireEvent.click(screen.getByRole("radio", { name: "Max" }));
    expect(screen.getByRole("radio", { name: "Max" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.getByRole("dialog", { name: "模型与推理强度" })).toBeTruthy();
    fireEvent.keyDown(screen.getByRole("radio", { name: "Max" }), { key: "Escape" });
    respond(started, finished);
    fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
    await waitFor(() => expect(gateway.aguiStart).toHaveBeenCalledTimes(1));
    expect(gateway.aguiStart.mock.calls[0]?.[0]).toMatchObject({
      agent: "swarm",
      input: {
        threadId: props.threadId,
        forwardedProps: { modelName: "model-b", reasoningEffort: "max", mode: "full" },
        messages: expect.arrayContaining([
          { id: expect.any(String), role: "user", content: "保留草稿" },
        ]),
      },
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    fireEvent.click(await screen.findByRole("option", { name: "Model C" }));
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    expect(screen.queryByRole("radiogroup", { name: "推理强度" })).toBeNull();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    respond(started, finished);
    await send("无需推理档位");
    await waitFor(() => expect(gateway.aguiStart).toHaveBeenCalledTimes(2));
    expect(gateway.aguiStart.mock.calls[1]?.[0]?.input.forwardedProps).toEqual({
      modelName: "model-c",
      mode: "full",
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    fireEvent.click(await screen.findByRole("option", { name: "Model B" }));
    expect(screen.getByRole("combobox", { name: "选择模型" }).textContent).toContain("Max");
  });

  it("surfaces catalog failures, retries explicitly, and does not invent an empty catalog", async () => {
    gateway.modelsRead.mockRejectedValueOnce(new Error("Model catalog unavailable"));
    render(<ConversationSurface {...props} />);
    await screen.findByRole("button", { name: "梳理任务思路" });
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Model catalog unavailable");
    gateway.modelsRead.mockResolvedValueOnce({ models: [], current: {} });
    fireEvent.click(screen.getByRole("button", { name: "重新加载模型" }));
    await screen.findByText("此 Harness 未提供可选模型");
    expect(screen.queryByRole("option")).toBeNull();
    expect(gateway.modelsRead).toHaveBeenCalledTimes(2);
  });

  it("uses the native default when the selected effort is unsupported by the new model", async () => {
    gateway.modelsRead.mockResolvedValueOnce({
      ...catalog,
      models: [
        ...catalog.models,
        {
          id: "native-default",
          name: "Native default",
          defaultEffort: "high",
          efforts: [
            { id: "medium", name: "Medium" },
            { id: "high", name: "High" },
          ],
        },
      ],
    });
    render(<ConversationSurface {...props} />);
    await screen.findByText("Model A");
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    fireEvent.click(await screen.findByRole("option", { name: "Native default" }));
    expect(screen.getByRole("combobox", { name: "选择模型" }).textContent).toContain("High");
    respond(started, finished);
    await send("采用原生默认强度");
    await waitFor(() => expect(gateway.aguiStart).toHaveBeenCalledTimes(1));
    expect(gateway.aguiStart.mock.calls[0]?.[0]?.input.forwardedProps).toEqual({
      modelName: "native-default",
      reasoningEffort: "high",
    });
  });

  it("returns keyboard navigation from Thinking to the model list without sending", async () => {
    render(<ConversationSurface {...props} />);
    await screen.findByText("Model A");
    const trigger = screen.getByRole("combobox", { name: "选择模型" });
    fireEvent.keyDown(trigger, { key: "ArrowDown" });
    fireEvent.click(await screen.findByRole("option", { name: "Model B" }));
    fireEvent.click(trigger);
    const low = screen.getByRole("radio", { name: "Low" });
    act(() => low.focus());
    fireEvent.keyDown(low, { key: "ArrowUp" });
    const navigation = screen.getByRole("combobox", { name: "模型导航" });
    expect(document.activeElement).toBe(navigation);
    fireEvent.keyDown(navigation, { key: "Enter", keyCode: 13 });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(trigger.textContent).toContain("Model A");
    expect(gateway.aguiStart).not.toHaveBeenCalled();
  });

  it("separates generic tool arguments and results while preserving a native failure", async () => {
    await i18n.changeLanguage("en");
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "Inspect the index" },
        {
          id: "call",
          role: "assistant",
          _tool: { kind: "other", status: "failed" },
          toolCalls: [
            {
              id: "index",
              type: "function",
              function: { name: "read_index", arguments: '{"path":"index.json"}' },
            },
          ],
        },
        { id: "result", role: "tool", toolCallId: "index", content: "Permission denied" },
      ],
    });
    render(<ConversationSurface {...props} />);
    fireEvent.click(await screen.findByRole("button", { name: "Used tools" }));
    const trigger = screen.getByRole("button", { name: "read_index Failed" });
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(trigger);
    expect(screen.getByText("Arguments")).toBeTruthy();
    expect(screen.getByText('{"path":"index.json"}')).toBeTruthy();
    expect(screen.getByText("Result")).toBeTruthy();
    expect(screen.getByText("Permission denied")).toBeTruthy();
    expect(screen.queryByText("Complete")).toBeNull();
    expect(screen.queryByText("Success")).toBeNull();
    expect(screen.queryByRole("button", { name: /Allow|Deny/ })).toBeNull();
    expect(gateway.aguiStart).not.toHaveBeenCalled();
  });
  it.each(
    ["read_memory", "load_memory"].flatMap((action) => [
      { action, harness: "codex", live: false },
      { action, harness: "pi", live: false },
      { action, harness: "pi", live: true },
    ]),
  )(
    "opens a pinned source from $harness $action (live=$live) and preserves the draft",
    async ({ action, harness, live }) => {
      await i18n.changeLanguage("en");
      const source = {
        title: "Execution evidence",
        resource: "urn:swarmx:execution:11111111-1111-4111-8111-111111111111",
      };
      const concept = {
        id: "finding",
        revision: "sha256:0123456789abcdef",
        metadata: {
          title: "Finding",
          status: "draft",
          sources: [source],
        },
      };
      const data = {
        action,
        data: action === "read_memory" ? concept : { concepts: [concept] },
      };
      const result = harness === "pi" ? data : { result: { structuredContent: data }, error: null };
      const toolName = harness === "pi" ? "memory" : "mcp__swarmx__memory";
      gateway.sessionsHistory.mockResolvedValue({
        supported: true,
        messages: live
          ? []
          : [
              { id: "u1", role: "user", content: "Retrieve the saved finding" },
              {
                id: "tool",
                role: "assistant",
                toolCalls: [
                  {
                    id: "memory-read",
                    type: "function",
                    function: { name: toolName, arguments: "{}" },
                  },
                ],
              },
              {
                id: "read",
                role: "tool",
                toolCallId: "memory-read",
                content: JSON.stringify(result),
              },
              {
                id: "answer",
                role: "assistant",
                content: "## Recorded finding\n\nRetrieved without recomputing.",
              },
            ],
      });
      if (live)
        respond(
          started,
          { type: "TOOL_CALL_START", toolCallId: "memory-read", toolCallName: toolName },
          { type: "TOOL_CALL_ARGS", toolCallId: "memory-read", delta: "{}" },
          { type: "TOOL_CALL_END", toolCallId: "memory-read" },
          {
            type: "TOOL_CALL_RESULT",
            messageId: "read",
            toolCallId: "memory-read",
            content: JSON.stringify(result),
          },
          { type: "TEXT_MESSAGE_START", messageId: "answer", role: "assistant" },
          {
            type: "TEXT_MESSAGE_CONTENT",
            messageId: "answer",
            delta: "Retrieved without recomputing.",
          },
          { type: "TEXT_MESSAGE_END", messageId: "answer" },
          finished,
        );
      const open = vi.fn();
      window.addEventListener("swarmx:open-source", open);
      render(<ConversationSurface {...props} harness={harness} />);
      if (live) await send("Retrieve the saved finding");
      const saved = await screen.findByRole("region", { name: "Saved concept" });
      expect(within(saved).getByText("draft")).toBeTruthy();
      const draft = screen.getByRole("textbox", { name: "Send message" });
      fireEvent.change(draft, { target: { value: "Keep this draft" } });
      fireEvent.click(within(saved).getByRole("button", { name: /Execution evidence/ }));
      expect(open).toHaveBeenCalledWith(expect.objectContaining({ detail: { source } }));
      expect((draft as HTMLTextAreaElement).value).toBe("Keep this draft");
      expect(gateway.aguiStart).toHaveBeenCalledTimes(live ? 1 : 0);
      window.removeEventListener("swarmx:open-source", open);
    },
  );

  it.each([false, true])(
    "shows one saved concept under the final answer (phases=%s)",
    async (phases) => {
      await i18n.changeLanguage("en");
      const result = {
        action: "read_memory",
        data: {
          id: "finding",
          revision: "v1",
          metadata: {
            title: "Finding",
            status: "draft",
            sources: [],
          },
        },
      };
      gateway.sessionsHistory.mockResolvedValue({
        supported: true,
        messages: [
          { id: "u", role: "user", content: "Read finding" },
          {
            id: "tool",
            role: "assistant",
            toolCalls: [
              { id: "read", type: "function", function: { name: "memory", arguments: "{}" } },
            ],
          },
          { id: "result", role: "tool", toolCallId: "read", content: JSON.stringify(result) },
          {
            id: "comment",
            role: "assistant",
            content: "Checking the finding",
            ...(phases ? { _meta: { turnId: "turn", phase: "commentary", durationMs: 1000 } } : {}),
          },
          {
            id: "final",
            role: "assistant",
            content: "Final answer",
            ...(phases
              ? { _meta: { turnId: "turn", phase: "final_answer", durationMs: 1000 } }
              : {}),
          },
        ],
      });
      render(<ConversationSurface {...props} />);
      await screen.findByText("Final answer");
      if (phases) fireEvent.click(screen.getByRole("button", { name: "Worked for 1s" }));
      expect(await screen.findAllByRole("region", { name: "Saved concept" })).toHaveLength(1);
    },
  );

  it.each(["pi", "codex"].flatMap((harness) => [false, true].map((live) => ({ harness, live }))))(
    "keeps historical $harness domain results readable without local domain actions (live=$live)",
    async ({ harness, live }) => {
      const data = { data: { artifact: { id: "figure", projectId: "study" } } };
      const native = {
        type: "mcpToolCall",
        id: "figure-read",
        server: "swarmx",
        tool: "science_figure",
        status: "completed",
        arguments: {},
        appContext: null,
        pluginId: null,
        readOnlyHint: true,
        result: {
          content: [{ type: "text", text: JSON.stringify(data) }],
          structuredContent: data,
          _meta: null,
        },
        error: null,
        durationMs: 1,
      } satisfies Extract<ThreadItem, { type: "mcpToolCall" }>;
      const result = harness === "pi" ? data : native;
      gateway.sessionsHistory.mockResolvedValue({
        supported: true,
        messages: live
          ? []
          : [
              { id: "input", role: "user", content: "Retrieve the figure" },
              {
                id: "call",
                role: "assistant",
                toolCalls: [
                  {
                    id: "figure-read",
                    type: "function",
                    function: { name: "science_figure", arguments: "{}" },
                  },
                ],
              },
              {
                id: "result",
                role: "tool",
                toolCallId: "figure-read",
                content: JSON.stringify(result),
              },
            ],
      });
      if (live)
        respond(
          started,
          { type: "TOOL_CALL_START", toolCallId: "figure-read", toolCallName: "science_figure" },
          { type: "TOOL_CALL_ARGS", toolCallId: "figure-read", delta: "{}" },
          { type: "TOOL_CALL_END", toolCallId: "figure-read" },
          {
            type: "TOOL_CALL_RESULT",
            messageId: "result",
            toolCallId: "figure-read",
            content: JSON.stringify(result),
          },
          finished,
        );
      const open = vi.fn();
      window.addEventListener("swarmx:open-source", open);
      render(<ConversationSurface {...props} harness={harness} />);
      if (live) await send("Retrieve the figure");
      fireEvent.click(await screen.findByRole("button", { name: "调用工具" }));
      fireEvent.click(screen.getByRole("button", { name: "science_figure 完成" }));
      expect(screen.queryByRole("button", { name: "在侧栏中查看" })).toBeNull();
      expect(screen.getByText(/"artifact"/)).toBeTruthy();
      expect(open).not.toHaveBeenCalled();
      window.removeEventListener("swarmx:open-source", open);
    },
  );
  it.each(["legacy", "codex", "claude", "hermes"])(
    "groups historical tools and renders %s shell output",
    async (format) => {
      await i18n.changeLanguage("en");
      const tool = (id: string, name: string, kind: string, command?: string, exitCode = 0) => [
        {
          id: `call:${id}`,
          role: "assistant",
          _tool: { kind, status: exitCode === 0 ? "completed" : "failed" },
          toolCalls: [
            {
              id,
              type: "function",
              function: { name, arguments: JSON.stringify(command ? { command } : {}) },
            },
          ],
        },
        {
          id: `result:${id}`,
          role: "tool",
          toolCallId: id,
          content: JSON.stringify(
            format === "codex" && command
              ? {
                  type: "commandExecution",
                  aggregatedOutput: `output from ${id}\nnext line`,
                  exitCode,
                }
              : format === "claude" && command
                ? { stdout: `output from ${id}`, stderr: "next line", interrupted: false }
                : format === "hermes" && command
                  ? { result: { output: `output from ${id}\nnext line`, exit_code: exitCode } }
                  : {
                      formatted_output: `output from ${id}\nnext line`,
                      exit_code: exitCode,
                    },
          ),
        },
      ];
      gateway.sessionsHistory.mockResolvedValue({
        supported: true,
        messages: [
          { id: "u1", role: "user", content: "Check files" },
          ...tool("read", "Read file 'README.md'", "read"),
          ...tool("shell", "pnpm test", "execute", "pnpm test"),
          { id: "commentary", role: "assistant", content: "Now checking a failure" },
          ...tool("fail", "exit 2", "execute", "exit 2", 2),
          { id: "reasoning", role: "reasoning", content: "Check another file" },
          ...tool("another", "Read file 'package.json'", "read"),
          { id: "answer", role: "assistant", content: "Final result" },
        ],
      });
      const { container } = render(<ConversationSurface {...props} />);
      const trigger = await screen.findByRole("button", { name: "Read files, ran commands" });
      const group = trigger.closest(".tool-group");
      if (!group) throw new Error("Missing tool group");
      expect(container.querySelectorAll(".tool-group")).toHaveLength(2);
      expect(screen.queryByText("Check another file")).toBeNull();
      expect(trigger.getAttribute("aria-expanded")).toBe("false");
      expect(within(group).queryByText("$ pnpm test")).toBeNull();
      fireEvent.click(trigger);
      expect(trigger.getAttribute("aria-expanded")).toBe("true");
      expect(within(group).getAllByText("Success")).toHaveLength(2);
      expect(within(group).getByText("$ pnpm test")).toBeTruthy();
      expect(within(group).getByText(/output from shell/).textContent).toBe(
        "output from shell\nnext line",
      );
      expect(group.textContent).not.toContain("formatted_output");
      fireEvent.click(screen.getByRole("button", { name: "Ran commands, read files" }));
      const failed = screen.getByText("$ exit 2").closest('[data-slot="terminal-block"]');
      expect(failed?.textContent).toContain("Failed");
      expect(failed?.textContent).toContain("exit 2");
      expect(failed?.textContent).not.toContain("Success");
      expect(screen.getByText("Final result").closest(".tool-group")).toBeNull();
      fireEvent.click(trigger);
      expect(trigger.getAttribute("aria-expanded")).toBe("false");
      expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
    },
  );

  it("keeps unrecognized command results visible in the generic tool view", async () => {
    await i18n.changeLanguage("en");
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "user", role: "user", content: "Inspect output" },
        {
          id: "call",
          role: "assistant",
          toolCalls: [
            {
              id: "command",
              type: "function",
              function: {
                name: "terminal",
                arguments: JSON.stringify({ command: "inspect" }),
              },
            },
          ],
        },
        {
          id: "result",
          role: "tool",
          toolCallId: "command",
          content: JSON.stringify({ detail: "Native result remains available" }),
        },
        { id: "answer", role: "assistant", content: "Inspection finished" },
      ],
    });
    const { container } = render(<ConversationSurface {...props} />);
    await screen.findByText("Inspection finished");
    fireEvent.click(screen.getByRole("button", { name: "Used tools" }));
    fireEvent.click(screen.getByRole("button", { name: /terminal/ }));
    expect(screen.getByText(/Native result remains available/)).toBeTruthy();
    expect(container.querySelector('[data-slot="terminal-block"]')).toBeNull();
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("groups streaming calls across native data events and never labels cancelled tools successful", async () => {
    await i18n.changeLanguage("en");
    const { container } = render(<ConversationSurface {...props} />);
    await send("Check commands");
    const native = (id: string, kind: string) => ({
      type: "CUSTOM",
      name: "swarmx.activity",
      value: {
        type: "tool",
        toolCallId: id,
        kind,
        status: "in_progress",
      },
    });
    const startTool = (id: string, command: string) => [
      { type: "TOOL_CALL_START", toolCallId: id, toolCallName: command },
      { type: "TOOL_CALL_ARGS", toolCallId: id, delta: JSON.stringify({ command }) },
      { type: "TOOL_CALL_END", toolCallId: id },
    ];
    await emit(
      started,
      { type: "TEXT_MESSAGE_START", messageId: "intro", role: "assistant" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "intro", delta: "Starting tool work" },
      { type: "TEXT_MESSAGE_END", messageId: "intro" },
      native("read", "read"),
      ...startTool("read", "cat README.md"),
      {
        type: "TOOL_CALL_RESULT",
        messageId: "r1",
        toolCallId: "read",
        content: JSON.stringify({ formatted_output: "Read output", exit_code: 0 }),
      },
      {
        type: "CUSTOM",
        name: "swarmx.activity",
        value: {
          type: "tool",
          toolCallId: "read",
          status: "completed",
        },
      },
      native("shell", "execute"),
      ...startTool("shell", "pnpm test"),
    );
    fireEvent.click(await screen.findByRole("button", { name: "Read files, ran commands" }));
    expect(container.querySelectorAll(".tool-group")).toHaveLength(1);
    expect(screen.getByText("Starting tool work").closest(".tool-group")).toBeNull();
    const shell = () => screen.getByText("$ pnpm test").closest('[data-slot="terminal-block"]');
    expect(shell()?.textContent).toContain("Running");
    expect(shell()?.textContent).not.toContain("Success");
    fireEvent.click(screen.getByRole("button", { name: "Stop generation" }));
    await waitFor(() => expect(shell()?.textContent).toContain("Stopped"));
    expect(shell()?.textContent).not.toContain("Success");
    expect(container.querySelectorAll(".tool-group")).toHaveLength(1);
    expect(gateway.aguiCancel).toHaveBeenCalledWith({
      agent: props.agentId,
      threadId: props.threadId,
    });
  });

  it("renders avatar-free Demo messages, hydrates Markdown and copies the assistant reply", async () => {
    const reply =
      "## 研究进展\n\n- 已整理数据\n- 待验证结果\n\n| 内容 | 进展 |\n| --- | --- |\n| 数据 | 已完成 |";
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "研究进展？" },
        {
          id: "tool",
          role: "assistant",
          toolCalls: [
            { id: "call", type: "function", function: { name: "read_file", arguments: "{}" } },
          ],
        },
        { id: "result", role: "tool", toolCallId: "call", content: "done" },
        { id: "a1", role: "assistant", content: reply },
      ],
    });
    const { container } = render(<ConversationSurface {...props} />);
    expect(await screen.findByRole("heading", { name: "研究进展" })).toBeTruthy();
    expect(container.querySelector(".message-avatar")).toBeNull();
    expect(screen.queryByText("原生会话")).toBeNull();
    const userMessage = screen.getByText("研究进展？").closest('[data-role="user"]');
    expect(userMessage?.textContent).toBe("研究进展？");
    const assistantMessage = screen
      .getByRole("heading", { name: "研究进展" })
      .closest('[data-role="assistant"]');
    const actions = screen.getByRole("button", { name: "复制回复" }).closest(".answer-actions");
    expect(actions?.parentElement).toBe(assistantMessage);
    expect(actions?.previousElementSibling?.querySelector(".aui-md")).toBeTruthy();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByRole("cell", { name: "已完成" })).toBeTruthy();
    expect(screen.queryByText("正在处理…")).toBeNull();
    expect(gateway.sessionsHistory).toHaveBeenCalledWith({
      agent: "swarm",
      sessionId: "codex:session",
    });
    fireEvent.click(screen.getByRole("button", { name: "复制回复" }));
    await waitFor(() => expect(copy).toHaveBeenCalledWith(reply));
    expect(await screen.findByText("已复制")).toBeTruthy();
    expect(screen.queryByRole("complementary", { name: "执行轨迹" })).toBeNull();
  });

  it("keeps the reply action placeholder mounted as hover shows and hides the copy button", async () => {
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "a1", role: "assistant", content: "Earlier reply" },
        { id: "u2", role: "user", content: "Next question" },
        { id: "a2", role: "assistant", content: "Latest reply" },
      ],
    });
    render(<ConversationSurface {...props} />);
    const message = (await screen.findByText("Earlier reply")).closest(
      '[data-role="assistant"]',
    ) as HTMLElement;
    const placeholder = message.querySelector(".answer-actions");
    expect(placeholder).not.toBeNull();
    expect(within(message).queryByRole("button", { name: "复制回复" })).toBeNull();
    fireEvent.mouseEnter(message);
    const button = within(message).getByRole("button", { name: "复制回复" });
    expect(button.closest(".answer-actions")).toBe(placeholder);
    fireEvent.click(button);
    await waitFor(() => expect(copy).toHaveBeenCalledWith("Earlier reply"));
    fireEvent.mouseLeave(message);
    expect(within(message).queryByRole("button", { name: "复制回复" })).toBeNull();
    expect(message.querySelector(".answer-actions")).toBe(placeholder);
    expect(screen.getByText("Next question")).toBeTruthy();
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("folds completed Codex work by turn, preserves final answers and expands independently", async () => {
    const message = (id: string, phase: string, turnId: string, durationMs: number | null) => ({
      id,
      role: "assistant",
      content: id,
      _meta: { phase, turnId, durationMs },
    });
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "First question" },
        message("First commentary", "commentary", "t1", 1027731),
        { id: "r1", role: "reasoning", content: "Reasoning alongside commentary" },
        {
          id: "tool",
          role: "assistant",
          toolCalls: [
            { id: "call", type: "function", function: { name: "read_file", arguments: "{}" } },
          ],
        },
        { id: "result", role: "tool", toolCallId: "call", content: "done" },
        message("More commentary", "commentary", "t1", 1027731),
        message("First final", "final_answer", "t1", 1027731),
        { id: "u2", role: "user", content: "Second question" },
        message("Second commentary", "commentary", "t2", 3601000),
        message("Second final", "final_answer", "t2", 3601000),
        { id: "u3", role: "user", content: "Interrupted question" },
        message("Unfinished commentary", "commentary", "t3", null),
        message("Resumed commentary", "commentary", "t3-next", 0),
        message("Resumed final", "final_answer", "t3-next", 0),
        { id: "u4", role: "user", content: "Untagged question" },
        { id: "plain", role: "assistant", content: "Untagged reply" },
        { id: "u5", role: "user", content: "Unknown timing" },
        message("Untimed commentary", "commentary", "t5", null),
        message("Untimed final", "final_answer", "t5", null),
      ],
    });
    render(<ConversationSurface {...props} />);
    const first = await workPanel("Worked for 17m 8s");
    const second = await workPanel("Worked for 1h 0m 1s");
    expect(first.dataset.state).toBe("closed");
    expect(second.dataset.state).toBe("closed");
    expect(within(first).queryByText("First commentary")).toBeNull();
    expect(within(first).queryByText("More commentary")).toBeNull();
    for (const text of ["First final", "Second final", "Unfinished commentary", "Untagged reply"])
      expect(screen.getByText(text).closest('[data-slot="reasoning-root"]')).toBeNull();
    expect(screen.getByText("Worked for 0s")).toBeTruthy();
    expect(screen.getByText("Worked")).toBeTruthy();
    fireEvent.click(within(first).getByText("Worked for 17m 8s"));
    expect(first.dataset.state).toBe("open");
    expect(second.dataset.state).toBe("closed");
    for (const text of ["First commentary", "More commentary", "Unfinished commentary"]) {
      const message = screen.getByText(text).closest('[data-role="assistant"]') as HTMLElement;
      fireEvent.mouseEnter(message);
      expect(within(message).queryByRole("button", { name: "复制回复", hidden: true })).toBeNull();
      fireEvent.mouseLeave(message);
    }
    const final = screen.getByText("First final").closest('[data-role="assistant"]') as HTMLElement;
    fireEvent.mouseEnter(final);
    fireEvent.click(within(final).getByRole("button", { name: "复制回复" }));
    await waitFor(() => expect(copy).toHaveBeenCalledWith("First final"));
    fireEvent.mouseLeave(final);
    expect(screen.queryByText("Reasoning alongside commentary")).toBeNull();
    expect(first.querySelector('[data-slot="reasoning-root"]')).toBeNull();
    fireEvent.click(within(first).getByRole("button", { name: "调用工具" }));
    expect(within(first).getByText("read_file")).toBeTruthy();
    fireEvent.click(within(first).getByText("Worked for 17m 8s"));
    expect(first.dataset.state).toBe("closed");
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("shows native reasoning duration without an expandable disclosure or hidden message spacing", async () => {
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "Reason through this" },
        { id: "r1", role: "reasoning", content: "First reasoning block" },
        { id: "r2", role: "reasoning", content: "Second reasoning block" },
        {
          id: "a1",
          role: "assistant",
          content: "Reasoned answer",
          _meta: {
            phase: "final_answer",
            turnId: "reasoned",
            durationMs: 45000,
          },
        },
        { id: "u2", role: "user", content: "Unfinished turn" },
        { id: "r3", role: "reasoning", content: "Unfinished reasoning" },
        { id: "u3", role: "user", content: "Unmarked turn" },
        { id: "r4", role: "reasoning", content: "Unmarked reasoning" },
        { id: "a2", role: "assistant", content: "Unmarked answer" },
      ],
    });
    const { container } = render(<ConversationSurface {...props} />);
    expect(await screen.findByText("Reasoned answer")).toBeTruthy();
    expect(screen.getByText("Unmarked answer")).toBeTruthy();
    const duration = screen.getByText("Worked for 45s");
    expect(duration.closest("button")).toBeNull();
    expect(duration.hasAttribute("aria-expanded")).toBe(false);
    expect(duration.compareDocumentPosition(screen.getByText("Reasoned answer")) & 4).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Worked|思考过程/ })).toBeNull();
    for (const text of [
      "First reasoning block",
      "Second reasoning block",
      "Unfinished reasoning",
      "Unmarked reasoning",
    ])
      expect(screen.queryByText(text)).toBeNull();
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(2);
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("hides context compaction between turns and inside answers while keeping each native duration", async () => {
    const compaction = {
      id: "compact",
      type: "function",
      function: { name: "contextCompaction", arguments: "{}" },
    };
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "First question" },
        {
          id: "a1",
          role: "assistant",
          content: "First answer",
          _meta: { phase: "final_answer", turnId: "first", durationMs: 72000 },
        },
        { id: "c1", role: "assistant", toolCalls: [compaction] },
        { id: "result", role: "tool", toolCallId: "compact", content: "Context compacted" },
        { id: "u2", role: "user", content: "Second question" },
        {
          id: "a2",
          role: "assistant",
          content: "Second answer",
          toolCalls: [{ ...compaction, id: "compact-again" }],
          _meta: { phase: "final_answer", turnId: "second", durationMs: 130000 },
        },
      ],
    });
    const { container } = render(<ConversationSurface {...props} />);
    expect(await screen.findByText("Second answer")).toBeTruthy();
    for (const [label, answer] of [
      ["Worked for 1m 12s", "First answer"],
      ["Worked for 2m 10s", "Second answer"],
    ] as const) {
      const duration = screen.getByText(label);
      expect(duration.closest("button")).toBeNull();
      expect(duration.compareDocumentPosition(screen.getByText(answer)) & 4).toBeTruthy();
      fireEvent.click(duration);
    }
    expect(screen.queryByRole("button", { name: /Worked|调用工具/ })).toBeNull();
    expect(screen.queryByText("contextCompaction")).toBeNull();
    expect(screen.queryByText("Context compacted")).toBeNull();
    expect(container.querySelector('[data-slot="reasoning-root"]')).toBeNull();
    expect(container.querySelectorAll('[data-role="assistant"]')).toHaveLength(2);
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("keeps tools in the work disclosure when a turn has no commentary", async () => {
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [
        { id: "u1", role: "user", content: "Read a file" },
        { id: "r1", role: "reasoning", content: "Hidden tool reasoning" },
        {
          id: "tool",
          role: "assistant",
          toolCalls: [
            {
              id: "compact",
              type: "function",
              function: { name: "contextCompaction", arguments: "{}" },
            },
            { id: "call", type: "function", function: { name: "read_file", arguments: "{}" } },
          ],
        },
        { id: "result", role: "tool", toolCallId: "call", content: "done" },
        {
          id: "final",
          role: "assistant",
          content: "Read complete",
          _meta: { phase: "final_answer", turnId: "tools", durationMs: 2300 },
        },
      ],
    });
    render(<ConversationSurface {...props} />);
    const work = await workPanel("Worked for 2s");
    expect(work.dataset.state).toBe("closed");
    expect(screen.getByText("Read complete").closest('[data-slot="reasoning-root"]')).toBeNull();
    fireEvent.click(within(work).getByRole("button", { name: "Worked for 2s" }));
    fireEvent.click(within(work).getByRole("button", { name: "调用工具" }));
    expect(within(work).getByText("read_file")).toBeTruthy();
    expect(screen.queryByText("contextCompaction")).toBeNull();
    expect(screen.queryByText("Hidden tool reasoning")).toBeNull();
    expect(within(work).queryByRole("button", { name: "复制回复", hidden: true })).toBeNull();
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it("uses Reasoning for commentary Markdown and preserves manual toggles through final answers", async () => {
    render(<ConversationSurface {...props} />);
    await send("Inspect carefully");
    const native = (messageId: string, phase: string) => ({
      type: "CUSTOM",
      name: "swarmx.activity",
      value: { type: "message", messageId, phase, turnId: "manual" },
    });
    await emit(
      started,
      native("commentary", "commentary"),
      { type: "TEXT_MESSAGE_START", messageId: "commentary", role: "assistant" },
      {
        type: "TEXT_MESSAGE_CONTENT",
        messageId: "commentary",
        delta: "## Plan\n\n- **Inspect** the files",
      },
    );
    const trigger = await screen.findByRole("button", { name: "正在处理…" });
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    expect(await screen.findByRole("heading", { name: "Plan" })).toBeTruthy();
    await waitFor(() => expect(screen.getByRole("listitem").textContent).toBe("Inspect the files"));
    fireEvent.click(trigger);
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    await emit({
      type: "TEXT_MESSAGE_CONTENT",
      messageId: "commentary",
      delta: "\n- Verify the result",
    });
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("heading", { name: "Plan" })).toBeNull();
    fireEvent.click(trigger);
    expect(await screen.findByText("Verify the result")).toBeTruthy();
    expect(trigger.getAttribute("aria-expanded")).toBe("true");
    await emit(
      { type: "TEXT_MESSAGE_END", messageId: "commentary" },
      native("answer", "final_answer"),
      { type: "TEXT_MESSAGE_START", messageId: "answer", role: "assistant" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "answer", delta: "Checked the files" },
      { type: "TEXT_MESSAGE_END", messageId: "answer" },
      finished,
    );
    const completed = await screen.findByRole("button", { name: "Worked" });
    expect(completed.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("heading", { name: "Plan" })).toBeTruthy();
    expect(screen.getByText("Verify the result")).toBeTruthy();
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });

  it.each(["commentary", "reasoning"] as const)(
    "shows live %s with one work disclosure or a status, then preserves final answers",
    async (kind) => {
      render(<ConversationSurface {...props} />);
      const native = (messageId: string, phase: string) => ({
        type: "CUSTOM",
        name: "swarmx.activity",
        value: { type: "message", messageId, phase, turnId: "live" },
      });
      await send("Start work");
      await emit(
        started,
        {
          type: "CUSTOM",
          name: "swarmx.activity",
          value: {
            type: "tool",
            toolCallId: "compact",
            kind: "other",
            status: "in_progress",
          },
        },
        { type: "TOOL_CALL_START", toolCallId: "compact", toolCallName: "contextCompaction" },
        {
          type: "TOOL_CALL_ARGS",
          toolCallId: "compact",
          delta: JSON.stringify({ type: "contextCompaction", id: "compact" }),
        },
        { type: "TOOL_CALL_END", toolCallId: "compact" },
        {
          type: "TOOL_CALL_RESULT",
          toolCallId: "compact",
          messageId: "result",
          content: JSON.stringify({ type: "contextCompaction", id: "compact" }),
        },
        ...(kind === "commentary"
          ? [native("work", "commentary")]
          : [{ type: "REASONING_START", messageId: "work" }]),
        {
          type: kind === "commentary" ? "TEXT_MESSAGE_START" : "REASONING_MESSAGE_START",
          messageId: "work",
          role: kind === "commentary" ? "assistant" : "reasoning",
        },
        {
          type: kind === "commentary" ? "TEXT_MESSAGE_CONTENT" : "REASONING_MESSAGE_CONTENT",
          messageId: "work",
          delta: `Live ${kind}`,
        },
        {
          type: kind === "commentary" ? "TEXT_MESSAGE_END" : "REASONING_MESSAGE_END",
          messageId: "work",
        },
        ...(kind === "reasoning" ? [{ type: "REASONING_END", messageId: "work" }] : []),
      );
      expect(screen.queryByRole("button", { name: "调用工具" })).toBeNull();
      expect(screen.queryByText("contextCompaction")).toBeNull();
      if (kind === "commentary") {
        const live = await workPanel("正在处理…");
        expect(live.dataset.state).toBe("open");
        expect(await within(live).findByText("Live commentary")).toBeTruthy();
      } else {
        expect(screen.queryByText("Live reasoning")).toBeNull();
        expect(screen.queryByRole("button", { name: /Worked|思考/ })).toBeNull();
        expect(await screen.findByText("正在处理…")).toBeTruthy();
      }
      await emit(
        native("final", "final_answer"),
        { type: "TEXT_MESSAGE_START", messageId: "final", role: "assistant" },
        { type: "TEXT_MESSAGE_CONTENT", messageId: "final", delta: "Live final" },
      );
      if (kind === "commentary") {
        const work = await workPanel("Worked");
        expect(work.dataset.state).toBe("closed");
        expect(within(work).queryByText("Live commentary")).toBeNull();
      }
      expect(
        (await screen.findByText("Live final")).closest('[data-slot="reasoning-root"]'),
      ).toBeNull();
      await emit(
        { type: "TEXT_MESSAGE_END", messageId: "final" },
        {
          type: "CUSTOM",
          name: "swarmx.activity",
          value: { type: "message", turnId: "live", durationMs: 74809 },
        },
        finished,
      );
      if (kind === "commentary") {
        const work = await workPanel("Worked for 1m 15s");
        expect(work.dataset.state).toBe("closed");
        fireEvent.click(within(work).getByRole("button", { name: "Worked for 1m 15s" }));
        const message = within(work)
          .getByText("Live commentary")
          .closest('[data-role="assistant"]') as HTMLElement;
        fireEvent.mouseEnter(message);
        expect(within(work).queryByRole("button", { name: "复制回复", hidden: true })).toBeNull();
        fireEvent.mouseLeave(message);
      } else {
        expect(screen.getByText("Worked for 1m 15s").closest("button")).toBeNull();
        expect(screen.queryByRole("button", { name: /Worked/ })).toBeNull();
        expect(screen.queryByText("Live reasoning")).toBeNull();
      }
      fireEvent.click(screen.getByRole("button", { name: "复制回复" }));
      await waitFor(() => expect(copy).toHaveBeenCalledWith("Live final"));
    },
  );

  it("fills a suggested draft without sending and streams through the AG-UI bridge", async () => {
    render(<ConversationSurface {...props} />);
    fireEvent.click(await screen.findByRole("button", { name: "梳理任务思路" }));
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toContain("任务目标");
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
    respond(
      started,
      { type: "TEXT_MESSAGE_START", messageId: "answer", role: "assistant" },
      { type: "TEXT_MESSAGE_CONTENT", messageId: "answer", delta: "这是流式回复。" },
      { type: "TEXT_MESSAGE_END", messageId: "answer" },
      finished,
    );
    fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
    expect(await screen.findByText("这是流式回复。")).toBeTruthy();
    expect(gateway.aguiStart.mock.calls[0]?.[0]).toMatchObject({
      agent: "swarm",
      input: {
        threadId: "codex:session",
        messages: [{ role: "user", content: expect.stringContaining("任务目标") }],
      },
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
  });

  it("keeps streaming when the side pane opens or closes and aborts only when Stop is clicked", async () => {
    const { rerender } = render(<ConversationSurface {...props} />);
    await send("持续运行");
    await emit(started);
    const stop = await screen.findByRole("button", { name: "停止生成" });
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(screen.getByRole("combobox", { name: "选择模型" }).hasAttribute("disabled")).toBe(true);
    expect(screen.queryByRole("button", { name: "发送消息" })).toBeNull();
    rerender(<ConversationSurface {...props} panelOpen sidePanel={<TracePanel />} />);
    expect(screen.getByRole("button", { name: "停止生成" })).toBe(stop);
    expect(gateway.aguiCancel).not.toHaveBeenCalled();
    rerender(<ConversationSurface {...props} />);
    expect(screen.getByRole("button", { name: "停止生成" })).toBe(stop);
    expect(gateway.aguiCancel).not.toHaveBeenCalled();
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
    fireEvent.click(stop);
    await waitFor(() =>
      expect(gateway.aguiCancel).toHaveBeenCalledWith({
        agent: "swarm",
        threadId: "codex:session",
      }),
    );
    expect(await screen.findByRole("button", { name: "发送消息" })).toBeTruthy();
  });

  it("requires interaction fields, preserves a declined boolean, and resumes the native request", async () => {
    render(<ConversationSurface {...props} />);
    respond(started, {
      ...finished,
      outcome: {
        type: "interrupt",
        interrupts: [
          {
            id: "approval",
            reason: "input_required",
            message: "确认分析参数",
            responseSchema: {
              type: "object",
              description: '{"command":"inspect dataset","path":"/project/data file.txt"}',
              properties: {
                count: { type: "integer", title: "样本数量", minimum: 1 },
                allow: { type: "boolean", title: "允许写入" },
                password: { type: "string", format: "password", title: "凭据" },
              },
              required: ["count", "allow"],
            },
          },
        ],
      },
    });
    await send("分析数据");
    const number = await screen.findByRole("spinbutton", { name: "样本数量" });
    const scope = screen.getByText('{"command":"inspect dataset","path":"/project/data file.txt"}');
    expect(scope.tagName).toBe("PRE");
    expect(scope.getAttribute("contenteditable")).toBeNull();
    const password = screen.getByLabelText("凭据");
    expect(password.getAttribute("type")).toBe("password");
    fireEvent.change(password, { target: { value: "private value" } });
    expect(screen.getByRole("textbox").hasAttribute("disabled")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "继续" }));
    expect(gateway.aguiStart).toHaveBeenCalledTimes(1);
    fireEvent.change(number, { target: { value: "3" } });
    respond(started, finished);
    fireEvent.click(screen.getByRole("button", { name: "继续" }));
    await waitFor(() => expect(gateway.aguiStart).toHaveBeenCalledTimes(2));
    expect(gateway.aguiStart.mock.calls[1]?.[0]).toMatchObject({
      input: {
        resume: [
          {
            interruptId: "approval",
            status: "resolved",
            payload: { count: 3, allow: false, password: "private value" },
          },
        ],
      },
    });
    await waitFor(() => expect(screen.queryByText("确认分析参数")).toBeNull());
  });

  it("shows history failures and prevents a send with missing native history", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    gateway.sessionsHistory.mockRejectedValueOnce(new Error("History is unavailable"));
    render(<ConversationSurface {...props} />);
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("History is unavailable");
    expect(alert.textContent).toContain("重新加载后即可继续此对话。");
    expect(screen.getByRole("textbox").hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "发送消息" }).hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      false,
    );
    fireEvent.keyDown(screen.getByRole("button", { name: "选择 Harness" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Claude" }));
    expect(props.onHarnessChange).toHaveBeenCalledWith("claude");
    log.mockRestore();
  });

  it("retries failed history without replacing the chat", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const conflict = "History is unavailable";
    gateway.sessionsHistory.mockRejectedValueOnce(new Error(conflict));
    render(
      <ConversationSurface
        {...props}
        panelOpen
        sidePanel={<aside>来源面板</aside>}
        source={{ resource: "urn:swarmx:execution:11111111-1111-4111-8111-111111111111" }}
      />,
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("无法加载对话历史");
    expect(alert.textContent).toContain(conflict);
    expect(alert.textContent).toContain("重新加载后即可继续此对话。");
    expect(alert.textContent).not.toContain('{"error":');
    const input = screen.getByRole("textbox") as HTMLTextAreaElement;
    await act(async () =>
      window.dispatchEvent(new CustomEvent("swarmx:compose", { detail: "保留来源草稿" })),
    );
    const failedRetry = Promise.withResolvers<unknown>();
    gateway.sessionsHistory.mockReturnValueOnce(failedRetry.promise);
    const retry = screen.getByRole("button", { name: "重新加载历史" });
    fireEvent.click(retry);
    expect(retry.hasAttribute("disabled")).toBe(true);
    fireEvent.click(retry);
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(2);
    await act(async () => failedRetry.reject(new Error("Still unavailable")));
    expect((await screen.findByRole("alert")).textContent).toContain("Still unavailable");
    expect(input.hasAttribute("disabled")).toBe(true);
    expect(retry.hasAttribute("disabled")).toBe(false);
    gateway.sessionsHistory.mockResolvedValueOnce({
      supported: true,
      messages: [
        { id: "old-user", role: "user", content: "原来的问题" },
        { id: "old-answer", role: "assistant", content: "原来的回答" },
      ],
    });
    fireEvent.click(retry);
    await screen.findByText("原来的回答");
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("textbox")).toBe(input);
    expect(input.value).toBe("保留来源草稿");
    expect(input.hasAttribute("disabled")).toBe(false);
    expect(screen.getByText("来源面板")).toBeTruthy();
    expect(screen.getByRole("button", { name: "发送消息" }).hasAttribute("disabled")).toBe(false);
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(3);
    log.mockRestore();
  });

  it("keeps loaded history visible when another Codex instance blocks a send", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [{ id: "saved", role: "assistant", content: "已加载的历史" }],
    });
    render(<ConversationSurface {...props} />);
    await screen.findByText("已加载的历史");
    const conflict = "Internal error: thread session already has an active writer";
    respond(started, { type: "RUN_ERROR", message: conflict });
    await send("继续");
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(conflict);
    expect(alert.textContent).toContain(
      "该对话的写入权限正被另一个 Codex 实例持有，暂时无法在这里发送消息；历史记录仍可查看。",
    );
    expect(screen.getByText("已加载的历史")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "重新加载历史" })).toBeNull();
    log.mockRestore();
  });

  it("restores the same normal chat after closing a source pane without losing history or draft", async () => {
    gateway.sessionsHistory.mockResolvedValue({
      supported: true,
      messages: [{ id: "old", role: "assistant", content: "已有对话" }],
    });
    const { rerender } = render(<ConversationSurface {...props} />);
    const message = (await screen.findByText("已有对话")).closest(".assistant-message");
    const draft = screen.getByRole("textbox");
    fireEvent.change(draft, { target: { value: "保留草稿" } });
    await act(async () =>
      rerender(
        <ConversationSurface
          {...props}
          panelOpen
          sidePanel={<TracePanel />}
          source={{ resource: "urn:swarmx:execution:11111111-1111-4111-8111-111111111111" }}
        />,
      ),
    );
    expect(screen.getByText("运行与调用")).toBeTruthy();
    expect(draft.getAttribute("placeholder")).toBe("询问这些记录…");
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("保留草稿");
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
    await act(async () => i18n.changeLanguage("en"));
    expect(screen.getByRole("button", { name: "Send message" })).toBeTruthy();
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("保留草稿");
    await act(async () =>
      window.dispatchEvent(new CustomEvent("swarmx:compose", { detail: "Edit artifact abc" })),
    );
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe(
      "保留草稿\n\nEdit artifact abc",
    );
    rerender(<ConversationSurface {...props} />);
    expect(screen.getByRole("textbox")).toBe(draft);
    expect(screen.getByText("已有对话").closest(".assistant-message")).toBe(message);
    expect(draft.getAttribute("placeholder")).toBe("Describe a task or ask a question…");
    expect(screen.queryByRole("button", { name: "Add source context" })).toBeNull();
    expect((draft as HTMLTextAreaElement).value).toBe("保留草稿\n\nEdit artifact abc");
    expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  });
});

it("allows resumed conversation when the agent advertises no history replay", async () => {
  gateway.sessionsHistory.mockResolvedValue({ supported: false });
  const threadId = "acp:geepilot:existing";
  const incoming = { type: "RUN_STARTED", threadId, runId: "run" };
  const done = { type: "RUN_FINISHED", threadId, runId: "run" };
  render(<ConversationSurface {...props} harness="acp" threadId={threadId} />);
  const notice = await screen.findByText(
    "此 Agent 不提供历史消息；你可以继续对话，这里仅显示本次打开后的消息。",
  );
  expect(notice).toBeTruthy();
  expect(screen.queryByRole("alert")).toBeNull();
  expect(screen.queryByText("今天想探索什么？")).toBeNull();
  expect(screen.queryByRole("button", { name: "重新加载历史" })).toBeNull();
  respond(
    incoming,
    { type: "TEXT_MESSAGE_START", messageId: "answer", role: "assistant" },
    { type: "TEXT_MESSAGE_CONTENT", messageId: "answer", delta: "Current reply" },
    { type: "TEXT_MESSAGE_END", messageId: "answer" },
    done,
  );
  await send("Continue existing session");
  expect(await screen.findByText("Current reply")).toBeTruthy();
  expect(gateway.aguiStart).toHaveBeenLastCalledWith(
    expect.objectContaining({ input: expect.objectContaining({ threadId }) }),
  );
  expect(
    screen.getByText("此 Agent 不提供历史消息；你可以继续对话，这里仅显示本次打开后的消息。"),
  ).toBeTruthy();
  expect(gateway.sessionsHistory).toHaveBeenCalledTimes(1);
  expect(gateway.sessionsCreate).not.toHaveBeenCalled();
});
