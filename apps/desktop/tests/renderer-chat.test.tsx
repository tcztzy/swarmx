// @vitest-environment jsdom

import { type AGUIEvent, EventSchemas, EventType } from "@ag-ui/core";
import { EventEncoder } from "@ag-ui/encoder";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ExecutionRecord } from "../src/execution-record.js";
import { ConversationSurface, scienceTarget } from "../src/renderer/chat.js";
import { i18n } from "../src/renderer/i18n.js";
import { TracePanel } from "../src/renderer/trace.js";

const fetchMock = vi.fn<typeof fetch>();
const catalogMock = vi.fn<typeof fetch>();
const logsMock = vi.fn<typeof fetch>();
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
const encoder = new EventEncoder();
const props = {
  agentId: "swarm",
  harness: "codex",
  harnesses: ["codex", "claude"],
  onHarnessChange: vi.fn(),
  harnessDisabled: false,
  threadId: "codex:session",
  workspace: "research",
};
const started = { type: "RUN_STARTED", threadId: props.threadId, runId: "run" };
const finished = { type: "RUN_FINISHED", threadId: props.threadId, runId: "run" };

function events(...items: unknown[]) {
  return items.map((item) => encoder.encode(EventSchemas.parse(item))).join("");
}

function response(...items: unknown[]) {
  return new Response(events(...items), { headers: { "content-type": "text/event-stream" } });
}

beforeEach(async () => {
  await i18n.changeLanguage("zh");
  vi.stubGlobal("fetch", (...args: Parameters<typeof fetch>) =>
    String(args[0]).startsWith("/api/v1/models")
      ? catalogMock(...args)
      : String(args[0]).startsWith("/api/v1/logs")
        ? logsMock(...args)
        : fetchMock(...args),
  );
  catalogMock.mockImplementation(async () => Response.json(catalog));
  logsMock.mockImplementation(async () =>
    Response.json({ events: [], nextAfter: 0, activeRunIds: [] }),
  );
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
  fetchMock.mockResolvedValue(Response.json([]));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
  fetchMock.mockReset();
  catalogMock.mockReset();
  logsMock.mockReset();
});

async function send(text: string) {
  const input = await screen.findByRole("textbox", { name: "发送消息" });
  await waitFor(() => expect(input.hasAttribute("disabled")).toBe(false));
  fireEvent.change(input, { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
}

async function agentCard(name: string) {
  const title = await screen.findByText(name);
  const card = title.closest("details");
  if (!card) throw new Error("Missing child card");
  return card;
}

function delegationLog() {
  const records: ExecutionRecord[] = [];
  const starts = new Map<string, string>();
  function append(
    name: string,
    event: AGUIEvent,
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
      type: EventType.TOOL_CALL_START,
      toolCallId: `call-${name}`,
      toolCallName: "swarm",
    });
    append(
      parent,
      {
        type: EventType.TOOL_CALL_ARGS,
        toolCallId: `call-${name}`,
        delta: JSON.stringify({ action: "send_message", agentId: name, text: `任务 ${name}` }),
      },
      {},
      cause,
    );
    const id = append(
      name,
      {
        type: EventType.RUN_STARTED,
        threadId: `claude:${name}`,
        runId: `run-${name}`,
        input: {
          ...runLogInput(name),
          messages: [{ id: `prompt-${name}`, role: "user", content: `任务 ${name}` }],
        },
      },
      {},
      cause,
    );
    starts.set(name, id);
  }
  function finish(name: string, interruptionRequested = false) {
    append(name, {
      type: EventType.RUN_FINISHED,
      threadId: `claude:${name}`,
      runId: `run-${name}`,
      result: { interruptionRequested },
    });
  }
  starts.set(
    "parent",
    append("parent", {
      type: EventType.RUN_STARTED,
      threadId: props.threadId,
      runId: "run-parent",
    }),
  );
  child("worker-a");
  child("worker-b");
  return { records, append, child, finish };
}

function runLogInput(name: string) {
  return {
    threadId: `claude:${name}`,
    runId: `run-${name}`,
    tools: [],
    context: [],
    state: {},
    forwardedProps: {},
  };
}

describe("assistant-ui conversation", () => {
  it("does not count an ACP token limit as a completed child", async () => {
    const log = delegationLog();
    log.append("worker-a", {
      type: EventType.RUN_FINISHED,
      threadId: "claude:worker-a",
      runId: "run-worker-a",
      result: { stopReason: "max_tokens" },
    });
    log.finish("worker-b");
    logsMock.mockResolvedValue(
      Response.json({ events: log.records, nextAfter: log.records.length, activeRunIds: [] }),
    );
    render(<ConversationSurface {...props} />);
    expect(within(await agentCard("worker-a")).getByText("已停止")).toBeTruthy();
    expect(within(await agentCard("worker-b")).getByText("已完成")).toBeTruthy();
  });
  it("follows journal pagination and refreshes a running child from the last cursor", async () => {
    const log = delegationLog();
    while (log.records.length < 200)
      log.append("worker-a", {
        type: EventType.RAW,
        source: "claude",
        event: { chunk: log.records.length },
      });
    log.finish("worker-b");
    let activeRunIds = ["run-worker-a"];
    logsMock.mockImplementation(async (input) => {
      const after = Number(new URL(String(input), "http://localhost").searchParams.get("after"));
      const events = log.records.filter((record) => record.seq > after).slice(0, 200);
      return Response.json({ events, nextAfter: events.at(-1)?.seq ?? after, activeRunIds });
    });
    render(<ConversationSurface {...props} />);
    expect(within(await agentCard("worker-b")).getByText("已完成")).toBeTruthy();
    expect(String(logsMock.mock.calls[1]?.[0])).toContain("after=200");
    const stop = log.append("worker-a", {
      type: EventType.CUSTOM,
      name: "swarmx.run.interrupt_requested",
      value: {},
    });
    log.append(
      "worker-a",
      {
        type: EventType.CUSTOM,
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
    expect(String(logsMock.mock.calls[2]?.[0])).toContain("after=201");
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
      { type: EventType.RAW, source: "claude", event: { original: "native payload" } },
      { "gen_ai.response.model": "actual-model" },
    );
    log.append("worker-b", {
      type: EventType.TEXT_MESSAGE_CHUNK,
      messageId: "answer",
      role: "assistant",
      delta: "**审查完成**",
    });
    log.finish("worker-b");
    log.child("nested-reviewer", "worker-a");
    logsMock.mockImplementation(async () =>
      Response.json({
        events: log.records,
        nextAfter: log.records.length,
        activeRunIds: ["run-worker-a", "run-nested-reviewer"],
      }),
    );
    fetchMock.mockResolvedValueOnce(
      Response.json([{ id: "parent", role: "assistant", content: "父会话内容" }]),
    );
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
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("sends child steering and cancellation to an execution ID and preserves rejected input", async () => {
    const log = delegationLog();
    log.finish("worker-b");
    logsMock.mockImplementation(async () =>
      Response.json({
        events: log.records,
        nextAfter: log.records.length,
        activeRunIds: ["run-worker-a"],
      }),
    );
    render(<ConversationSurface {...props} />);
    const card = await agentCard("worker-a");
    fireEvent.click(within(card).getByText("worker-a"));
    const draft = await within(card).findByRole("textbox", { name: "给 worker-a 补充指令" });
    fireEvent.change(draft, { target: { value: "补充检查误差" } });
    fetchMock.mockResolvedValueOnce(
      Response.json({ error: "Native steering rejected" }, { status: 409 }),
    );
    fireEvent.click(within(card).getByRole("button", { name: "发送指令" }));
    await within(card).findByRole("alert");
    expect((draft as HTMLTextAreaElement).value).toBe("补充检查误差");
    fetchMock.mockResolvedValueOnce(Response.json({ runId: "run-worker-a" }));
    fireEvent.click(within(card).getByRole("button", { name: "发送指令" }));
    await within(card).findByText("补充指令已发送");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/runs/run-worker-a",
      expect.objectContaining({ body: JSON.stringify({ action: "steer", text: "补充检查误差" }) }),
    );
    expect((draft as HTMLTextAreaElement).value).toBe("");
    fetchMock.mockResolvedValueOnce(Response.json({ runId: "run-worker-a" }));
    fireEvent.click(within(card).getByRole("button", { name: "停止 worker-a" }));
    await within(card).findByText("已请求停止");
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/runs/run-worker-a",
      expect.objectContaining({ body: JSON.stringify({ action: "cancel" }) }),
    );
  });

  it("restores unknown, failed and stopped runs without fabricating successful completion", async () => {
    const log = delegationLog();
    log.append("worker-b", { type: EventType.RUN_ERROR, message: "child failed" });
    log.child("stopped");
    log.finish("stopped", true);
    logsMock.mockImplementation(async () =>
      Response.json({ events: log.records, nextAfter: log.records.length, activeRunIds: [] }),
    );
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
      type: EventType.CUSTOM,
      name: "swarmx.interaction.requested",
      value: { id: "approve", title: "Permission", schema: {} },
    });
    log.finish("worker-b");
    logsMock.mockImplementation(async () =>
      Response.json({
        events: log.records,
        nextAfter: log.records.length,
        activeRunIds: ["run-worker-a"],
      }),
    );
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
    catalogMock.mockResolvedValueOnce(
      Response.json({
        ...catalog,
        modes: [
          { id: "plan", name: "Plan" },
          { id: "full", name: "Full access" },
        ],
        current: { ...catalog.current, mode: "plan" },
      }),
    );
    fetchMock.mockResolvedValueOnce(
      Response.json([{ id: "old", role: "assistant", content: "已有历史" }]),
    );
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
    fetchMock.mockResolvedValueOnce(response(started, finished));
    fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toMatchObject({
      threadId: props.threadId,
      forwardedProps: { modelName: "model-b", reasoningEffort: "max", mode: "full" },
      messages: expect.arrayContaining([
        { id: expect.any(String), role: "user", content: "保留草稿" },
      ]),
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    fireEvent.click(await screen.findByRole("option", { name: "Model C" }));
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    expect(screen.queryByRole("radiogroup", { name: "推理强度" })).toBeNull();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    fetchMock.mockResolvedValueOnce(response(started, finished));
    await send("无需推理档位");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(JSON.parse(String(fetchMock.mock.calls[2]?.[1]?.body)).forwardedProps).toEqual({
      modelName: "model-c",
      mode: "full",
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    fireEvent.click(await screen.findByRole("option", { name: "Model B" }));
    expect(screen.getByRole("combobox", { name: "选择模型" }).textContent).toContain("Max");
  });

  it("surfaces catalog failures, retries explicitly, and does not invent an empty catalog", async () => {
    catalogMock.mockResolvedValueOnce(new Response("Model catalog unavailable", { status: 503 }));
    render(<ConversationSurface {...props} />);
    await screen.findByRole("button", { name: "梳理研究思路" });
    fireEvent.click(screen.getByRole("combobox", { name: "选择模型" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Model catalog unavailable");
    catalogMock.mockResolvedValueOnce(Response.json({ models: [], current: {} }));
    fireEvent.click(screen.getByRole("button", { name: "重新加载模型" }));
    await screen.findByText("此 Harness 未提供可选模型");
    expect(screen.queryByRole("option")).toBeNull();
    expect(catalogMock).toHaveBeenCalledTimes(2);
  });
  it("hydrates native history, renders Markdown and copies the assistant reply", async () => {
    const reply =
      "## 研究进展\n\n- 已整理数据\n- 待验证结果\n\n| 内容 | 进展 |\n| --- | --- |\n| 数据 | 已完成 |";
    fetchMock.mockResolvedValueOnce(
      Response.json([
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
      ]),
    );
    render(<ConversationSurface {...props} />);
    expect(await screen.findByRole("heading", { name: "研究进展" })).toBeTruthy();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(screen.getByRole("cell", { name: "已完成" })).toBeTruthy();
    expect(screen.queryByText("正在处理…")).toBeNull();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/sessions/codex%3Asession?agent=swarm",
      undefined,
    );
    fireEvent.click(screen.getByRole("button", { name: "复制回复" }));
    await waitFor(() => expect(copy).toHaveBeenCalledWith(reply));
    expect(await screen.findByText("已复制")).toBeTruthy();
    expect(screen.queryByRole("complementary", { name: "执行轨迹" })).toBeNull();
  });

  it("fills a suggested draft without sending and streams through the existing AG-UI endpoint", async () => {
    render(<ConversationSurface {...props} />);
    fireEvent.click(await screen.findByRole("button", { name: "梳理研究思路" }));
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toContain("研究目标");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    fetchMock.mockResolvedValueOnce(
      response(
        started,
        { type: "TEXT_MESSAGE_START", messageId: "answer", role: "assistant" },
        { type: "TEXT_MESSAGE_CONTENT", messageId: "answer", delta: "这是流式回复。" },
        { type: "TEXT_MESSAGE_END", messageId: "answer" },
        finished,
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "发送消息" }));
    expect(await screen.findByText("这是流式回复。")).toBeTruthy();
    const request = fetchMock.mock.calls[1]?.[1];
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/ag-ui?agent=swarm");
    expect(JSON.parse(String(request?.body))).toMatchObject({
      threadId: "codex:session",
      messages: [{ role: "user", content: expect.stringContaining("研究目标") }],
    });
    await waitFor(() => expect(screen.queryByRole("button", { name: "停止生成" })).toBeNull());
  });

  it("shows only Stop during a run and aborts its stream when clicked", async () => {
    render(<ConversationSurface {...props} />);
    let signal: AbortSignal | null | undefined;
    fetchMock.mockImplementationOnce(async (_url, init) => {
      signal = init?.signal;
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(events(started)));
          signal?.addEventListener("abort", () =>
            controller.error(new DOMException("Stopped", "AbortError")),
          );
        },
      });
      return new Response(stream, { headers: { "content-type": "text/event-stream" } });
    });
    await send("持续运行");
    const stop = await screen.findByRole("button", { name: "停止生成" });
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      true,
    );
    expect(screen.getByRole("combobox", { name: "选择模型" }).hasAttribute("disabled")).toBe(true);
    expect(screen.queryByRole("button", { name: "发送消息" })).toBeNull();
    fireEvent.click(stop);
    await waitFor(() => expect(signal?.aborted).toBe(true));
    expect(await screen.findByRole("button", { name: "发送消息" })).toBeTruthy();
  });

  it("requires interaction fields, preserves a declined boolean, and resumes the native request", async () => {
    render(<ConversationSurface {...props} />);
    fetchMock.mockResolvedValueOnce(
      response(started, {
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
                properties: {
                  count: { type: "integer", title: "样本数量", minimum: 1 },
                  allow: { type: "boolean", title: "允许写入" },
                },
                required: ["count", "allow"],
              },
            },
          ],
        },
      }),
    );
    await send("分析数据");
    const number = await screen.findByRole("spinbutton", { name: "样本数量" });
    expect(screen.getByRole("textbox").hasAttribute("disabled")).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "继续" }));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    fireEvent.change(number, { target: { value: "3" } });
    fetchMock.mockResolvedValueOnce(response(started, finished));
    fireEvent.click(screen.getByRole("button", { name: "继续" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
    expect(JSON.parse(String(fetchMock.mock.calls[2]?.[1]?.body))).toMatchObject({
      resume: [
        { interruptId: "approval", status: "resolved", payload: { count: 3, allow: false } },
      ],
    });
    await waitFor(() => expect(screen.queryByText("确认分析参数")).toBeNull());
  });

  it("shows history failures and prevents a send with missing native history", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    fetchMock.mockResolvedValueOnce(new Response("History is unavailable", { status: 503 }));
    render(<ConversationSurface {...props} />);
    expect((await screen.findByRole("alert")).textContent).toContain("History is unavailable");
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

  it("opens the trace pane without remounting the conversation or losing the draft", async () => {
    const { rerender } = render(<ConversationSurface {...props} />);
    await screen.findByRole("button", { name: "梳理研究思路" });
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "保留草稿" } });
    await act(async () =>
      rerender(<ConversationSurface {...props} panelOpen sidePanel={<TracePanel />} />),
    );
    expect(screen.getByText("任务开始后，执行轨迹会显示在这里。")).toBeTruthy();
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("保留草稿");
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => i18n.changeLanguage("en"));
    expect(screen.getByRole("button", { name: "Send message" })).toBeTruthy();
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("保留草稿");
    await act(async () =>
      window.dispatchEvent(new CustomEvent("swarmx:compose", { detail: "Edit artifact abc" })),
    );
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe(
      "保留草稿\n\nEdit artifact abc",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("opens exact scientific artifacts from structured and JSON tool results", () => {
    const artifact = { id: "figure", projectId: "study" };
    expect(scienceTarget({ data: { artifact } })).toEqual({
      artifactId: "figure",
      projectId: "study",
    });
    expect(scienceTarget(JSON.stringify({ data: { ...artifact, kind: "figure" } }))).toEqual({
      artifactId: "figure",
      projectId: "study",
    });
    expect(scienceTarget({ data: { id: "study", kind: "project" } })).toEqual({
      projectId: "study",
    });
    expect(scienceTarget("not a scientific result")).toEqual({});
  });
});
