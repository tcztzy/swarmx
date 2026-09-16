// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HarnessPicker, type HarnessProps } from "../src/renderer/agent-controls.js";
import { App } from "../src/renderer/app.js";
import { i18n } from "../src/renderer/i18n.js";
import { DEFAULT_POLICY } from "../src/settings.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

vi.mock("../src/renderer/trace.js", () => ({ TracePanel: () => <p>Trace fixture</p> }));

vi.mock("../src/renderer/chat.js", () => ({
  ConversationSurface: ({
    threadId,
    agentId,
    harnessDisabled,
    sidePanel,
    panelOpen,
    source,
    ...harness
  }: HarnessProps & {
    threadId: string;
    agentId: string;
    harnessDisabled: boolean;
    sidePanel?: ReactNode;
    panelOpen?: boolean;
    source?: { resource: string };
  }) => (
    <>
      <div data-testid="conversation">
        {agentId}:{threadId}
      </div>
      <HarnessPicker {...harness} disabled={harnessDisabled} />
      <input aria-label="draft fixture" defaultValue="保留研究内容" />
      {source && <span>{source.resource}</span>}
      {panelOpen && sidePanel}
    </>
  ),
}));

const bootstrap = {
  agents: ["swarm", "codex", "claude"],
  defaultHarness: "codex",
  language: null,
  sessions: [
    { sessionId: "codex:one", title: "Review RNA results", updatedAt: "2026-09-05T10:00:00Z" },
    { sessionId: "codex:two", title: "整理实验记录" },
  ],
  cwd: "/research",
};
const settings = {
  policy: DEFAULT_POLICY,
  environment: null,
  cwd: "/research",
};
const memory = {
  action: "memory_status",
  data: {
    settings: {
      enabled: true,
      autoReview: true,
      writeApproval: false,
      reviewInterval: 10,
      reviewHarness: "codex",
    },
    note: { content: "", revision: `sha256:${"0".repeat(64)}`, limit: 1375 },
    pending: [],
    review: { state: "idle", message: "", sessionId: null },
  },
};
let gateway: BridgeHarness;

beforeEach(async () => {
  await i18n.changeLanguage("zh");
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  gateway = installBridge();
  gateway.bootstrap.mockResolvedValue(bootstrap);
  gateway.sessionsList.mockResolvedValue(bootstrap.sessions);
  gateway.sessionsCreate.mockResolvedValue({ sessionId: "codex:new" });
  gateway.settingsRead.mockResolvedValue(settings);
  gateway.environmentRead.mockResolvedValue({
    state: "missing",
    environment: null,
    log: "",
    activeProcesses: 0,
  });
  gateway.scienceWorkspace.mockResolvedValue({
    projects: [],
    notebooks: [],
    artifacts: [],
    documents: [],
    figures: [],
    records: [],
    relations: [],
    experiments: [],
    runs: [],
    exports: [],
  });
  gateway.tool.mockResolvedValue(memory);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("desktop task navigation", () => {
  it("keeps chat mounted beside assets/observability and restores it after full-page settings and language changes", async () => {
    render(<App />);
    const conversation = await screen.findByTestId("conversation");
    const draft = screen.getByRole("textbox", { name: "draft fixture" });
    fireEvent.click(screen.getByRole("button", { name: "科研资产", exact: true }));
    await screen.findByRole("complementary", { name: "科研资产侧栏" });
    expect(screen.getByTestId("conversation")).toBe(conversation);
    fireEvent.click(screen.getByRole("button", { name: "观测与溯源", exact: true }));
    await screen.findByRole("tab", { name: "RO-Crate 图谱" });
    expect(screen.getByText("Trace fixture")).toBeTruthy();
    await act(async () =>
      window.dispatchEvent(
        new CustomEvent("swarmx:open-research", {
          detail: { source: { resource: "sx:a/figure@1" } },
        }),
      ),
    );
    await screen.findByRole("heading", { name: "来源检查" });
    expect(screen.getByText("sx:a/figure@1")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "关闭侧栏" }));
    expect(screen.queryByText("sx:a/figure@1")).toBeNull();
    expect(screen.getByTestId("conversation")).toBe(conversation);
    expect(screen.getByRole("textbox", { name: "draft fixture" })).toBe(draft);
    fireEvent.click(screen.getByRole("button", { name: "观测与溯源", exact: true }));
    await screen.findByRole("tab", { name: "RO-Crate 图谱" });
    fireEvent.click(screen.getByRole("button", { name: "展开侧栏" }));
    fireEvent.click(screen.getByRole("button", { name: "设置", exact: true }));
    await screen.findByRole("heading", { name: "设置", level: 2 });
    await screen.findByRole("heading", { name: "执行与权限" });
    expect(screen.getByRole("heading", { name: "运行环境" })).toBeTruthy();
    expect(await screen.findByLabelText("用户偏好")).toBeTruthy();
    expect(screen.getByText("/research")).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "draft fixture" })).toBeNull();
    expect(screen.getByTestId("conversation")).toBe(conversation);
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    await screen.findByRole("button", { name: "Back to conversation" });
    expect(document.documentElement.lang).toBe("en");
    expect(i18n.language).toBe("en");
    expect(gateway.languageWrite).toHaveBeenCalledWith({ language: "en" });
    fireEvent.click(screen.getByRole("button", { name: "Back to conversation" }));
    expect(screen.getByRole("textbox", { name: "draft fixture" })).toBe(draft);
    expect((draft as HTMLInputElement).value).toBe("保留研究内容");
    expect(screen.getByRole("button", { name: "Observe", exact: true })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "RO-Crate graph" })).toBeTruthy();
    expect(gateway.bootstrap).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Settings", exact: true })).toBeTruthy();
  });
  it("offers task navigation and one settings entry without directory management", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    expect(
      screen.getAllByRole("navigation").map((element) => element.getAttribute("aria-label")),
    ).toEqual(["任务列表"]);
    expect(screen.getAllByRole("button", { name: "设置", exact: true })).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "设置", exact: true }));
    await screen.findByText("/research");
    expect(screen.getByRole("heading", { name: "工作目录" })).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "工作目录" })).toBeNull();
  });
  it("searches native titles, selects tasks, and creates a session through the IPC bridge", async () => {
    render(<App />);
    await screen.findByRole("button", { name: /Review RNA results/ });
    fireEvent.change(screen.getByRole("searchbox", { name: "搜索任务" }), {
      target: { value: " rna " },
    });
    expect(screen.queryByRole("button", { name: "整理实验记录" })).toBeNull();
    expect(screen.getByRole("button", { name: /Review RNA results/ })).toBeTruthy();
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "missing" } });
    expect(screen.getByText("没有匹配的任务")).toBeTruthy();
    fireEvent.change(screen.getByRole("searchbox"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "整理实验记录" }));
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:two");

    gateway.sessionsCreate.mockResolvedValueOnce({ sessionId: "codex:new" });
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    await waitFor(() =>
      expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:new"),
    );
    expect(gateway.sessionsCreate).toHaveBeenLastCalledWith({ agent: "swarm" });
  });

  it("keeps create errors visible alongside the selected conversation", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    gateway.sessionsCreate.mockRejectedValueOnce(new Error("Native Agent unavailable"));
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Native Agent unavailable");
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:one");
  });

  it("ignores late session lists from a previously selected Agent", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    const old = Promise.withResolvers<unknown>();
    gateway.sessionsList.mockReturnValueOnce(old.promise);
    fireEvent.keyDown(screen.getByRole("button", { name: "选择 Harness" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Claude" }));
    await waitFor(() => expect(gateway.sessionsList).toHaveBeenCalledWith({ agent: "claude" }));
    gateway.sessionsList.mockResolvedValueOnce([{ sessionId: "codex:now", title: "当前任务" }]);
    fireEvent.keyDown(screen.getByRole("button", { name: "选择 Harness" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Codex" }));
    await screen.findByRole("button", { name: "当前任务" });
    await act(async () => old.resolve([{ sessionId: "claude:stale", title: "过期任务" }]));
    expect(screen.queryByRole("button", { name: "过期任务" })).toBeNull();
    expect(screen.getByTestId("conversation").textContent).toBe("codex:codex:now");
  });

  it("locks Harness until a pending native session creation completes", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    const created = Promise.withResolvers<unknown>();
    gateway.sessionsCreate.mockReturnValueOnce(created.promise);
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      true,
    );
    await act(async () => created.resolve({ sessionId: "codex:new" }));
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:new");
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      false,
    );
  });

  it("refreshes native titles without replacing the selected task and can reopen navigation", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    fireEvent.click(screen.getByRole("button", { name: "整理实验记录" }));
    gateway.bootstrap.mockResolvedValueOnce(bootstrap);
    fireEvent.click(screen.getByRole("button", { name: "刷新任务" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "刷新任务" }).hasAttribute("disabled")).toBe(false),
    );
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:two");
    fireEvent.click(screen.getByRole("button", { name: "收起侧栏" }));
    expect(screen.queryByRole("searchbox")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "展开侧栏" }));
    expect(screen.getByRole("searchbox", { name: "搜索任务" })).toBeTruthy();
  });
});
