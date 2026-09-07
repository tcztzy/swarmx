// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HarnessPicker, type HarnessProps } from "../src/renderer/agent-controls.js";
import { App } from "../src/renderer/app.js";
import { i18n } from "../src/renderer/i18n.js";
import { DEFAULT_POLICY } from "../src/settings.js";

vi.mock("../src/renderer/trace.js", () => ({ TracePanel: () => <p>Trace fixture</p> }));

vi.mock("../src/renderer/chat.js", () => ({
  ConversationSurface: ({
    threadId,
    agentId,
    harnessDisabled,
    sidePanel,
    panelOpen,
    ...harness
  }: HarnessProps & {
    threadId: string;
    agentId: string;
    harnessDisabled: boolean;
    sidePanel?: ReactNode;
    panelOpen?: boolean;
  }) => (
    <>
      <div data-testid="conversation">
        {agentId}:{threadId}
      </div>
      <HarnessPicker {...harness} disabled={harnessDisabled} />
      <input aria-label="draft fixture" defaultValue="保留研究内容" />
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
  workspace: { id: "workspace", label: "research", root: "/research" },
  projects: [{ id: "workspace", label: "research", root: "/research" }],
};
const fetchMock = vi.fn<typeof fetch>();

beforeEach(async () => {
  await i18n.changeLanguage("zh");
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  fetchMock.mockResolvedValue(Response.json(bootstrap));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

describe("desktop task navigation", () => {
  it("keeps chat mounted beside assets/observability and restores it after full-page settings and language changes", async () => {
    fetchMock.mockImplementation(async (path, init) =>
      Response.json(
        path === "/api/v1/language"
          ? JSON.parse(String(init?.body))
          : path === "/api/v1/bootstrap"
            ? bootstrap
            : path === "/api/v1/settings"
              ? {
                  policy: DEFAULT_POLICY,
                  environment: null,
                  workspace: { ...bootstrap.workspace, root: "/research" },
                }
              : path === "/api/v1/memory"
                ? {
                    settings: {
                      enabled: true,
                      autoReview: true,
                      writeApproval: false,
                      reviewInterval: 10,
                      reviewHarness: "codex",
                    },
                    notes: [],
                    pending: [],
                    review: { state: "idle", message: "", sessionId: null },
                  }
                : path === "/api/v1/environment"
                  ? { state: "missing", environment: null, log: "", activeProcesses: 0 }
                  : {
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
                    },
      ),
    );
    render(<App />);
    const conversation = await screen.findByTestId("conversation");
    const draft = screen.getByRole("textbox", { name: "draft fixture" });
    fireEvent.click(screen.getByRole("button", { name: "科研资产", exact: true }));
    await screen.findByRole("complementary", { name: "科研资产侧栏" });
    expect(screen.getByTestId("conversation")).toBe(conversation);
    fireEvent.click(screen.getByRole("button", { name: "观测与溯源", exact: true }));
    await screen.findByRole("tab", { name: "RO-Crate 图谱" });
    expect(screen.getByText("Trace fixture")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "通用设置" }));
    await screen.findByRole("heading", { name: "通用设置", level: 2 });
    expect(screen.queryByText("项目目录")).toBeNull();
    expect(screen.queryByText("运行环境")).toBeNull();
    expect(screen.queryByRole("textbox", { name: "draft fixture" })).toBeNull();
    expect(screen.getByTestId("conversation")).toBe(conversation);
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    await screen.findByRole("button", { name: "Back to conversation" });
    expect(document.documentElement.lang).toBe("en");
    expect(i18n.language).toBe("en");
    expect(
      fetchMock.mock.calls.some(
        ([path, init]) =>
          path === "/api/v1/language" && JSON.parse(String(init?.body)).language === "en",
      ),
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Back to conversation" }));
    expect(screen.getByRole("textbox", { name: "draft fixture" })).toBe(draft);
    expect((draft as HTMLInputElement).value).toBe("保留研究内容");
    expect(screen.getByRole("button", { name: "Observe", exact: true })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "RO-Crate graph" })).toBeTruthy();
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/v1/bootstrap")).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Project settings" }));
    await screen.findByText("/research");
    expect(screen.queryByRole("textbox", { name: "Directory path" })).toBeNull();
    expect(screen.queryByLabelText("Interface language")).toBeNull();
    expect(screen.getByRole("heading", { name: "Environment" })).toBeTruthy();
  });
  it("searches native titles, selects tasks, and creates a session through the existing endpoint", async () => {
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

    fetchMock.mockResolvedValueOnce(Response.json({ sessionId: "codex:new" }));
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    await waitFor(() =>
      expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:new"),
    );
    expect(fetchMock).toHaveBeenLastCalledWith("/api/v1/sessions?agent=swarm", { method: "POST" });
  });

  it("keeps create errors visible alongside the selected conversation", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    fetchMock.mockResolvedValueOnce(new Response("Native Agent unavailable", { status: 503 }));
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Native Agent unavailable");
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:one");
  });

  it("ignores late session lists from a previously selected Agent", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    const old = Promise.withResolvers<Response>();
    fetchMock.mockReturnValueOnce(old.promise);
    fireEvent.keyDown(screen.getByRole("button", { name: "选择 Harness" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Claude" }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("/api/v1/sessions?agent=claude", undefined),
    );
    fetchMock.mockResolvedValueOnce(Response.json([{ sessionId: "codex:now", title: "当前任务" }]));
    fireEvent.keyDown(screen.getByRole("button", { name: "选择 Harness" }), { key: "Enter" });
    fireEvent.click(await screen.findByRole("menuitemradio", { name: "Codex" }));
    await screen.findByRole("button", { name: "当前任务" });
    await act(async () =>
      old.resolve(Response.json([{ sessionId: "claude:stale", title: "过期任务" }])),
    );
    expect(screen.queryByRole("button", { name: "过期任务" })).toBeNull();
    expect(screen.getByTestId("conversation").textContent).toBe("codex:codex:now");
  });

  it("locks Harness until a pending native session creation completes", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    const created = Promise.withResolvers<Response>();
    fetchMock.mockReturnValueOnce(created.promise);
    fireEvent.click(screen.getByRole("button", { name: "新建任务" }));
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      true,
    );
    await act(async () => created.resolve(Response.json({ sessionId: "codex:new" })));
    expect(screen.getByTestId("conversation").textContent).toBe("swarm:codex:new");
    expect(screen.getByRole("button", { name: "选择 Harness" }).hasAttribute("disabled")).toBe(
      false,
    );
  });

  it("refreshes native titles without replacing the selected task and can reopen navigation", async () => {
    render(<App />);
    await screen.findByTestId("conversation");
    fireEvent.click(screen.getByRole("button", { name: "整理实验记录" }));
    fetchMock.mockResolvedValueOnce(Response.json(bootstrap));
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
