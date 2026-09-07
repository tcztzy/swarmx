// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MemoryStatusSchema } from "../src/memory.js";
import { i18n, t } from "../src/renderer/i18n.js";
import { MemorySettings } from "../src/renderer/memory.js";

vi.mock("../src/renderer/research-graph.js", () => ({
  GraphView: ({ label }: { label: string }) => <div aria-label={label} />,
}));
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it.each(["zh", "en"])(
  "edits notes, resolves approvals and loads prerequisites in %s",
  async (language) => {
    await i18n.changeLanguage(language);
    const revision = `sha256:${"0".repeat(64)}`;
    const id = "11111111-1111-4111-8111-111111111111";
    const status = MemoryStatusSchema.parse({
      settings: {},
      notes: [{ target: "workspace", content: "Chinese first", revision, limit: 2200 }],
      pending: [
        {
          id,
          createdAt: "2026-09-06T00:00:00Z",
          origin: "review",
          sessionId: "codex:test",
          operation: {
            action: "update_core_memory",
            request: {
              target: "workspace",
              content: "Use pinned dependencies",
              expectedRevision: revision,
            },
          },
        },
      ],
      review: { state: "idle", message: "", sessionId: null },
    });
    const nodes = ["environment", "analysis"].map((id) => ({
      id,
      revision,
      title: id,
      description: id,
      type: "Playbook",
      scope: "workspace",
      status: "draft",
      stale: id === "analysis",
    }));
    const graph = {
      nodes,
      edges: [{ source: "analysis", target: "environment", revision, stale: true }],
    };
    let conflict = true;
    const fetchMock = vi.fn<typeof fetch>(async (path, init) => {
      if (String(path).includes("/pending/")) {
        if (conflict)
          return Response.json(
            { error: "Core memory changed; read it again before editing." },
            { status: 409 },
          );
        status.pending = [];
        return Response.json({ action: "approve" });
      }
      if (path === "/api/v1/memory/notes") return Response.json({});
      if (path === "/api/v1/memory/settings") return Response.json(JSON.parse(String(init?.body)));
      if (path === "/api/v1/memory/review") {
        status.review = { state: "completed", sessionId: "codex:test", message: "0" };
        return Response.json({ state: "running" }, { status: 202 });
      }
      if (path === "/api/v1/memory/graph") return Response.json(graph);
      if (String(path).startsWith("/api/v1/memory/concept?"))
        return Response.json({
          graph,
          concepts: nodes.map((node) => ({
            id: node.id,
            revision,
            metadata: node,
            body: `${node.id} source`,
          })),
        });
      if (path === "/api/v1/memory") return Response.json(status);
      throw new Error(`Unexpected request: ${path}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<MemorySettings scope="project" sessionId="codex:test" />);
    await screen.findByRole("heading", {
      name: language === "zh" ? "记忆与知识" : "Memory and knowledge",
    });
    await screen.findByText("Use pinned dependencies");
    expect(
      screen.getByText(
        new RegExp(
          new Date("2026-09-06T00:00:00Z").toLocaleDateString(language).replaceAll("/", "\\/"),
        ),
      ),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText(t("工作区笔记")), {
      target: { value: "Prefer bilingual explanations" },
    });
    fireEvent.click(screen.getByRole("button", { name: t("保存笔记") }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/v1/memory/notes",
        expect.objectContaining({
          method: "PUT",
          body: JSON.stringify({
            target: "workspace",
            content: "Prefer bilingual explanations",
            expectedRevision: revision,
          }),
        }),
      ),
    );
    await waitFor(() =>
      expect(
        (screen.getByRole("button", { name: t("批准保存") }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole("button", { name: t("批准保存") }));
    expect((await screen.findByRole("alert")).textContent).toContain("Core memory changed");
    expect(screen.getByText("Use pinned dependencies")).toBeTruthy();
    conflict = false;
    fireEvent.click(screen.getByRole("button", { name: t("批准保存") }));
    await screen.findByText(t("没有待确认的修改。"));
    fireEvent.click(screen.getByRole("button", { name: t("立即复盘当前对话") }));
    await screen.findByText(t("复盘完成：{{count}} 项修改", { count: 0 }));
    fireEvent.click(screen.getByRole("button", { name: t("浏览 Vault 依赖图") }));
    const select = await screen.findByLabelText(t("查看概念"));
    expect(select.textContent).toContain(t("需要复核"));
    fireEvent.change(select, { target: { value: "analysis" } });
    await screen.findByText("analysis source");
    expect(screen.getByText("environment source")).toBeTruthy();
  },
);
