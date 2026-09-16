// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { MemoryStatusSchema } from "../src/memory.js";
import { i18n, t } from "../src/renderer/i18n.js";
import { MemorySettings } from "../src/renderer/memory.js";

interface Call {
  readonly name: string;
  readonly args: { action: string; request: unknown };
}

let calls: Call[] = [];
let handler: (call: Call) => Promise<unknown> = async () => {
  throw new Error("Unexpected tool call");
};

vi.mock("../src/renderer/bridge.js", () => ({
  tool: (
    name: string,
    args: { action: string; request: unknown },
    schema: { parse(value: unknown): unknown },
  ) => handler({ name, args }).then((data) => schema.parse(data)),
}));
vi.mock("../src/renderer/research-graph.js", () => ({
  GraphView: ({ label }: { label: string }) => <div aria-label={label} />,
}));
afterEach(() => {
  cleanup();
  calls = [];
});

it.each(["zh", "en"])(
  "edits notes, resolves approvals and loads prerequisites in %s",
  async (language) => {
    await i18n.changeLanguage(language);
    const revision = `sha256:${"0".repeat(64)}`;
    const id = "11111111-1111-4111-8111-111111111111";
    const status = MemoryStatusSchema.parse({
      settings: {},
      note: { content: "Chinese first", revision, limit: 1375 },
      pending: [
        {
          id,
          createdAt: "2026-09-06T00:00:00Z",
          origin: "review",
          sessionId: "codex:test",
          operation: {
            action: "update_core_memory",
            request: {
              content: "Use pinned dependencies",
              expectedRevision: revision,
            },
          },
        },
      ],
      review: { state: "idle", message: "", sessionId: null },
    });
    const nodes = ["environment", "analysis"].map((nodeId) => ({
      id: nodeId,
      revision,
      title: nodeId,
      description: nodeId,
      type: "Playbook",
      status: "draft",
      stale: nodeId === "analysis",
    }));
    const graph = {
      nodes,
      edges: [{ source: "analysis", target: "environment", revision, stale: true }],
    };
    let conflict = true;
    handler = async ({ args }) => {
      calls.push({ name: "memory", args });
      switch (args.action) {
        case "memory_status":
          return status;
        case "graph_memory":
          return graph;
        case "load_memory":
          return {
            graph,
            concepts: nodes.map((node) => ({
              id: node.id,
              revision,
              metadata: node,
              body: `${node.id} source`,
            })),
          };
        case "update_core_memory":
          return { content: "Prefer bilingual explanations", revision, limit: 1375 };
        case "memory_review":
          status.review = { state: "completed", sessionId: "codex:test", message: "0" };
          return { state: "running" };
        case "memory_decide":
          if (conflict) throw new Error("Core memory changed; read it again before editing.");
          status.pending = [];
          return { decision: "approve" };
        default:
          throw new Error(`Unexpected memory action ${args.action}`);
      }
    };

    render(<MemorySettings sessionId="codex:test" />);
    await screen.findByRole("heading", {
      name: language === "zh" ? "记忆与知识" : "Memory and knowledge",
    });
    fireEvent.change(await screen.findByLabelText(t("用户偏好")), {
      target: { value: "Prefer bilingual explanations" },
    });
    fireEvent.click(screen.getByRole("button", { name: t("保存笔记") }));
    await waitFor(() =>
      expect(calls).toContainEqual({
        name: "memory",
        args: {
          action: "update_core_memory",
          request: {
            content: "Prefer bilingual explanations",
            expectedRevision: revision,
          },
        },
      }),
    );
    await screen.findByText("Use pinned dependencies");
    expect(
      screen.getByText(
        new RegExp(
          new Date("2026-09-06T00:00:00Z").toLocaleDateString(language).replaceAll("/", "\\/"),
        ),
      ),
    ).toBeTruthy();
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
