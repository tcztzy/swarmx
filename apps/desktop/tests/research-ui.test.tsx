// @vitest-environment jsdom
import { RO_CRATE_CONTEXT, type RoCrateMetadataDocument } from "@swarmx/science/types";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FigureStudio } from "../src/renderer/figure-studio.js";
import { i18n } from "../src/renderer/i18n.js";
import { ResearchPanel } from "../src/renderer/research.js";
import { crateGraph } from "../src/renderer/research-graph.js";
import { SettingsPage } from "../src/renderer/settings.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const fetchMock = vi.fn<typeof fetch>();
beforeEach(async () => {
  await i18n.changeLanguage("zh");
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.resetAllMocks();
});

describe("research workbench", () => {
  it("shows recorded failure after refresh and edits the code cell instead of its output", async () => {
    const provenance = { eventId: "event", journalSeq: 1, sessionId: "renderer" };
    const identity = { createdAt: 0, updatedAt: 0, revision: 2, provenance };
    const source = "raise ValueError('invalid measurement')";
    const cell = {
      executionCount: 1,
      executionTimeMs: 10,
      outputArtifactIds: [],
      inputArtifactIds: [],
      runtimeEnvironment: {},
      relatedClaimIds: [],
      relatedExperimentIds: [],
      outputs: [],
    };
    const notebook = {
      ...identity,
      id: "notebook",
      projectId: "project",
      kind: "notebook",
      title: "Analysis",
      cells: [
        { ...cell, id: "code", kind: "code", source },
        { ...cell, id: "output", kind: "output", source: "ValueError: invalid measurement" },
      ],
    };
    const snapshot = {
      projects: [{ ...identity, id: "project", kind: "project", title: "Project" }],
      notebooks: [notebook],
      artifacts: [],
      documents: [],
      figures: [],
      records: [],
      relations: [],
      experiments: [],
      runs: [],
      exports: [],
    };
    const crate = {
      "@context": RO_CRATE_CONTEXT,
      "@graph": [
        {
          "@id": "ro-crate-metadata.json",
          "@type": "CreativeWork",
          about: { "@id": "./" },
          conformsTo: { "@id": "https://w3id.org/ro/crate/1.3" },
        },
        {
          "@id": "./",
          "@type": "Dataset",
          name: "Project",
          description: "Test fixture",
          datePublished: "2026-09-05T00:00:00Z",
          license: "MIT",
          hasPart: [{ "@id": "urn:uuid:notebook" }],
        },
        { "@id": "urn:uuid:notebook", "@type": "SoftwareSourceCode", name: "Analysis" },
      ],
    };
    fetchMock.mockImplementation(async (path) =>
      Response.json(
        String(path).startsWith("/api/v1/research-object")
          ? crate
          : String(path).startsWith("/api/v1/notebook-executions")
            ? [
                {
                  id: "execution",
                  notebookId: "notebook",
                  cellId: "code",
                  executionCount: 1,
                  status: "failed",
                  stdout: { text: "", truncated: false },
                  stderr: { text: "ValueError: invalid measurement", truncated: false },
                  outputs: [],
                  exitCode: 1,
                  signal: null,
                  durationMs: 10,
                  environment: {},
                  artifact: null,
                  provenance,
                },
              ]
            : snapshot,
      ),
    );
    const { rerender } = render(<ResearchPanel mode="assets" onClose={vi.fn()} canCompose />);
    const notebookButton = await screen.findByRole("button", { name: /^Analysis/ });
    expect(notebookButton.textContent?.replace(/\s/gu, "")).toContain("1次执行");
    fireEvent.click(screen.getByRole("button", { name: "在对话中生成" }));
    expect(screen.getByRole("status").textContent).toContain("已添加到对话草稿");
    rerender(<ResearchPanel mode="observe" onClose={vi.fn()} />);
    expect(screen.queryByText("已添加到对话草稿，请补充要求后发送。")).toBeNull();
    fireEvent.click(screen.getByRole("tab", { name: "运行记录" }));
    await screen.findByText("失败");
    expect(screen.queryByText(/尚无运行记录/)).toBeNull();
    rerender(<ResearchPanel mode="assets" onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /^Analysis/ }));
    expect(((await screen.findByLabelText("Python 图像代码")) as HTMLTextAreaElement).value).toBe(
      source,
    );
    expect((screen.getByLabelText("输出文件") as HTMLInputElement).value).toBe("");
    expect(
      (screen.getByRole("button", { name: "运行并生成图像" }) as HTMLButtonElement).disabled,
    ).toBe(true);
    const editor = screen.getByLabelText("Python 图像代码");
    fireEvent.change(editor, { target: { value: "print('unsaved correction')" } });
    rerender(<ResearchPanel mode="observe" onClose={vi.fn()} />);
    expect(screen.getByRole("tab", { name: "运行记录" })).toBeTruthy();
    expect(screen.queryByRole("textbox", { name: "Python 图像代码" })).toBeNull();
    rerender(<ResearchPanel mode="assets" onClose={vi.fn()} />);
    expect(screen.getByLabelText("Python 图像代码")).toBe(editor);
    expect((editor as HTMLTextAreaElement).value).toBe("print('unsaved correction')");
  });

  it("saves typed settings and leaves actionable setup failures visible", async () => {
    const settings = {
      policy: DEFAULT_POLICY,
      environment: null,
      workspace: { id: "workspace", label: "Research", root: "/research" },
    };
    fetchMock.mockImplementation(async (path, init) => {
      if (path === "/api/v1/memory")
        return Response.json({
          settings: {},
          notes: [],
          pending: [],
          review: { state: "idle", message: "", sessionId: null },
        });
      if (init?.method === "PUT")
        return Response.json({ policy: JSON.parse(String(init.body)), environment: null });
      if (init?.method === "POST")
        return Response.json({ error: "Docker daemon is unavailable" }, { status: 500 });
      return Response.json(
        path === "/api/v1/settings"
          ? settings
          : { state: "missing", environment: null, log: "", activeProcesses: 0 },
      );
    });
    render(<SettingsPage project={settings.workspace} />);
    await screen.findByRole("button", { name: "保存权限" });
    fireEvent.change(screen.getByLabelText("科研容器文件访问"), { target: { value: "read-only" } });
    fireEvent.click(screen.getByLabelText("修改记忆"));
    fireEvent.click(screen.getByLabelText("允许通过 Swarm 委派任务"));
    fireEvent.change(screen.getByLabelText("CPU 核数"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
    await screen.findByText("权限已保存，对下一次执行生效。");
    const saved = fetchMock.mock.calls.find(([, init]) => init?.method === "PUT");
    expect(JSON.parse(String(saved?.[1]?.body))).toEqual({
      ...DEFAULT_POLICY,
      filesystem: "read-only",
      tools: ["memory.read", "science.read", "science.write"],
      delegation: false,
      cpus: 3,
    });
    fireEvent.click(screen.getByRole("button", { name: "设置运行环境" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Docker daemon");
    expect(
      (screen.getByRole("button", { name: "设置运行环境" }) as HTMLButtonElement).disabled,
    ).toBe(false);
  });

  it("cancels a figure request and retains its source for correction", async () => {
    const notebook = {
      id: "notebook",
      projectId: "project",
      kind: "notebook" as const,
      title: "Analysis",
      cells: [],
      createdAt: 0,
      updatedAt: 0,
      revision: 1,
      provenance: { eventId: "created", journalSeq: 1, sessionId: "renderer" },
    };
    fetchMock.mockImplementation(
      async (_path, init) =>
        new Promise<Response>((_resolve, reject) =>
          init?.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          ),
        ),
    );
    const onResult = vi.fn();
    render(
      <FigureStudio
        projectId="project"
        notebook={notebook}
        source="print('measured data')"
        outputPath="figure.png"
        artifacts={[]}
        onClose={vi.fn()}
        onResult={onResult}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "运行并生成图像" }));
    await screen.findByRole("button", { name: "停止执行" });
    fireEvent.click(screen.getByRole("button", { name: "停止执行" }));
    expect((await screen.findByRole("alert")).textContent).toBe("执行已取消。");
    expect((screen.getByLabelText("Python 图像代码") as HTMLTextAreaElement).value).toBe(
      "print('measured data')",
    );
    expect(onResult).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(
        (screen.getByRole("button", { name: "运行并生成图像" }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    expect(fetchMock.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
  });

  it("preserves RO-Crate identities and relation names while filtering a selected neighborhood", () => {
    const document: RoCrateMetadataDocument = {
      "@context": RO_CRATE_CONTEXT,
      "@graph": [
        { "@id": "ro-crate-metadata.json", "@type": "CreativeWork", about: { "@id": "./" } },
        {
          "@id": "./",
          "@type": "Dataset",
          name: "Project",
          hasPart: [{ "@id": "urn:uuid:figure" }],
        },
        {
          "@id": "urn:uuid:figure",
          "@type": "ImageObject",
          name: "Response figure",
          isBasedOn: [{ "@id": "urn:uuid:data" }],
        },
        { "@id": "urn:uuid:data", "@type": "Dataset", name: "Measured data" },
        { "@id": "urn:uuid:other", "@type": "CreativeWork", name: "Unrelated note" },
      ],
    };
    const graph = crateGraph(document, "", "urn:uuid:figure", true);
    expect(graph.nodes.map((node) => node.id)).toEqual(["./", "urn:uuid:figure", "urn:uuid:data"]);
    expect(
      graph.edges.map(({ source, target, label }) => ({ source, target, label })),
    ).toContainEqual({ source: "urn:uuid:figure", target: "urn:uuid:data", label: "isBasedOn" });
    const searched = crateGraph(document, "MEASURED", "", false);
    expect(searched.nodes.map((node) => node.id)).toEqual(["urn:uuid:data"]);
    expect(searched.edges).toEqual([]);
  });
});
