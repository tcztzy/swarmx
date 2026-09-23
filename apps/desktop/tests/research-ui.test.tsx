// @vitest-environment jsdom
import { createResearchObject } from "@swarmx/science";
import { RO_CRATE_CONTEXT, type RoCrateMetadataDocument } from "@swarmx/science/types";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../src/renderer/app.js";
import { FigureStudio } from "../src/renderer/figure-studio.js";
import { i18n } from "../src/renderer/i18n.js";
import { ResearchPanel } from "../src/renderer/research.js";
import { crateGraph } from "../src/renderer/research-graph.js";
import { SettingsPage } from "../src/renderer/settings.js";
import { DEFAULT_POLICY } from "../src/settings.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

vi.mock("../src/renderer/chat.js", () => ({
  ConversationSurface: ({ sidePanel }: { sidePanel?: ReactNode }) => sidePanel,
}));
vi.mock("../src/renderer/trace.js", () => ({ TracePanel: () => null }));

it("preserves a running figure and its unsaved code when opening Work", async () => {
  gateway.bootstrap.mockResolvedValue({
    agents: ["pi"],
    defaultHarness: "pi",
    sessions: [],
    cwd: "/research",
    language: "zh",
  });
  const pending = Promise.withResolvers<never>();
  gateway.tool.mockReturnValue(pending.promise);
  render(<App />);
  fireEvent.click(await screen.findByRole("button", { name: "科研资产", exact: true }));
  fireEvent.click(await screen.findByRole("button", { name: /^Plotting run/ }));
  const editor = await screen.findByLabelText("Python 图像代码");
  fireEvent.change(editor, { target: { value: "print('unsaved')" } });
  fireEvent.change(screen.getByLabelText("输出文件"), { target: { value: "figure.png" } });
  fireEvent.click(screen.getByRole("button", { name: "运行并生成图像" }));
  await waitFor(() => expect(gateway.tool).toHaveBeenCalledTimes(1));
  try {
    fireEvent.click(screen.getByRole("button", { name: "长期工作", exact: true }));
    await screen.findByRole("complementary", { name: "长期工作侧栏" });
    expect(gateway.cancelTool).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "科研资产", exact: true }));
    expect(screen.getByLabelText("Python 图像代码")).toBe(editor);
    expect((editor as HTMLTextAreaElement).value).toBe("print('unsaved')");
  } finally {
    pending.reject(new Error("Local fixture finished"));
  }
});

const workbenchIdentity = {
  createdAt: 0,
  updatedAt: 0,
  revision: 1,
  provenance: { eventId: "event", journalSeq: 1, sessionId: "renderer" },
};
const workbenchFigure = {
  ...workbenchIdentity,
  id: "figure",
  projectId: "project",
  kind: "figure" as const,
  title: "Registered figure",
  digest: `sha256:${"a".repeat(64)}`,
  mime: "image/png",
  size: 10,
  creator: { kind: "session" as const, sessionId: "renderer" },
  runId: null,
  environment: {},
  license: null,
  sourceEntityIds: [],
};
const workbenchCode = {
  id: "code",
  kind: "code" as const,
  source: "print('measured output')",
  executionCount: 1,
  executionTimeMs: 10,
  inputArtifactIds: [],
  outputArtifactIds: ["figure"],
  runtimeEnvironment: {},
  relatedClaimIds: [],
  relatedExperimentIds: [],
  outputs: [],
};
const workbenchNotebook = {
  ...workbenchIdentity,
  id: "plot",
  projectId: "project",
  kind: "notebook" as const,
  title: "Plotting run",
  cells: [workbenchCode],
};
const workbenchSnapshot = {
  projects: [{ ...workbenchIdentity, id: "project", kind: "project" as const, title: "Project" }],
  artifacts: [workbenchFigure],
  notebooks: [workbenchNotebook],
  documents: [],
  figures: [],
  records: [],
  relations: [],
  experiments: [],
  runs: [],
  exports: [],
};
const workbenchExecution = {
  id: "execution",
  notebookId: "plot",
  cellId: "code",
  source: workbenchCode.source,
  executionCount: 1,
  status: "succeeded" as const,
  stdout: { text: "measured output", truncated: false },
  stderr: { text: "", truncated: false },
  outputs: [],
  exitCode: 0,
  signal: null,
  durationMs: 10,
  environment: {},
  inputArtifactIds: [],
  artifact: workbenchFigure,
  provenance: workbenchIdentity.provenance,
};

let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("zh");
  vi.stubGlobal("matchMedia", () => ({ matches: true }));
  gateway = installBridge();
  gateway.scienceWorkspace.mockResolvedValue(workbenchSnapshot);
  gateway.scienceResearchObject.mockResolvedValue(
    createResearchObject(workbenchSnapshot, "project"),
  );
  gateway.scienceNotebookExecutions.mockResolvedValue([workbenchExecution]);
  gateway.scienceArtifactPreview.mockResolvedValue({
    kind: "image",
    artifactId: workbenchFigure.id,
    digest: workbenchFigure.digest,
    size: workbenchFigure.size,
    mime: "image/png",
    dataUrl: "data:image/png;base64,AAAA",
  });
  gateway.tool.mockImplementation((payload: { args: { action: string } }) =>
    Promise.resolve(
      payload.args.action === "memory_status"
        ? {
            action: "memory_status",
            data: {
              settings: {},
              note: { content: "", revision: `sha256:${"0".repeat(64)}`, limit: 1375 },
              pending: [],
              review: { state: "idle", message: "", sessionId: null },
            },
          }
        : { action: "graph_memory", data: { nodes: [], edges: [] } },
    ),
  );
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
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
    gateway.scienceWorkspace.mockResolvedValue(snapshot);
    gateway.scienceResearchObject.mockResolvedValue(crate);
    gateway.scienceNotebookExecutions.mockResolvedValue([
      {
        id: "execution",
        notebookId: "notebook",
        cellId: "code",
        source,
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
    ]);
    const { rerender } = render(<ResearchPanel mode="assets" onClose={vi.fn()} canCompose />);
    const notebookButton = await screen.findByRole("button", { name: /^Analysis/ });
    expect(notebookButton.textContent?.replace(/\s/gu, "")).toContain("1次执行");
    fireEvent.click(screen.getByRole("button", { name: "在对话中生成" }));
    expect(screen.getByRole("status").textContent).toContain("已添加到对话草稿");
    rerender(<ResearchPanel mode="observe" onClose={vi.fn()} />);
    expect(screen.queryByText("已添加到对话草稿，请补充要求后发送。")).toBeNull();
    fireEvent.keyDown(screen.getByRole("tab", { name: "运行记录" }), { key: "Enter" });
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
      cwd: "/research",
    };
    gateway.settingsRead.mockResolvedValue(settings);
    gateway.environmentRead.mockResolvedValue({
      state: "missing",
      environment: null,
      log: "",
      activeProcesses: 0,
    });
    gateway.settingsUpdate.mockImplementation(async (policy: unknown) =>
      Promise.resolve({
        policy: { ...DEFAULT_POLICY, ...(policy as object) },
        environment: null,
        cwd: settings.cwd,
      }),
    );
    gateway.environmentAct.mockRejectedValue(new Error("Docker daemon is unavailable"));
    render(<SettingsPage />);
    await screen.findByRole("button", { name: "保存权限" });
    fireEvent.change(screen.getByLabelText("科研容器文件访问"), { target: { value: "read-only" } });
    fireEvent.click(screen.getByLabelText("修改记忆"));
    fireEvent.click(screen.getByLabelText("允许通过 Swarm 委派任务"));
    fireEvent.change(screen.getByLabelText("CPU 核数"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
    await screen.findByText("权限已保存，对下一次执行生效。");
    expect(gateway.settingsUpdate).toHaveBeenCalledWith({
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
    gateway.tool.mockImplementation(
      (payload: { requestId: string }) =>
        new Promise((_resolve, reject) => {
          gateway.cancelTool.mockImplementationOnce(async ({ requestId }) => {
            if (requestId === payload.requestId) reject(new Error("Execution cancelled."));
            return { cancelled: true };
          });
        }),
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
    expect(gateway.cancelTool).toHaveBeenCalledTimes(1);
    expect(gateway.tool).toHaveBeenCalledWith(
      expect.objectContaining({ name: "science_notebook" }),
    );
    await waitFor(() =>
      expect(
        (screen.getByRole("button", { name: "运行并生成图像" }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
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

  it("labels the assets region without a dangling tab trigger reference", async () => {
    const { container } = render(<ResearchPanel mode="assets" onClose={vi.fn()} />);
    await screen.findByText("科研资产");
    const panel = container.querySelector('[data-slot="tabs-content"]');
    expect(panel?.getAttribute("aria-labelledby")).toBeNull();
    expect(panel?.getAttribute("aria-label")).toBe("科研资产侧栏");
  });

  it("reopens the editor for the newly selected notebook output", async () => {
    const oldSource = "print('old input check')";
    const data = {
      ...workbenchSnapshot,
      notebooks: [
        workbenchNotebook,
        {
          ...workbenchNotebook,
          id: "check",
          title: "Input check",
          cells: [{ ...workbenchCode, source: oldSource, outputArtifactIds: [] }],
        },
      ],
    };
    gateway.scienceWorkspace.mockResolvedValue(data);
    gateway.scienceResearchObject.mockResolvedValue(createResearchObject(data, "project"));
    gateway.tool.mockRejectedValue(new Error("Stop after recording payload"));
    const { rerender } = render(<ResearchPanel mode="assets" onClose={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: /^Input check/ }));
    expect((screen.getByLabelText("Python 图像代码") as HTMLTextAreaElement).value).toBe(oldSource);
    rerender(<ResearchPanel mode="observe" onClose={vi.fn()} />);
    fireEvent.keyDown(screen.getByRole("tab", { name: "运行记录" }), { key: "Enter" });
    fireEvent.click(await screen.findByText(/Plotting run · #1/));
    fireEvent.click(await screen.findByRole("button", { name: "查看输出成果" }));
    fireEvent.click(await screen.findByRole("button", { name: "编辑代码并生成新版本" }));
    rerender(<ResearchPanel mode="assets" onClose={vi.fn()} />);
    expect((screen.getByLabelText("Python 图像代码") as HTMLTextAreaElement).value).toBe(
      workbenchCode.source,
    );
    fireEvent.change(screen.getByLabelText("输出文件"), { target: { value: "figure.png" } });
    fireEvent.click(screen.getByRole("button", { name: "运行并生成图像" }));
    await waitFor(() => expect(gateway.tool).toHaveBeenCalled());
    const payload = JSON.stringify(gateway.tool.mock.calls[0]?.[0]);
    expect(payload).toContain('"notebookId":"plot"');
    expect(payload).toContain(workbenchCode.source);
  });
});
