import { describe, expect, it, vi } from "vitest";
import { TYPERT_REMOTE } from "../../../science/core/src/remote.js";
import { apply } from "../src/client/index.js";

describe("V17/V46 Science client registration", () => {
  it("mounts its Remote contribution and registers the view inside remote.science injection", async () => {
    const disposeRemote = vi.fn();
    const mounted = vi.fn(() => Promise.resolve(disposeRemote));
    const registrations: Array<{ options: Record<string, unknown>; component: unknown }> = [];
    const register = vi.fn((options: Record<string, unknown>, component: unknown) => {
      registrations.push({ options, component });
      return vi.fn();
    });
    const injectSlot = vi.fn((_name: string, callback: () => void) => callback());
    const injectService = vi.fn((_services: string[], callback: (scope: typeof context) => void) =>
      callback(context),
    );
    const context = {
      inject: injectService,
      remote: {
        $mount: mounted,
        science: {
          createDocument: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          createFigure: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          createNotebook: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          createProject: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          executeNotebookCell: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          defineExperiment: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          startRun: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          finishRun: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          exportProject: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          getWorkspace: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          importArtifact: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          modifyDocument: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          modifyFigureCode: vi.fn(() => Promise.resolve({ ok: true, value: {} })),
          previewArtifact: vi.fn(() =>
            Promise.resolve({
              ok: true,
              value: {
                kind: "text",
                artifactId: "artifact-1",
                digest: `sha256:${"a".repeat(64)}`,
                mime: "text/plain",
                size: 5,
                text: "hello",
              },
            }),
          ),
        },
      },
      sideView: { dismiss: vi.fn(), open: vi.fn() },
      slots: { inject: injectSlot, register },
    };

    const dispose = await apply(context as never);

    expect(mounted).toHaveBeenCalledWith(TYPERT_REMOTE);
    expect(injectService).toHaveBeenCalledWith(["remote.science"], expect.any(Function));
    expect(injectSlot).toHaveBeenCalledWith("conversation.view", expect.any(Function));
    expect(injectSlot).toHaveBeenCalledWith("side-view.content", expect.any(Function));
    expect(injectSlot).toHaveBeenCalledWith("side-view.tool.actions", expect.any(Function));
    expect(register).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "side-view.tool.actions",
        id: "science-artifact",
      }),
      expect.any(Function),
    );
    expect(register).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "side-view.content",
        key: "science-artifact",
      }),
      expect.any(Function),
    );
    expect(register).toHaveBeenCalledWith(
      expect.objectContaining({
        name: "conversation.view",
        id: "science",
        order: 20,
      }),
      expect.any(Function),
    );
    const registration = registrations.find(({ options }) => options.name === "conversation.view")
      ?.options as
      | {
          inject(sessionId: string): Record<string, (...args: never[]) => Promise<unknown>>;
          store?: unknown;
        }
      | undefined;
    if (!registration) throw new Error("Science view registration was not captured");
    const injected = registration.inject("session-1");
    const createDocument = injected.createDocument;
    const executeCell = injected.executeCell;
    const importArtifact = injected.importArtifact;
    const modifyDocument = injected.modifyDocument;
    const createFigure = injected.createFigure;
    const modifyFigureCode = injected.modifyFigureCode;
    const defineExperiment = injected.defineExperiment;
    const startRun = injected.startRun;
    const finishRun = injected.finishRun;
    const exportProject = injected.exportProject;
    const openArtifact = injected.openArtifact;
    if (
      !createDocument ||
      !executeCell ||
      !importArtifact ||
      !modifyDocument ||
      !createFigure ||
      !modifyFigureCode ||
      !defineExperiment ||
      !startRun ||
      !finishRun ||
      !exportProject ||
      !openArtifact
    ) {
      throw new Error("Science studio injectors were not captured");
    }
    await createDocument("project-1", "paper.typ", "= Results");
    await importArtifact("project-1", "measurements.csv", "YSxiCjEsMgo=");
    await executeCell(
      "notebook-1",
      "print('saved')",
      {
        relativePath: "result.csv",
        kind: "dataset",
        title: "result.csv",
        mime: "text/csv",
        license: null,
      },
      ["artifact-1"],
    );
    await modifyDocument({
      documentId: "document-1",
      expectedRevision: 1,
      action: "accept",
      proposalId: "proposal-1",
    });
    await createFigure("project-1", "Accuracy", "matplotlib", "plt.plot(x, y)");
    await modifyFigureCode({
      figureId: "figure-1",
      expectedRevision: 1,
      action: "propose",
      objectIds: ["line-1"],
      proposedCode: "plt.plot(x, y, linewidth=2)",
      instruction: "Increase line width",
      reasoning: "The selected line is difficult to see.",
    });
    await defineExperiment("project-1", "Benchmark", "Repeat analysis", "python run.py");
    await startRun("experiment-1", 2);
    await finishRun("run-1", 1);
    await exportProject("project-1");
    openArtifact({
      id: "artifact-1",
      projectId: "project-1",
      kind: "figure",
      title: "umap.png",
      digest: `sha256:${"a".repeat(64)}`,
      mime: "image/png",
      size: 100,
      creator: { kind: "session", sessionId: "session-1" },
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
      createdAt: 1,
      updatedAt: 1,
      revision: 1,
      provenance: { eventId: "event-1", journalSeq: 1, sessionId: "session-1" },
    });
    expect(context.remote.science.createDocument).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ projectId: "project-1", name: "paper.typ", content: "= Results" }),
      undefined,
    );
    expect(context.remote.science.executeNotebookCell).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        notebookId: "notebook-1",
        source: "print('saved')",
        inputArtifactIds: ["artifact-1"],
        outputArtifact: {
          relativePath: "result.csv",
          kind: "dataset",
          title: "result.csv",
          mime: "text/csv",
          license: null,
        },
      }),
      undefined,
    );
    expect(context.remote.science.importArtifact).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        projectId: "project-1",
        name: "measurements.csv",
        dataBase64: "YSxiCjEsMgo=",
      }),
      undefined,
    );
    expect(context.remote.science.modifyDocument).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        documentId: "document-1",
        expectedRevision: 1,
        action: "accept",
        proposalId: "proposal-1",
      }),
      undefined,
    );
    expect(context.remote.science.createFigure).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        projectId: "project-1",
        title: "Accuracy",
        library: "matplotlib",
        code: "plt.plot(x, y)",
        artifactId: null,
      }),
      undefined,
    );
    expect(context.remote.science.modifyFigureCode).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ figureId: "figure-1", action: "propose", objectIds: ["line-1"] }),
      undefined,
    );
    expect(context.remote.science.defineExperiment).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ projectId: "project-1", protocol: "python run.py" }),
      undefined,
    );
    expect(context.remote.science.startRun).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ experimentId: "experiment-1", expectedRevision: 2 }),
      undefined,
    );
    expect(context.remote.science.finishRun).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ runId: "run-1", status: "succeeded" }),
      undefined,
    );
    expect(context.remote.science.exportProject).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({ projectId: "project-1" }),
      undefined,
    );
    expect(context.sideView.open).toHaveBeenCalledWith(
      "session-1",
      expect.objectContaining({
        id: "science-artifact:artifact-1",
        kind: "science-artifact",
      }),
    );

    const contentRegistration = registrations.find(
      ({ options }) => options.name === "side-view.content" && options.key === "science-artifact",
    )?.options as
      | {
          inject(sessionId: string): {
            loadPreview(artifactId: string, signal?: AbortSignal): Promise<unknown>;
            openInScience(target: Record<string, string>): boolean;
          };
        }
      | undefined;
    if (!contentRegistration) throw new Error("Artifact Side View registration was not captured");
    const content = contentRegistration.inject("session-1");
    await content.loadPreview("artifact-1");
    expect(context.remote.science.previewArtifact).toHaveBeenCalledWith(
      "session-1",
      { artifactId: "artifact-1" },
      undefined,
    );
    const target = {
      kind: "artifact",
      artifactId: "artifact-1",
      projectId: "project-1",
      surface: "artifacts",
    };
    expect(content.openInScience(target)).toBe(false);
    expect(context.sideView.dismiss).not.toHaveBeenCalled();

    const navigation = injected.navigation as unknown as { mount(): () => void };
    const unmount = navigation.mount();
    expect(content.openInScience(target)).toBe(true);
    expect(context.sideView.dismiss).toHaveBeenCalledWith("session-1");
    unmount();

    expect(registrations.every(({ options }) => options.name !== "conversation.session")).toBe(
      true,
    );
    expect(registration.store).toBeUndefined();
    await dispose();
    expect(disposeRemote).toHaveBeenCalledOnce();
  });
});
