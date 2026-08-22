import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type {} from "@deepseek-ai/dsh-client-ui-conversation/client";
import { TYPERT_REMOTE } from "@swarmx/dsh-science/remote";
import type { ExecuteNotebookCellRequest, ScienceArtifact } from "@swarmx/dsh-science/types";
import type {} from "@swarmx/dsh-ui-conversation/client";
import {
  ScienceArtifactSideView,
  scienceArtifactSideViewEntry,
} from "./science-artifact-side-view.js";
import { ScienceNavigationController } from "./science-navigation.js";
import { ScienceToolArtifactAction } from "./science-tool-artifact.js";
import { ScienceWorkspace } from "./science-workspace.js";

export const inject = ["slots", "remote", "sideView"];

function remoteValue<T>(
  result: { ok: true; value: T } | { ok: false; error: { message: string } },
): T {
  if (result.ok) return result.value;
  throw new Error(result.error.message);
}

/** Mount the strict Science Remote and add one independent conversation view. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
  const disposeRemote = await ctx.remote.$mount(TYPERT_REMOTE);
  const navigation = new ScienceNavigationController();
  ctx.inject(["remote.science"], (scienceCtx) => {
    scienceCtx.slots.inject("side-view.content", () =>
      scienceCtx.slots.register(
        {
          name: "side-view.content",
          key: "science-artifact",
          inject: (sessionId: SessionId) => ({
            loadPreview: async (artifactId: string, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.previewArtifact(sessionId, { artifactId }, signal),
              ),
            openInScience: (target) => {
              const mounted = navigation.open(sessionId, target);
              if (mounted) scienceCtx.sideView.dismiss(sessionId);
              return mounted;
            },
          }),
        },
        ScienceArtifactSideView,
      ),
    );
    scienceCtx.slots.inject("side-view.tool.actions", () =>
      scienceCtx.slots.register(
        {
          name: "side-view.tool.actions",
          id: "science-artifact",
          order: 10,
          inject: (sessionId: SessionId) => ({
            openArtifact: (artifact: ScienceArtifact) =>
              scienceCtx.sideView.open(sessionId, scienceArtifactSideViewEntry(artifact)),
          }),
        },
        ScienceToolArtifactAction,
      ),
    );
    return scienceCtx.slots.inject("conversation.view", () =>
      scienceCtx.slots.register(
        {
          name: "conversation.view",
          id: "science",
          order: 20,
          label: () => "Science",
          inject: (sessionId: SessionId) => ({
            loadWorkspace: async (signal?: AbortSignal) =>
              remoteValue(await scienceCtx.remote.science.getWorkspace(sessionId, signal)),
            createProject: async (title: string, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.createProject(
                  sessionId,
                  { requestId: crypto.randomUUID(), title },
                  signal,
                ),
              ),
            createNotebook: async (projectId: string, title: string, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.createNotebook(
                  sessionId,
                  { requestId: crypto.randomUUID(), projectId, title },
                  signal,
                ),
              ),
            executeCell: async (
              notebookId: string,
              source: string,
              outputArtifact: ExecuteNotebookCellRequest["outputArtifact"],
              inputArtifactIds: readonly string[] = [],
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.executeNotebookCell(
                  sessionId,
                  {
                    requestId: crypto.randomUUID(),
                    notebookId,
                    source,
                    inputArtifactIds: [...inputArtifactIds],
                    outputArtifact,
                  },
                  signal,
                ),
              ),
            importArtifact: async (
              projectId: string,
              name: string,
              dataBase64: string,
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.importArtifact(
                  sessionId,
                  { requestId: crypto.randomUUID(), projectId, name, dataBase64 },
                  signal,
                ),
              ),
            createDocument: async (
              projectId: string,
              name: string,
              content: string,
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.createDocument(
                  sessionId,
                  { requestId: crypto.randomUUID(), projectId, name, content },
                  signal,
                ),
              ),
            modifyDocument: async (request, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.modifyDocument(
                  sessionId,
                  { requestId: crypto.randomUUID(), ...request },
                  signal,
                ),
              ),
            createFigure: async (
              projectId: string,
              title: string,
              library: "matplotlib" | "seaborn" | "ggplot2" | "plotly",
              code: string,
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.createFigure(
                  sessionId,
                  {
                    requestId: crypto.randomUUID(),
                    projectId,
                    title,
                    library,
                    code,
                    artifactId: null,
                  },
                  signal,
                ),
              ),
            modifyFigureCode: async (request, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.modifyFigureCode(
                  sessionId,
                  { requestId: crypto.randomUUID(), ...request },
                  signal,
                ),
              ),
            defineExperiment: async (
              projectId: string,
              title: string,
              summary: string,
              protocol: string,
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.defineExperiment(
                  sessionId,
                  {
                    requestId: crypto.randomUUID(),
                    projectId,
                    title,
                    summary,
                    protocol,
                    hypothesisIds: [],
                    tags: [],
                  },
                  signal,
                ),
              ),
            startRun: async (
              experimentId: string,
              expectedRevision: number,
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.startRun(
                  sessionId,
                  {
                    requestId: crypto.randomUUID(),
                    experimentId,
                    expectedRevision,
                    environment: {},
                  },
                  signal,
                ),
              ),
            finishRun: async (runId: string, expectedRevision: number, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.finishRun(
                  sessionId,
                  {
                    requestId: crypto.randomUUID(),
                    runId,
                    expectedRevision,
                    status: "succeeded",
                    metrics: {},
                    artifactIds: [],
                    notes: "Finished from Science Workspace",
                  },
                  signal,
                ),
              ),
            exportProject: async (projectId: string, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.exportProject(
                  sessionId,
                  { requestId: crypto.randomUUID(), projectId },
                  signal,
                ),
              ),
            openArtifact: (artifact) =>
              scienceCtx.sideView.open(sessionId, scienceArtifactSideViewEntry(artifact)),
            navigation: {
              getSnapshot: () => navigation.getSnapshot(sessionId),
              subscribe: (listener: () => void) => navigation.subscribe(sessionId, listener),
              mount: () => navigation.mount(sessionId),
            },
          }),
        },
        ScienceWorkspace,
      ),
    );
  });
  return async () => {
    navigation.dispose();
    await disposeRemote();
  };
}
