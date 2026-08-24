import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type {} from "@deepseek-ai/dsh-client-ui-input-trigger/client";
import { TYPERT_REMOTE } from "@swarmx/dsh-science/remote";
import type { ScienceArtifact, ScienceImageAnnotation } from "@swarmx/dsh-science/types";
import type { SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import {
  annotationReferenceSource,
  imageCommentAnnotation,
  insertAnnotationReference,
  paperCommentAnnotation,
} from "./annotation-reference.js";
import {
  artifactPayload,
  ScienceArtifactSideView,
  scienceArtifactSideViewEntry,
  withScienceArtifactAnnotations,
} from "./science-artifact-side-view.js";
import { ScienceConversationArtifacts } from "./science-conversation-artifacts.js";
import { registerScienceDeliverables } from "./science-deliverables.js";
import {
  pdfFigureSideViewEntry,
  SciencePdfFigureSideView,
  ScienceTypstSideView,
} from "./science-typst-side-view.js";

export const inject = [
  "conversation",
  "conversationEvents",
  "inputTriggers",
  "remote",
  "sessions",
  "sideView",
  "slots",
  "workspaces",
];

function remoteValue<T>(
  result: { ok: true; value: T } | { ok: false; error: { message: string } },
): T {
  if (result.ok) return result.value;
  throw new Error(result.error.message);
}

/** Mount strict Science Remote and keep its UI inside Chat plus DetailsPanel. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
  registerScienceDeliverables(ctx);
  const disposeRemote = await ctx.remote.$mount(TYPERT_REMOTE);
  const disposeAnnotationSource = ctx.inputTriggers.registerSource(annotationReferenceSource());
  const annotations = new Map<string, readonly ScienceImageAnnotation[]>();
  const annotationKey = (sessionId: SessionId, artifactId: string) =>
    `${sessionId}\u0000${artifactId}`;

  ctx.inject(["remote.science"], (scienceCtx) => {
    const openArtifact = (sessionId: SessionId, artifact: ScienceArtifact) =>
      scienceCtx.sideView.open(
        sessionId,
        scienceArtifactSideViewEntry(
          artifact,
          annotations.get(annotationKey(sessionId, artifact.id)) ?? [],
        ),
      );
    const loadPreview = (sessionId: SessionId, artifactId: string, signal?: AbortSignal) =>
      scienceCtx.remote.science
        .previewArtifact(sessionId, { artifactId }, signal)
        .then(remoteValue);
    const loadTypst = (sessionId: SessionId, relativePath: string, signal?: AbortSignal) =>
      scienceCtx.remote.science
        .previewTypstDocument(sessionId, { relativePath }, signal)
        .then(remoteValue);

    scienceCtx.slots.inject("side-view.content", () =>
      scienceCtx.slots.register(
        {
          name: "side-view.content",
          key: "science-artifact",
          inject: (sessionId: SessionId) => ({
            loadPreview: (artifactId: string, signal?: AbortSignal) =>
              loadPreview(sessionId, artifactId, signal),
            loadResearchObject: async (projectId: string, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.getResearchObject(sessionId, { projectId }, signal),
              ),
            updateAnnotations: (
              entry: SideViewEntry,
              nextAnnotations: readonly ScienceImageAnnotation[],
            ) => {
              const artifact = artifactPayload(entry);
              const nextEntry = withScienceArtifactAnnotations(entry, nextAnnotations);
              if (artifact === null || nextEntry === null) return;
              annotations.set(annotationKey(sessionId, artifact.artifactId), nextAnnotations);
              scienceCtx.sideView.open(sessionId, nextEntry);
            },
            addAnnotationToConversation: (annotation: ScienceImageAnnotation) =>
              insertAnnotationReference(
                scienceCtx.conversation,
                scienceCtx.sessions,
                sessionId,
                imageCommentAnnotation(annotation),
              ),
          }),
        },
        ScienceArtifactSideView,
      ),
    );
    scienceCtx.slots.inject("side-view.content", () =>
      scienceCtx.slots.register(
        {
          name: "side-view.content",
          key: "science-typst",
          inject: (sessionId: SessionId) => ({
            loadPreview: (relativePath: string, signal?: AbortSignal) =>
              loadTypst(sessionId, relativePath, signal),
            updateSource: async (
              request: {
                readonly relativePath: string;
                readonly expectedSourceRevision: string;
                readonly source: string;
              },
              signal?: AbortSignal,
            ) =>
              remoteValue(
                await scienceCtx.remote.science.updateTypstSource(sessionId, request, signal),
              ),
            resolveSourceAtPoint: async (request, signal?: AbortSignal) =>
              remoteValue(
                await scienceCtx.remote.science.resolveTypstSourceAtPoint(
                  sessionId,
                  request,
                  signal,
                ),
              ),
            addAnnotationToConversation: (annotation) =>
              insertAnnotationReference(
                scienceCtx.conversation,
                scienceCtx.sessions,
                sessionId,
                paperCommentAnnotation(annotation),
              ),
            openFigure: (preview, locator) =>
              scienceCtx.sideView.open(sessionId, pdfFigureSideViewEntry(preview, locator)),
          }),
        },
        ScienceTypstSideView,
      ),
    );
    scienceCtx.slots.inject("side-view.content", () =>
      scienceCtx.slots.register(
        {
          name: "side-view.content",
          key: "science-pdf-figure",
          inject: (sessionId: SessionId) => ({
            loadPreview: (relativePath: string, signal?: AbortSignal) =>
              loadTypst(sessionId, relativePath, signal),
            addAnnotationToConversation: (annotation) =>
              insertAnnotationReference(
                scienceCtx.conversation,
                scienceCtx.sessions,
                sessionId,
                paperCommentAnnotation(annotation),
              ),
          }),
        },
        SciencePdfFigureSideView,
      ),
    );
    return scienceCtx.slots.inject("conversation.chat.turnTail.items", () =>
      scienceCtx.slots.register(
        {
          name: "conversation.chat.turnTail.items",
          id: "science-artifacts",
          inject: (sessionId: SessionId) => ({
            loadPreview: (artifactId: string, signal?: AbortSignal) =>
              loadPreview(sessionId, artifactId, signal),
            openArtifact: (artifact: ScienceArtifact) => openArtifact(sessionId, artifact),
          }),
        },
        ScienceConversationArtifacts,
      ),
    );
  });

  return async () => {
    annotations.clear();
    disposeAnnotationSource();
    await disposeRemote();
  };
}
