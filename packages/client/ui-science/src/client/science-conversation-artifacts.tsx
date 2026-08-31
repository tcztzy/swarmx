import type { ChatSnapshot, ToolCallBlock } from "@deepseek-ai/dsh-client-ui-chat/client";
import { IconDataOutline16 } from "@deepseek-ai/dsh-client-ui-primitives";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import type { ScienceArtifact, ScienceArtifactPreview } from "@swarmx/dsh-science/types";
import { useEffect, useMemo, useState } from "react";
import css from "./science-conversation-artifacts.module.css";
import { scienceArtifactFromToolCall } from "./science-tool-artifact.js";

function rootFromNode(node: unknown): ToolCallBlock | null {
  if (typeof node !== "object" || node === null) return null;
  if (!("kind" in node) || node.kind !== "tool-call" || !("data" in node)) return null;
  const data = node.data;
  if (typeof data !== "object" || data === null || !("root" in data)) return null;
  const root = data.root;
  if (typeof root !== "object" || root === null || !("callId" in root)) return null;
  return typeof root.callId === "string" ? (root as ToolCallBlock) : null;
}

function toolBlocks(root: ToolCallBlock): readonly ToolCallBlock[] {
  return [root, ...root.subCalls.flatMap(toolBlocks)];
}

/** Recover ordered, same-Session artifact results from one public Chat turn. */
export function scienceArtifactsInTurn(
  snapshot: ChatSnapshot,
  turn: number,
  sessionId: SessionId,
): readonly ScienceArtifact[] {
  const artifacts: ScienceArtifact[] = [];
  const seen = new Set<string>();
  for (const key of snapshot.locations.getTurn(turn)) {
    const root = rootFromNode(snapshot.nodes.get(key));
    if (root === null) continue;
    for (const block of toolBlocks(root)) {
      const artifact = scienceArtifactFromToolCall(block, sessionId);
      if (artifact === null || seen.has(artifact.id)) continue;
      seen.add(artifact.id);
      artifacts.push(artifact);
    }
  }
  return artifacts;
}

interface ScienceConversationArtifactCardsProps {
  readonly artifacts: readonly ScienceArtifact[];
  readonly loadPreview: (
    artifactId: string,
    signal?: AbortSignal,
  ) => Promise<ScienceArtifactPreview>;
  readonly openArtifact: (artifact: ScienceArtifact) => void;
}

function ScienceConversationArtifactCard({
  artifact,
  loadPreview,
  openArtifact,
}: Omit<ScienceConversationArtifactCardsProps, "artifacts"> & {
  readonly artifact: ScienceArtifact;
}) {
  const [preview, setPreview] = useState<ScienceArtifactPreview | null>(null);
  const [previewError, setPreviewError] = useState(false);
  useEffect(() => {
    if (!artifact.mime.startsWith("image/")) return;
    const controller = new AbortController();
    setPreviewError(false);
    void loadPreview(artifact.id, controller.signal).then(
      (value) => {
        if (!controller.signal.aborted) setPreview(value);
      },
      () => {
        if (!controller.signal.aborted) setPreviewError(true);
      },
    );
    return () => controller.abort();
  }, [artifact.id, artifact.mime, loadPreview]);
  const image = preview?.kind === "image" ? preview.dataUrl : null;

  return (
    <li>
      <button
        type="button"
        className={css.card}
        data-science-artifact-card="true"
        aria-label={`Open artifact details: ${artifact.title}`}
        onClick={() => openArtifact(artifact)}
      >
        <span className={css.preview} aria-hidden="true">
          {previewError ? (
            <span className={css.previewError}>Preview unavailable</span>
          ) : image === null ? (
            <IconDataOutline16 />
          ) : (
            <img src={image} alt="" decoding="async" />
          )}
        </span>
        <span className={css.identity} title={artifact.title}>
          {artifact.title}
        </span>
      </button>
    </li>
  );
}

/** Claude-style generated-artifact group rendered beneath one closing answer. */
export function ScienceConversationArtifactCards({
  artifacts,
  loadPreview,
  openArtifact,
}: ScienceConversationArtifactCardsProps) {
  if (artifacts.length === 0) return null;
  return (
    <section className={css.root} aria-label="Generated science artifacts">
      <p className={css.count}>GENERATED · {artifacts.length}</p>
      <ul className={css.grid}>
        {artifacts.map((artifact) => (
          <ScienceConversationArtifactCard
            key={artifact.id}
            artifact={artifact}
            loadPreview={loadPreview}
            openArtifact={openArtifact}
          />
        ))}
      </ul>
    </section>
  );
}

interface ScienceConversationArtifactsInjected {
  readonly loadPreview: ScienceConversationArtifactCardsProps["loadPreview"];
  readonly openArtifact: ScienceConversationArtifactCardsProps["openArtifact"];
}

type ScienceConversationArtifactsProps = PropsRuntime<"conversation.chat.turnTail.items"> &
  ScienceConversationArtifactsInjected;

/** Turn-tail adapter from public Chat snapshot to the generated-artifact card group. */
export function ScienceConversationArtifacts({
  turn,
  sessionId,
  useChat,
  loadPreview,
  openArtifact,
}: ScienceConversationArtifactsProps) {
  const snapshot = useChat((value: ChatSnapshot) => value);
  const artifacts = useMemo(
    () => scienceArtifactsInTurn(snapshot, turn, sessionId),
    [sessionId, snapshot, turn],
  );
  return (
    <ScienceConversationArtifactCards
      artifacts={artifacts}
      loadPreview={loadPreview}
      openArtifact={openArtifact}
    />
  );
}
