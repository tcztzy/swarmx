import type { SessionId, ToolCallBlock } from "@deepseek-ai/dsh-client-runtime/client";
import { type ScienceArtifact, scienceArtifactSchema } from "@swarmx/dsh-science/types";

const MAX_SCIENCE_RESULT_CHARS = 100_000;
const SCIENCE_TOOL_NAMES = new Set([
  "science_notebook",
  "science_write",
  "science_figure",
  "science_experiment",
  "science_record",
  "science_query",
  "science_export",
]);

/** Strictly recover an artifact locator from one aggregate Science Tool result. */
export function scienceArtifactFromToolCall(
  block: ToolCallBlock,
  sessionId: SessionId,
): ScienceArtifact | null {
  if (
    !("kind" in block) ||
    block.isError ||
    block.call === null ||
    !SCIENCE_TOOL_NAMES.has(block.call.name)
  ) {
    return null;
  }
  const text =
    block.content.length === 1 && block.content[0]?.type === "text" ? block.content[0].text : null;
  if (text === null || text.length > MAX_SCIENCE_RESULT_CHARS) return null;
  try {
    const result = JSON.parse(text) as {
      readonly locator?: {
        readonly sessionId?: unknown;
        readonly toolCallId?: unknown;
        readonly entityKind?: unknown;
        readonly entityId?: unknown;
        readonly journalSeq?: unknown;
      };
      readonly data?: unknown;
    };
    const locator = result.locator;
    const artifact = scienceArtifactSchema.safeParse(result.data);
    if (
      !artifact.success ||
      locator?.sessionId !== sessionId ||
      locator.toolCallId !== block.callId ||
      locator.entityKind !== artifact.data.kind ||
      locator.entityId !== artifact.data.id ||
      locator.journalSeq !== artifact.data.provenance.journalSeq
    ) {
      return null;
    }
    return artifact.data;
  } catch {
    return null;
  }
}
