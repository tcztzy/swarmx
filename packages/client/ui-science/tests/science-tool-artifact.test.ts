import { describe, expect, it } from "vitest";
import { scienceArtifactFromToolCall } from "../src/client/science-tool-artifact.js";

const artifact = {
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
  provenance: { eventId: "event-1", journalSeq: 4, sessionId: "session-1" },
};

function block(value: unknown) {
  return {
    kind: "tool-result",
    callId: "call-1",
    call: { name: "science_record", argsRaw: "{}" },
    content: [{ type: "text", text: JSON.stringify(value) }],
    subCalls: [],
  };
}

describe("V54 Science Tool artifact locator", () => {
  it("accepts one same-Session aggregate Science artifact result", () => {
    expect(
      scienceArtifactFromToolCall(
        block({
          classification: "fact",
          summary: "Registered artifact",
          locator: {
            sessionId: "session-1",
            toolCallId: "call-1",
            entityKind: "figure",
            entityId: "artifact-1",
            journalSeq: 4,
          },
          data: artifact,
        }) as never,
        "session-1" as never,
      ),
    ).toEqual(artifact);
  });

  it("rejects cross-Session, non-artifact, malformed, and oversized results", () => {
    const result = {
      locator: {
        sessionId: "session-2",
        toolCallId: "call-1",
        entityKind: "figure",
        entityId: "artifact-1",
        journalSeq: 4,
      },
      data: artifact,
    };
    expect(scienceArtifactFromToolCall(block(result) as never, "session-1" as never)).toBeNull();
    expect(
      scienceArtifactFromToolCall(
        block({
          ...result,
          locator: { ...result.locator, sessionId: "session-1", entityKind: "run" },
        }) as never,
        "session-1" as never,
      ),
    ).toBeNull();
    expect(
      scienceArtifactFromToolCall(block({ nope: true }) as never, "session-1" as never),
    ).toBeNull();
    expect(
      scienceArtifactFromToolCall(
        { ...block(result), content: [{ type: "text", text: "x".repeat(100_001) }] } as never,
        "session-1" as never,
      ),
    ).toBeNull();
  });
});
