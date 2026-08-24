import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { ScienceArtifact } from "../../../science/core/src/contracts.js";
import {
  ScienceConversationArtifactCards,
  scienceArtifactsInTurn,
} from "../src/client/science-conversation-artifacts.js";

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconDataOutline16: () => <span data-icon="data" />,
}));

function artifact(id: string, title: string, kind: "figure" | "model"): ScienceArtifact {
  return {
    id,
    projectId: "project-1",
    kind,
    title,
    digest: `sha256:${(id === "artifact-1" ? "a" : "b").repeat(64)}`,
    mime: kind === "figure" ? "image/png" : "chemical/x-pdb",
    size: 100,
    creator: { kind: "session", sessionId: "session-1" },
    runId: null,
    environment: {},
    license: null,
    sourceEntityIds: [],
    createdAt: 1,
    updatedAt: 1,
    revision: 1,
    provenance: {
      eventId: `event-${id}`,
      journalSeq: id === "artifact-1" ? 4 : 5,
      sessionId: "session-1",
    },
  };
}

function resultBlock(value: ScienceArtifact, callId: string) {
  return {
    kind: "tool-result",
    callId,
    call: { name: "science_record", argsRaw: "{}" },
    content: [
      {
        type: "text",
        text: JSON.stringify({
          classification: "fact",
          summary: "Registered artifact",
          locator: {
            sessionId: "session-1",
            toolCallId: callId,
            entityKind: value.kind,
            entityId: value.id,
            journalSeq: value.provenance.journalSeq,
          },
          data: value,
        }),
      },
    ],
    subCalls: [],
  };
}

describe("V57 conversation Science artifacts", () => {
  it("collects ordered valid artifacts from one turn and deduplicates artifact ids", () => {
    const figure = artifact("artifact-1", "umap.png", "figure");
    const model = artifact("artifact-2", "structure.pdb", "model");
    const nodes = new Map([
      ["node-1", { kind: "tool-call", data: { root: resultBlock(figure, "call-1") } }],
      ["node-2", { kind: "tool-call", data: { root: resultBlock(model, "call-2") } }],
      ["node-3", { kind: "tool-call", data: { root: resultBlock(figure, "call-1") } }],
    ]);
    const snapshot = {
      chat: {
        nodes,
        locations: { getTurn: (turn: number) => (turn === 3 ? [...nodes.keys()] : []) },
      },
    };

    expect(scienceArtifactsInTurn(snapshot as never, 3, "session-1" as never)).toEqual([
      figure,
      model,
    ]);
    expect(scienceArtifactsInTurn(snapshot as never, 4, "session-1" as never)).toEqual([]);
    expect(scienceArtifactsInTurn(snapshot as never, 3, "session-2" as never)).toEqual([]);
  });

  it("renders one complete keyboard-operable card per artifact", () => {
    const markup = renderToStaticMarkup(
      <ScienceConversationArtifactCards
        artifacts={[
          artifact("artifact-1", "umap.png", "figure"),
          artifact("artifact-2", "structure.pdb", "model"),
        ]}
        loadPreview={() => new Promise(() => undefined)}
        openArtifact={vi.fn()}
      />,
    );

    expect(markup).toContain("GENERATED · 2");
    expect(markup).toContain("umap.png");
    expect(markup).toContain("structure.pdb");
    expect(markup).toContain('aria-label="Open artifact details: umap.png"');
    expect(markup.match(/data-science-artifact-card="true"/gu)).toHaveLength(2);
  });
});
