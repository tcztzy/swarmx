import { describe, expect, it, vi } from "vitest";
import {
  basename,
  filesForClosing,
  isTypstPaperPath,
  referencedTypstPaths,
  scienceDeliverablesDefinition,
  scienceTurnFileMentions,
  typstPaperSideViewEntry,
} from "../src/client/science-deliverables.js";

describe("V94 Typst-aware produced file links", () => {
  it("publishes location data under the exact conversation node kind", () => {
    const location = scienceDeliverablesDefinition.buildLocationData?.(
      {
        state: { turn: 1, calls: new Map(), produced: [] },
      } as never,
      "turn",
    );
    expect(location).toMatchObject({ key: scienceDeliverablesDefinition.kind });
  });

  it("routes only Typst papers to a metadata-only workbench entry", () => {
    expect(isTypstPaperPath("paper.typ")).toBe(true);
    expect(isTypstPaperPath("papers/main.TYPST")).toBe(true);
    expect(isTypstPaperPath("paper.pdf")).toBe(false);
    expect(typstPaperSideViewEntry("papers/main.typ")).toEqual({
      id: "science-typst:papers/main.typ",
      kind: "science-typst",
      title: "main.typ",
      mode: "workbench",
      payload: { relativePath: "papers/main.typ" },
    });
    expect(JSON.stringify(typstPaperSideViewEntry("papers/main.typ"))).not.toContain("/Users/");
    expect(() => typstPaperSideViewEntry("papers\\main.typ")).toThrow(
      "Typst Side View path must be workspace-relative",
    );
    expect(() => typstPaperSideViewEntry("papers//main.typ")).toThrow(
      "Typst Side View path must be workspace-relative",
    );
  });

  it("preserves exact-path and unique-basename resolution without guessing ambiguity", () => {
    const openTypst = vi.fn();
    const openFile = vi.fn();
    const mentions = scienceTurnFileMentions(
      ["papers/main.typ", "data/results.csv", "other/results.csv"],
      openTypst,
      openFile,
      (path) => `Open ${path}`,
    );

    mentions.resolve("main.typ")?.open();
    mentions.resolve("data/results.csv")?.open();
    mentions.resolveLink?.("./papers/main.typ")?.open();
    expect(mentions.resolve("results.csv")).toBeUndefined();
    expect(mentions.resolveLink?.("./results.csv")).toBeUndefined();
    expect(mentions.resolveLink?.("./papers\\main.typ")).toBeUndefined();
    expect(mentions.resolveLink?.("https://example.com/main.typ")).toBeUndefined();
    expect(openTypst).toHaveBeenCalledTimes(2);
    expect(openTypst).toHaveBeenLastCalledWith("papers/main.typ");
    expect(openFile).toHaveBeenCalledWith("data/results.csv");
    expect(basename("papers/main.typ")).toBe("main.typ");
  });

  it("V109 recovers one safe Typst entry from replayed Tool and Assistant evidence", () => {
    expect(
      referencedTypstPaths(
        "Updated [paper](./docs/swarmx-introduction.typ), not /tmp/private.typ or ../escape.typ.",
      ),
    ).toEqual(["docs/swarmx-introduction.typ"]);

    const start = scienceDeliverablesDefinition.start(
      {} as never,
      { event: { type: "turn/start", seq: 0, data: { turn: 1 } } } as never,
      {} as never,
    );
    const afterTool = scienceDeliverablesDefinition.update(
      { state: start } as never,
      {
        event: {
          type: "tool/call",
          seq: 1,
          data: {
            turn: 1,
            step: 1,
            callId: "call-1",
            name: "bash",
            arguments: '{"cmd":"cat > docs/swarmx-introduction.typ"}',
          },
        },
      } as never,
    );
    const afterAssistant = scienceDeliverablesDefinition.update(
      { state: afterTool } as never,
      {
        event: {
          type: "assistant/message",
          seq: 2,
          data: {
            turn: 1,
            step: 1,
            message: {
              role: "assistant",
              content: [
                {
                  type: "text",
                  text: "Updated `docs/swarmx-introduction.typ`.",
                },
              ],
            },
          },
        },
      } as never,
    );
    const location = scienceDeliverablesDefinition.buildLocationData?.(
      { state: afterAssistant } as never,
      "turn",
    );

    expect(location).toMatchObject({
      value: {
        referenced: [
          { path: "docs/swarmx-introduction.typ", seq: 1 },
          { path: "docs/swarmx-introduction.typ", seq: 2 },
        ],
      },
    });
    expect(
      filesForClosing(location?.kind === "turn" ? (location.value as never) : undefined, 2),
    ).toEqual(["docs/swarmx-introduction.typ"]);
  });
});
