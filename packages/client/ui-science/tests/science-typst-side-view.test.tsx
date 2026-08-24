import { describe, expect, it, vi } from "vitest";
import type { TypstDocumentPreview } from "../../../science/core/src/contracts.js";
import type { SideViewEntry } from "../../ui-conversation/src/client/side-view.js";
import {
  figurePayload,
  isCurrentFigurePreview,
  pdfFigureSideViewEntry,
  ScienceTypstSideView,
  typstPayload,
} from "../src/client/science-typst-side-view.js";

const SOURCE_REVISION = `sha256:${"a".repeat(64)}` as const;
const PDF_REVISION = `sha256:${"b".repeat(64)}` as const;

const preview: TypstDocumentPreview = {
  relativePath: "papers/main.typ",
  title: "main.typ",
  source: "= Paper",
  sourceRevision: SOURCE_REVISION,
  status: "ready",
  diagnostics: [],
  pdfBase64: "JVBERg==",
  pdfRevision: PDF_REVISION,
  pdfSourceRevision: SOURCE_REVISION,
  pdfSize: 4,
  compiledAt: 1,
};

const entry = (relativePath: string): SideViewEntry => ({
  id: `science-typst:${relativePath}`,
  kind: "science-typst",
  title: relativePath,
  mode: "workbench",
  payload: { relativePath },
});

const injected = {
  loadPreview: vi.fn(),
  updateSource: vi.fn(),
  resolveSourceAtPoint: vi.fn(),
  addAnnotationToConversation: vi.fn(),
  openFigure: vi.fn(),
};

describe("V110 per-paper Typst workbench identity", () => {
  it("accepts only the shared canonical Typst path contract", () => {
    expect(typstPayload(entry("papers/main.typ"))).toEqual({
      relativePath: "papers/main.typ",
    });
    expect(typstPayload(entry("papers\\main.typ"))).toBeNull();
    expect(typstPayload(entry("papers//main.typ"))).toBeNull();
  });

  it("keys the stateful editor lifecycle by the authorized paper path", () => {
    const first = ScienceTypstSideView({ ...injected, entry: entry("papers/a.typ") } as never);
    const second = ScienceTypstSideView({ ...injected, entry: entry("papers/b.typ") } as never);

    expect(first.key).toBe("papers/a.typ");
    expect(second.key).toBe("papers/b.typ");
    expect(second.key).not.toBe(first.key);
  });
});

describe("V93 PDF figure workbench identity", () => {
  it("requires the same compiled PDF and source revisions", () => {
    const figureEntry = pdfFigureSideViewEntry(preview, {
      page: 1,
      figureIndex: 0,
      rect: { x: 0.1, y: 0.2, width: 0.3, height: 0.4 },
    });
    const payload = figurePayload(figureEntry);
    if (payload === null) throw new Error("Figure entry did not round-trip");

    expect(isCurrentFigurePreview(payload, preview)).toBe(true);
    expect(
      isCurrentFigurePreview(payload, {
        ...preview,
        pdfSourceRevision: `sha256:${"c".repeat(64)}`,
      }),
    ).toBe(false);
  });
});
