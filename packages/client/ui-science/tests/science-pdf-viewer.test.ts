import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  figureRegionFromTransform,
  normalizeInverseSearchClick,
  normalizePageRectangle,
  pointInsideRegion,
  positionFloatingEditor,
} from "../src/client/science-pdf-geometry.js";

describe("V90/V92/V93 PDF.js paper geometry", () => {
  it("normalizes a PDF text selection inside one page and rejects outside geometry", () => {
    expect(
      normalizePageRectangle(
        { left: 100, top: 50, width: 800, height: 1_000 },
        { left: 180, top: 250, width: 400, height: 40 },
      ),
    ).toEqual({ x: 0.1, y: 0.2, width: 0.5, height: 0.04 });
    expect(
      normalizePageRectangle(
        { left: 100, top: 50, width: 800, height: 1_000 },
        { left: 50, top: 250, width: 400, height: 40 },
      ),
    ).toBeNull();
  });

  it("derives bounded figure regions from PDF image transforms and maps figure-local points", () => {
    const region = figureRegionFromTransform([400, 0, 0, 250, 80, 120], 800, 1_000);
    expect(region).toEqual({ x: 0.1, y: 0.12, width: 0.5, height: 0.25 });
    expect(pointInsideRegion(region, { x: 0.35, y: 0.245 })).toEqual({ x: 0.5, y: 0.5 });
    expect(pointInsideRegion(region, { x: 0.05, y: 0.2 })).toBeNull();
  });

  it("V98 positions the annotation editor beside its anchor and flips above at the viewport edge", () => {
    const boundary = { left: 100, top: 50, width: 800, height: 650 };
    const editor = { width: 320, height: 180 };

    expect(positionFloatingEditor({ x: 450, y: 200 }, editor, boundary, 12)).toEqual({
      left: 290,
      top: 212,
    });
    expect(positionFloatingEditor({ x: 880, y: 670 }, editor, boundary, 12)).toEqual({
      left: 568,
      top: 478,
    });
  });

  it("V104 resolves only an ordinary text click to a bounded normalized page point", () => {
    const page = { left: 100, top: 50, width: 800, height: 1_000 };
    expect(
      normalizeInverseSearchClick({
        page,
        clientX: 300,
        clientY: 300,
        detail: 1,
        selectionCollapsed: true,
        targetInTextLayer: true,
      }),
    ).toEqual({ x: 0.25, y: 0.25 });
    expect(
      normalizeInverseSearchClick({
        page,
        clientX: 300,
        clientY: 300,
        detail: 1,
        selectionCollapsed: false,
        targetInTextLayer: true,
      }),
    ).toBeNull();
    expect(
      normalizeInverseSearchClick({
        page,
        clientX: 300,
        clientY: 300,
        detail: 0,
        selectionCollapsed: true,
        targetInTextLayer: true,
      }),
    ).toBeNull();
    expect(
      normalizeInverseSearchClick({
        page,
        clientX: 300,
        clientY: 300,
        detail: 2,
        selectionCollapsed: true,
        targetInTextLayer: true,
      }),
    ).toBeNull();
    expect(
      normalizeInverseSearchClick({
        page,
        clientX: 300,
        clientY: 300,
        detail: 1,
        selectionCollapsed: true,
        targetInTextLayer: false,
      }),
    ).toBeNull();
  });

  it("V106 wires selection to pointerup and inverse search to the counted click event", () => {
    const source = readFileSync(
      new URL("../src/client/science-pdf-viewer.tsx", import.meta.url),
      "utf8",
    );

    expect(source).toContain("onPointerUp={(event) => onSelectText(");
    expect(source).toContain("onClick={(event) => onTextClick(");
    expect(source).toContain(
      "function pageClick(event: MouseEvent<HTMLDivElement>, page: number, pageElement: HTMLDivElement)",
    );
  });
});
