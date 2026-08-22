import { createHash } from "node:crypto";
import type { FigureLibrary, FigureObject, FigureObjectKind } from "./contracts.js";

const OBJECT_PATTERNS: ReadonlyArray<{
  readonly kind: FigureObjectKind;
  readonly pattern: RegExp;
}> = [
  { kind: "legend", pattern: /legend/iu },
  { kind: "image-layer", pattern: /(?:imshow|image\s*\(|add_image)/iu },
  { kind: "annotation", pattern: /(?:annotate|annotation|\btext\s*\(|\btitle\s*\()/iu },
  { kind: "point", pattern: /(?:scatterplot|\bscatter\s*\(|geom_point)/iu },
  { kind: "line", pattern: /(?:\.plot\s*\(|geom_line|add_scatter.*mode\s*=\s*["']lines)/iu },
  { kind: "axis", pattern: /(?:[xy](?:label|lim|ticks?)\s*\(|axis|scale_[xy])/iu },
  { kind: "data-series", pattern: /(?:ggplot|\baes\s*\(|add_(?:bar|trace)|\bbar\s*\()/iu },
];

export function codeHash(code: string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(code).digest("hex")}`;
}

export function inferFigureObjects(
  code: string,
  _library: FigureLibrary,
  createId: () => string,
): FigureObject[] {
  const counters = new Map<FigureObjectKind, number>();
  const objects: FigureObject[] = [];
  for (const line of code.matchAll(/[^\n]+/gu)) {
    if (line.index === undefined) continue;
    const match = OBJECT_PATTERNS.find(({ pattern }) => pattern.test(line[0]));
    if (!match) continue;
    const count = (counters.get(match.kind) ?? 0) + 1;
    counters.set(match.kind, count);
    objects.push({
      id: createId(),
      kind: match.kind,
      label: `${match.kind.replace("-", " ")} ${count}`,
      codeRange: { start: line.index, end: line.index + line[0].length },
    });
    if (objects.length >= 200) break;
  }
  if (objects.length > 0) return objects;
  return [
    {
      id: createId(),
      kind: "data-series",
      label: "data series 1",
      codeRange: { start: 0, end: Math.max(1, code.length) },
    },
  ];
}

export function remapFigureObjects(
  objects: readonly FigureObject[],
  selectedIds: ReadonlySet<string>,
  selection: { readonly start: number; readonly end: number },
  replacementLength: number,
): FigureObject[] {
  const delta = replacementLength - (selection.end - selection.start);
  return objects.map((object) => {
    if (selectedIds.has(object.id)) {
      return {
        ...object,
        codeRange: { start: selection.start, end: selection.start + replacementLength },
      };
    }
    if (object.codeRange.start >= selection.end) {
      return {
        ...object,
        codeRange: {
          start: object.codeRange.start + delta,
          end: object.codeRange.end + delta,
        },
      };
    }
    return object;
  });
}
