export interface NormalizedPdfRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

interface ClientRectLike {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

interface NormalizedPoint {
  readonly x: number;
  readonly y: number;
}

interface InverseSearchClick {
  readonly page: ClientRectLike;
  readonly clientX: number;
  readonly clientY: number;
  readonly detail: number;
  readonly selectionCollapsed: boolean;
  readonly targetInTextLayer: boolean;
}

interface SizeLike {
  readonly width: number;
  readonly height: number;
}

export interface FloatingEditorPosition {
  readonly left: number;
  readonly top: number;
}

const round = (value: number) => Math.round(value * 1_000_000) / 1_000_000;

export function boundedPdfRect(
  x: number,
  y: number,
  width: number,
  height: number,
): NormalizedPdfRect | null {
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
  const left = Math.max(0, Math.min(1, x));
  const top = Math.max(0, Math.min(1, y));
  const right = Math.max(0, Math.min(1, x + width));
  const bottom = Math.max(0, Math.min(1, y + height));
  if (right <= left || bottom <= top) return null;
  return {
    x: round(left),
    y: round(top),
    width: round(right - left),
    height: round(bottom - top),
  };
}

export function normalizePageRectangle(
  page: ClientRectLike,
  selection: ClientRectLike,
): NormalizedPdfRect | null {
  if (page.width <= 0 || page.height <= 0 || selection.width <= 0 || selection.height <= 0) {
    return null;
  }
  const epsilon = 0.5;
  if (
    selection.left < page.left - epsilon ||
    selection.top < page.top - epsilon ||
    selection.left + selection.width > page.left + page.width + epsilon ||
    selection.top + selection.height > page.top + page.height + epsilon
  ) {
    return null;
  }
  return boundedPdfRect(
    (selection.left - page.left) / page.width,
    (selection.top - page.top) / page.height,
    selection.width / page.width,
    selection.height / page.height,
  );
}

export function normalizeInverseSearchClick(click: InverseSearchClick): NormalizedPoint | null {
  const { page } = click;
  if (
    click.detail !== 1 ||
    !click.selectionCollapsed ||
    !click.targetInTextLayer ||
    page.width <= 0 ||
    page.height <= 0 ||
    ![click.clientX, click.clientY, page.left, page.top, page.width, page.height].every(
      Number.isFinite,
    ) ||
    click.clientX < page.left ||
    click.clientX > page.left + page.width ||
    click.clientY < page.top ||
    click.clientY > page.top + page.height
  ) {
    return null;
  }
  return {
    x: round((click.clientX - page.left) / page.width),
    y: round((click.clientY - page.top) / page.height),
  };
}

/** Convert an image unit-square transform into a normalized page rectangle. */
export function figureRegionFromTransform(
  transform: readonly number[],
  pageWidth: number,
  pageHeight: number,
): NormalizedPdfRect | null {
  if (transform.length !== 6 || pageWidth <= 0 || pageHeight <= 0) return null;
  const matrix = transform as readonly [number, number, number, number, number, number];
  const [a, b, c, d, e, f] = matrix;
  if (matrix.some((value) => !Number.isFinite(value))) return null;
  const xs = [e, a + e, c + e, a + c + e];
  const ys = [f, b + f, d + f, b + d + f];
  const left = Math.min(...xs) / pageWidth;
  const top = Math.min(...ys) / pageHeight;
  return boundedPdfRect(
    left,
    top,
    (Math.max(...xs) - Math.min(...xs)) / pageWidth,
    (Math.max(...ys) - Math.min(...ys)) / pageHeight,
  );
}

export function pointInsideRegion(
  region: NormalizedPdfRect | null,
  point: NormalizedPoint,
): NormalizedPoint | null {
  if (
    region === null ||
    point.x < region.x ||
    point.x > region.x + region.width ||
    point.y < region.y ||
    point.y > region.y + region.height
  ) {
    return null;
  }
  return {
    x: round((point.x - region.x) / region.width),
    y: round((point.y - region.y) / region.height),
  };
}

export function positionFloatingEditor(
  anchor: NormalizedPoint,
  editor: SizeLike,
  boundary: ClientRectLike,
  gap = 10,
): FloatingEditorPosition | null {
  if (
    ![anchor.x, anchor.y, editor.width, editor.height, boundary.left, boundary.top, gap].every(
      Number.isFinite,
    ) ||
    editor.width <= 0 ||
    editor.height <= 0 ||
    boundary.width <= 0 ||
    boundary.height <= 0 ||
    gap < 0
  ) {
    return null;
  }
  const minLeft = boundary.left + gap;
  const minTop = boundary.top + gap;
  const maxLeft = Math.max(minLeft, boundary.left + boundary.width - gap - editor.width);
  const maxTop = Math.max(minTop, boundary.top + boundary.height - gap - editor.height);
  const left = Math.max(minLeft, Math.min(maxLeft, anchor.x - editor.width / 2));
  const below = anchor.y + gap;
  const above = anchor.y - gap - editor.height;
  const top =
    below <= maxTop ? below : above >= minTop ? above : Math.max(minTop, Math.min(maxTop, below));
  return { left: round(left), top: round(top) };
}

export function roundPdfCoordinate(value: number): number {
  return round(value);
}
