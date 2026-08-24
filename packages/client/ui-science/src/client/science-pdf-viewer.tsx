import {
  GlobalWorkerOptions,
  getDocument,
  type PDFDocumentLoadingTask,
  type PDFDocumentProxy,
  type PDFPageProxy,
  type RenderTask,
  TextLayer,
} from "pdfjs-dist";
import pdfWorkerSource from "pdfjs-dist/build/pdf.worker.mjs?raw";
import {
  type CSSProperties,
  type FormEvent,
  type MouseEvent,
  type PointerEvent,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  boundedPdfRect,
  type NormalizedPdfRect,
  normalizeInverseSearchClick,
  normalizePageRectangle,
  positionFloatingEditor,
  roundPdfCoordinate,
} from "./science-pdf-geometry.js";
import css from "./science-pdf-viewer.module.css";

const pdfWorkerUrl = URL.createObjectURL(new Blob([pdfWorkerSource], { type: "text/javascript" }));
GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export type { NormalizedPdfRect } from "./science-pdf-geometry.js";
export {
  figureRegionFromTransform,
  normalizePageRectangle,
  pointInsideRegion,
} from "./science-pdf-geometry.js";

export interface PdfTextAnnotationRequest {
  readonly page: number;
  readonly rect: NormalizedPdfRect;
  readonly selectedText: string;
  readonly comment: string;
}

export interface PdfFigureAnnotationRequest {
  readonly page: number;
  readonly figureIndex: number;
  readonly rect: NormalizedPdfRect;
  readonly x: number;
  readonly y: number;
  readonly comment: string;
}

export interface PdfFigureLocator {
  readonly page: number;
  readonly figureIndex: number;
  readonly rect: NormalizedPdfRect;
}

export interface PdfSourcePointRequest {
  readonly page: number;
  readonly x: number;
  readonly y: number;
}

interface SciencePdfViewerProps {
  readonly pdfBase64: string;
  readonly title: string;
  readonly focus?: PdfFigureLocator;
  readonly onAnnotateText?: (request: PdfTextAnnotationRequest) => void;
  readonly onAnnotateFigure: (request: PdfFigureAnnotationRequest) => void;
  readonly onOpenFigure?: (request: PdfFigureLocator) => void;
  readonly onResolveSource?: (request: PdfSourcePointRequest) => void;
}

type PendingAnnotation =
  | ({ readonly kind: "text"; readonly anchor: ClientPoint } & Omit<
      PdfTextAnnotationRequest,
      "comment"
    >)
  | ({ readonly kind: "figure"; readonly anchor: ClientPoint } & Omit<
      PdfFigureAnnotationRequest,
      "comment"
    >);

interface ClientPoint {
  readonly x: number;
  readonly y: number;
}

function regionFromRecordedImage(coordinates: ArrayLike<number>, offset: number) {
  const xs = [coordinates[offset], coordinates[offset + 2], coordinates[offset + 4]] as const;
  const ys = [coordinates[offset + 1], coordinates[offset + 3], coordinates[offset + 5]] as const;
  if (
    xs.some((value) => value === undefined || !Number.isFinite(value)) ||
    ys.some((value) => value === undefined || !Number.isFinite(value))
  ) {
    return null;
  }
  const values = [...xs, ...ys] as readonly number[];
  const [x0, x1, x2, y0, y1, y2] = values as readonly [
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  const left = Math.min(x0, x1, x2);
  const top = Math.min(y0, y1, y2);
  return boundedPdfRect(left, top, Math.max(x0, x1, x2) - left, Math.max(y0, y1, y2) - top);
}

function recordedFigureRegions(coordinates: unknown): readonly NormalizedPdfRect[] {
  if (
    coordinates === null ||
    typeof coordinates !== "object" ||
    !("length" in coordinates) ||
    typeof coordinates.length !== "number"
  ) {
    return [];
  }
  const values = coordinates as ArrayLike<number>;
  const regions: NormalizedPdfRect[] = [];
  for (let offset = 0; offset + 5 < values.length && regions.length < 100; offset += 6) {
    const region = regionFromRecordedImage(values, offset);
    if (region !== null && region.width * region.height >= 0.0005) regions.push(region);
  }
  return regions;
}

function base64Bytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function usePdfDocument(pdfBase64: string) {
  const [document, setDocument] = useState<PDFDocumentProxy | Error | null>(null);
  useEffect(() => {
    let active = true;
    let task: PDFDocumentLoadingTask | null = null;
    try {
      task = getDocument({ data: base64Bytes(pdfBase64) });
      void task.promise.then(
        (next) => {
          if (active) setDocument(next);
          else void task?.destroy();
        },
        (error: unknown) => {
          if (active) setDocument(error instanceof Error ? error : new Error("Unable to open PDF"));
        },
      );
    } catch (error) {
      setDocument(error instanceof Error ? error : new Error("Unable to decode PDF"));
    }
    return () => {
      active = false;
      if (task !== null) void task.destroy();
    };
  }, [pdfBase64]);
  return document;
}

function useNearViewport(root: HTMLDivElement | null): boolean {
  const [near, setNear] = useState(false);
  useEffect(() => {
    if (root === null) return;
    if (typeof IntersectionObserver === "undefined") {
      setNear(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => setNear(entry?.isIntersecting ?? false),
      { rootMargin: "800px 0px" },
    );
    observer.observe(root);
    return () => observer.disconnect();
  }, [root]);
  return near;
}

interface PdfPageProps {
  readonly document: PDFDocumentProxy;
  readonly pageNumber: number;
  readonly scale: number;
  readonly focus?: PdfFigureLocator;
  readonly onFigureClick: (event: MouseEvent<HTMLButtonElement>, locator: PdfFigureLocator) => void;
  readonly onFigureDoubleClick: (locator: PdfFigureLocator) => void;
  readonly onSelectText: (
    event: PointerEvent<HTMLDivElement>,
    page: number,
    pageElement: HTMLDivElement,
  ) => void;
  readonly onTextClick: (
    event: MouseEvent<HTMLDivElement>,
    page: number,
    pageElement: HTMLDivElement,
  ) => void;
}

function PdfPage({
  document,
  pageNumber,
  scale,
  focus,
  onFigureClick,
  onFigureDoubleClick,
  onSelectText,
  onTextClick,
}: PdfPageProps) {
  const [root, setRoot] = useState<HTMLDivElement | null>(null);
  const near = useNearViewport(root);
  const canvas = useRef<HTMLCanvasElement>(null);
  const textLayer = useRef<HTMLDivElement>(null);
  const [aspectRatio, setAspectRatio] = useState(0.7727);
  const [viewportSize, setViewportSize] = useState({ width: 612, height: 792, userUnit: 1 });
  const [regions, setRegions] = useState<readonly NormalizedPdfRect[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!near || canvas.current === null || textLayer.current === null) return;
    let active = true;
    let page: PDFPageProxy | null = null;
    let renderTask: RenderTask | null = null;
    let layer: TextLayer | null = null;
    void document
      .getPage(pageNumber)
      .then(async (loadedPage) => {
        if (!active || canvas.current === null || textLayer.current === null) return;
        page = loadedPage;
        const viewport = loadedPage.getViewport({ scale });
        const visibleWidth = viewport.width * (focus?.rect.width ?? 1);
        const visibleHeight = viewport.height * (focus?.rect.height ?? 1);
        setAspectRatio(visibleWidth / visibleHeight);
        setViewportSize({
          width: visibleWidth,
          height: visibleHeight,
          userUnit: loadedPage.userUnit,
        });
        const outputScale = Math.min(window.devicePixelRatio || 1, 2);
        const target = canvas.current;
        const renderTarget =
          focus === undefined ? target : globalThis.document.createElement("canvas");
        renderTarget.width = Math.floor(viewport.width * outputScale);
        renderTarget.height = Math.floor(viewport.height * outputScale);
        const context = renderTarget.getContext("2d", { alpha: false });
        if (context === null) throw new Error("Canvas is unavailable");
        renderTask = loadedPage.render({
          canvas: renderTarget,
          canvasContext: context,
          viewport,
          transform: outputScale === 1 ? undefined : [outputScale, 0, 0, outputScale, 0, 0],
          recordImages: true,
        });
        await renderTask.promise;
        if (!active) return;
        setRegions(recordedFigureRegions(loadedPage.imageCoordinates));
        if (focus !== undefined) {
          const sourceX = Math.floor(renderTarget.width * focus.rect.x);
          const sourceY = Math.floor(renderTarget.height * focus.rect.y);
          const sourceWidth = Math.max(1, Math.floor(renderTarget.width * focus.rect.width));
          const sourceHeight = Math.max(1, Math.floor(renderTarget.height * focus.rect.height));
          target.width = sourceWidth;
          target.height = sourceHeight;
          target.style.aspectRatio = `${sourceWidth} / ${sourceHeight}`;
          const cropContext = target.getContext("2d", { alpha: false });
          if (cropContext === null) throw new Error("Canvas is unavailable");
          cropContext.drawImage(
            renderTarget,
            sourceX,
            sourceY,
            sourceWidth,
            sourceHeight,
            0,
            0,
            sourceWidth,
            sourceHeight,
          );
        } else {
          target.style.aspectRatio = `${viewport.width} / ${viewport.height}`;
        }
        if (focus === undefined) {
          textLayer.current.replaceChildren();
          layer = new TextLayer({
            textContentSource: await loadedPage.getTextContent(),
            container: textLayer.current,
            viewport,
          });
          await layer.render();
        }
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Unable to render PDF page");
      });
    return () => {
      active = false;
      renderTask?.cancel();
      layer?.cancel();
      page?.cleanup();
    };
  }, [document, focus, near, pageNumber, scale]);

  const visibleRegions =
    focus === undefined
      ? regions.map((rect, figureIndex) => ({
          rect,
          locator: { page: pageNumber, figureIndex, rect },
        }))
      : [
          {
            rect: { x: 0, y: 0, width: 1, height: 1 },
            locator: {
              page: pageNumber,
              figureIndex: focus.figureIndex,
              rect: focus.rect,
            },
          },
        ];
  const pageStyle = {
    aspectRatio,
    width: `${viewportSize.width}px`,
    height: `${viewportSize.height}px`,
    "--scale-factor": String(scale),
    "--user-unit": String(viewportSize.userUnit),
    "--total-scale-factor": "calc(var(--scale-factor) * var(--user-unit))",
    "--scale-round-x": "1px",
    "--scale-round-y": "1px",
  } as CSSProperties;
  return (
    <section className={css.pageSection} aria-label={`Page ${pageNumber}`}>
      <span className={css.pageNumber}>Page {pageNumber}</span>
      {/* biome-ignore lint/a11y/useKeyWithClickEvents: inverse search requires a PDF spatial point; the Source toggle is the keyboard path. */}
      <div
        ref={setRoot}
        className={focus === undefined ? css.page : css.focusPage}
        data-pdf-page={pageNumber}
        style={pageStyle}
        onPointerUp={(event) => onSelectText(event, pageNumber, event.currentTarget)}
        onClick={(event) => onTextClick(event, pageNumber, event.currentTarget)}
      >
        <canvas ref={canvas} className={css.canvas} aria-label={`${document.numPages}-page PDF`} />
        <div ref={textLayer} className={`textLayer ${css.textLayer}`} />
        {visibleRegions.map(({ rect, locator }) => {
          return (
            <button
              key={`${rect.x}:${rect.y}:${locator.figureIndex}`}
              type="button"
              className={focus === undefined ? css.figureRegion : css.focusFigureRegion}
              style={{
                left: `${rect.x * 100}%`,
                top: `${rect.y * 100}%`,
                width: `${rect.width * 100}%`,
                height: `${rect.height * 100}%`,
              }}
              aria-label={
                focus === undefined
                  ? `Annotate figure ${locator.figureIndex + 1}; double-click to open`
                  : `Annotate figure ${locator.figureIndex + 1}`
              }
              onClick={(event) => onFigureClick(event, locator)}
              onDoubleClick={() => onFigureDoubleClick(locator)}
            />
          );
        })}
        {error !== null && <p className={css.pageError}>{error}</p>}
      </div>
    </section>
  );
}

export function SciencePdfViewer({
  pdfBase64,
  title,
  focus,
  onAnnotateText,
  onAnnotateFigure,
  onOpenFigure,
  onResolveSource,
}: SciencePdfViewerProps) {
  const document = usePdfDocument(pdfBase64);
  const [scale, setScale] = useState(focus === undefined ? 1.15 : 2);
  const [pending, setPending] = useState<PendingAnnotation | null>(null);
  const [comment, setComment] = useState("");
  const [editorPosition, setEditorPosition] = useState<{
    readonly left: number;
    readonly top: number;
  } | null>(null);
  const viewer = useRef<HTMLElement>(null);
  const editorForm = useRef<HTMLFormElement>(null);
  const editor = useRef<HTMLTextAreaElement>(null);
  const clickTimer = useRef<number | null>(null);
  const pages = useMemo(
    () =>
      document instanceof Error || document === null
        ? []
        : Array.from({ length: document.numPages }, (_, index) => index + 1),
    [document],
  );

  useEffect(() => {
    if (pending !== null) editor.current?.focus();
  }, [pending]);
  useLayoutEffect(() => {
    const viewerElement = viewer.current;
    const editorElement = editorForm.current;
    if (pending === null || viewerElement === null || editorElement === null) {
      setEditorPosition(null);
      return;
    }
    const update = () => {
      const viewerBounds = viewerElement.getBoundingClientRect();
      const left = Math.max(0, viewerBounds.left);
      const top = Math.max(0, viewerBounds.top);
      const right = Math.min(window.innerWidth, viewerBounds.right);
      const bottom = Math.min(window.innerHeight, viewerBounds.bottom);
      setEditorPosition(
        positionFloatingEditor(pending.anchor, editorElement.getBoundingClientRect(), {
          left,
          top,
          width: right - left,
          height: bottom - top,
        }),
      );
    };
    update();
    window.addEventListener("resize", update);
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(viewerElement);
    observer?.observe(editorElement);
    return () => {
      window.removeEventListener("resize", update);
      observer?.disconnect();
    };
  }, [pending]);
  useEffect(
    () => () => {
      if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
    },
    [],
  );

  function selectText(
    event: PointerEvent<HTMLDivElement>,
    page: number,
    pageElement: HTMLDivElement,
  ) {
    if (focus !== undefined || onAnnotateText === undefined) return;
    const selection = window.getSelection();
    if (selection === null || selection.isCollapsed || selection.rangeCount !== 1) return;
    const selectedText = selection.toString().trim().slice(0, 8_000);
    const range = selection.getRangeAt(0);
    if (!pageElement.contains(range.startContainer) || !pageElement.contains(range.endContainer)) {
      return;
    }
    const rect = normalizePageRectangle(
      pageElement.getBoundingClientRect(),
      range.getBoundingClientRect(),
    );
    if (rect === null || selectedText.length === 0) return;
    setComment("");
    setEditorPosition(null);
    setPending({
      kind: "text",
      page,
      rect,
      selectedText,
      anchor: { x: event.clientX, y: event.clientY },
    });
  }

  function pageClick(event: MouseEvent<HTMLDivElement>, page: number, pageElement: HTMLDivElement) {
    if (focus !== undefined || onResolveSource === undefined) return;
    const selection = window.getSelection();
    const target = event.target;
    const point = normalizeInverseSearchClick({
      page: pageElement.getBoundingClientRect(),
      clientX: event.clientX,
      clientY: event.clientY,
      detail: event.detail,
      selectionCollapsed: selection?.isCollapsed ?? true,
      targetInTextLayer:
        target instanceof Element && target.closest(".textLayer")?.parentElement === pageElement,
    });
    if (point !== null) onResolveSource({ page, ...point });
  }

  function figurePoint(event: MouseEvent<HTMLButtonElement>, locator: PdfFigureLocator) {
    if (event.detail === 0) {
      const bounds = event.currentTarget.getBoundingClientRect();
      setComment("");
      setEditorPosition(null);
      setPending({
        kind: "figure",
        ...locator,
        x: 0.5,
        y: 0.5,
        anchor: { x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2 },
      });
      return;
    }
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    const y = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
    const anchor = { x: event.clientX, y: event.clientY };
    if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
    clickTimer.current = window.setTimeout(() => {
      setComment("");
      setEditorPosition(null);
      setPending({
        kind: "figure",
        ...locator,
        x: roundPdfCoordinate(x),
        y: roundPdfCoordinate(y),
        anchor,
      });
      clickTimer.current = null;
    }, 220);
  }

  function openFigure(locator: PdfFigureLocator) {
    if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
    clickTimer.current = null;
    setPending(null);
    onOpenFigure?.(locator);
  }

  function submitAnnotation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending === null) return;
    const trimmed = comment.trim();
    if (trimmed.length === 0) return;
    const { anchor: _anchor, ...annotation } = pending;
    if (annotation.kind === "text") onAnnotateText?.({ ...annotation, comment: trimmed });
    else onAnnotateFigure({ ...annotation, comment: trimmed });
    setPending(null);
    setComment("");
    window.getSelection()?.removeAllRanges();
  }

  return (
    <section ref={viewer} className={css.root} aria-label={`${title} PDF preview`}>
      {focus === undefined && (
        <header className={css.toolbar}>
          <span>
            {document instanceof Error || document === null ? "PDF" : `${document.numPages} pages`}
          </span>
          <span className={css.zoomControls}>
            <button type="button" onClick={() => setScale((value) => Math.max(0.75, value - 0.15))}>
              −
            </button>
            <output>{Math.round(scale * 100)}%</output>
            <button type="button" onClick={() => setScale((value) => Math.min(2.5, value + 0.15))}>
              +
            </button>
          </span>
        </header>
      )}
      <div className={css.document}>
        {document === null ? (
          <p role="status" className={css.state}>
            Opening PDF…
          </p>
        ) : document instanceof Error ? (
          <p role="alert" className={css.state}>
            {document.message}
          </p>
        ) : (
          pages
            .filter((page) => focus === undefined || page === focus.page)
            .map((page) => (
              <PdfPage
                key={page}
                document={document}
                pageNumber={page}
                scale={scale}
                {...(focus === undefined ? {} : { focus })}
                onFigureClick={figurePoint}
                onFigureDoubleClick={openFigure}
                onSelectText={selectText}
                onTextClick={pageClick}
              />
            ))
        )}
      </div>
      {pending !== null && (
        <form
          ref={editorForm}
          role="dialog"
          aria-modal={false}
          aria-label="Paper annotation"
          className={css.annotationEditor}
          style={
            editorPosition === null
              ? { visibility: "hidden" }
              : { left: `${editorPosition.left}px`, top: `${editorPosition.top}px` }
          }
          onSubmit={submitAnnotation}
        >
          <label htmlFor="paper-annotation-comment">
            {pending.kind === "text"
              ? `Discuss selected text on page ${pending.page}`
              : `Discuss figure ${pending.figureIndex + 1} on page ${pending.page}`}
          </label>
          {pending.kind === "text" && <blockquote>{pending.selectedText}</blockquote>}
          <textarea
            ref={editor}
            id="paper-annotation-comment"
            maxLength={2_000}
            value={comment}
            placeholder="What should change?"
            onChange={(event) => setComment(event.currentTarget.value)}
          />
          <span className={css.editorActions}>
            <button type="button" onClick={() => setPending(null)}>
              Cancel
            </button>
            <button type="submit" disabled={comment.trim().length === 0}>
              Add to Chat
            </button>
          </span>
        </form>
      )}
    </section>
  );
}
