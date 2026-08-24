import {
  type SciencePaperAnnotation,
  sciencePaperAnnotationSchema,
  type TypstDocumentPreview,
  type TypstSourceTarget,
  type TypstSourceUpdate,
  typstRelativePathSchema,
} from "@swarmx/dsh-science/types";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type { NormalizedPdfRect } from "./science-pdf-geometry.js";
import type {
  PdfFigureAnnotationRequest,
  PdfFigureLocator,
  PdfSourcePointRequest,
  PdfTextAnnotationRequest,
} from "./science-pdf-viewer.js";
import css from "./science-typst-side-view.module.css";

const SciencePdfViewer = lazy(async () => ({
  default: (await import("./science-pdf-viewer.js")).SciencePdfViewer,
}));

interface TypstPayload {
  readonly relativePath: string;
}

interface FigurePayload extends TypstPayload, PdfFigureLocator {
  readonly sourceRevision: `sha256:${string}`;
  readonly pdfRevision: `sha256:${string}`;
}

type CompiledTypstPreview = TypstDocumentPreview & {
  readonly pdfBase64: string;
  readonly pdfRevision: `sha256:${string}`;
  readonly pdfSourceRevision: `sha256:${string}`;
};

interface TypstInjected {
  readonly loadPreview: (
    relativePath: string,
    signal?: AbortSignal,
  ) => Promise<TypstDocumentPreview>;
  readonly updateSource: (
    request: {
      readonly relativePath: string;
      readonly expectedSourceRevision: string;
      readonly source: string;
    },
    signal?: AbortSignal,
  ) => Promise<TypstSourceUpdate>;
  readonly resolveSourceAtPoint: (
    request: {
      readonly relativePath: string;
      readonly pdfRevision: string;
      readonly page: number;
      readonly x: number;
      readonly y: number;
    },
    signal?: AbortSignal,
  ) => Promise<TypstSourceTarget | null>;
  readonly addAnnotationToConversation: (annotation: SciencePaperAnnotation) => boolean;
  readonly openFigure: (preview: TypstDocumentPreview, locator: PdfFigureLocator) => void;
}

interface FigureInjected {
  readonly loadPreview: (
    relativePath: string,
    signal?: AbortSignal,
  ) => Promise<TypstDocumentPreview>;
  readonly addAnnotationToConversation: (annotation: SciencePaperAnnotation) => boolean;
}

const SHA256 = /^sha256:[0-9a-f]{64}$/u;

function objectPayload(entry: SideViewEntry): Readonly<Record<string, unknown>> | null {
  const payload = entry.payload;
  return typeof payload === "object" && payload !== null && !Array.isArray(payload)
    ? (payload as Readonly<Record<string, unknown>>)
    : null;
}

export function typstPayload(entry: SideViewEntry): TypstPayload | null {
  const payload = objectPayload(entry);
  if (payload === null) return null;
  const path = typstRelativePathSchema.safeParse(payload.relativePath);
  return path.success ? { relativePath: path.data } : null;
}

function normalizedRect(value: unknown): NormalizedPdfRect | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return null;
  const rect = value as Readonly<Record<string, unknown>>;
  const numbers = [rect.x, rect.y, rect.width, rect.height];
  if (numbers.some((number) => typeof number !== "number" || !Number.isFinite(number))) return null;
  const [x, y, width, height] = numbers as [number, number, number, number];
  if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > 1 || y + height > 1) {
    return null;
  }
  return { x, y, width, height };
}

export function figurePayload(entry: SideViewEntry): FigurePayload | null {
  const payload = objectPayload(entry);
  const paper = typstPayload(entry);
  const rect = normalizedRect(payload?.rect);
  if (
    payload === null ||
    paper === null ||
    rect === null ||
    typeof payload.page !== "number" ||
    !Number.isInteger(payload.page) ||
    payload.page < 1 ||
    typeof payload.figureIndex !== "number" ||
    !Number.isInteger(payload.figureIndex) ||
    payload.figureIndex < 0 ||
    typeof payload.sourceRevision !== "string" ||
    !SHA256.test(payload.sourceRevision) ||
    typeof payload.pdfRevision !== "string" ||
    !SHA256.test(payload.pdfRevision)
  ) {
    return null;
  }
  return {
    ...paper,
    page: payload.page,
    figureIndex: payload.figureIndex,
    rect,
    sourceRevision: payload.sourceRevision as `sha256:${string}`,
    pdfRevision: payload.pdfRevision as `sha256:${string}`,
  };
}

export function pdfFigureSideViewEntry(
  preview: TypstDocumentPreview,
  locator: PdfFigureLocator,
): SideViewEntry {
  if (preview.pdfRevision === null || preview.pdfSourceRevision === null) {
    throw new TypeError("Figure tabs require a compiled PDF");
  }
  return {
    id: `science-pdf-figure:${preview.relativePath}:${locator.page}:${locator.figureIndex}`,
    kind: "science-pdf-figure",
    title: `Figure ${locator.figureIndex + 1} · ${preview.title}`,
    mode: "workbench",
    payload: {
      relativePath: preview.relativePath,
      sourceRevision: preview.pdfSourceRevision,
      pdfRevision: preview.pdfRevision,
      page: locator.page,
      figureIndex: locator.figureIndex,
      rect: {
        x: locator.rect.x,
        y: locator.rect.y,
        width: locator.rect.width,
        height: locator.rect.height,
      },
    },
  };
}

export function isCurrentFigurePreview(
  payload: FigurePayload,
  preview: TypstDocumentPreview,
): preview is CompiledTypstPreview {
  return (
    preview.pdfBase64 !== null &&
    preview.pdfRevision === payload.pdfRevision &&
    preview.pdfSourceRevision === payload.sourceRevision
  );
}

function annotation(
  preview: TypstDocumentPreview,
  request: PdfTextAnnotationRequest | PdfFigureAnnotationRequest,
): SciencePaperAnnotation | null {
  if (preview.pdfRevision === null || preview.pdfSourceRevision === null) return null;
  const common = {
    version: 1 as const,
    id: crypto.randomUUID(),
    relativePath: preview.relativePath,
    title: preview.title,
    sourceRevision: preview.pdfSourceRevision,
    pdfRevision: preview.pdfRevision,
    page: request.page,
    rect: request.rect,
    comment: request.comment,
    createdAt: Date.now(),
  };
  return sciencePaperAnnotationSchema.parse(
    "selectedText" in request
      ? { ...common, kind: "text", selectedText: request.selectedText }
      : {
          ...common,
          kind: "figure-point",
          figureIndex: request.figureIndex,
          x: request.x,
          y: request.y,
        },
  );
}

function Diagnostics({ preview }: { readonly preview: TypstDocumentPreview }) {
  if (preview.diagnostics.length === 0) return null;
  return (
    <details className={css.diagnostics} open={preview.status === "error"}>
      <summary>
        {preview.status === "stale" ? "Showing last successful PDF" : "Typst diagnostics"}
      </summary>
      <pre>{preview.diagnostics.join("\n")}</pre>
    </details>
  );
}

export function ScienceTypstSideView({
  entry,
  loadPreview,
  updateSource,
  resolveSourceAtPoint,
  addAnnotationToConversation,
  openFigure,
}: SideViewContentOwnerProps & TypstInjected) {
  const payload = typstPayload(entry);
  if (payload === null) {
    return (
      <p role="alert" className={css.state}>
        Invalid Typst paper link.
      </p>
    );
  }
  return (
    <ScienceTypstPaperWorkbench
      key={payload.relativePath}
      relativePath={payload.relativePath}
      loadPreview={loadPreview}
      updateSource={updateSource}
      resolveSourceAtPoint={resolveSourceAtPoint}
      addAnnotationToConversation={addAnnotationToConversation}
      openFigure={openFigure}
    />
  );
}

function ScienceTypstPaperWorkbench({
  relativePath,
  loadPreview,
  updateSource,
  resolveSourceAtPoint,
  addAnnotationToConversation,
  openFigure,
}: TypstInjected & { readonly relativePath: string }) {
  const [preview, setPreview] = useState<TypstDocumentPreview | Error | null>(null);
  const [mode, setMode] = useState<"source" | "pdf">("pdf");
  const [draft, setDraft] = useState("");
  const [savedSource, setSavedSource] = useState("");
  const [baselineRevision, setBaselineRevision] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [activePath, setActivePath] = useState<string | null>(null);
  const [activeTitle, setActiveTitle] = useState<string | null>(null);
  const [caretOffset, setCaretOffset] = useState<number | null>(null);
  const activeSave = useRef(false);
  const activeResolution = useRef<AbortController | null>(null);
  const draftRef = useRef("");
  const sourceEditor = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let disposed = false;
    let controller: AbortController | null = null;
    let timer: number | null = null;
    const poll = async () => {
      if (disposed) return;
      if (document.hidden) {
        timer = window.setTimeout(poll, 750);
        return;
      }
      controller = new AbortController();
      try {
        const next = await loadPreview(relativePath, controller.signal);
        if (!disposed) setPreview(next);
      } catch (error) {
        if (!disposed && !(error instanceof DOMException && error.name === "AbortError")) {
          setPreview(error instanceof Error ? error : new Error("Unable to load Typst preview"));
        }
      } finally {
        controller = null;
        if (!disposed) timer = window.setTimeout(poll, 750);
      }
    };
    void poll();
    return () => {
      disposed = true;
      controller?.abort();
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [loadPreview, relativePath]);

  useEffect(() => {
    if (preview === null || preview instanceof Error) return;
    if (activePath === null) {
      setActivePath(preview.relativePath);
      setActiveTitle(preview.title);
      setDraft(preview.source);
      draftRef.current = preview.source;
      setSavedSource(preview.source);
      setBaselineRevision(preview.sourceRevision);
      setSaveError(null);
    } else if (
      activePath === preview.relativePath &&
      (baselineRevision === null || (!dirty && baselineRevision !== preview.sourceRevision))
    ) {
      setDraft(preview.source);
      draftRef.current = preview.source;
      setSavedSource(preview.source);
      setBaselineRevision(preview.sourceRevision);
      setSaveError(null);
    } else if (
      activePath === preview.relativePath &&
      dirty &&
      !activeSave.current &&
      baselineRevision !== preview.sourceRevision
    ) {
      setSaveError(
        "The source changed outside this editor. Copy your draft, reload, and merge the changes.",
      );
    }
  }, [activePath, baselineRevision, dirty, preview]);

  const save = useCallback(async (): Promise<boolean> => {
    if (!dirty) return true;
    if (activePath === null || baselineRevision === null || activeSave.current) return false;
    activeSave.current = true;
    setSaving(true);
    setSaveError(null);
    const source = draftRef.current;
    try {
      const updated = await updateSource({
        relativePath: activePath,
        expectedSourceRevision: baselineRevision,
        source,
      });
      setBaselineRevision(updated.sourceRevision);
      setSavedSource(updated.source);
      setSaveError(null);
      if (draftRef.current === source) {
        setDraft(updated.source);
        draftRef.current = updated.source;
        setDirty(false);
      } else {
        setDirty(true);
      }
      return true;
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Unable to save Typst source");
      return false;
    } finally {
      activeSave.current = false;
      setSaving(false);
    }
  }, [activePath, baselineRevision, dirty, updateSource]);

  useEffect(
    () => () => {
      activeResolution.current?.abort();
    },
    [],
  );

  useEffect(() => {
    if (mode !== "source" || caretOffset === null) return;
    const editor = sourceEditor.current;
    if (editor === null) return;
    const offset = Math.min(caretOffset, editor.value.length);
    editor.focus();
    editor.setSelectionRange(offset, offset);
    const before = editor.value.slice(0, offset).split("\n").length - 1;
    const lines = Math.max(1, editor.value.split("\n").length - 1);
    editor.scrollTop = (before / lines) * Math.max(0, editor.scrollHeight - editor.clientHeight);
    setCaretOffset(null);
  }, [caretOffset, mode]);

  const openSourceAtPoint = useCallback(
    async (point: PdfSourcePointRequest) => {
      if (preview === null || preview instanceof Error || preview.pdfRevision === null) return;
      if (!(await save())) return;
      activeResolution.current?.abort();
      const controller = new AbortController();
      activeResolution.current = controller;
      setSaveError(null);
      try {
        const target = await resolveSourceAtPoint(
          {
            relativePath: preview.relativePath,
            pdfRevision: preview.pdfRevision,
            ...point,
          },
          controller.signal,
        );
        if (target === null) {
          setSaveError("No editable Typst source is attached to this PDF position.");
          return;
        }
        setActivePath(target.relativePath);
        setActiveTitle(target.title);
        setDraft(target.source);
        draftRef.current = target.source;
        setSavedSource(target.source);
        setBaselineRevision(target.sourceRevision);
        setDirty(false);
        setMode("source");
        setCaretOffset(target.offset);
      } catch (error) {
        if (!controller.signal.aborted) {
          setSaveError(error instanceof Error ? error.message : "Unable to locate Typst source");
        }
      } finally {
        if (activeResolution.current === controller) activeResolution.current = null;
      }
    },
    [preview, resolveSourceAtPoint, save],
  );

  useEffect(() => {
    if (!dirty || saveError !== null) return;
    const timer = window.setTimeout(() => void save(), 500);
    return () => window.clearTimeout(timer);
  }, [dirty, save, saveError]);

  if (preview === null)
    return (
      <p role="status" className={css.state}>
        Starting Typst preview…
      </p>
    );
  if (preview instanceof Error)
    return (
      <p role="alert" className={css.state}>
        {preview.message}
      </p>
    );

  const addAnnotation = (request: PdfTextAnnotationRequest | PdfFigureAnnotationRequest) => {
    const next = annotation(preview, request);
    if (next !== null && !addAnnotationToConversation(next)) {
      setSaveError("Open this session's Chat to add the annotation.");
    }
  };

  return (
    <section className={css.root}>
      <header className={css.header}>
        <span className={css.fileIdentity}>
          <small>Typst paper</small>
          <strong
            title={mode === "source" ? (activePath ?? preview.relativePath) : preview.relativePath}
          >
            {mode === "source" ? (activeTitle ?? preview.title) : preview.title}
          </strong>
        </span>
        <span className={css.headerActions}>
          {mode === "source" && (
            <span className={css.saveState}>
              {saving ? "Saving…" : dirty ? "Unsaved" : "Saved"}
            </span>
          )}
          <span className={css.toggle} aria-label="Paper view">
            <button
              type="button"
              aria-pressed={mode === "source"}
              onClick={() => setMode("source")}
            >
              Source
            </button>
            <button type="button" aria-pressed={mode === "pdf"} onClick={() => setMode("pdf")}>
              PDF
            </button>
          </span>
        </span>
      </header>
      {saveError !== null && (
        <p role="alert" className={css.error}>
          {saveError}
        </p>
      )}
      {mode === "source" ? (
        <div className={css.sourcePane}>
          <textarea
            ref={sourceEditor}
            aria-label="Typst source"
            spellCheck={false}
            value={draft}
            onChange={(event) => {
              const source = event.currentTarget.value;
              setDraft(source);
              draftRef.current = source;
              setDirty(source !== savedSource);
              setSaveError(null);
            }}
          />
          <footer>
            <span>Changes save automatically and trigger Typst.</span>
            <button type="button" disabled={!dirty || saving} onClick={() => void save()}>
              Save now
            </button>
          </footer>
        </div>
      ) : preview.pdfBase64 === null ? (
        <div className={css.emptyPdf}>
          <p role="status">
            {preview.status === "compiling" ? "Compiling paper…" : "No PDF is available."}
          </p>
          <Diagnostics preview={preview} />
        </div>
      ) : (
        <div className={css.pdfPane}>
          <Diagnostics preview={preview} />
          <Suspense
            fallback={
              <p role="status" className={css.state}>
                Loading PDF.js…
              </p>
            }
          >
            <SciencePdfViewer
              pdfBase64={preview.pdfBase64}
              title={preview.title}
              onAnnotateText={addAnnotation}
              onAnnotateFigure={addAnnotation}
              onOpenFigure={(locator) => openFigure(preview, locator)}
              onResolveSource={(point) => void openSourceAtPoint(point)}
            />
          </Suspense>
        </div>
      )}
    </section>
  );
}

export function SciencePdfFigureSideView({
  entry,
  loadPreview,
  addAnnotationToConversation,
}: SideViewContentOwnerProps & FigureInjected) {
  const payload = figurePayload(entry);
  const relativePath = payload?.relativePath;
  const [preview, setPreview] = useState<TypstDocumentPreview | Error | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  useEffect(() => {
    if (relativePath === undefined) return;
    const controller = new AbortController();
    void loadPreview(relativePath, controller.signal).then(setPreview, (error: unknown) => {
      if (!controller.signal.aborted) {
        setPreview(error instanceof Error ? error : new Error("Unable to load figure"));
      }
    });
    return () => controller.abort();
  }, [loadPreview, relativePath]);

  if (payload === null)
    return (
      <p role="alert" className={css.state}>
        Invalid paper figure link.
      </p>
    );
  if (preview === null)
    return (
      <p role="status" className={css.state}>
        Opening figure…
      </p>
    );
  if (preview instanceof Error)
    return (
      <p role="alert" className={css.state}>
        {preview.message}
      </p>
    );
  if (!isCurrentFigurePreview(payload, preview)) {
    return (
      <p role="alert" className={css.state}>
        This figure belongs to an older PDF revision. Re-open it from the current paper preview.
      </p>
    );
  }
  return (
    <section className={css.figureRoot}>
      <header className={css.figureHeader}>
        <span>
          <small>Paper figure workbench</small>
          <strong>
            Figure {payload.figureIndex + 1} · page {payload.page}
          </strong>
        </span>
        <p>Click a point and send an edit request to Chat.</p>
      </header>
      {notice !== null && <p className={css.error}>{notice}</p>}
      <Suspense
        fallback={
          <p role="status" className={css.state}>
            Loading PDF.js…
          </p>
        }
      >
        <SciencePdfViewer
          pdfBase64={preview.pdfBase64}
          title={entry.title}
          focus={payload}
          onAnnotateFigure={(request) => {
            const next = annotation(preview, request);
            if (next !== null) {
              setNotice(
                addAnnotationToConversation(next)
                  ? "Annotation added to Chat."
                  : "Open this session's Chat to add the annotation.",
              );
            }
          }}
        />
      </Suspense>
    </section>
  );
}
