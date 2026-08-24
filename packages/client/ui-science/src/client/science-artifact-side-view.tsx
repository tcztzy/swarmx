import {
  IconBranchOutline16,
  IconEditOutline16,
  IconEllipsisOutline16,
  IconFullscreenOutline16,
  IconSendOutline16,
  IconTrashOutline16,
  Menu,
  type MenuItem,
} from "@deepseek-ai/dsh-client-ui-primitives";
import {
  type RoCrateEntity,
  type RoCrateMetadataDocument,
  roCrateEntityId,
  type ScienceArtifact,
  type ScienceArtifactPreview,
  type ScienceImageAnnotation,
  scienceImageAnnotationSchema,
} from "@swarmx/dsh-science/types";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import {
  type ComponentType,
  type FormEvent,
  type MouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import css from "./science-artifact-side-view.module.css";

export interface ScienceArtifactPayload {
  readonly artifactId: string;
  readonly projectId: string;
  readonly artifactKind: string;
  readonly title: string;
  readonly digest: string;
  readonly mime: string;
  readonly size: number;
  readonly runId: string | null;
  readonly sourceEntityIds: readonly string[];
  readonly revision: number;
  readonly journalSeq: number;
  readonly annotations: readonly ScienceImageAnnotation[];
}

export const SCIENCE_ARTIFACT_ACTIONS = [
  { id: "provenance", label: "Provenance" },
] as const satisfies readonly MenuItem[];

export function artifactPayload(entry: SideViewEntry): ScienceArtifactPayload | null {
  const payload = entry.payload;
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  const values = payload as { readonly [key: string]: unknown };
  const sourceEntityIds = values.sourceEntityIds;
  const annotations = values.annotations;
  if (
    typeof values.artifactId !== "string" ||
    typeof values.projectId !== "string" ||
    typeof values.artifactKind !== "string" ||
    typeof values.title !== "string" ||
    typeof values.digest !== "string" ||
    typeof values.mime !== "string" ||
    typeof values.size !== "number" ||
    (values.runId !== null && typeof values.runId !== "string") ||
    !Array.isArray(sourceEntityIds) ||
    !sourceEntityIds.every((value) => typeof value === "string") ||
    typeof values.revision !== "number" ||
    typeof values.journalSeq !== "number" ||
    !Array.isArray(annotations)
  ) {
    return null;
  }
  const parsedAnnotations = annotations.map((annotation) =>
    scienceImageAnnotationSchema.safeParse(annotation),
  );
  if (
    parsedAnnotations.some(({ success }) => !success) ||
    parsedAnnotations.some(
      (parsed) =>
        parsed.success &&
        (parsed.data.artifactId !== values.artifactId ||
          parsed.data.projectId !== values.projectId ||
          parsed.data.title !== values.title ||
          parsed.data.digest !== values.digest ||
          parsed.data.mime !== values.mime),
    )
  ) {
    return null;
  }
  return {
    artifactId: values.artifactId,
    projectId: values.projectId,
    artifactKind: values.artifactKind,
    title: values.title,
    digest: values.digest,
    mime: values.mime,
    size: values.size,
    runId: values.runId,
    sourceEntityIds,
    revision: values.revision,
    journalSeq: values.journalSeq,
    annotations: parsedAnnotations.map((parsed) => {
      if (!parsed.success) throw new Error("Unreachable invalid annotation");
      return parsed.data;
    }),
  };
}

/** Create a bounded locator; artifact bytes and host paths remain behind the host service. */
export function scienceArtifactSideViewEntry(
  artifact: ScienceArtifact,
  annotations: readonly ScienceImageAnnotation[] = [],
): SideViewEntry {
  return {
    id: `science-artifact:${artifact.id}`,
    kind: "science-artifact",
    title: artifact.title,
    mode: "workbench",
    payload: {
      artifactId: artifact.id,
      projectId: artifact.projectId,
      artifactKind: artifact.kind,
      title: artifact.title,
      digest: artifact.digest,
      mime: artifact.mime,
      size: artifact.size,
      runId: artifact.runId,
      sourceEntityIds: artifact.sourceEntityIds,
      revision: artifact.revision,
      journalSeq: artifact.provenance.journalSeq,
      annotations: scienceImageAnnotationSchema.array().max(100).parse(annotations),
    },
  };
}

export function withScienceArtifactAnnotations(
  entry: SideViewEntry,
  annotations: readonly ScienceImageAnnotation[],
): SideViewEntry | null {
  const artifact = artifactPayload(entry);
  if (artifact === null) return null;
  const next = scienceImageAnnotationSchema.array().max(100).parse(annotations);
  return { ...entry, payload: { ...artifact, annotations: next } };
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

export function ScienceArtifactPreviewView({
  preview,
}: {
  readonly preview: ScienceArtifactPreview | Error | null;
}) {
  if (preview === null) return <p role="status">Loading preview…</p>;
  if (preview instanceof Error) return <p role="alert">{preview.message}</p>;
  if (preview.kind === "table") return <ScienceArtifactTablePreview preview={preview} />;
  if (preview.kind === "text") return <pre className={css.previewText}>{preview.text}</pre>;
  if (preview.kind === "image") {
    return <img className={css.previewImage} src={preview.dataUrl} alt="Artifact preview" />;
  }
  return (
    <p className={css.previewUnavailable}>
      Preview unavailable:{" "}
      {preview.reason === "too-large" ? "file is too large" : "format is unsupported"}.
    </p>
  );
}

type TablePreview = Extract<ScienceArtifactPreview, { kind: "table" }>;
type ImagePreview = Extract<ScienceArtifactPreview, { kind: "image" }>;
type TableGrid = ComponentType<{ readonly preview: TablePreview }>;

function ScienceArtifactTablePreview({ preview }: { readonly preview: TablePreview }) {
  const [grid, setGrid] = useState<TableGrid | Error | null>(null);
  useEffect(() => {
    let active = true;
    void import("./science-table-grid.js").then(
      (module) => {
        if (active) setGrid(() => module.ScienceTableGrid);
      },
      () => {
        if (active) setGrid(new Error("Interactive table renderer unavailable"));
      },
    );
    return () => {
      active = false;
    };
  }, []);
  const Grid = grid instanceof Error ? null : grid;

  return (
    <section className={css.tablePreview} data-science-table-grid="true">
      <p>
        {preview.rowCount} rows × {preview.columns.length} columns
        {preview.truncated ? " · bounded preview" : ""}
      </p>
      {grid instanceof Error ? (
        <p role="alert">{grid.message}</p>
      ) : Grid === null ? (
        <p role="status">Loading interactive table…</p>
      ) : (
        <Grid preview={preview} />
      )}
    </section>
  );
}

interface ImagePointRequest {
  readonly clientX: number;
  readonly clientY: number;
  readonly rect: Pick<DOMRect, "height" | "left" | "top" | "width">;
  readonly naturalWidth: number;
  readonly naturalHeight: number;
}

export function normalizedImagePoint(request: ImagePointRequest): { x: number; y: number } | null {
  if (
    request.rect.width <= 0 ||
    request.rect.height <= 0 ||
    request.naturalWidth <= 0 ||
    request.naturalHeight <= 0
  ) {
    return null;
  }
  const scale = Math.min(
    request.rect.width / request.naturalWidth,
    request.rect.height / request.naturalHeight,
  );
  const width = request.naturalWidth * scale;
  const height = request.naturalHeight * scale;
  const left = request.rect.left + (request.rect.width - width) / 2;
  const top = request.rect.top + (request.rect.height - height) / 2;
  if (
    request.clientX < left ||
    request.clientX > left + width ||
    request.clientY < top ||
    request.clientY > top + height
  ) {
    return null;
  }
  return {
    x: Math.min(1, Math.max(0, (request.clientX - left) / width)),
    y: Math.min(1, Math.max(0, (request.clientY - top) / height)),
  };
}

interface ScienceAnnotatedImageProps {
  readonly artifact: ScienceArtifactPayload;
  readonly preview: ImagePreview;
  readonly onAnnotationsChange: (annotations: readonly ScienceImageAnnotation[]) => void;
  readonly onAddToConversation: (annotation: ScienceImageAnnotation) => boolean;
}

function ScienceAnnotatedImage({
  artifact,
  preview,
  onAnnotationsChange,
  onAddToConversation,
}: ScienceAnnotatedImageProps) {
  const image = useRef<HTMLImageElement>(null);
  const editor = useRef<HTMLTextAreaElement>(null);
  const [pendingPoint, setPendingPoint] = useState<{ x: number; y: number } | null>(null);
  const [comment, setComment] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const selected = artifact.annotations.find(({ id }) => id === selectedId) ?? null;
  useEffect(() => {
    if (pendingPoint !== null) editor.current?.focus();
  }, [pendingPoint]);

  function openEditor(point: { x: number; y: number }, annotation?: ScienceImageAnnotation) {
    setPendingPoint(point);
    setComment(annotation?.comment ?? "");
    setEditingId(annotation?.id ?? null);
    setSelectedId(null);
    setNotice(null);
  }

  function selectPoint(event: MouseEvent<HTMLButtonElement>) {
    const element = image.current;
    if (!element) return;
    if (event.detail === 0) {
      openEditor({ x: 0.5, y: 0.5 });
      return;
    }
    const point = normalizedImagePoint({
      clientX: event.clientX,
      clientY: event.clientY,
      rect: event.currentTarget.getBoundingClientRect(),
      naturalWidth: element.naturalWidth,
      naturalHeight: element.naturalHeight,
    });
    if (point !== null) openEditor(point);
  }

  function saveAnnotation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pendingPoint === null) return;
    const trimmed = comment.trim();
    if (trimmed.length === 0) return;
    if (editingId !== null) {
      const next = artifact.annotations.map((annotation) =>
        annotation.id === editingId ? { ...annotation, comment: trimmed } : annotation,
      );
      onAnnotationsChange(next);
      setSelectedId(editingId);
      setPendingPoint(null);
      setEditingId(null);
      return;
    }
    const annotation = scienceImageAnnotationSchema.parse({
      version: 1,
      id: crypto.randomUUID(),
      artifactId: artifact.artifactId,
      projectId: artifact.projectId,
      title: artifact.title,
      digest: artifact.digest,
      mime: artifact.mime,
      x: pendingPoint.x,
      y: pendingPoint.y,
      comment: trimmed,
      createdAt: Date.now(),
    });
    onAnnotationsChange([...artifact.annotations, annotation]);
    const inserted = onAddToConversation(annotation);
    setNotice(
      inserted ? "Annotation added to Chat." : "Open this Session's Chat to add the annotation.",
    );
    setSelectedId(annotation.id);
    setPendingPoint(null);
  }

  function removeAnnotation(annotation: ScienceImageAnnotation) {
    onAnnotationsChange(artifact.annotations.filter(({ id }) => id !== annotation.id));
    setSelectedId(null);
    setNotice(null);
  }

  function addToConversation(annotation: ScienceImageAnnotation) {
    setNotice(
      onAddToConversation(annotation)
        ? "Annotation added to Chat."
        : "Open this Session's Chat to add the annotation.",
    );
  }

  return (
    <div className={css.annotationRoot}>
      <div className={css.imageFrame}>
        <div className={css.imageStage}>
          <img
            ref={image}
            className={css.annotatedImage}
            src={preview.dataUrl}
            alt={artifact.title}
          />
          <button
            type="button"
            className={css.annotationCanvas}
            aria-label="Add annotation to image"
            onClick={selectPoint}
          />
          {artifact.annotations.map((annotation, index) => (
            <button
              key={annotation.id}
              type="button"
              className={css.annotationPin}
              style={{ left: `${annotation.x * 100}%`, top: `${annotation.y * 100}%` }}
              aria-label={`Open annotation ${index + 1}: ${annotation.comment}`}
              onClick={() => {
                setSelectedId(annotation.id);
                setPendingPoint(null);
                setNotice(null);
              }}
            >
              {index + 1}
            </button>
          ))}
          {pendingPoint !== null && (
            <form className={css.annotationEditor} onSubmit={saveAnnotation}>
              <label htmlFor={`annotation-${artifact.artifactId}`}>
                {editingId === null ? "Add annotation" : "Edit annotation"}
              </label>
              <textarea
                ref={editor}
                id={`annotation-${artifact.artifactId}`}
                maxLength={2_000}
                placeholder="What do you want to discuss here?"
                value={comment}
                onChange={(event) => setComment(event.currentTarget.value)}
              />
              <div className={css.annotationEditorActions}>
                <button
                  type="button"
                  onClick={() => {
                    setPendingPoint(null);
                    setEditingId(null);
                  }}
                >
                  Cancel
                </button>
                <button type="submit" disabled={comment.trim().length === 0}>
                  {editingId === null ? "Save" : "Update"}
                </button>
              </div>
            </form>
          )}
          {selected !== null && (
            <aside className={css.annotationPopover} aria-label="Image annotation">
              <header>
                <strong>Comment {artifact.annotations.indexOf(selected) + 1}</strong>
                <button
                  type="button"
                  aria-label="Close annotation"
                  onClick={() => setSelectedId(null)}
                >
                  ×
                </button>
              </header>
              <p>{selected.comment}</p>
              <time dateTime={new Date(selected.createdAt).toISOString()}>
                {new Date(selected.createdAt).toLocaleDateString()}
              </time>
              <div className={css.annotationActions}>
                <button type="button" onClick={() => addToConversation(selected)}>
                  <IconSendOutline16 />
                  Add to Chat
                </button>
                <button type="button" onClick={() => openEditor(selected, selected)}>
                  <IconEditOutline16 />
                  Edit
                </button>
                <button type="button" onClick={() => removeAnnotation(selected)}>
                  <IconTrashOutline16 />
                  Delete
                </button>
              </div>
            </aside>
          )}
        </div>
      </div>
      {notice !== null && <p className={css.annotationNotice}>{notice}</p>}
    </div>
  );
}

function entityTypes(entity: RoCrateEntity): readonly string[] {
  const types = entity["@type"];
  return Array.isArray(types) ? types : [types];
}

function referenceIds(value: unknown): string[] {
  const values = Array.isArray(value) ? value : value === undefined ? [] : [value];
  return values.flatMap((candidate) => {
    if (
      typeof candidate === "object" &&
      candidate !== null &&
      !Array.isArray(candidate) &&
      "@id" in candidate &&
      typeof candidate["@id"] === "string"
    ) {
      return [candidate["@id"]];
    }
    return [];
  });
}

export function ScienceArtifactProvenanceView({
  artifactId,
  researchObject,
  onClose,
}: {
  readonly artifactId: string;
  readonly researchObject: RoCrateMetadataDocument | Error | null;
  readonly onClose: () => void;
}) {
  const artifactRoId = roCrateEntityId(artifactId);
  const graph =
    researchObject instanceof Error || researchObject === null ? [] : researchObject["@graph"];
  const artifact = graph.find((entity) => entity["@id"] === artifactRoId);
  const sources = referenceIds(artifact?.isBasedOn)
    .map((id) => graph.find((entity) => entity["@id"] === id))
    .filter((entity): entity is RoCrateEntity => entity !== undefined);
  const actions = graph.filter(
    (entity) =>
      entityTypes(entity).some((type) => type.endsWith("Action")) &&
      [...referenceIds(entity.object), ...referenceIds(entity.result)].includes(artifactRoId),
  );
  return (
    <section className={css.provenancePanel} data-artifact-provenance="true">
      <header className={css.provenanceHeader}>
        <div>
          <p className={css.eyebrow}>Artifact</p>
          <h3>Provenance</h3>
        </div>
        <button type="button" onClick={onClose}>
          Back to artifact
        </button>
      </header>
      {researchObject === null ? (
        <p role="status">Loading provenance…</p>
      ) : researchObject instanceof Error ? (
        <p role="alert">{researchObject.message}</p>
      ) : artifact === undefined ? (
        <p role="alert">Artifact is missing from the Research Object.</p>
      ) : (
        <div className={css.provenanceBody}>
          <section>
            <h4>Lineage</h4>
            <ol className={css.provenanceEntities}>
              {[artifact, ...sources].map((entity) => (
                <li key={entity["@id"]}>
                  <IconBranchOutline16 />
                  <span className={css.provenanceEntityCopy}>
                    <strong>{entity.name ?? entity["@id"]}</strong>
                    <small>{entityTypes(entity).join(" · ")}</small>
                  </span>
                </li>
              ))}
            </ol>
          </section>
          {sources.length > 0 && (
            <section>
              <h4>Relations</h4>
              <ul className={css.provenanceRelations}>
                {sources.map((source) => (
                  <li key={`${artifactRoId}:isBasedOn:${source["@id"]}`}>
                    <span className={css.provenanceRelationEndpoint}>{artifact.name}</span>
                    <strong>is based on</strong>
                    <span className={css.provenanceRelationEndpoint}>
                      {source.name ?? source["@id"]}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}
          <section>
            <h4>Activity</h4>
            <ol className={css.provenanceEvents}>
              {actions.map((action) => (
                <li key={action["@id"]}>
                  <span className={css.provenanceEventLabel}>
                    {action.name ?? entityTypes(action).join(" · ")}
                  </span>
                  {action.identifier?.startsWith("science-journal:") && (
                    <small>Journal #{action.identifier.slice("science-journal:".length)}</small>
                  )}
                </li>
              ))}
            </ol>
          </section>
        </div>
      )}
    </section>
  );
}

interface ScienceArtifactSideViewInjected {
  readonly loadPreview: (
    artifactId: string,
    signal?: AbortSignal,
  ) => Promise<ScienceArtifactPreview>;
  readonly loadResearchObject: (
    projectId: string,
    signal?: AbortSignal,
  ) => Promise<RoCrateMetadataDocument>;
  readonly updateAnnotations: (
    entry: SideViewEntry,
    annotations: readonly ScienceImageAnnotation[],
  ) => void;
  readonly addAnnotationToConversation: (annotation: ScienceImageAnnotation) => boolean;
}

/** Artifact workbench with bounded preview, on-demand provenance, and image point references. */
export function ScienceArtifactSideView({
  entry,
  loadPreview,
  loadResearchObject,
  updateAnnotations,
  addAnnotationToConversation,
}: SideViewContentOwnerProps & ScienceArtifactSideViewInjected) {
  const root = useRef<HTMLElement>(null);
  const artifact = artifactPayload(entry);
  const artifactId = artifact?.artifactId ?? null;
  const projectId = artifact?.projectId ?? null;
  const [preview, setPreview] = useState<ScienceArtifactPreview | Error | null>(null);
  const [researchObject, setResearchObject] = useState<RoCrateMetadataDocument | Error | null>(
    null,
  );
  const [surface, setSurface] = useState<"artifact" | "provenance">("artifact");
  const [actionsOpen, setActionsOpen] = useState(false);
  const [fullscreenError, setFullscreenError] = useState<string | null>(null);
  useEffect(() => {
    if (artifactId === null) return;
    const controller = new AbortController();
    setPreview(null);
    setFullscreenError(null);
    setSurface("artifact");
    setResearchObject(null);
    void loadPreview(artifactId, controller.signal).then(setPreview, (error: unknown) => {
      if (!controller.signal.aborted) {
        setPreview(error instanceof Error ? error : new Error("Artifact preview failed"));
      }
    });
    return () => controller.abort();
  }, [artifactId, loadPreview]);
  useEffect(() => {
    if (projectId === null || surface !== "provenance" || researchObject !== null) return;
    const controller = new AbortController();
    void loadResearchObject(projectId, controller.signal).then(
      setResearchObject,
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setResearchObject(
            error instanceof Error ? error : new Error("Research Object failed to load"),
          );
        }
      },
    );
    return () => controller.abort();
  }, [loadResearchObject, projectId, researchObject, surface]);
  if (artifact === null) return <p>Artifact locator is invalid.</p>;

  async function openFullscreen() {
    const element = root.current;
    if (element === null) return;
    try {
      await element.requestFullscreen();
      setFullscreenError(null);
    } catch {
      setFullscreenError("Fullscreen is unavailable in this window.");
    }
  }

  return (
    <article ref={root} className={css.root} data-artifact-file-view="true">
      <header className={css.fileHeader}>
        <div>
          <h2>{artifact.title}</h2>
        </div>
        <div className={css.headerActions}>
          <Menu
            open={actionsOpen}
            anchor={
              <button
                type="button"
                className={css.iconButton}
                aria-label="More actions"
                aria-expanded={actionsOpen}
                onClick={() => setActionsOpen((open) => !open)}
              >
                <IconEllipsisOutline16 />
              </button>
            }
            items={SCIENCE_ARTIFACT_ACTIONS.map((item) => ({
              ...item,
              icon: <IconBranchOutline16 />,
            }))}
            align="end"
            portal
            compact
            onClose={() => setActionsOpen(false)}
            onSelect={(id) => {
              setActionsOpen(false);
              if (id === "provenance") setSurface("provenance");
            }}
          />
          <button
            type="button"
            className={css.iconButton}
            aria-label="Open fullscreen"
            onClick={() => void openFullscreen()}
          >
            <IconFullscreenOutline16 />
          </button>
        </div>
      </header>
      {fullscreenError !== null && <p role="alert">{fullscreenError}</p>}
      {surface === "provenance" ? (
        <ScienceArtifactProvenanceView
          artifactId={artifact.artifactId}
          researchObject={researchObject}
          onClose={() => setSurface("artifact")}
        />
      ) : (
        <>
          <section className={css.previewPanel}>
            {preview !== null && !(preview instanceof Error) && preview.kind === "image" ? (
              <ScienceAnnotatedImage
                artifact={artifact}
                preview={preview}
                onAnnotationsChange={(annotations) => updateAnnotations(entry, annotations)}
                onAddToConversation={addAnnotationToConversation}
              />
            ) : (
              <ScienceArtifactPreviewView preview={preview} />
            )}
          </section>
          <details className={css.fileDetails}>
            <summary>File details</summary>
            <dl className={css.metadata}>
              <div>
                <dt>Media type</dt>
                <dd>{artifact.mime}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd>{formatBytes(artifact.size)}</dd>
              </div>
              <div>
                <dt>Revision</dt>
                <dd>{artifact.revision}</dd>
              </div>
              <div>
                <dt>Provenance</dt>
                <dd>Journal #{artifact.journalSeq}</dd>
              </div>
              <div>
                <dt>Digest</dt>
                <dd className={css.digest}>{artifact.digest}</dd>
              </div>
              {artifact.runId !== null && (
                <div>
                  <dt>Run</dt>
                  <dd>{artifact.runId}</dd>
                </div>
              )}
            </dl>
            {artifact.sourceEntityIds.length > 0 && (
              <section className={css.sources}>
                <h3>Source entities</h3>
                <ul>
                  {artifact.sourceEntityIds.map((entityId) => (
                    <li key={entityId}>{entityId}</li>
                  ))}
                </ul>
              </section>
            )}
          </details>
        </>
      )}
    </article>
  );
}
