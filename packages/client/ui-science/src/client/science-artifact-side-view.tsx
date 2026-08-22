import type { ScienceArtifact, ScienceArtifactPreview } from "@swarmx/dsh-science/types";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import { type ComponentType, useEffect, useState } from "react";
import css from "./science-artifact-side-view.module.css";
import type { ScienceWorkbenchTarget } from "./science-navigation.js";

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
}

export function artifactPayload(entry: SideViewEntry): ScienceArtifactPayload | null {
  const payload = entry.payload;
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  const values = payload as { readonly [key: string]: unknown };
  const sourceEntityIds = values.sourceEntityIds;
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
    typeof values.journalSeq !== "number"
  ) {
    return null;
  }
  return payload as unknown as ScienceArtifactPayload;
}

/** Create a bounded locator; artifact bytes and host paths remain behind the host service. */
export function scienceArtifactSideViewEntry(artifact: ScienceArtifact): SideViewEntry {
  return {
    id: `science-artifact:${artifact.id}`,
    kind: "science-artifact",
    title: artifact.title,
    mode: "inspect",
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
    },
  };
}

export function scienceArtifactWorkbenchTarget(
  entry: SideViewEntry,
): ScienceWorkbenchTarget | null {
  const artifact = artifactPayload(entry);
  return artifact === null
    ? null
    : {
        kind: "artifact",
        artifactId: artifact.artifactId,
        projectId: artifact.projectId,
        surface: "artifacts",
      };
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

interface ScienceArtifactSideViewInjected {
  readonly loadPreview: (
    artifactId: string,
    signal?: AbortSignal,
  ) => Promise<ScienceArtifactPreview>;
  readonly openInScience: (target: ScienceWorkbenchTarget) => boolean;
}

/** Metadata plus a bounded Host-authorized preview in the generic keyed Side View slot. */
export function ScienceArtifactSideView({
  entry,
  loadPreview,
  openInScience,
}: SideViewContentOwnerProps & ScienceArtifactSideViewInjected) {
  const artifact = artifactPayload(entry);
  const artifactId = artifact?.artifactId ?? null;
  const [preview, setPreview] = useState<ScienceArtifactPreview | Error | null>(null);
  const [handoffPending, setHandoffPending] = useState(false);
  useEffect(() => {
    if (artifactId === null) return;
    const controller = new AbortController();
    setPreview(null);
    setHandoffPending(false);
    void loadPreview(artifactId, controller.signal).then(setPreview, (error: unknown) => {
      if (!controller.signal.aborted) {
        setPreview(error instanceof Error ? error : new Error("Artifact preview failed"));
      }
    });
    return () => controller.abort();
  }, [artifactId, loadPreview]);
  if (artifact === null) return <p>Artifact locator is invalid.</p>;
  const target = scienceArtifactWorkbenchTarget(entry);

  return (
    <article className={css.root} data-artifact-file-view="true">
      <header className={css.fileHeader}>
        <div>
          <p className={css.eyebrow}>{artifact.artifactKind}</p>
          <h2>{artifact.title}</h2>
        </div>
        {target !== null && (
          <button type="button" onClick={() => setHandoffPending(!openInScience(target))}>
            Open in Science
          </button>
        )}
      </header>
      {handoffPending && (
        <p className={css.handoffStatus} role="status">
          Target ready. Select the Science tab to open the fullscreen workbench.
        </p>
      )}
      <section className={css.previewPanel}>
        <h3>Preview</h3>
        <ScienceArtifactPreviewView preview={preview} />
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
    </article>
  );
}
