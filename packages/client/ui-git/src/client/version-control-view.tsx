import {
  IconBranchOutline16,
  IconRefreshOutline16,
  Tooltip,
} from "@deepseek-ai/dsh-client-ui-primitives";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import {
  type DvcUiInspection,
  type DvcUiStatusSummary,
  dvcUiInspectionSchema,
} from "@swarmx/dsh-ui-dvc/contracts";
import { useCallback, useEffect, useState } from "react";
import { z } from "zod";
import { type GitUiRepositorySnapshot, gitUiSnapshotSchema } from "../contracts.js";
import css from "./version-control-view.module.css";

const versionControlSnapshotSchema = z.strictObject({
  git: gitUiSnapshotSchema,
  dvc: dvcUiInspectionSchema.nullable(),
});

export type VersionControlSnapshot = z.infer<typeof versionControlSnapshotSchema>;

export function versionControlSideViewEntry(snapshot: VersionControlSnapshot): SideViewEntry {
  const parsed = versionControlSnapshotSchema.parse(snapshot);
  return {
    id: "version-control",
    kind: "version-control",
    title: "Version Control",
    mode: "inspect",
    payload: parsed as SideViewEntry["payload"],
  };
}

interface VersionControlLoader {
  readonly load: (signal?: AbortSignal) => Promise<VersionControlSnapshot>;
}

type VersionControlHeaderActionProps = PropsRuntime<"conversation.session.header.actions"> &
  VersionControlLoader & {
    readonly open: (snapshot: VersionControlSnapshot) => void;
  };

function gitChangeCount(snapshot: GitUiRepositorySnapshot): number {
  return snapshot.staged + snapshot.unstaged + snapshot.untracked + snapshot.conflicted;
}

function changeSummary(snapshot: VersionControlSnapshot | Error): string {
  if (snapshot instanceof Error || snapshot.git.kind !== "repository") return "Repository status";
  const gitChanges = gitChangeCount(snapshot.git);
  const gitLabel = `${gitChanges} Git ${gitChanges === 1 ? "change" : "changes"}`;
  if (snapshot.dvc === null) return gitLabel;
  const dvcChanges = snapshot.dvc.data.entries + snapshot.dvc.pipeline.entries;
  return `${gitLabel} · ${dvcChanges} DVC ${dvcChanges === 1 ? "change" : "changes"}`;
}

export function VersionControlHeaderAction({ load, open }: VersionControlHeaderActionProps) {
  const [snapshot, setSnapshot] = useState<VersionControlSnapshot | Error | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal).then(setSnapshot, (error: unknown) => {
      if (!controller.signal.aborted) {
        setSnapshot(
          error instanceof Error ? error : new Error("Version Control could not be read"),
        );
      }
    });
    return () => controller.abort();
  }, [load]);

  if (!(snapshot instanceof Error) && snapshot?.git.kind === "not-repository") return null;
  const repository =
    !(snapshot instanceof Error) && snapshot?.git.kind === "repository" ? snapshot.git : null;
  const changes = repository ? gitChangeCount(repository) : 0;
  const label =
    snapshot instanceof Error || snapshot?.git.kind === "unavailable"
      ? "Version Control unavailable"
      : repository
        ? `Version control for ${repository.branch ?? repository.head.slice(0, 8)}`
        : "Loading Version Control";
  return (
    <Tooltip label={label} side="bottom">
      <button
        type="button"
        className={css.headerAction}
        aria-label={label}
        disabled={snapshot === null}
        onClick={() => {
          if (snapshot !== null && !(snapshot instanceof Error)) open(snapshot);
        }}
      >
        <IconBranchOutline16 size={16} />
        {repository?.branch && <span>{repository.branch}</span>}
        {changes > 0 && <span className={css.changeCount}>{changes}</span>}
      </button>
    </Tooltip>
  );
}

function RepositoryStatus({ snapshot }: { readonly snapshot: GitUiRepositorySnapshot }) {
  return (
    <>
      <div className={css.repositorySummary}>
        <div>
          <h3>{snapshot.branch ?? "Detached HEAD"}</h3>
          <p>{snapshot.upstream ?? snapshot.head.slice(0, 12)}</p>
        </div>
        <span className={snapshot.clean ? css.clean : css.changed}>
          {snapshot.clean ? "Clean" : "Changed"}
        </span>
      </div>
      <div className={css.meta}>
        {snapshot.ahead !== null && <span>Ahead {snapshot.ahead}</span>}
        {snapshot.behind !== null && <span>Behind {snapshot.behind}</span>}
        <span>Staged {snapshot.staged}</span>
        <span>Modified {snapshot.unstaged}</span>
        <span>Untracked {snapshot.untracked}</span>
        {snapshot.conflicted > 0 && <span>Conflicts {snapshot.conflicted}</span>}
      </div>
      {snapshot.entries.length === 0 ? (
        <p className={css.empty}>Working tree clean.</p>
      ) : (
        <ul className={css.files}>
          {snapshot.entries.map((entry) => (
            <li
              className={css.file}
              key={`${entry.kind}:${entry.previousPath ?? ""}:${entry.path}`}
            >
              <span className={css.status}>{`${entry.index}${entry.worktree}`}</span>
              <span className={css.path}>
                {entry.previousPath && (
                  <span className={css.previous}>{entry.previousPath} → </span>
                )}
                {entry.path}
              </span>
            </li>
          ))}
        </ul>
      )}
      {snapshot.truncated && <p className={css.notice}>Additional changed files are not shown.</p>}
    </>
  );
}

function shortDigest(value: string | null): string {
  return value === null ? "Not present" : value.slice(0, "sha256:".length + 12);
}

function StatusRows({ summary }: { readonly summary: DvcUiStatusSummary }) {
  if (summary.categories.length === 0) return null;
  return (
    <ul className={css.statusRows}>
      {summary.categories.map((category) => (
        <li key={category.name}>
          <span>{category.name}</span>
          <strong>{category.count}</strong>
        </li>
      ))}
    </ul>
  );
}

function DvcStatus({ inspection }: { readonly inspection: DvcUiInspection }) {
  return (
    <>
      <div className={css.dvcLead}>
        <div>
          <strong>
            {inspection.pipeline.entries === 0 ? "Pipeline clean" : "Pipeline changed"}
          </strong>
          <p>
            {inspection.pipeline.entries === 0
              ? "No pipeline changes"
              : `${inspection.pipeline.entries} items need review`}
          </p>
        </div>
        <span className={inspection.pipeline.entries === 0 ? css.clean : css.changed}>
          DVC {inspection.version}
        </span>
      </div>
      <StatusRows summary={inspection.pipeline} />
      <div className={css.dataRow}>
        <span>Tracked data</span>
        <strong>{inspection.data.entries === 0 ? "No changes" : inspection.data.entries}</strong>
      </div>
      <StatusRows summary={inspection.data} />
      <dl className={css.manifests}>
        <div>
          <dt>Root</dt>
          <dd>{inspection.root === "." ? "Repository root" : inspection.root}</dd>
        </div>
        <div>
          <dt>dvc.yaml</dt>
          <dd>{shortDigest(inspection.dvcYamlDigest)}</dd>
        </div>
        <div>
          <dt>dvc.lock</dt>
          <dd>{shortDigest(inspection.dvcLockDigest)}</dd>
        </div>
      </dl>
    </>
  );
}

function initialSnapshot(entry: SideViewEntry): VersionControlSnapshot | Error {
  const parsed = versionControlSnapshotSchema.safeParse(entry.payload);
  return parsed.success ? parsed.data : new Error("Version Control payload is invalid");
}

type VersionControlViewProps = SideViewContentOwnerProps & VersionControlLoader;

export function VersionControlView({ entry, load }: VersionControlViewProps) {
  const [snapshot, setSnapshot] = useState<VersionControlSnapshot | Error>(() =>
    initialSnapshot(entry),
  );
  const [refreshing, setRefreshing] = useState(false);
  const refresh = useCallback(() => {
    const controller = new AbortController();
    setRefreshing(true);
    void load(controller.signal).then(
      (next) => {
        setSnapshot(next);
        setRefreshing(false);
      },
      (error: unknown) => {
        setSnapshot(
          error instanceof Error ? error : new Error("Version Control could not be read"),
        );
        setRefreshing(false);
      },
    );
  }, [load]);

  return (
    <section className={css.panel} aria-label="Version Control">
      <header className={css.statusBar}>
        <span>{changeSummary(snapshot)}</span>
        <Tooltip label="Refresh Version Control" side="bottom">
          <button
            type="button"
            className={css.refresh}
            aria-label="Refresh Version Control"
            disabled={refreshing}
            onClick={refresh}
          >
            <IconRefreshOutline16 size={16} />
          </button>
        </Tooltip>
      </header>
      {snapshot instanceof Error ? (
        <p className={css.error} role="alert">
          {snapshot.message}
        </p>
      ) : snapshot.git.kind === "repository" ? (
        <>
          <details className={css.disclosure} open>
            <summary>
              <span className={css.summaryLabel}>Changes</span>
              <span className={css.summaryCount}>{gitChangeCount(snapshot.git)}</span>
            </summary>
            <div className={css.disclosureContent}>
              <RepositoryStatus snapshot={snapshot.git} />
            </div>
          </details>
          {snapshot.dvc !== null && (
            <details className={css.disclosure} open>
              <summary>
                <span className={css.summaryLabel}>DVC</span>
                <span className={css.summaryCount}>
                  {snapshot.dvc.data.entries + snapshot.dvc.pipeline.entries}
                </span>
              </summary>
              <div className={css.disclosureContent}>
                <DvcStatus inspection={snapshot.dvc} />
              </div>
            </details>
          )}
        </>
      ) : (
        <p className={snapshot.git.kind === "unavailable" ? css.error : css.empty}>
          {snapshot.git.message}
        </p>
      )}
    </section>
  );
}
