import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XtermTerminal } from "@xterm/xterm";
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  ChevronLeft,
  File,
  FileCode2,
  Folder,
  GitCompareArrows,
  Globe2,
  Home,
  Loader2,
  PanelRight,
  Plus,
  RefreshCw,
  RotateCw,
  Search,
  Terminal as TerminalIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { errorMessage } from "./text-utils.js";
import { cx, rightPanelVariants } from "./ui-primitives.js";

type TerminalStatus = "idle" | "starting" | "running" | "exited" | "error";
type ReviewStatusTone = "added" | "deleted" | "renamed" | "modified";

const TERMINAL_STATUS_CLASS = {
  idle: "is-idle",
  starting: "is-starting",
  running: "is-running",
  exited: "is-exited",
  error: "is-error",
} satisfies Record<TerminalStatus, string>;

const REVIEW_STATUS_CLASS = {
  added: "is-added",
  deleted: "is-deleted",
  renamed: "is-renamed",
  modified: "is-modified",
} satisfies Record<ReviewStatusTone, string>;

const REVIEW_STATUS_TONE = {
  Added: "added",
  Deleted: "deleted",
  Renamed: "renamed",
  Modified: "modified",
} satisfies Record<ReturnType<typeof reviewStatusLabel>, ReviewStatusTone>;

const REVIEW_LINE_CLASS = {
  addition: "is-addition",
  deletion: "is-deletion",
  context: "is-context",
} satisfies Record<ParsedDiffLine["kind"], string>;

export type WorkspaceTool = "review" | "terminal" | "browser" | "files";

export interface WorkspaceReviewFile {
  path: string;
  status: string;
  patch: string;
  binary: boolean;
  additions: number;
  deletions: number;
  truncated: boolean;
}

export interface WorkspaceReviewSnapshot {
  root: string;
  branch?: string | null;
  isRepository: boolean;
  files: WorkspaceReviewFile[];
  truncated: boolean;
  error?: string;
}

export interface WorkspaceDirectoryEntry {
  name: string;
  path: string;
  kind: "directory" | "file" | "symlink" | "other";
  size?: number;
}

export interface WorkspaceDirectoryListing {
  root: string;
  path: string;
  entries: WorkspaceDirectoryEntry[];
  truncated: boolean;
}

export interface WorkspaceFilePreview {
  root: string;
  path: string;
  content: string;
  size: number;
  binary: boolean;
  truncated: boolean;
}

export interface BrowserBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BrowserState {
  id: string;
  url: string;
  title: string;
  loading: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  error?: string;
}

export interface WorkspacePanelApi {
  getWorkspaceReview(cwd?: string): Promise<WorkspaceReviewSnapshot>;
  listWorkspaceDirectory(path?: string, cwd?: string): Promise<WorkspaceDirectoryListing>;
  readWorkspaceFile(path: string, cwd?: string): Promise<WorkspaceFilePreview>;
  createTerminal(params: {
    id: string;
    cwd: string;
    cols?: number;
    rows?: number;
  }): Promise<{ id: string; pid: number }>;
  writeTerminal(id: string, data: string): Promise<{ written: boolean }>;
  resizeTerminal(id: string, cols: number, rows: number): Promise<{ resized: boolean }>;
  killTerminal(id: string): Promise<{ killed: boolean }>;
  onTerminalData(listener: (event: { id: string; data: string }) => void): () => void;
  onTerminalExit(
    listener: (event: { id: string; exitCode: number; signal?: number }) => void,
  ): () => void;
  createBrowser(params?: {
    id?: string;
    url?: string;
    bounds?: BrowserBounds;
    visible?: boolean;
  }): Promise<BrowserState>;
  navigateBrowser(id: string, url: string): Promise<BrowserState>;
  backBrowser(id: string): Promise<BrowserState>;
  forwardBrowser(id: string): Promise<BrowserState>;
  reloadBrowser(id: string): Promise<BrowserState>;
  setBrowserBounds(id: string, bounds: BrowserBounds): Promise<{ updated: boolean }>;
  setBrowserVisible(id: string, visible: boolean): Promise<{ updated: boolean }>;
  destroyBrowser(id: string): Promise<{ destroyed: boolean }>;
  onBrowserState(listener: (state: BrowserState) => void): () => void;
}

const TOOL_DEFINITIONS: Array<{
  id: WorkspaceTool;
  label: string;
  shortcut: string;
  icon: typeof GitCompareArrows;
}> = [
  { id: "review", label: "Review", shortcut: "⌃⇧G", icon: GitCompareArrows },
  { id: "terminal", label: "Terminal", shortcut: "⌘`", icon: TerminalIcon },
  { id: "browser", label: "Browser", shortcut: "⌘T", icon: Globe2 },
  { id: "files", label: "Files", shortcut: "⌘P", icon: Folder },
];

export function WorkspacePanel({
  api,
  cwd,
  onClose,
}: {
  api: WorkspacePanelApi;
  cwd: string;
  onClose: () => void;
}) {
  const [activeTool, setActiveTool] = useState<WorkspaceTool | null>(null);
  const [visitedTools, setVisitedTools] = useState<Set<WorkspaceTool>>(() => new Set());

  const selectTool = useCallback((tool: WorkspaceTool) => {
    setVisitedTools((visited) => {
      if (visited.has(tool)) return visited;
      const next = new Set(visited);
      next.add(tool);
      return next;
    });
    setActiveTool(tool);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      let tool: WorkspaceTool | null = null;
      if (event.ctrlKey && event.shiftKey && key === "g") tool = "review";
      if (event.metaKey && key === "`") tool = "terminal";
      if (event.metaKey && key === "t") tool = "browser";
      if (event.metaKey && key === "p") tool = "files";
      if (!tool) return;
      event.preventDefault();
      selectTool(tool);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [selectTool]);

  return (
    <aside className={rightPanelVariants({ kind: "workspace" })} aria-label="Right panel">
      {activeTool && (
        <header className="workspace-panel__header [min-width:0] [height:46px] [flex:0_0_46px] [padding:0_8px] [display:flex] [align-items:center] [gap:6px] [border-bottom:1px_solid_var(--border-subtle)] [background:color-mix(in_srgb,_var(--card-solid)_94%,_var(--background))]">
          <button
            type="button"
            className="workspace-panel__home [display:inline-grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:1px_solid_transparent] [border-radius:7px] [cursor:pointer] [width:30px] [height:30px] [flex:0_0_30px] [padding:0] [&_svg]:[width:15px] [&_svg]:[height:15px]"
            onClick={() => setActiveTool(null)}
            aria-label="Workspace tools home"
            title="Workspace tools"
          >
            <Home aria-hidden="true" />
          </button>
          <div
            className="workspace-panel__tabs [min-width:0] [flex:1] [display:flex] [align-items:center] [gap:2px] [&_button]:[place-items:center] [&_button]:[color:var(--muted-foreground)] [&_button]:[background:transparent] [&_button]:[border:1px_solid_transparent] [&_button]:[border-radius:7px] [&_button]:[cursor:pointer] [&_button]:[min-width:0] [&_button]:[height:30px] [&_button]:[padding:0_8px] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:6px] [&_button]:[font-size:10.5px] [&_button_svg]:[width:13px] [&_button_svg]:[height:13px] [&_button_svg]:[flex:0_0_auto] [&_button_span]:[overflow:hidden] [&_button_span]:[text-overflow:ellipsis] [&_button_span]:[white-space:nowrap] max-860:[&_button]:[padding:0_6px] max-860:[&_button_span]:[display:none]"
            role="tablist"
            aria-label="Workspace tools"
          >
            {TOOL_DEFINITIONS.map((tool) => {
              const Icon = tool.icon;
              return (
                <button
                  key={tool.id}
                  type="button"
                  role="tab"
                  aria-selected={activeTool === tool.id}
                  className={activeTool === tool.id ? "is-active" : undefined}
                  onClick={() => selectTool(tool.id)}
                >
                  <Icon aria-hidden="true" />
                  <span>{tool.label}</span>
                </button>
              );
            })}
          </div>
          <button
            type="button"
            className="workspace-panel__close [margin-left:auto] [display:inline-grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:1px_solid_transparent] [border-radius:7px] [cursor:pointer] [width:30px] [height:30px] [flex:0_0_30px] [padding:0] [&_svg]:[width:15px] [&_svg]:[height:15px]"
            onClick={onClose}
            aria-label="Close right panel"
            title="Close right panel"
          >
            <PanelRight aria-hidden="true" />
          </button>
        </header>
      )}

      {activeTool === null ? (
        <ToolLauncher onSelect={selectTool} />
      ) : (
        <div className="workspace-panel__views [flex:1] [overflow:hidden] [min-width:0] [min-height:0] [height:100%]">
          {visitedTools.has("review") && (
            <div
              className="workspace-panel__view [min-width:0] [min-height:0] [height:100%]"
              hidden={activeTool !== "review"}
            >
              <ReviewTool key={cwd} api={api} cwd={cwd} active={activeTool === "review"} />
            </div>
          )}
          {visitedTools.has("terminal") && (
            <div
              className="workspace-panel__view [min-width:0] [min-height:0] [height:100%]"
              hidden={activeTool !== "terminal"}
            >
              <TerminalTool api={api} cwd={cwd} active={activeTool === "terminal"} />
            </div>
          )}
          {visitedTools.has("browser") && (
            <div
              className="workspace-panel__view [min-width:0] [min-height:0] [height:100%]"
              hidden={activeTool !== "browser"}
            >
              <BrowserTool api={api} active={activeTool === "browser"} />
            </div>
          )}
          {visitedTools.has("files") && (
            <div
              className="workspace-panel__view [min-width:0] [min-height:0] [height:100%]"
              hidden={activeTool !== "files"}
            >
              <FilesTool key={cwd} api={api} cwd={cwd} active={activeTool === "files"} />
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

function ToolLauncher({ onSelect }: { onSelect: (tool: WorkspaceTool) => void }) {
  return (
    <nav
      className="workspace-panel__launcher [width:min(100%_-_48px,_540px)] [margin:auto] [display:grid] [gap:4px] [&_button]:[width:100%] [&_button]:[min-height:58px] [&_button]:[padding:0_14px] [&_button]:[display:grid] [&_button]:[grid-template-columns:24px_minmax(0,_1fr)_auto] [&_button]:[align-items:center] [&_button]:[gap:12px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:1px_solid_transparent] [&_button]:[border-radius:9px] [&_button]:[cursor:pointer] [&_button]:[text-align:left] [&_svg]:[width:18px] [&_svg]:[height:18px] [&_svg]:[color:var(--muted-foreground)] [&_span]:[font-size:14px] [&_span]:[font-weight:540] [&_kbd]:[min-width:30px] [&_kbd]:[padding:3px_7px] [&_kbd]:[color:var(--muted-foreground)] [&_kbd]:[background:var(--input)] [&_kbd]:[border:0] [&_kbd]:[border-radius:999px] [&_kbd]:[font-family:var(--font-sans)] [&_kbd]:[font-size:10px] [&_kbd]:[text-align:center]"
      aria-label="Open workspace tool"
    >
      {TOOL_DEFINITIONS.map((tool) => {
        const Icon = tool.icon;
        return (
          <button key={tool.id} type="button" onClick={() => onSelect(tool.id)}>
            <Icon aria-hidden="true" />
            <span>{tool.label}</span>
            <kbd>{tool.shortcut}</kbd>
          </button>
        );
      })}
    </nav>
  );
}

function ReviewTool({
  api,
  cwd,
  active,
}: {
  api: WorkspacePanelApi;
  cwd: string;
  active: boolean;
}) {
  const [snapshot, setSnapshot] = useState<WorkspaceReviewSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSnapshot(await api.getWorkspaceReview(cwd));
    } catch (reason) {
      setError(errorMessage(reason));
    } finally {
      setLoading(false);
    }
  }, [api, cwd]);

  useEffect(() => {
    if (active && snapshot === null && !loading) void refresh();
  }, [active, loading, refresh, snapshot]);

  const totals = useMemo(
    () =>
      snapshot?.files.reduce(
        (result, file) => ({
          additions: result.additions + file.additions,
          deletions: result.deletions + file.deletions,
        }),
        { additions: 0, deletions: 0 },
      ) ?? { additions: 0, deletions: 0 },
    [snapshot],
  );

  return (
    <section
      className="review-tool [min-width:0] [min-height:0] [height:100%] [display:grid] [grid-template-rows:42px_minmax(0,_1fr)] [overflow:hidden]"
      aria-label="Review changes"
    >
      <div className="workspace-tool__toolbar [min-width:0] [padding:0_9px_0_12px] [display:flex] [align-items:center] [gap:8px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_strong]:[font-size:11.5px] [&_strong]:[font-weight:650]">
        <div>
          <strong>Changes</strong>
          <span>{snapshot?.branch || "Working tree"}</span>
        </div>
        <div
          className="review-tool__summary [&_span]:[color:var(--muted-foreground)] [&_b]:[color:var(--success)] [&_b]:[font-style:normal] [&_i]:[color:var(--danger)] [&_i]:[font-style:normal] [&_i]:[font-weight:650] ![flex:0_0_auto] ![display:flex] ![align-items:center] ![gap:7px] [font-size:10px]"
          aria-label="Change summary"
        >
          <span>{snapshot?.files.length ?? 0} files</span>
          <b>+{totals.additions}</b>
          <i>−{totals.deletions}</i>
        </div>
        <IconButton label="Refresh changes" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw
            className={loading ? "is-spinning [animation:spin_0.9s_linear_infinite]" : undefined}
            aria-hidden="true"
          />
        </IconButton>
      </div>

      <div className="review-tool__body [min-width:0] [min-height:0] [overflow:auto] [background:color-mix(in_srgb,_var(--background)_78%,_var(--card-solid))]">
        {loading && snapshot === null ? (
          <ToolState
            icon={Loader2}
            title="Loading changes"
            detail="Reading the working tree…"
            spin
          />
        ) : error || snapshot?.error ? (
          <ToolState
            icon={GitCompareArrows}
            title="Review unavailable"
            detail={error ?? snapshot?.error ?? "Unable to read changes."}
          />
        ) : snapshot && !snapshot.isRepository ? (
          <ToolState
            icon={GitCompareArrows}
            title="Not a Git repository"
            detail="Open a Git workspace to review local changes."
          />
        ) : snapshot?.files.length === 0 ? (
          <ToolState
            icon={GitCompareArrows}
            title="No local changes"
            detail="The working tree is clean."
          />
        ) : (
          <>
            {snapshot?.truncated && (
              <p className="workspace-tool__notice [margin:8px] [padding:7px_9px] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-size:10px]">
                Large review truncated to a safe preview.
              </p>
            )}
            <div className="review-tool__files [padding:10px] [display:grid] [gap:10px]">
              {snapshot?.files.map((file, fileIndex) => (
                <ReviewFile
                  key={`${file.status}:${file.path}`}
                  file={file}
                  defaultExpanded={fileIndex === 0 && file.additions + file.deletions <= 800}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </section>
  );
}

function ReviewFile({
  file,
  defaultExpanded,
}: {
  file: WorkspaceReviewFile;
  defaultExpanded: boolean;
}) {
  const hunks = useMemo(() => parseUnifiedPatch(file.patch), [file.patch]);
  const [expanded, setExpanded] = useState(defaultExpanded);
  return (
    <article className="review-file [min-width:0] [overflow:hidden] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:8px]">
      <header className="review-file__header [position:sticky] [top:0] [z-index:2] [min-width:0] [height:38px] [padding:0_10px] [display:flex] [align-items:center] [gap:7px] [background:color-mix(in_srgb,_var(--card-solid)_94%,_var(--foreground))] [border-bottom:1px_solid_var(--border-subtle)] [&_>_svg]:[width:13px] [&_>_svg]:[height:13px] [&_>_svg]:[flex:0_0_auto] [&_>_svg]:[color:var(--muted-foreground)] [&_strong]:[min-width:0] [&_strong]:[flex:1] [&_strong]:[overflow:hidden] [&_strong]:[font-family:var(--font-mono)] [&_strong]:[font-size:10.5px] [&_strong]:[font-weight:580] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap]">
        <button
          type="button"
          className="review-file__toggle [width:24px] [height:24px] [margin-left:-5px] [padding:0] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:5px] [cursor:pointer] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_svg]:[transition:transform_var(--duration-fast)_var(--ease-out)] [&[aria-expanded='false']_svg]:[transform:rotate(-90deg)]"
          aria-label={`${expanded ? "Collapse" : "Expand"} ${file.path}`}
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <ChevronDown aria-hidden="true" />
        </button>
        <FileCode2 aria-hidden="true" />
        <strong title={file.path}>{file.path}</strong>
        <span
          className={cx(
            "review-file__status [padding:2px_5px] [color:var(--muted)] [background:var(--input)] [border-radius:4px] [font-size:8.5px] [font-weight:650] [text-transform:uppercase]",
            REVIEW_STATUS_CLASS[reviewStatusTone(file.status)],
          )}
        >
          {reviewStatusLabel(file.status)}
        </span>
        <span className="review-file__stats [display:flex] [gap:5px] [font-size:9px] [&_b]:[color:var(--success)] [&_b]:[font-style:normal] [&_i]:[color:var(--danger)] [&_i]:[font-style:normal] [&_i]:[font-weight:650]">
          <b>+{file.additions}</b>
          <i>−{file.deletions}</i>
        </span>
      </header>
      {!expanded ? null : file.binary ? (
        <p className="review-file__binary [margin:0] [padding:18px] [color:var(--muted-foreground)] [font-size:10.5px] [text-align:center]">
          Binary file changed
        </p>
      ) : hunks.length === 0 ? (
        <p className="review-file__binary [margin:0] [padding:18px] [color:var(--muted-foreground)] [font-size:10.5px] [text-align:center]">
          No text preview available
        </p>
      ) : (
        <div className="review-file__diff [width:100%] [overflow-x:auto] [font-family:var(--font-mono)] [font-size:10.5px] [line-height:19px] [&_table]:[min-width:100%] [&_table]:[border-spacing:0] [&_table]:[border-collapse:collapse] [&_tbody]:[display:block] [&_tbody]:[min-width:max-content]">
          <table aria-label={`Diff for ${file.path}`}>
            {hunks.map((hunk) => (
              <tbody key={hunk.id} className="review-hunk">
                <tr className="review-hunk__header [color:color-mix(in_srgb,_var(--accent)_82%,_var(--foreground))] [background:color-mix(in_srgb,_var(--accent-muted)_70%,_transparent)] [min-width:max-content] [display:grid] [grid-template-columns:42px_42px_minmax(360px,_1fr)] [&_>_td]:[border-right:1px_solid_color-mix(in_srgb,_var(--border-subtle)_75%,_transparent)] [&_td]:[padding:0] [&_td_>_code]:[display:block] [&_td_>_code]:[padding:0_9px] [&_td_>_code]:[white-space:pre]">
                  <td />
                  <td />
                  <td>
                    <code>{hunk.header}</code>
                  </td>
                </tr>
                {hunk.lines.map((line) => (
                  <tr
                    key={line.id}
                    className={cx(
                      "review-line [min-width:max-content] [display:grid] [grid-template-columns:42px_42px_minmax(360px,_1fr)] [&_td]:[padding:0] [&_td_>_code]:[display:block] [&_td_>_code]:[padding:0_9px] [&_td_>_code]:[white-space:pre] [&_code_>_span]:[display:inline-block] [&_code_>_span]:[width:14px] [&_code_>_span]:[color:var(--muted-foreground)] [&_code_>_span]:[user-select:none] [&.is-addition_code_>_span]:[color:var(--success)] [&.is-deletion_code_>_span]:[color:var(--danger)]",
                      REVIEW_LINE_CLASS[line.kind],
                    )}
                  >
                    <td className="review-line__number [padding-right:7px] [color:var(--muted-foreground)] [user-select:none] [text-align:right] [border-right:1px_solid_color-mix(in_srgb,_var(--border-subtle)_75%,_transparent)]">
                      {line.oldLine ?? ""}
                    </td>
                    <td className="review-line__number [padding-right:7px] [color:var(--muted-foreground)] [user-select:none] [text-align:right] [border-right:1px_solid_color-mix(in_srgb,_var(--border-subtle)_75%,_transparent)]">
                      {line.newLine ?? ""}
                    </td>
                    <td>
                      <code>
                        <span aria-hidden="true">{line.marker}</span>
                        {line.content}
                      </code>
                    </td>
                  </tr>
                ))}
              </tbody>
            ))}
          </table>
        </div>
      )}
      {file.truncated && (
        <p className="review-file__truncated [padding:5px_9px] [background:var(--input)] [border-top:1px_solid_var(--border-subtle)] [text-align:left] [margin:0] [color:var(--muted-foreground)] [font-size:10.5px]">
          Preview truncated
        </p>
      )}
    </article>
  );
}

export interface ParsedDiffLine {
  id: string;
  kind: "addition" | "deletion" | "context";
  marker: "+" | "-" | " ";
  content: string;
  oldLine?: number;
  newLine?: number;
}

export interface ParsedDiffHunk {
  id: string;
  header: string;
  lines: ParsedDiffLine[];
}

export function parseUnifiedPatch(patch: string): ParsedDiffHunk[] {
  const hunks: ParsedDiffHunk[] = [];
  let current: ParsedDiffHunk | null = null;
  let oldLine = 0;
  let newLine = 0;

  for (const rawLine of patch.split("\n")) {
    const match = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(rawLine);
    if (match) {
      oldLine = Number(match[1]);
      newLine = Number(match[2]);
      current = { id: `${oldLine}:${newLine}:${hunks.length}`, header: rawLine, lines: [] };
      hunks.push(current);
      continue;
    }
    if (!current || rawLine === "\\ No newline at end of file") continue;
    const marker = rawLine[0];
    const content = rawLine.slice(1);
    if (marker === "+") {
      current.lines.push({
        id: `addition:${newLine}:${current.lines.length}`,
        kind: "addition",
        marker,
        content,
        newLine,
      });
      newLine += 1;
    } else if (marker === "-") {
      current.lines.push({
        id: `deletion:${oldLine}:${current.lines.length}`,
        kind: "deletion",
        marker,
        content,
        oldLine,
      });
      oldLine += 1;
    } else if (marker === " ") {
      current.lines.push({
        id: `context:${oldLine}:${newLine}:${current.lines.length}`,
        kind: "context",
        marker,
        content,
        oldLine,
        newLine,
      });
      oldLine += 1;
      newLine += 1;
    }
  }
  return hunks;
}

function TerminalTool({
  api,
  cwd,
  active,
}: {
  api: WorkspacePanelApi;
  cwd: string;
  active: boolean;
}) {
  const terminalElementRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XtermTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const terminalIdRef = useRef<string | null>(null);
  const startingRef = useRef(false);
  const readyRef = useRef(false);
  const activeRef = useRef(active);
  const pendingInputRef = useRef("");
  const fitRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<TerminalStatus>("idle");

  const startTerminal = useCallback(async () => {
    const terminal = terminalRef.current;
    const fitAddon = fitAddonRef.current;
    if (!terminal || !fitAddon || terminalIdRef.current || startingRef.current) return;
    startingRef.current = true;
    setStatus("starting");
    fitAddon.fit();
    const id = requestId("terminal");
    terminalIdRef.current = id;
    try {
      await api.createTerminal({ id, cwd, cols: terminal.cols, rows: terminal.rows });
      readyRef.current = true;
      setStatus("running");
      if (pendingInputRef.current) {
        const input = pendingInputRef.current;
        pendingInputRef.current = "";
        await api.writeTerminal(id, input);
      }
      terminal.focus();
    } catch (reason) {
      if (terminalIdRef.current === id) terminalIdRef.current = null;
      readyRef.current = false;
      setStatus("error");
      terminal.writeln(`\r\nUnable to start terminal: ${plainText(errorMessage(reason))}`);
    } finally {
      startingRef.current = false;
    }
  }, [api, cwd]);

  const newTerminal = useCallback(async () => {
    const id = terminalIdRef.current;
    terminalIdRef.current = null;
    readyRef.current = false;
    pendingInputRef.current = "";
    if (id) await api.killTerminal(id);
    terminalRef.current?.reset();
    setStatus("idle");
    await startTerminal();
  }, [api, startTerminal]);

  useEffect(() => {
    const element = terminalElementRef.current;
    if (!element) return;
    const terminal = new XtermTerminal({
      cursorBlink: true,
      cursorStyle: "bar",
      fontFamily:
        '"SFMono-Regular", "SF Mono", "Cascadia Code", Consolas, "Liberation Mono", Menlo, monospace',
      fontSize: 12.5,
      lineHeight: 1.25,
      minimumContrastRatio: 4.5,
      screenReaderMode: true,
      scrollback: 5_000,
      theme: terminalTheme(),
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(element);
    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    let lastDimensions = "";
    const fit = () => {
      if (!activeRef.current || element.offsetWidth === 0 || element.offsetHeight === 0) return;
      fitAddon.fit();
      const dimensions = `${terminal.cols}:${terminal.rows}`;
      if (lastDimensions === dimensions) return;
      lastDimensions = dimensions;
      const id = terminalIdRef.current;
      if (id) void api.resizeTerminal(id, terminal.cols, terminal.rows);
    };
    fitRef.current = fit;
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fit);
    observer?.observe(element);
    const input = terminal.onData((data) => {
      const id = terminalIdRef.current;
      if (!id || !readyRef.current) {
        pendingInputRef.current += data;
        return;
      }
      void api.writeTerminal(id, data);
    });
    const removeData = api.onTerminalData((event) => {
      if (event.id === terminalIdRef.current) terminal.write(event.data);
    });
    const removeExit = api.onTerminalExit((event) => {
      if (event.id !== terminalIdRef.current) return;
      terminalIdRef.current = null;
      readyRef.current = false;
      setStatus("exited");
      terminal.writeln(`\r\n[Process exited with code ${event.exitCode}]`);
    });
    const media = window.matchMedia?.("(prefers-color-scheme: light)");
    const updateTheme = () => {
      terminal.options.theme = terminalTheme();
    };
    media?.addEventListener("change", updateTheme);
    return () => {
      const id = terminalIdRef.current;
      terminalIdRef.current = null;
      readyRef.current = false;
      if (id) void api.killTerminal(id);
      observer?.disconnect();
      input.dispose();
      removeData();
      removeExit();
      media?.removeEventListener("change", updateTheme);
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
      fitRef.current = () => undefined;
    };
  }, [api]);

  useEffect(() => {
    activeRef.current = active;
    if (!active) return;
    const frame = window.requestAnimationFrame(() => {
      fitRef.current();
      void startTerminal();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [active, startTerminal]);

  return (
    <section
      className={String.raw`terminal-tool [background:var(--background)] [min-width:0] [min-height:0] [height:100%] [display:grid] [grid-template-rows:42px_minmax(0,_1fr)] [overflow:hidden] [&_.workspace-tool\_\_toolbar]:[color:var(--foreground)] [&_.workspace-tool\_\_toolbar]:[background:var(--card-solid)]`}
      aria-label="Terminal"
    >
      <div className="workspace-tool__toolbar [min-width:0] [padding:0_9px_0_12px] [display:flex] [align-items:center] [gap:8px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_strong]:[font-size:11.5px] [&_strong]:[font-weight:650]">
        <div>
          <strong>Terminal</strong>
          <span title={cwd}>{projectName(cwd)}</span>
        </div>
        <span
          className={cx(
            "terminal-tool__status [padding:2px_6px] [color:var(--muted-foreground)] [background:rgba(255,_255,_255,_0.05)] [border-radius:999px] [font-size:9px] [text-transform:capitalize]",
            TERMINAL_STATUS_CLASS[status],
          )}
        >
          {status}
        </span>
        <IconButton
          label="New terminal"
          onClick={() => void newTerminal()}
          disabled={status === "starting"}
        >
          <Plus aria-hidden="true" />
        </IconButton>
      </div>
      <div
        ref={terminalElementRef}
        className="terminal-tool__viewport [min-width:0] [min-height:0] [overflow:hidden] [padding:10px_12px] [background:var(--background)]"
        aria-label="Right panel terminal"
      />
      <span
        className="sr-only [position:absolute] [width:1px] [height:1px] [padding:0] [overflow:hidden] [clip:rect(0,_0,_0,_0)] [white-space:nowrap] [border:0]"
        aria-live="polite"
      >
        Terminal {status}
      </span>
    </section>
  );
}

function BrowserTool({ api, active }: { api: WorkspacePanelApi; active: boolean }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const browserIdRef = useRef<string | null>(null);
  const initialAddressRef = useRef("https://www.google.com");
  const [state, setState] = useState<BrowserState | null>(null);
  const [address, setAddress] = useState(initialAddressRef.current);
  const [error, setError] = useState<string | null>(null);

  const updateBounds = useCallback(() => {
    const id = browserIdRef.current;
    const element = viewportRef.current;
    if (!id || !active || !element) return;
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    void api.setBrowserBounds(id, {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    });
  }, [active, api]);

  useEffect(() => {
    const removeState = api.onBrowserState((next) => {
      if (next.id !== browserIdRef.current) return;
      setState(next);
      if (next.url) setAddress(next.url);
      setError(next.error ?? null);
    });
    return removeState;
  }, [api]);

  useEffect(() => {
    if (!active) {
      const id = browserIdRef.current;
      if (id) void api.setBrowserVisible(id, false);
      return;
    }
    let cancelled = false;
    const open = async () => {
      try {
        let id = browserIdRef.current;
        if (!id) {
          const created = await api.createBrowser({
            url: initialAddressRef.current,
            visible: true,
          });
          if (cancelled) {
            await api.destroyBrowser(created.id);
            return;
          }
          id = created.id;
          browserIdRef.current = id;
          setState(created);
        } else {
          await api.setBrowserVisible(id, true);
        }
        window.requestAnimationFrame(updateBounds);
      } catch (reason) {
        setError(errorMessage(reason));
      }
    };
    void open();
    return () => {
      cancelled = true;
    };
  }, [active, api, updateBounds]);

  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return;
    const observer =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateBounds);
    observer?.observe(element);
    window.addEventListener("resize", updateBounds);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateBounds);
    };
  }, [updateBounds]);

  useEffect(
    () => () => {
      const id = browserIdRef.current;
      browserIdRef.current = null;
      if (id) void api.destroyBrowser(id);
    },
    [api],
  );

  const navigate = async () => {
    const id = browserIdRef.current;
    if (!id) return;
    setError(null);
    try {
      setState(await api.navigateBrowser(id, browserInputUrl(address)));
    } catch (reason) {
      setError(errorMessage(reason));
    }
  };

  return (
    <section
      className="browser-tool [grid-template-rows:44px_2px_auto_minmax(0,_1fr)] [background:var(--background)] [min-width:0] [min-height:0] [height:100%] [display:grid] [overflow:hidden]"
      aria-label="Browser"
    >
      <form
        className="browser-tool__toolbar [min-width:0] [padding:0_8px] [display:flex] [align-items:center] [gap:3px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)]"
        onSubmit={(event) => {
          event.preventDefault();
          void navigate();
        }}
      >
        <IconButton
          label="Back"
          disabled={!state?.canGoBack}
          onClick={() => browserIdRef.current && void api.backBrowser(browserIdRef.current)}
        >
          <ArrowLeft aria-hidden="true" />
        </IconButton>
        <IconButton
          label="Forward"
          disabled={!state?.canGoForward}
          onClick={() => browserIdRef.current && void api.forwardBrowser(browserIdRef.current)}
        >
          <ArrowRight aria-hidden="true" />
        </IconButton>
        <IconButton
          label="Reload"
          disabled={!browserIdRef.current}
          onClick={() => browserIdRef.current && void api.reloadBrowser(browserIdRef.current)}
        >
          <RotateCw
            className={
              state?.loading ? "is-spinning [animation:spin_0.9s_linear_infinite]" : undefined
            }
            aria-hidden="true"
          />
        </IconButton>
        <label className="browser-tool__address [min-width:0] [height:30px] [margin-left:3px] [padding:0_8px] [flex:1] [display:flex] [align-items:center] [gap:6px] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [&_svg]:[width:12px] [&_svg]:[height:12px] [&_svg]:[flex:0_0_auto] [&_svg]:[color:var(--muted-foreground)] [&_input]:[min-width:0] [&_input]:[width:100%] [&_input]:[color:var(--foreground)] [&_input]:[background:transparent] [&_input]:[border:0] [&_input]:[outline:0] [&_input]:[font-size:10.5px]">
          <Search aria-hidden="true" />
          <span className="sr-only [position:absolute] [width:1px] [height:1px] [padding:0] [overflow:hidden] [clip:rect(0,_0,_0,_0)] [white-space:nowrap] [border:0]">
            Address or search
          </span>
          <input
            value={address}
            onChange={(event) => setAddress(event.target.value)}
            aria-label="Address or search"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
          />
        </label>
      </form>
      {state?.loading && (
        <div
          className="browser-tool__progress [height:2px] [background:linear-gradient(90deg,_transparent,_var(--accent),_transparent)] [background-size:200%_100%] [animation:browser-progress_1.1s_linear_infinite]"
          aria-label="Page loading"
        />
      )}
      {error && (
        <p className="browser-tool__error [margin:7px_8px_0] [color:var(--danger)] [background:var(--danger-muted)] [padding:7px_9px] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-size:10px]">
          {error}
        </p>
      )}
      <div
        ref={viewportRef}
        className="browser-tool__viewport [min-width:0] [min-height:0] [background:#ffffff]"
        aria-label="Browser page"
      />
    </section>
  );
}

function FilesTool({ api, cwd, active }: { api: WorkspacePanelApi; cwd: string; active: boolean }) {
  const [listing, setListing] = useState<WorkspaceDirectoryListing | null>(null);
  const [preview, setPreview] = useState<WorkspaceFilePreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const openDirectory = useCallback(
    async (path = "") => {
      setLoading(true);
      setError(null);
      try {
        setListing(await api.listWorkspaceDirectory(path, cwd));
      } catch (reason) {
        setError(errorMessage(reason));
      } finally {
        setLoading(false);
      }
    },
    [api, cwd],
  );

  const openFile = useCallback(
    async (path: string) => {
      setLoading(true);
      setError(null);
      try {
        setPreview(await api.readWorkspaceFile(path, cwd));
      } catch (reason) {
        setError(errorMessage(reason));
      } finally {
        setLoading(false);
      }
    },
    [api, cwd],
  );

  useEffect(() => {
    if (active && listing === null && !loading) void openDirectory();
  }, [active, listing, loading, openDirectory]);

  const parent = parentPath(listing?.path ?? "");
  return (
    <section
      className="files-tool [min-width:0] [min-height:0] [height:100%] [display:grid] [grid-template-rows:42px_minmax(0,_1fr)] [overflow:hidden]"
      aria-label="Files"
    >
      <div className="workspace-tool__toolbar [min-width:0] [padding:0_9px_0_12px] [display:flex] [align-items:center] [gap:8px] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_strong]:[font-size:11.5px] [&_strong]:[font-weight:650]">
        <div>
          <strong>Files</strong>
          <span title={listing?.root}>{projectName(listing?.root ?? "Workspace")}</span>
        </div>
        <IconButton
          label="Refresh files"
          disabled={loading}
          onClick={() => void openDirectory(listing?.path ?? "")}
        >
          <RefreshCw
            className={loading ? "is-spinning [animation:spin_0.9s_linear_infinite]" : undefined}
            aria-hidden="true"
          />
        </IconButton>
      </div>
      <div className="files-tool__layout [min-width:0] [min-height:0] [display:grid] [grid-template-columns:minmax(150px,_34%)_minmax(0,_1fr)] [overflow:hidden] max-860:[grid-template-columns:minmax(126px,_40%)_minmax(0,_1fr)]">
        <nav
          className="files-tool__browser [border-right:1px_solid_var(--border-subtle)] [background:color-mix(in_srgb,_var(--card-solid)_92%,_var(--background))] [min-width:0] [min-height:0] [overflow:auto]"
          aria-label="Workspace files"
        >
          <div
            className="files-tool__path [position:sticky] [top:0] [z-index:1] [height:36px] [padding:0_7px] [display:flex] [align-items:center] [gap:5px] [background:var(--card-solid)] [border-bottom:1px_solid_var(--border-subtle)] [&_span]:[min-width:0] [&_span]:[overflow:hidden] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-family:var(--font-mono)] [&_span]:[font-size:9.5px] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap]"
            title={listing?.path || "/"}
          >
            <IconButton
              label="Parent directory"
              disabled={!listing?.path}
              onClick={() => void openDirectory(parent)}
            >
              <ChevronLeft aria-hidden="true" />
            </IconButton>
            <span>{listing?.path || projectName(listing?.root ?? "Workspace")}</span>
          </div>
          {error ? (
            <p className="files-tool__error [margin:8px] [padding:7px_9px] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-size:10px]">
              {error}
            </p>
          ) : loading && listing === null ? (
            <ToolState icon={Loader2} title="Loading files" detail="Reading workspace…" spin />
          ) : (
            <ul className="files-tool__entries [margin:0] [padding:5px] [list-style:none] [&_button]:[width:100%] [&_button]:[height:30px] [&_button]:[padding:0_7px] [&_button]:[display:grid] [&_button]:[grid-template-columns:15px_minmax(0,_1fr)_auto] [&_button]:[align-items:center] [&_button]:[gap:6px] [&_button]:[color:var(--muted)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:5px] [&_button]:[cursor:pointer] [&_button]:[text-align:left] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_svg]:[color:var(--muted-foreground)] [&_span]:[min-width:0] [&_span]:[overflow:hidden] [&_span]:[font-size:10.5px] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:8.5px]">
              {listing?.entries.map((entry) => (
                <li key={entry.path}>
                  <button
                    type="button"
                    className={preview?.path === entry.path ? "is-selected" : undefined}
                    disabled={entry.kind !== "directory" && entry.kind !== "file"}
                    onClick={() =>
                      entry.kind === "directory"
                        ? void openDirectory(entry.path)
                        : void openFile(entry.path)
                    }
                    title={entry.path}
                  >
                    {entry.kind === "directory" ? (
                      <Folder aria-hidden="true" />
                    ) : (
                      <File aria-hidden="true" />
                    )}
                    <span>{entry.name}</span>
                    {entry.kind === "file" && entry.size !== undefined && (
                      <small>{formatBytes(entry.size)}</small>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>
        <div
          className="files-tool__preview [background:color-mix(in_srgb,_var(--background)_84%,_var(--card-solid))] [min-width:0] [min-height:0] [overflow:auto] [&_>_header]:[position:sticky] [&_>_header]:[top:0] [&_>_header]:[z-index:1] [&_>_header]:[height:36px] [&_>_header]:[padding:0_10px] [&_>_header]:[display:flex] [&_>_header]:[align-items:center] [&_>_header]:[justify-content:space-between] [&_>_header]:[gap:8px] [&_>_header]:[background:var(--card-solid)] [&_>_header]:[border-bottom:1px_solid_var(--border-subtle)] [&_>_header_strong]:[min-width:0] [&_>_header_strong]:[overflow:hidden] [&_>_header_strong]:[font-family:var(--font-mono)] [&_>_header_strong]:[font-size:9.5px] [&_>_header_strong]:[font-weight:570] [&_>_header_strong]:[text-overflow:ellipsis] [&_>_header_strong]:[white-space:nowrap] [&_>_header_span]:[color:var(--muted-foreground)] [&_>_header_span]:[font-size:8.5px] [&_>_p]:[color:var(--muted-foreground)] [&_>_p]:[font-size:8.5px] [&_>_pre]:[min-width:max-content] [&_>_pre]:[margin:0] [&_>_pre]:[padding:7px_0_18px] [&_>_pre]:[font-family:var(--font-mono)] [&_>_pre]:[font-size:10px] [&_>_pre]:[line-height:18px] [&_>_pre_>_span]:[display:grid] [&_>_pre_>_span]:[grid-template-columns:44px_minmax(320px,_1fr)] [&_>_pre_i]:[padding-right:9px] [&_>_pre_i]:[color:var(--muted-foreground)] [&_>_pre_i]:[border-right:1px_solid_var(--border-subtle)] [&_>_pre_i]:[font-style:normal] [&_>_pre_i]:[user-select:none] [&_>_pre_i]:[text-align:right] [&_>_pre_code]:[padding:0_10px] [&_>_pre_code]:[white-space:pre] [&_>_p]:[margin:0] [&_>_p]:[padding:6px_10px] [&_>_p]:[background:var(--input)] [&_>_p]:[border-top:1px_solid_var(--border-subtle)]"
          aria-label="File preview"
        >
          {!preview ? (
            <ToolState
              icon={FileCode2}
              title="Select a file"
              detail="Open a text file to preview it."
            />
          ) : preview.binary ? (
            <ToolState
              icon={File}
              title="Binary file"
              detail={`${preview.path} · ${formatBytes(preview.size)}`}
            />
          ) : (
            <>
              <header>
                <strong title={preview.path}>{preview.path}</strong>
                <span>{formatBytes(preview.size)}</span>
              </header>
              <pre>
                {preview.content.split("\n").map((line, index) => (
                  // biome-ignore lint/suspicious/noArrayIndexKey: The source line number is the stable identity here.
                  <span key={`${index}:${line}`}>
                    <i>{index + 1}</i>
                    <code>{line || " "}</code>
                  </span>
                ))}
              </pre>
              {preview.truncated && <p>Preview truncated</p>}
            </>
          )}
        </div>
      </div>
    </section>
  );
}

function IconButton({
  label,
  children,
  ...props
}: Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "type" | "aria-label"> & {
  label: string;
}) {
  return (
    <button
      type="button"
      className="workspace-tool__icon-button [display:inline-grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:1px_solid_transparent] [border-radius:7px] [cursor:pointer] [width:30px] [height:30px] [flex:0_0_30px] [padding:0] [&_svg]:[width:15px] [&_svg]:[height:15px]"
      aria-label={label}
      title={label}
      {...props}
    >
      {children}
    </button>
  );
}

function ToolState({
  icon: Icon,
  title,
  detail,
  spin = false,
}: {
  icon: typeof GitCompareArrows;
  title: string;
  detail: string;
  spin?: boolean;
}) {
  return (
    <div className="workspace-tool__state [min-height:100%] [padding:30px] [display:flex] [flex-direction:column] [align-items:center] [justify-content:center] [color:var(--muted-foreground)] [text-align:center] [&_>_svg]:[width:24px] [&_>_svg]:[height:24px] [&_>_svg]:[margin-bottom:12px] [&_>_svg]:[opacity:0.72] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:12.5px] [&_strong]:[font-weight:620] [&_span]:[max-width:320px] [&_span]:[margin-top:4px] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.45]">
      <Icon
        className={spin ? "is-spinning [animation:spin_0.9s_linear_infinite]" : undefined}
        aria-hidden="true"
      />
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  );
}

function browserInputUrl(value: string): string {
  const trimmed = value.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  if (/^(localhost|\d{1,3}(?:\.\d{1,3}){3})(:\d+)?(?:\/|$)/i.test(trimmed)) {
    return `http://${trimmed}`;
  }
  if (/^[\w.-]+\.[a-z]{2,}(?::\d+)?(?:\/|$)/i.test(trimmed)) return `https://${trimmed}`;
  return `https://www.google.com/search?q=${encodeURIComponent(trimmed)}`;
}

function reviewStatusLabel(status: string): "Added" | "Deleted" | "Renamed" | "Modified" {
  const code = status.replace(/\s/g, "")[0]?.toUpperCase();
  if (code === "A" || code === "?") return "Added";
  if (code === "D") return "Deleted";
  if (code === "R") return "Renamed";
  return "Modified";
}

function reviewStatusTone(status: string): ReviewStatusTone {
  return REVIEW_STATUS_TONE[reviewStatusLabel(status)];
}

function terminalTheme() {
  const light = window.matchMedia?.("(prefers-color-scheme: light)").matches;
  return light
    ? {
        background: "#ffffff",
        foreground: "#20242c",
        cursor: "#087c9b",
        selectionBackground: "#cfeef5",
        red: "#c33535",
        green: "#087c55",
        blue: "#0969da",
      }
    : {
        background: "#090b10",
        foreground: "#e7eaf0",
        cursor: "#95e9ff",
        selectionBackground: "#274652",
        red: "#f87171",
        green: "#34d399",
        blue: "#60a5fa",
      };
}

function requestId(prefix: string): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}

function parentPath(path: string): string {
  const parts = path.split("/").filter(Boolean);
  parts.pop();
  return parts.join("/");
}

function projectName(path: string): string {
  return (
    path
      .replace(/[\\/]+$/, "")
      .split(/[\\/]/)
      .filter(Boolean)
      .at(-1) || "Workspace"
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${Math.round(bytes / 1_024)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

function plainText(value: string): string {
  return Array.from(value, (character) => {
    const codePoint = character.codePointAt(0) ?? 0;
    return codePoint <= 31 || (codePoint >= 127 && codePoint <= 159) ? " " : character;
  })
    .join("")
    .trim();
}
