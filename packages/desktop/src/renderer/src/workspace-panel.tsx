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
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
    <aside className="runtime-right-panel workspace-panel" aria-label="Right panel">
      {activeTool && (
        <header className="workspace-panel__header">
          <button
            type="button"
            className="workspace-panel__home"
            onClick={() => setActiveTool(null)}
            aria-label="Workspace tools home"
            title="Workspace tools"
          >
            <Home aria-hidden="true" />
          </button>
          <div className="workspace-panel__tabs" role="tablist" aria-label="Workspace tools">
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
            className="workspace-panel__close"
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
        <div className="workspace-panel__views">
          {visitedTools.has("review") && (
            <div className="workspace-panel__view" hidden={activeTool !== "review"}>
              <ReviewTool key={cwd} api={api} cwd={cwd} active={activeTool === "review"} />
            </div>
          )}
          {visitedTools.has("terminal") && (
            <div className="workspace-panel__view" hidden={activeTool !== "terminal"}>
              <TerminalTool api={api} cwd={cwd} active={activeTool === "terminal"} />
            </div>
          )}
          {visitedTools.has("browser") && (
            <div className="workspace-panel__view" hidden={activeTool !== "browser"}>
              <BrowserTool api={api} active={activeTool === "browser"} />
            </div>
          )}
          {visitedTools.has("files") && (
            <div className="workspace-panel__view" hidden={activeTool !== "files"}>
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
    <nav className="workspace-panel__launcher" aria-label="Open workspace tool">
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
    <section className="review-tool" aria-label="Review changes">
      <div className="workspace-tool__toolbar">
        <div>
          <strong>Changes</strong>
          <span>{snapshot?.branch || "Working tree"}</span>
        </div>
        <div className="review-tool__summary" aria-label="Change summary">
          <span>{snapshot?.files.length ?? 0} files</span>
          <b>+{totals.additions}</b>
          <i>−{totals.deletions}</i>
        </div>
        <IconButton label="Refresh changes" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={loading ? "is-spinning" : undefined} aria-hidden="true" />
        </IconButton>
      </div>

      <div className="review-tool__body">
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
              <p className="workspace-tool__notice">Large review truncated to a safe preview.</p>
            )}
            <div className="review-tool__files">
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
    <article className="review-file">
      <header className="review-file__header">
        <button
          type="button"
          className="review-file__toggle"
          aria-label={`${expanded ? "Collapse" : "Expand"} ${file.path}`}
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
        >
          <ChevronDown aria-hidden="true" />
        </button>
        <FileCode2 aria-hidden="true" />
        <strong title={file.path}>{file.path}</strong>
        <span className={`review-file__status is-${reviewStatusTone(file.status)}`}>
          {reviewStatusLabel(file.status)}
        </span>
        <span className="review-file__stats">
          <b>+{file.additions}</b>
          <i>−{file.deletions}</i>
        </span>
      </header>
      {!expanded ? null : file.binary ? (
        <p className="review-file__binary">Binary file changed</p>
      ) : hunks.length === 0 ? (
        <p className="review-file__binary">No text preview available</p>
      ) : (
        <div className="review-file__diff">
          <table aria-label={`Diff for ${file.path}`}>
            {hunks.map((hunk) => (
              <tbody key={hunk.id} className="review-hunk">
                <tr className="review-hunk__header">
                  <td />
                  <td />
                  <td>
                    <code>{hunk.header}</code>
                  </td>
                </tr>
                {hunk.lines.map((line) => (
                  <tr key={line.id} className={`review-line is-${line.kind}`}>
                    <td className="review-line__number">{line.oldLine ?? ""}</td>
                    <td className="review-line__number">{line.newLine ?? ""}</td>
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
      {file.truncated && <p className="review-file__truncated">Preview truncated</p>}
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
  const [status, setStatus] = useState<"idle" | "starting" | "running" | "exited" | "error">(
    "idle",
  );

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
    <section className="terminal-tool" aria-label="Terminal">
      <div className="workspace-tool__toolbar">
        <div>
          <strong>Terminal</strong>
          <span title={cwd}>{projectName(cwd)}</span>
        </div>
        <span className={`terminal-tool__status is-${status}`}>{status}</span>
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
        className="terminal-tool__viewport"
        aria-label="Right panel terminal"
      />
      <span className="sr-only" aria-live="polite">
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
    <section className="browser-tool" aria-label="Browser">
      <form
        className="browser-tool__toolbar"
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
          <RotateCw className={state?.loading ? "is-spinning" : undefined} aria-hidden="true" />
        </IconButton>
        <label className="browser-tool__address">
          <Search aria-hidden="true" />
          <span className="sr-only">Address or search</span>
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
      {state?.loading && <div className="browser-tool__progress" aria-label="Page loading" />}
      {error && <p className="browser-tool__error">{error}</p>}
      <div ref={viewportRef} className="browser-tool__viewport" aria-label="Browser page" />
    </section>
  );
}

function FilesTool({
  api,
  cwd,
  active,
}: {
  api: WorkspacePanelApi;
  cwd: string;
  active: boolean;
}) {
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
    <section className="files-tool" aria-label="Files">
      <div className="workspace-tool__toolbar">
        <div>
          <strong>Files</strong>
          <span title={listing?.root}>{projectName(listing?.root ?? "Workspace")}</span>
        </div>
        <IconButton
          label="Refresh files"
          disabled={loading}
          onClick={() => void openDirectory(listing?.path ?? "")}
        >
          <RefreshCw className={loading ? "is-spinning" : undefined} aria-hidden="true" />
        </IconButton>
      </div>
      <div className="files-tool__layout">
        <nav className="files-tool__browser" aria-label="Workspace files">
          <div className="files-tool__path" title={listing?.path || "/"}>
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
            <p className="files-tool__error">{error}</p>
          ) : loading && listing === null ? (
            <ToolState icon={Loader2} title="Loading files" detail="Reading workspace…" spin />
          ) : (
            <ul className="files-tool__entries">
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
        <div className="files-tool__preview" aria-label="File preview">
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
      className="workspace-tool__icon-button"
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
    <div className="workspace-tool__state">
      <Icon className={spin ? "is-spinning" : undefined} aria-hidden="true" />
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

function reviewStatusLabel(status: string): string {
  const code = status.replace(/\s/g, "")[0]?.toUpperCase();
  if (code === "A" || code === "?") return "Added";
  if (code === "D") return "Deleted";
  if (code === "R") return "Renamed";
  return "Modified";
}

function reviewStatusTone(status: string): string {
  return reviewStatusLabel(status).toLowerCase();
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
