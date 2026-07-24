import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XtermTerminal } from "@xterm/xterm";
import { Plus, Terminal as TerminalIcon, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./renderer-api.js";
import { errorMessage, projectName } from "./text-utils.js";
import { Button, cx } from "./ui-primitives.js";

export function RuntimeBottomPanel({
  active,
  cwd,
  onClose,
}: {
  active: boolean;
  cwd: string;
  onClose: () => void;
}) {
  const terminalElementRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XtermTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const terminalIdRef = useRef<string | null>(null);
  const startingRef = useRef(false);
  const readyRef = useRef(false);
  const activeRef = useRef(active);
  const disposedRef = useRef(false);
  const pendingInputRef = useRef("");
  const fitAndResizeRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<"idle" | "starting" | "running" | "exited" | "error">(
    "idle",
  );

  const startTerminal = useCallback(async () => {
    const terminal = terminalRef.current;
    const fitAddon = fitAddonRef.current;
    if (!terminal || !fitAddon || startingRef.current || terminalIdRef.current) return;

    startingRef.current = true;
    readyRef.current = false;
    setStatus("starting");
    fitAddon.fit();
    const id = terminalRequestId();
    terminalIdRef.current = id;

    try {
      await api.createTerminal({ id, cwd, cols: terminal.cols, rows: terminal.rows });
      if (disposedRef.current || terminalIdRef.current !== id) {
        await api.killTerminal(id);
        return;
      }
      readyRef.current = true;
      setStatus("running");
      if (pendingInputRef.current) {
        const pendingInput = pendingInputRef.current;
        pendingInputRef.current = "";
        await api.writeTerminal(id, pendingInput);
      }
      terminal.focus();
    } catch (error) {
      if (terminalIdRef.current === id) terminalIdRef.current = null;
      pendingInputRef.current = "";
      setStatus("error");
      terminal.writeln(`\r\nUnable to start terminal: ${plainTerminalError(errorMessage(error))}`);
    } finally {
      startingRef.current = false;
    }
  }, [cwd]);

  const newTerminal = useCallback(async () => {
    const currentId = terminalIdRef.current;
    terminalIdRef.current = null;
    readyRef.current = false;
    pendingInputRef.current = "";
    if (currentId) await api.killTerminal(currentId);
    terminalRef.current?.reset();
    setStatus("idle");
    await startTerminal();
  }, [startTerminal]);

  useEffect(() => {
    disposedRef.current = false;
    const element = terminalElementRef.current;
    if (!element) return;

    const terminal = new XtermTerminal({
      allowTransparency: false,
      cursorBlink: true,
      cursorStyle: "bar",
      fontFamily:
        '"SFMono-Regular", "SF Mono", "Cascadia Code", Consolas, "Liberation Mono", Menlo, monospace',
      fontSize: 12.5,
      lineHeight: 1.25,
      minimumContrastRatio: 4.5,
      screenReaderMode: true,
      scrollback: 5_000,
      theme: internalTerminalTheme(),
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(element);
    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    let lastDimensions = "";
    const fitAndResize = () => {
      if (!activeRef.current || !element.offsetWidth || !element.offsetHeight) return;
      fitAddon.fit();
      const dimensions = `${terminal.cols}:${terminal.rows}`;
      if (dimensions === lastDimensions) return;
      lastDimensions = dimensions;
      const id = terminalIdRef.current;
      if (id) void api.resizeTerminal(id, terminal.cols, terminal.rows);
    };
    fitAndResizeRef.current = fitAndResize;

    const terminalInput = terminal.onData((data) => {
      const id = terminalIdRef.current;
      if (!id || !readyRef.current) {
        pendingInputRef.current += data;
        return;
      }
      void api.writeTerminal(id, data);
    });
    const removeDataListener = api.onTerminalData((event) => {
      if (event.id === terminalIdRef.current) terminal.write(event.data);
    });
    const removeExitListener = api.onTerminalExit((event) => {
      if (event.id !== terminalIdRef.current) return;
      terminalIdRef.current = null;
      readyRef.current = false;
      setStatus("exited");
      terminal.writeln(`\r\n[Process exited with code ${event.exitCode}]`);
    });
    const media =
      typeof window.matchMedia === "function"
        ? window.matchMedia("(prefers-color-scheme: light)")
        : null;
    const updateTheme = () => {
      terminal.options.theme = internalTerminalTheme();
    };
    media?.addEventListener("change", updateTheme);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fitAndResize);
    resizeObserver?.observe(element);

    return () => {
      disposedRef.current = true;
      const id = terminalIdRef.current;
      terminalIdRef.current = null;
      readyRef.current = false;
      if (id) void api.killTerminal(id);
      resizeObserver?.disconnect();
      media?.removeEventListener("change", updateTheme);
      terminalInput.dispose();
      removeDataListener();
      removeExitListener();
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
      fitAndResizeRef.current = () => undefined;
    };
  }, []);

  useEffect(() => {
    activeRef.current = active;
    if (!active) return;
    const frame = window.requestAnimationFrame(() => {
      fitAndResizeRef.current();
      void startTerminal();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [active, startTerminal]);

  return (
    <section className="runtime-bottom-panel" aria-label="Bottom panel">
      <div className="terminal-panel__tabbar">
        <div className="terminal-panel__tabs" role="tablist" aria-label="Terminals">
          <button type="button" className="terminal-panel__tab" role="tab" aria-selected="true">
            <TerminalIcon aria-hidden="true" />
            <span>{projectName(cwd)}</span>
            <span className={cx("terminal-panel__status", `is-${status}`)} aria-hidden="true" />
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => void newTerminal()}
            disabled={status === "starting"}
            title="New terminal"
            aria-label="New terminal"
          >
            <Plus aria-hidden="true" />
          </Button>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          title="Close bottom panel"
          aria-label="Close bottom panel"
        >
          <X aria-hidden="true" />
        </Button>
      </div>
      <div
        ref={terminalElementRef}
        className="terminal-panel__viewport"
        aria-label="Internal terminal"
      />
      <span className="sr-only" aria-live="polite">
        Terminal {status}
      </span>
    </section>
  );
}

function terminalRequestId(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `terminal-${Date.now()}-${Math.random().toString(36).slice(2)}`
  );
}

function plainTerminalError(message: string): string {
  return [...message]
    .map((character) => {
      const code = character.charCodeAt(0);
      return code < 32 || (code >= 127 && code <= 159) ? " " : character;
    })
    .join("")
    .trim();
}

function internalTerminalTheme() {
  if (
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
  ) {
    return {
      background: "#ffffff",
      foreground: "#20242c",
      cursor: "#087c9b",
      cursorAccent: "#ffffff",
      selectionBackground: "#cfeef5",
      black: "#20242c",
      brightBlack: "#737e8e",
      red: "#c33535",
      green: "#087c55",
      yellow: "#9a6700",
      blue: "#0969da",
      magenta: "#8250df",
      cyan: "#087c9b",
      white: "#e7eaf0",
      brightWhite: "#17191f",
    };
  }
  return {
    background: "#0b0d12",
    foreground: "#e8eaf0",
    cursor: "#95e9ff",
    cursorAccent: "#0b0d12",
    selectionBackground: "#294451",
    black: "#151821",
    brightBlack: "#77808f",
    red: "#f87171",
    green: "#34d399",
    yellow: "#fbbf24",
    blue: "#60a5fa",
    magenta: "#c084fc",
    cyan: "#67e8f9",
    white: "#d5d9e2",
    brightWhite: "#ffffff",
  };
}
