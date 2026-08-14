import { FitAddon } from "@xterm/addon-fit";
import { type ITheme, Terminal as XtermTerminal } from "@xterm/xterm";
import { useCallback, useEffect, useRef, useState } from "react";
import type { DesktopTerminalApi } from "../../shared/desktop-api.js";
import { errorMessage } from "./text-utils.js";

export type TerminalStatus = "idle" | "starting" | "running" | "exited" | "error";

export const TERMINAL_STATUS_CLASS = {
  idle: "is-idle",
  starting: "is-starting",
  running: "is-running",
  exited: "is-exited",
  error: "is-error",
} satisfies Record<TerminalStatus, string>;

export function useTerminalController({
  api,
  active,
  cwd,
  getTheme,
}: {
  api: DesktopTerminalApi;
  active: boolean;
  cwd: string;
  getTheme: () => ITheme;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<XtermTerminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const requestedIdRef = useRef<string | null>(null);
  const liveIdRef = useRef<string | null>(null);
  const startingRef = useRef(false);
  const restartingRef = useRef(false);
  const restartPromiseRef = useRef<Promise<void> | null>(null);
  const readyRef = useRef(false);
  const activeRef = useRef(active);
  const pendingInputRef = useRef("");
  const generationRef = useRef(0);
  const getThemeRef = useRef(getTheme);
  const fitAndResizeRef = useRef<() => void>(() => undefined);
  const [status, setStatus] = useState<TerminalStatus>("idle");
  getThemeRef.current = getTheme;
  activeRef.current = active;

  const startTerminal = useCallback(
    async (allowWhileRestarting = false) => {
      const terminal = terminalRef.current;
      const fitAddon = fitAddonRef.current;
      if (
        !terminal ||
        !fitAddon ||
        startingRef.current ||
        requestedIdRef.current ||
        (restartingRef.current && !allowWhileRestarting)
      ) {
        return;
      }

      const generation = generationRef.current;
      const id = terminalRequestId();
      startingRef.current = true;
      readyRef.current = false;
      requestedIdRef.current = id;
      setStatus("starting");
      fitAddon.fit();

      try {
        await api.createTerminal({ id, cwd, cols: terminal.cols, rows: terminal.rows });
        if (generationRef.current !== generation || requestedIdRef.current !== id) {
          await api.killTerminal(id);
          return;
        }
        liveIdRef.current = id;
        readyRef.current = true;
        setStatus("running");
        if (pendingInputRef.current) {
          const pendingInput = pendingInputRef.current;
          pendingInputRef.current = "";
          await api.writeTerminal(id, pendingInput);
        }
        if (
          generationRef.current === generation &&
          requestedIdRef.current === id &&
          activeRef.current
        ) {
          terminal.focus();
        }
      } catch (error) {
        if (generationRef.current !== generation || requestedIdRef.current !== id) return;
        if (liveIdRef.current === id) {
          readyRef.current = true;
          setStatus("error");
          terminal.writeln(
            `\r\nUnable to write to terminal: ${plainTerminalError(errorMessage(error))}`,
          );
          return;
        }
        requestedIdRef.current = null;
        liveIdRef.current = null;
        readyRef.current = false;
        pendingInputRef.current = "";
        setStatus("error");
        terminal.writeln(
          `\r\nUnable to start terminal: ${plainTerminalError(errorMessage(error))}`,
        );
      } finally {
        if (generationRef.current === generation) startingRef.current = false;
      }
    },
    [api, cwd],
  );

  const startNewTerminal = useCallback((): Promise<void> => {
    if (restartPromiseRef.current) return restartPromiseRef.current;
    restartingRef.current = true;
    const generation = ++generationRef.current;
    const liveId = liveIdRef.current;
    if (!liveId) requestedIdRef.current = null;
    startingRef.current = false;
    readyRef.current = false;
    pendingInputRef.current = "";

    const restart = Promise.resolve().then(async () => {
      try {
        if (liveId) await api.killTerminal(liveId);
        if (generationRef.current !== generation) return;
        if (liveIdRef.current === liveId) {
          requestedIdRef.current = null;
          liveIdRef.current = null;
        }
        terminalRef.current?.reset();
        setStatus("idle");
        await startTerminal(true);
      } catch (error) {
        if (generationRef.current !== generation) return;
        if (liveId && liveIdRef.current === liveId) {
          requestedIdRef.current = liveId;
          readyRef.current = true;
        }
        setStatus("error");
        terminalRef.current?.writeln(
          `\r\nUnable to restart terminal: ${plainTerminalError(errorMessage(error))}`,
        );
      } finally {
        if (generationRef.current === generation) restartingRef.current = false;
        if (restartPromiseRef.current === restart) restartPromiseRef.current = null;
      }
    });
    restartPromiseRef.current = restart;
    return restart;
  }, [api, startTerminal]);

  useEffect(() => {
    const generation = ++generationRef.current;
    const element = viewportRef.current;
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
      theme: getThemeRef.current(),
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
      const id = liveIdRef.current;
      if (id) void api.resizeTerminal(id, terminal.cols, terminal.rows);
    };
    fitAndResizeRef.current = fitAndResize;
    const inputSubscription = terminal.onData((data) => {
      const id = requestedIdRef.current;
      if (!id || !readyRef.current) {
        pendingInputRef.current += data;
        return;
      }
      void api.writeTerminal(id, data).catch((error) => {
        if (requestedIdRef.current !== id) return;
        setStatus("error");
        terminal.writeln(
          `\r\nUnable to write to terminal: ${plainTerminalError(errorMessage(error))}`,
        );
      });
    });
    const removeDataListener = api.onTerminalData((event) => {
      if (event.id === requestedIdRef.current) terminal.write(event.data);
    });
    const removeExitListener = api.onTerminalExit((event) => {
      if (event.id !== requestedIdRef.current) return;
      requestedIdRef.current = null;
      liveIdRef.current = null;
      readyRef.current = false;
      setStatus("exited");
      terminal.writeln(`\r\n[Process exited with code ${event.exitCode}]`);
    });
    const media = window.matchMedia?.("(prefers-color-scheme: light)");
    const updateTheme = () => {
      terminal.options.theme = getThemeRef.current();
    };
    media?.addEventListener("change", updateTheme);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(fitAndResize);
    resizeObserver?.observe(element);

    return () => {
      generationRef.current += 1;
      const liveId = liveIdRef.current;
      requestedIdRef.current = null;
      liveIdRef.current = null;
      startingRef.current = false;
      restartingRef.current = false;
      restartPromiseRef.current = null;
      readyRef.current = false;
      pendingInputRef.current = "";
      if (liveId) void api.killTerminal(liveId);
      resizeObserver?.disconnect();
      media?.removeEventListener("change", updateTheme);
      inputSubscription.dispose();
      removeDataListener();
      removeExitListener();
      terminal.dispose();
      if (generationRef.current > generation) {
        terminalRef.current = null;
        fitAddonRef.current = null;
        fitAndResizeRef.current = () => undefined;
      }
    };
  }, [api]);

  useEffect(() => {
    if (!active) return;
    const frame = window.requestAnimationFrame(() => {
      fitAndResizeRef.current();
      void startTerminal();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [active, startTerminal]);

  return { viewportRef, status, startNewTerminal };
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
