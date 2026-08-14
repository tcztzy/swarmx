import { Plus, Terminal as TerminalIcon, X } from "lucide-react";
import { api } from "./renderer-api.js";
import { TERMINAL_STATUS_CLASS, useTerminalController } from "./terminal-controller.js";
import { projectName } from "./text-utils.js";
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
  const { viewportRef, status, startNewTerminal } = useTerminalController({
    api,
    active,
    cwd,
    getTheme: internalTerminalTheme,
  });

  return (
    <section
      className="runtime-bottom-panel [min-width:0] [height:clamp(180px,_29vh,_320px)] [min-height:0] [overflow:hidden] [display:grid] [grid-template-rows:40px_minmax(0,_1fr)] [border-top:1px_solid_var(--border-subtle)] [background:#0b0d12] [box-shadow:0_-12px_32px_rgba(0,_0,_0,_0.1)] [@media(prefers-color-scheme:light)]:[background:#ffffff] max-860:[grid-template-columns:1fr] max-680:[max-height:210px]"
      aria-label="Bottom panel"
    >
      <div className="terminal-panel__tabbar [justify-content:space-between] [border-bottom:1px_solid_var(--border-subtle)] [background:var(--card-solid)] [color:var(--muted-foreground)] [min-width:0] [display:flex] [align-items:center] [&_>_.button]:[width:34px] [&_>_.button]:[height:34px] [&_>_.button]:[margin:0_5px]">
        <div
          className="terminal-panel__tabs [height:100%] [flex:1] [min-width:0] [display:flex] [align-items:center] [&_>_.button]:[width:34px] [&_>_.button]:[height:34px] [&_>_.button]:[margin:0_5px]"
          role="tablist"
          aria-label="Terminals"
        >
          <button
            type="button"
            className="terminal-panel__tab [align-self:stretch] [width:min(220px,_40vw)] [padding:0_14px] [gap:8px] [border:0] [border-right:1px_solid_var(--border-subtle)] [background:rgba(255,_255,_255,_0.035)] [color:var(--foreground)] [cursor:default] [font-size:12px] [min-width:0] [display:flex] [align-items:center] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[flex:0_0_auto]"
            role="tab"
            aria-selected="true"
          >
            <TerminalIcon aria-hidden="true" />
            <span>{projectName(cwd)}</span>
            <span
              className={cx(
                "terminal-panel__status [width:6px] [height:6px] [margin-left:auto] [flex:0_0_auto] [border-radius:999px] [background:var(--muted-foreground)]",
                TERMINAL_STATUS_CLASS[status],
              )}
              aria-hidden="true"
            />
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => void startNewTerminal()}
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
        ref={viewportRef}
        className="terminal-panel__viewport [min-width:0] [min-height:0] [overflow:hidden] [padding:9px_12px_10px] [background:#0b0d12] [@media(prefers-color-scheme:light)]:[background:#ffffff]"
        aria-label="Internal terminal"
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
