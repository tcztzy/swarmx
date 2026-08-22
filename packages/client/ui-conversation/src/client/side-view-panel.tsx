import type { PropsRenderSlots, PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import { type KeyboardEvent, useCallback, useRef, useSyncExternalStore } from "react";
import type { ISideView } from "./side-view.js";
import css from "./side-view.module.css";

interface SideViewPanelInjected {
  readonly sideView: ISideView;
}

type SideViewPanelProps = PropsRuntime<"details"> &
  PropsRenderSlots<"side-view.content"> &
  SideViewPanelInjected;

/** Resolve standard horizontal-tab keyboard movement without touching DOM state. */
export function sideViewTabTarget(
  entryIds: readonly string[],
  activeId: string,
  key: string,
): string | null {
  if (entryIds.length === 0) return null;
  if (key === "Home") return entryIds[0] ?? null;
  if (key === "End") return entryIds.at(-1) ?? null;
  const delta =
    key === "ArrowRight" || key === "ArrowDown"
      ? 1
      : key === "ArrowLeft" || key === "ArrowUp"
        ? -1
        : 0;
  if (delta === 0) return null;
  const current = Math.max(0, entryIds.indexOf(activeId));
  return entryIds[(current + delta + entryIds.length) % entryIds.length] ?? null;
}

/** Generic tab shell occupying only the published details column. */
export function SideViewPanel({ sessionId, renderSlot, sideView }: SideViewPanelProps) {
  const tabRefs = useRef(new Map<string, HTMLButtonElement>());
  const subscribe = useCallback(
    (listener: () => void) => sideView.subscribe(sessionId, listener),
    [sessionId, sideView],
  );
  const snapshot = useCallback(() => sideView.getSnapshot(sessionId), [sessionId, sideView]);
  const state = useSyncExternalStore(subscribe, snapshot, snapshot);
  const active = state.entries.find((entry) => entry.id === state.activeId);
  const moveTab = (event: KeyboardEvent<HTMLButtonElement>, entryId: string) => {
    const target = sideViewTabTarget(
      state.entries.map((entry) => entry.id),
      entryId,
      event.key,
    );
    if (target === null) return;
    event.preventDefault();
    sideView.activate(sessionId, target);
    tabRefs.current.get(target)?.focus();
  };

  return (
    <section className={css.root} data-side-view data-mode={active?.mode ?? "inspect"}>
      <header className={css.header}>
        <div className={css.tabs} role="tablist" aria-label="Side View tabs">
          {state.entries.map((entry) => (
            <div className={css.tabItem} key={entry.id} data-active={entry.id === state.activeId}>
              <button
                type="button"
                className={css.tab}
                role="tab"
                aria-selected={entry.id === state.activeId}
                tabIndex={entry.id === state.activeId ? 0 : -1}
                ref={(element) => {
                  if (element === null) tabRefs.current.delete(entry.id);
                  else tabRefs.current.set(entry.id, element);
                }}
                onClick={() => sideView.activate(sessionId, entry.id)}
                onKeyDown={(event) => moveTab(event, entry.id)}
              >
                {entry.title}
              </button>
              <button
                type="button"
                className={css.tabClose}
                aria-label={`Close ${entry.title}`}
                onClick={() => sideView.close(sessionId, entry.id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        <button
          type="button"
          className={css.panelClose}
          aria-label="Close Side View"
          onClick={() => sideView.dismiss(sessionId)}
        >
          ×
        </button>
      </header>
      <div className={css.body} role="tabpanel">
        {active === undefined ? (
          <p className={css.empty}>Open a Tool or artifact to inspect it here.</p>
        ) : (
          renderSlot(
            "side-view.content",
            { entry: active },
            {
              entryKey: active.kind,
              fallback: <p className={css.empty}>No renderer is available for {active.kind}.</p>,
            },
          )
        )}
      </div>
    </section>
  );
}
