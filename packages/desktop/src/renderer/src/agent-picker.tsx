import type { DesktopComposerSelection } from "@swarmx/core";
import { resolveHarnessModelInventory } from "@swarmx/core/model-capabilities";
import { ChevronRight, CircleCheck, Plus, RefreshCw, Search, Sparkles, Trash2 } from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  ExtensionCapabilityInventory,
  ManualModelInput,
  ModelApiProtocol,
  ModelCatalogSummary,
} from "../../shared/desktop-api.js";
import { HarnessBrandIcon, type HarnessOption } from "./harness-presentation.js";
import {
  compareModelDisplayOrder,
  modelBrandPresentation,
  selectableModelReasoning,
} from "./model-display.js";
import { ProviderBrandIcon } from "./provider-presentation.js";
import { errorMessage } from "./text-utils.js";
import { cx } from "./ui-primitives.js";

type ExtensionProvider = ExtensionCapabilityInventory["providers"][number];

export interface ComposerModelOption {
  id: string;
  label: string;
  modelId: string;
  modelSupplyId?: string;
  runtimeModel: string;
  apiProtocol: string;
  providerId: string;
  providerLabel: string;
  providerGroup?: string;
  provider?: ExtensionProvider;
  manual?: boolean;
  reasoning?: {
    supportedEfforts: string[];
    defaultEffort?: string;
  };
}

export function AgentPicker({
  open,
  section,
  harnesses,
  selectedHarness,
  models,
  selectedModel,
  efforts,
  selectedEffort,
  modelStatusText,
  modelCatalog,
  modelCatalogRefreshing,
  modelCatalogError,
  disabled,
  label,
  onOpenChange,
  onSectionChange,
  onHarnessChange,
  onModelChange,
  onEffortChange,
  onRefreshModels,
  onAddManualModel,
  onRemoveManualModel,
}: {
  open: boolean;
  section: "harness" | "model" | "effort";
  harnesses: HarnessOption[];
  selectedHarness: HarnessOption;
  models: ComposerModelOption[];
  selectedModel: ComposerModelOption | null;
  efforts: string[];
  selectedEffort: string | null;
  modelStatusText: string;
  modelCatalog?: ModelCatalogSummary;
  modelCatalogRefreshing: boolean;
  modelCatalogError: string | null;
  disabled: boolean;
  label: string;
  onOpenChange: (open: boolean) => void;
  onSectionChange: (section: "harness" | "model" | "effort") => void;
  onHarnessChange: (harnessId: string) => void;
  onModelChange: (modelId: string) => void;
  onEffortChange: (effort: string) => void;
  onRefreshModels: () => Promise<void>;
  onAddManualModel: (input: ManualModelInput) => Promise<void>;
  onRemoveManualModel: (modelId: string) => Promise<void>;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [menuGeometry, setMenuGeometry] = useState<{
    inlineOffset: number;
    secondarySide: "left" | "right";
  }>({ inlineOffset: 0, secondarySide: "right" });
  const [modelQuery, setModelQuery] = useState("");
  const [manualModelOpen, setManualModelOpen] = useState(false);
  const [manualModelId, setManualModelId] = useState("");
  const [manualModelLabel, setManualModelLabel] = useState("");
  const [manualRuntimeModel, setManualRuntimeModel] = useState("");
  const [manualApiProtocol, setManualApiProtocol] = useState<ModelApiProtocol>("openai_responses");
  const [manualModelSaving, setManualModelSaving] = useState(false);
  const [manualModelError, setManualModelError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) onOpenChange(false);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [onOpenChange, open]);

  useEffect(() => {
    if (!open) setModelQuery("");
  }, [open]);

  useLayoutEffect(() => {
    if (!open) return;
    const root = rootRef.current;
    const menu = menuRef.current;
    if (!root || !menu) return;

    const updateGeometry = () => {
      const styles = window.getComputedStyle(menu);
      const cssPixels = (property: string, fallback: number) => {
        const value = Number.parseFloat(styles.getPropertyValue(property));
        return Number.isFinite(value) ? value : fallback;
      };
      const primaryWidth = cssPixels("--agent-picker-primary-width", 196);
      const secondaryWidth = cssPixels("--agent-picker-secondary-width", 236);
      const panelGap = cssPixels("--agent-picker-panel-gap", 6);
      const viewportMargin = 12;
      const anchorLeft = root.getBoundingClientRect().left;
      const maximumPrimaryLeft = Math.max(
        viewportMargin,
        window.innerWidth - viewportMargin - primaryWidth,
      );
      const primaryLeft = Math.min(Math.max(anchorLeft, viewportMargin), maximumPrimaryLeft);
      const availableRight =
        window.innerWidth - viewportMargin - (primaryLeft + primaryWidth + panelGap);
      const availableLeft = primaryLeft - viewportMargin - panelGap;
      const secondarySide =
        availableRight >= secondaryWidth || availableRight >= availableLeft ? "right" : "left";
      const inlineOffset = Math.round(primaryLeft - anchorLeft);

      setMenuGeometry((current) =>
        current.inlineOffset === inlineOffset && current.secondarySide === secondarySide
          ? current
          : { inlineOffset, secondarySide },
      );
    };

    updateGeometry();
    window.addEventListener("resize", updateGeometry);
    const resizeObserver =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateGeometry);
    resizeObserver?.observe(root);
    return () => {
      window.removeEventListener("resize", updateGeometry);
      resizeObserver?.disconnect();
    };
  }, [open]);

  const focusFirstPrimaryItem = useCallback(() => {
    window.requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector<HTMLButtonElement>(".agent-picker__primary button:not(:disabled)")
        ?.focus();
    });
  }, []);

  const handleMenuKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement;
      if (event.key === "Escape") {
        event.preventDefault();
        onOpenChange(false);
        triggerRef.current?.focus();
        return;
      }

      if (target instanceof HTMLInputElement) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          const modelItems = Array.from(
            menuRef.current?.querySelectorAll<HTMLButtonElement>(
              ".agent-picker__secondary button:not(:disabled)",
            ) ?? [],
          );
          if (modelItems.length > 0) {
            event.preventDefault();
            modelItems[event.key === "ArrowUp" ? modelItems.length - 1 : 0]?.focus();
          }
        }
        return;
      }

      if (event.key === "ArrowRight" && target.closest(".agent-picker__primary")) {
        const button = target.closest<HTMLButtonElement>("button:not(:disabled)");
        if (!button) return;
        event.preventDefault();
        button.click();
        window.requestAnimationFrame(() => {
          menuRef.current
            ?.querySelector<HTMLButtonElement>(".agent-picker__secondary button:not(:disabled)")
            ?.focus();
        });
        return;
      }

      if (event.key === "ArrowLeft" && target.closest(".agent-picker__secondary")) {
        event.preventDefault();
        menuRef.current
          ?.querySelector<HTMLButtonElement>(".agent-picker__primary .is-active:not(:disabled)")
          ?.focus();
        return;
      }

      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
      const items = Array.from(
        event.currentTarget.querySelectorAll<HTMLButtonElement>("button:not(:disabled)"),
      );
      if (items.length === 0) return;
      event.preventDefault();
      const currentIndex = items.indexOf(target.closest("button") as HTMLButtonElement);
      const nextIndex =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? items.length - 1
            : event.key === "ArrowUp"
              ? currentIndex <= 0
                ? items.length - 1
                : currentIndex - 1
              : currentIndex < 0 || currentIndex === items.length - 1
                ? 0
                : currentIndex + 1;
      items[nextIndex]?.focus();
    },
    [onOpenChange],
  );

  const submitManualModel = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setManualModelSaving(true);
    setManualModelError(null);
    try {
      await onAddManualModel({
        id: manualModelId,
        label: manualModelLabel || undefined,
        runtimeModel: manualRuntimeModel || undefined,
        apiProtocol: manualApiProtocol,
      });
      setManualModelId("");
      setManualModelLabel("");
      setManualRuntimeModel("");
      setManualModelOpen(false);
    } catch (error) {
      setManualModelError(errorMessage(error));
    } finally {
      setManualModelSaving(false);
    }
  };
  const removeManualModel = async (modelId: string) => {
    setManualModelError(null);
    try {
      await onRemoveManualModel(modelId);
    } catch (error) {
      setManualModelError(errorMessage(error));
    }
  };
  const providerErrorCount =
    modelCatalog?.providers.filter((provider) => provider.status === "error").length ?? 0;
  const discoveredModelCount =
    modelCatalog?.providers.reduce((total, provider) => total + provider.modelCount, 0) ?? 0;

  const primaryRows: Array<{
    id: "harness" | "model" | "effort";
    label: string;
    value: string;
    enabled: boolean;
  }> = [
    { id: "harness", label: "Harness", value: selectedHarness.label, enabled: true },
    {
      id: "model",
      label: "Model",
      value: selectedModel?.label ?? modelStatusText,
      enabled: true,
    },
    {
      id: "effort",
      label: "Effort",
      value: selectedEffort ? effortLabel(selectedEffort) : "Default",
      enabled: efforts.length > 0,
    },
  ];
  const modelGroups = groupComposerModels(models, modelQuery);
  const triggerModel =
    !disabled && selectedModel ? modelBrandPresentation(selectedModel) : undefined;

  return (
    <div className="agent-picker [position:relative] [min-width:0]" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="agent-picker__trigger [min-height:32px] [max-width:min(420px,_62vw)] [padding:5px_8px] [display:inline-flex] [align-items:center] [gap:7px] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:8px] [font:inherit] [font-size:12px] [font-weight:600] [cursor:pointer] [&_svg]:[width:15px] [&_svg]:[height:15px] [&_svg]:[flex:0_0_auto] [&_.harness-brand-icon]:[width:15px] [&_.harness-brand-icon]:[height:15px] [&_.harness-brand-icon]:[flex:0_0_auto] [&_.model-brand-icon]:[width:15px] [&_.model-brand-icon]:[height:15px] [&_.model-brand-icon]:[flex:0_0_auto] [&_.model-brand-icon]:[object-fit:contain]"
        aria-label="Choose agent"
        aria-haspopup="menu"
        aria-expanded={open}
        data-harness-id={selectedHarness.id}
        disabled={disabled}
        onClick={() => onOpenChange(!open)}
        onKeyDown={(event) => {
          if (event.key === "Escape" && open) {
            event.preventDefault();
            onOpenChange(false);
            return;
          }
          if (event.key !== "ArrowDown") return;
          event.preventDefault();
          onOpenChange(true);
          focusFirstPrimaryItem();
        }}
      >
        {triggerModel ? (
          <img
            className="model-brand-icon"
            src={triggerModel.iconUrl}
            alt=""
            aria-hidden="true"
            data-model-brand={triggerModel.brand}
          />
        ) : (
          <HarnessBrandIcon harness={selectedHarness} />
        )}
        <span className="agent-picker__trigger-label [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
          {triggerModel?.label ?? label}
        </span>
        {!disabled && selectedModel && selectedEffort && (
          <span className="agent-picker__trigger-effort [flex:0_0_auto] [color:var(--muted-foreground)] [font-weight:500] [opacity:0.72]">
            {effortLabel(selectedEffort)}
          </span>
        )}
        {!disabled && <ChevronRight aria-hidden="true" />}
      </button>
      {open && !disabled && (
        <div
          ref={menuRef}
          className={String.raw`agent-picker__menu [--agent-picker-primary-width:196px] [--agent-picker-secondary-width:270px] [--agent-picker-panel-gap:6px] [position:absolute] [z-index:40] [left:var(--agent-picker-inline-offset,_0px)] [bottom:calc(100%_+_8px)] [width:var(--agent-picker-primary-width)] [overflow:visible] [color:var(--foreground)] [&[data-secondary-side='left']_.agent-picker\_\_secondary]:[right:calc(100%_+_var(--agent-picker-panel-gap))] [&[data-secondary-side='left']_.agent-picker\_\_secondary]:[left:auto] max-520:[--agent-picker-primary-width:164px] max-520:[--agent-picker-secondary-width:214px]`}
          role="menu"
          aria-label="Agent composition"
          data-secondary-side={menuGeometry.secondarySide}
          style={
            {
              "--agent-picker-inline-offset": `${menuGeometry.inlineOffset}px`,
            } as React.CSSProperties
          }
          onKeyDown={handleMenuKeyDown}
        >
          <div
            className="agent-picker__primary [position:relative] [z-index:1] [width:var(--agent-picker-primary-width)] [height:fit-content] [box-sizing:border-box] [min-width:0] [padding:6px] [background:var(--popover,_var(--card-strong))] [border:1px_solid_var(--border)] [border-radius:13px] [box-shadow:0_18px_48px_rgba(0,_0,_0,_0.28),_var(--shadow-inset)]"
            data-testid="agent-picker-primary"
          >
            {primaryRows.map((row) => (
              <button
                key={row.id}
                type="button"
                role="menuitem"
                className={cx(
                  "agent-picker__row [width:100%] [min-height:44px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:inherit] [background:transparent] [border:0] [border-radius:8px] [text-align:left] [cursor:pointer] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[display:flex] [&_>_span]:[flex-direction:column] [&_>_span]:[gap:2px] [&_strong]:[font-size:12px] [&_strong]:[font-weight:610] [&_small]:[overflow:hidden] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[font-weight:500] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap] [&_>_svg]:[width:14px] [&_>_svg]:[height:14px] [&_>_svg]:[flex:0_0_auto] [&_>_.harness-brand-icon]:[width:14px] [&_>_.harness-brand-icon]:[height:14px] [&_>_.harness-brand-icon]:[flex:0_0_auto]",
                  section === row.id && "is-active",
                )}
                disabled={!row.enabled}
                onPointerEnter={() => row.enabled && onSectionChange(row.id)}
                onClick={() => row.enabled && onSectionChange(row.id)}
              >
                {row.id === "harness" && <HarnessBrandIcon harness={selectedHarness} />}
                <span>
                  <strong>{row.label}</strong>
                  <small>{row.value}</small>
                </span>
                <ChevronRight aria-hidden="true" />
              </button>
            ))}
          </div>
          <div
            className="agent-picker__secondary [position:absolute] [left:calc(100%_+_var(--agent-picker-panel-gap))] [bottom:0] [width:var(--agent-picker-secondary-width)] [max-height:min(360px,_56vh)] [overflow-y:auto] [overscroll-behavior:contain] [scrollbar-gutter:stable] [box-sizing:border-box] [min-width:0] [padding:6px] [background:var(--popover,_var(--card-strong))] [border:1px_solid_var(--border)] [border-radius:13px] [box-shadow:0_18px_48px_rgba(0,_0,_0,_0.28),_var(--shadow-inset)]"
            role="menu"
            aria-label={`${section} options`}
          >
            {section === "harness" &&
              harnesses.map((harness) => {
                return (
                  <button
                    key={harness.id}
                    type="button"
                    role="menuitemradio"
                    aria-checked={harness.id === selectedHarness.id}
                    aria-disabled={harness.disabled || undefined}
                    disabled={harness.disabled}
                    title={harness.disabledReason}
                    className={cx(
                      "agent-picker__option [width:100%] [min-height:44px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:inherit] [background:transparent] [border:0] [border-radius:8px] [text-align:left] [cursor:pointer] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[display:flex] [&_>_span]:[flex-direction:column] [&_>_span]:[gap:2px] [&_>_span]:[font-size:12px] [&_>_span]:[font-weight:610] [&_small]:[overflow:hidden] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[font-weight:500] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap] [&_>_svg]:[width:14px] [&_>_svg]:[height:14px] [&_>_svg]:[flex:0_0_auto]",
                      harness.id === selectedHarness.id && "is-selected",
                    )}
                    onClick={() => !harness.disabled && onHarnessChange(harness.id)}
                  >
                    <HarnessBrandIcon harness={harness} />
                    <span>
                      <span>{harness.label}</span>
                      {harness.disabledReason && <small>{harness.disabledReason}</small>}
                    </span>
                    {harness.id === selectedHarness.id && <CircleCheck aria-hidden="true" />}
                  </button>
                );
              })}
            {section === "model" && (
              <div className="agent-picker__model-list [min-width:0] [display:flex] [flex-direction:column] [gap:6px]">
                <div className="agent-picker__model-actions [margin:-2px_-2px_0] [display:grid] [grid-template-columns:repeat(2,_minmax(0,_1fr))] [gap:5px]">
                  <button
                    type="button"
                    className="agent-picker__model-action [min-width:0] [height:32px] [padding:0_8px] [display:flex] [align-items:center] [justify-content:center] [gap:6px] [color:var(--foreground)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [font-size:11px] [font-weight:620] [cursor:pointer] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_.is-spinning]:[animation:spin_0.8s_linear_infinite]"
                    disabled={modelCatalogRefreshing}
                    onClick={() => void onRefreshModels()}
                  >
                    <RefreshCw
                      aria-hidden="true"
                      className={
                        modelCatalogRefreshing
                          ? "is-spinning [animation:spin_0.9s_linear_infinite]"
                          : undefined
                      }
                    />
                    <span>{modelCatalogRefreshing ? "Refreshing" : "Refresh"}</span>
                  </button>
                  <button
                    type="button"
                    className="agent-picker__model-action [min-width:0] [height:32px] [padding:0_8px] [display:flex] [align-items:center] [justify-content:center] [gap:6px] [color:var(--foreground)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [font-size:11px] [font-weight:620] [cursor:pointer] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_.is-spinning]:[animation:spin_0.8s_linear_infinite]"
                    aria-expanded={manualModelOpen}
                    onClick={() => setManualModelOpen((current) => !current)}
                  >
                    <Plus aria-hidden="true" />
                    <span>Add model</span>
                  </button>
                </div>
                <output className="agent-picker__model-status [padding:1px_4px] [color:var(--muted-foreground)] [font-size:10px] [line-height:1.35]">
                  {modelCatalogRefreshing
                    ? "Refreshing Provider APIs…"
                    : providerErrorCount > 0
                      ? `${providerErrorCount} Provider refresh${providerErrorCount === 1 ? "" : "es"} failed; cached Models retained.`
                      : modelCatalog
                        ? `${discoveredModelCount} discovered · ${modelCatalog.manualModelIds.length} manual`
                        : "Provider discovery has not run yet."}
                </output>
                {(modelCatalogError || manualModelError) && (
                  <div
                    className="agent-picker__model-error [color:var(--danger)] [padding:1px_4px] [font-size:10px] [line-height:1.35]"
                    role="alert"
                  >
                    {manualModelError ?? modelCatalogError}
                  </div>
                )}
                {manualModelOpen && (
                  <form
                    className="agent-picker__manual-model [padding:8px] [display:grid] [gap:7px] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:9px] [&_label]:[min-width:0] [&_label]:[display:grid] [&_label]:[gap:3px] [&_label]:[color:var(--muted-foreground)] [&_label]:[font-size:10px] [&_label]:[font-weight:620] [&_input]:[box-sizing:border-box] [&_input]:[min-width:0] [&_input]:[width:100%] [&_input]:[height:30px] [&_input]:[padding:0_7px] [&_input]:[color:var(--foreground)] [&_input]:[background:var(--background)] [&_input]:[border:1px_solid_var(--border)] [&_input]:[border-radius:6px] [&_input]:[outline:0] [&_input]:[font:inherit] [&_input]:[font-size:11px] [&_select]:[box-sizing:border-box] [&_select]:[min-width:0] [&_select]:[width:100%] [&_select]:[height:30px] [&_select]:[padding:0_7px] [&_select]:[color:var(--foreground)] [&_select]:[background:var(--background)] [&_select]:[border:1px_solid_var(--border)] [&_select]:[border-radius:6px] [&_select]:[outline:0] [&_select]:[font:inherit] [&_select]:[font-size:11px]"
                    aria-label="Add manual model"
                    onSubmit={(event) => void submitManualModel(event)}
                    onKeyDown={(event) => event.stopPropagation()}
                  >
                    <label>
                      <span>Model ID</span>
                      <input
                        required
                        value={manualModelId}
                        placeholder="vendor-model-id"
                        onChange={(event) => setManualModelId(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Runtime model</span>
                      <input
                        value={manualRuntimeModel}
                        placeholder="Defaults to Model ID"
                        onChange={(event) => setManualRuntimeModel(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>Display name</span>
                      <input
                        value={manualModelLabel}
                        placeholder="Optional"
                        onChange={(event) => setManualModelLabel(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>API protocol</span>
                      <select
                        value={manualApiProtocol}
                        onChange={(event) =>
                          setManualApiProtocol(event.target.value as ModelApiProtocol)
                        }
                      >
                        <option value="openai_responses">OpenAI Responses</option>
                        <option value="openai_chat">OpenAI Chat</option>
                        <option value="anthropic">Anthropic</option>
                        <option value="ollama">Ollama</option>
                      </select>
                    </label>
                    <div className="agent-picker__manual-model-actions [display:flex] [justify-content:flex-end] [gap:5px] [&_button]:[min-height:28px] [&_button]:[padding:0_8px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:1px_solid_var(--border-subtle)] [&_button]:[border-radius:6px] [&_button]:[font-size:10.5px] [&_button]:[cursor:pointer] [&_button[type='submit']]:[color:var(--primary-foreground)] [&_button[type='submit']]:[background:var(--primary)] [&_button[type='submit']]:[border-color:transparent]">
                      <button type="button" onClick={() => setManualModelOpen(false)}>
                        Cancel
                      </button>
                      <button type="submit" disabled={manualModelSaving || !manualModelId.trim()}>
                        {manualModelSaving ? "Saving…" : "Save model"}
                      </button>
                    </div>
                  </form>
                )}
                {(modelCatalog?.manualModelIds.length ?? 0) > 0 && (
                  <div
                    className="agent-picker__manual-model-list [display:flex] [flex-wrap:wrap] [gap:4px] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_button]:[min-height:28px] [&_button]:[padding:0_8px] [&_button]:[background:transparent] [&_button]:[border:1px_solid_var(--border-subtle)] [&_button]:[border-radius:6px] [&_button]:[font-size:10.5px] [&_button]:[cursor:pointer] [&_button]:[max-width:100%] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:5px] [&_button]:[color:var(--muted-foreground)] [&_button_span]:[overflow:hidden] [&_button_span]:[text-overflow:ellipsis] [&_button_span]:[white-space:nowrap]"
                    aria-label="Manual models"
                  >
                    {modelCatalog?.manualModelIds.map((modelId) => (
                      <button
                        key={modelId}
                        type="button"
                        aria-label={`Remove manual model ${modelId}`}
                        onClick={() => void removeManualModel(modelId)}
                      >
                        <span>{modelId}</span>
                        <Trash2 aria-hidden="true" />
                      </button>
                    ))}
                  </div>
                )}
                {models.length > 0 ? (
                  <>
                    <label
                      className="agent-picker__model-search [min-width:0] [height:40px] [margin:1px_-6px_0] [padding:7px_9px] [display:flex] [align-items:center] [gap:8px] [color:var(--muted-foreground)] [background:var(--popover,_var(--card-strong))] [border-bottom:1px_solid_var(--border-subtle)] [border-radius:13px_13px_0_0] [&_svg]:[width:15px] [&_svg]:[height:15px] [&_svg]:[flex:0_0_auto] [&_input]:[min-width:0] [&_input]:[width:100%] [&_input]:[color:var(--foreground)] [&_input]:[background:transparent] [&_input]:[border:0] [&_input]:[outline:0] [&_input]:[font-size:12px]"
                      onKeyDown={(event) => event.stopPropagation()}
                    >
                      <Search aria-hidden="true" />
                      <input
                        type="search"
                        value={modelQuery}
                        placeholder="Search models"
                        aria-label="Search models"
                        onChange={(event) => setModelQuery(event.target.value)}
                      />
                    </label>
                    {modelGroups.map((group) => (
                      <fieldset
                        key={group.id}
                        className={String.raw`agent-picker__model-group [min-width:0] [margin:0] [padding:0] [display:flex] [flex-direction:column] [gap:1px] [border:0] [&_+_.agent-picker\_\_model-group]:[margin-top:7px] [&_+_.agent-picker\_\_model-group]:[padding-top:7px] [&_+_.agent-picker\_\_model-group]:[border-top:1px_solid_var(--border-subtle)] [&_.agent-picker\_\_option]:[min-height:38px] [&_.agent-picker\_\_option]:[padding-left:9px]`}
                      >
                        <legend
                          id={`model-provider-${domId(group.id)}`}
                          className={String.raw`agent-picker__model-group-label [padding:5px_9px_4px] [display:flex] [align-items:center] [gap:7px] [overflow:hidden] [color:var(--muted-foreground)] [font-size:11px] [font-weight:650] [line-height:1.2] [text-overflow:ellipsis] [white-space:nowrap] [&_.settings-provider-matrix\_\_icon]:[width:20px] [&_.settings-provider-matrix\_\_icon]:[height:20px] [&_.settings-provider-matrix\_\_icon]:[border-radius:6px] [&_.settings-provider-matrix\_\_icon_img]:[width:12px] [&_.settings-provider-matrix\_\_icon_img]:[height:12px] [&_.settings-provider-matrix\_\_icon_svg]:[width:12px] [&_.settings-provider-matrix\_\_icon_svg]:[height:12px]`}
                        >
                          <ProviderBrandIcon
                            label={group.label}
                            sourceId={group.id}
                            provider={group.provider}
                          />
                          <span>{group.label}</span>
                        </legend>
                        {group.subgroups.map((subgroup) => (
                          <div
                            key={subgroup.id}
                            className={String.raw`agent-picker__model-subgroup [min-width:0] [display:flex] [flex-direction:column] [gap:1px] [&_+_.agent-picker\_\_model-subgroup]:[margin-top:4px]`}
                            {...(subgroup.label
                              ? { role: "group", "aria-label": subgroup.label }
                              : {})}
                          >
                            {subgroup.label && (
                              <span className="agent-picker__model-subgroup-label [padding:3px_9px_2px_36px] [overflow:hidden] [color:var(--muted-foreground)] [font-size:9.5px] [font-weight:620] [line-height:1.2] [text-overflow:ellipsis] [white-space:nowrap]">
                                {subgroup.label}
                              </span>
                            )}
                            {subgroup.models.map((model) => (
                              <button
                                key={model.id}
                                type="button"
                                role="menuitemradio"
                                title={model.modelId}
                                aria-checked={model.id === selectedModel?.id}
                                className={cx(
                                  "agent-picker__option [width:100%] [min-height:44px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:inherit] [background:transparent] [border:0] [border-radius:8px] [text-align:left] [cursor:pointer] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[display:flex] [&_>_span]:[flex-direction:column] [&_>_span]:[gap:2px] [&_>_span]:[font-size:12px] [&_>_span]:[font-weight:610] [&_small]:[overflow:hidden] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[font-weight:500] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap] [&_>_svg]:[width:14px] [&_>_svg]:[height:14px] [&_>_svg]:[flex:0_0_auto]",
                                  model.id === selectedModel?.id && "is-selected",
                                )}
                                onClick={() => onModelChange(model.id)}
                              >
                                <span>
                                  <span>{model.label}</span>
                                  {model.manual && <small>Manual</small>}
                                </span>
                                {model.id === selectedModel?.id && (
                                  <CircleCheck aria-hidden="true" />
                                )}
                              </button>
                            ))}
                          </div>
                        ))}
                      </fieldset>
                    ))}
                    {modelGroups.length === 0 && (
                      <div className="agent-picker__empty [padding:22px_12px] [color:var(--muted-foreground)] [font-size:11.5px] [line-height:1.45] [text-align:center]">
                        No models match “{modelQuery}”
                      </div>
                    )}
                  </>
                ) : (
                  <div className="agent-picker__empty [padding:22px_12px] [color:var(--muted-foreground)] [font-size:11.5px] [line-height:1.45] [text-align:center]">
                    No compatible Models. Refresh Provider APIs or add one manually.
                  </div>
                )}
              </div>
            )}
            {section === "effort" &&
              efforts.map((effort) => (
                <button
                  key={effort}
                  type="button"
                  role="menuitemradio"
                  aria-checked={effort === selectedEffort}
                  className={cx(
                    "agent-picker__option [width:100%] [min-height:44px] [padding:7px_9px] [display:flex] [align-items:center] [gap:9px] [color:inherit] [background:transparent] [border:0] [border-radius:8px] [text-align:left] [cursor:pointer] [&_>_span]:[min-width:0] [&_>_span]:[flex:1] [&_>_span]:[display:flex] [&_>_span]:[flex-direction:column] [&_>_span]:[gap:2px] [&_>_span]:[font-size:12px] [&_>_span]:[font-weight:610] [&_small]:[overflow:hidden] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[font-weight:500] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap] [&_>_svg]:[width:14px] [&_>_svg]:[height:14px] [&_>_svg]:[flex:0_0_auto]",
                    effort === selectedEffort && "is-selected",
                  )}
                  onClick={() => onEffortChange(effort)}
                >
                  <Sparkles aria-hidden="true" />
                  <span>{effortLabel(effort)}</span>
                  {effort === selectedEffort && <CircleCheck aria-hidden="true" />}
                </button>
              ))}
            {section === "effort" && efforts.length === 0 && (
              <div className="agent-picker__empty [padding:22px_12px] [color:var(--muted-foreground)] [font-size:11.5px] [line-height:1.45] [text-align:center]">
                This model has no verified effort control
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function effortLabel(effort: string): string {
  if (effort === "xhigh") return "Extra High";
  return effort.charAt(0).toUpperCase() + effort.slice(1);
}

export function resolveComposerModelOptions(
  inventory: ExtensionCapabilityInventory,
  harnessId: string,
): ComposerModelOption[] {
  const providersById = new Map(inventory.providers.map((provider) => [provider.id, provider]));
  return resolveHarnessModelInventory({
    harnessId,
    models: inventory.models,
    supplies: inventory.modelSupplies,
    providers: inventory.providers,
    harnesses: inventory.harnesses,
  }).flatMap((model): ComposerModelOption[] => {
    const manual = inventory.modelCatalog?.manualModelIds.includes(model.modelId) ?? false;
    const supplies = preferredComposerModelSupplies(model.supplies, model.apiProtocol);
    if (supplies.length === 0) {
      return [
        {
          id: model.agentId,
          label: model.modelLabel,
          modelId: model.modelId,
          runtimeModel: model.runtimeModel,
          apiProtocol: model.apiProtocol,
          providerId: manual ? "manual" : "unrouted",
          providerLabel: manual ? "Manual" : "Other",
          manual,
          reasoning: composerReasoning(model.reasoning),
        },
      ];
    }
    return supplies.map((supply) => {
      const provider = providersById.get(supply.providerProfileId);
      return {
        id: `${model.agentId}@${supply.id}`,
        label: model.modelLabel,
        modelId: model.modelId,
        modelSupplyId: supply.id,
        runtimeModel: supply.runtimeModel,
        apiProtocol: supply.apiProtocol,
        providerId: supply.providerProfileId,
        providerLabel: supply.providerLabel ?? provider?.label ?? supply.providerProfileId,
        providerGroup: supply.providerGroup,
        provider,
        manual,
        reasoning: composerReasoning(supply.reasoning ?? model.reasoning),
      };
    });
  });
}

export function composerModelOptionId(
  harnessId: string,
  modelId: string,
  modelSupplyId?: string,
): string {
  const agentId = `${harnessId}:${modelId}`;
  return modelSupplyId ? `${agentId}@${modelSupplyId}` : agentId;
}

interface ResolvedComposerSupply {
  id: string;
  providerProfileId: string;
  providerLabel?: string;
  providerKind?: string;
  providerGroup?: string;
  runtimeModel: string;
  apiProtocol: string;
  reasoning?: {
    supportedEfforts: string[];
    defaultEffort?: string;
  };
}

function preferredComposerModelSupplies(
  supplies: ResolvedComposerSupply[],
  preferredApi: string,
): ResolvedComposerSupply[] {
  const selected = new Map<string, ResolvedComposerSupply>();
  for (const supply of supplies) {
    const key = `${supply.providerProfileId}\u0000${supply.providerGroup ?? ""}`;
    const current = selected.get(key);
    if (
      !current ||
      composerSupplyRank(supply, preferredApi) < composerSupplyRank(current, preferredApi)
    ) {
      selected.set(key, supply);
    }
  }
  return [...selected.values()];
}

function composerSupplyRank(supply: ResolvedComposerSupply, preferredApi: string): number {
  if (supply.providerKind && supply.apiProtocol === supply.providerKind) return 0;
  if (supply.apiProtocol === preferredApi) return 1;
  return 2;
}

function composerReasoning(
  reasoning: { supportedEfforts: string[]; defaultEffort?: string } | null | undefined,
): ComposerModelOption["reasoning"] {
  return selectableModelReasoning(reasoning);
}

interface ComposerModelSubgroup {
  id: string;
  label?: string;
  models: ComposerModelOption[];
}

interface ComposerModelGroup {
  id: string;
  label: string;
  provider?: ExtensionProvider;
  subgroups: ComposerModelSubgroup[];
}

type MutableComposerModelGroup = ComposerModelGroup & {
  subgroupMap: Map<string, ComposerModelSubgroup>;
};

export function canonicalDefaultComposerModel(
  models: ComposerModelOption[],
): ComposerModelOption | undefined {
  return groupComposerModels(models, "")[0]?.subgroups[0]?.models[0];
}

export function preferredComposerModel(
  models: ComposerModelOption[],
  selection: DesktopComposerSelection | undefined,
): ComposerModelOption | undefined {
  if (!selection) return undefined;
  const matchingModels = models.filter((model) => model.modelId === selection.modelId);
  if (selection.modelSupplyId) {
    const exactRoute = matchingModels.find(
      (model) => model.modelSupplyId === selection.modelSupplyId,
    );
    if (exactRoute) return exactRoute;
  }
  return canonicalDefaultComposerModel(matchingModels);
}

export function groupComposerModels(
  models: ComposerModelOption[],
  query: string,
): ComposerModelGroup[] {
  const normalizedQuery = query.trim().toLowerCase();
  const groups = new Map<string, MutableComposerModelGroup>();

  for (const model of models) {
    if (
      normalizedQuery &&
      !`${model.providerLabel} ${model.providerGroup ?? ""} ${model.apiProtocol} ${model.label} ${model.modelId} ${model.runtimeModel}`
        .toLowerCase()
        .includes(normalizedQuery)
    ) {
      continue;
    }
    const group: MutableComposerModelGroup = groups.get(model.providerId) ?? {
      id: model.providerId,
      label: model.providerLabel,
      provider: model.provider,
      subgroups: [],
      subgroupMap: new Map<string, ComposerModelSubgroup>(),
    };
    const subgroupId = model.providerGroup ?? "default";
    const subgroup: ComposerModelSubgroup = group.subgroupMap.get(subgroupId) ?? {
      id: subgroupId,
      label: model.providerGroup,
      models: [],
    };
    subgroup.models.push(model);
    group.subgroupMap.set(subgroupId, subgroup);
    if (!group.subgroups.includes(subgroup)) group.subgroups.push(subgroup);
    groups.set(model.providerId, group);
  }

  return [...groups.values()].map(({ subgroupMap: _subgroupMap, ...group }) => ({
    ...group,
    subgroups: group.subgroups.map((subgroup) => ({
      ...subgroup,
      models: [...subgroup.models].sort(compareModelDisplayOrder),
    })),
  }));
}

function domId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "-");
}
