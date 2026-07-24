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
    <div className="agent-picker" ref={rootRef}>
      <button
        ref={triggerRef}
        type="button"
        className="agent-picker__trigger"
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
        <span className="agent-picker__trigger-label">{triggerModel?.label ?? label}</span>
        {!disabled && selectedModel && selectedEffort && (
          <span className="agent-picker__trigger-effort">{effortLabel(selectedEffort)}</span>
        )}
        {!disabled && <ChevronRight aria-hidden="true" />}
      </button>
      {open && !disabled && (
        <div
          ref={menuRef}
          className="agent-picker__menu"
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
          <div className="agent-picker__primary" data-testid="agent-picker-primary">
            {primaryRows.map((row) => (
              <button
                key={row.id}
                type="button"
                role="menuitem"
                className={cx("agent-picker__row", section === row.id && "is-active")}
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
          <div className="agent-picker__secondary" role="menu" aria-label={`${section} options`}>
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
                      "agent-picker__option",
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
              <div className="agent-picker__model-list">
                <div className="agent-picker__model-actions">
                  <button
                    type="button"
                    className="agent-picker__model-action"
                    disabled={modelCatalogRefreshing}
                    onClick={() => void onRefreshModels()}
                  >
                    <RefreshCw
                      aria-hidden="true"
                      className={modelCatalogRefreshing ? "is-spinning" : undefined}
                    />
                    <span>{modelCatalogRefreshing ? "Refreshing" : "Refresh"}</span>
                  </button>
                  <button
                    type="button"
                    className="agent-picker__model-action"
                    aria-expanded={manualModelOpen}
                    onClick={() => setManualModelOpen((current) => !current)}
                  >
                    <Plus aria-hidden="true" />
                    <span>Add model</span>
                  </button>
                </div>
                <output className="agent-picker__model-status">
                  {modelCatalogRefreshing
                    ? "Refreshing Provider APIs…"
                    : providerErrorCount > 0
                      ? `${providerErrorCount} Provider refresh${providerErrorCount === 1 ? "" : "es"} failed; cached Models retained.`
                      : modelCatalog
                        ? `${discoveredModelCount} discovered · ${modelCatalog.manualModelIds.length} manual`
                        : "Provider discovery has not run yet."}
                </output>
                {(modelCatalogError || manualModelError) && (
                  <div className="agent-picker__model-error" role="alert">
                    {manualModelError ?? modelCatalogError}
                  </div>
                )}
                {manualModelOpen && (
                  <form
                    className="agent-picker__manual-model"
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
                    <div className="agent-picker__manual-model-actions">
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
                  <div className="agent-picker__manual-model-list" aria-label="Manual models">
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
                      className="agent-picker__model-search"
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
                      <fieldset key={group.id} className="agent-picker__model-group">
                        <legend
                          id={`model-provider-${domId(group.id)}`}
                          className="agent-picker__model-group-label"
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
                            className="agent-picker__model-subgroup"
                            {...(subgroup.label
                              ? { role: "group", "aria-label": subgroup.label }
                              : {})}
                          >
                            {subgroup.label && (
                              <span className="agent-picker__model-subgroup-label">
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
                                  "agent-picker__option",
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
                      <div className="agent-picker__empty">No models match “{modelQuery}”</div>
                    )}
                  </>
                ) : (
                  <div className="agent-picker__empty">
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
                  className={cx("agent-picker__option", effort === selectedEffort && "is-selected")}
                  onClick={() => onEffortChange(effort)}
                >
                  <Sparkles aria-hidden="true" />
                  <span>{effortLabel(effort)}</span>
                  {effort === selectedEffort && <CircleCheck aria-hidden="true" />}
                </button>
              ))}
            {section === "effort" && efforts.length === 0 && (
              <div className="agent-picker__empty">This model has no verified effort control</div>
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
