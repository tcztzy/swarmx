import type { AssistantRuntime } from "@assistant-ui/react";
import * as Menu from "@radix-ui/react-dropdown-menu";
import * as Popover from "@radix-ui/react-popover";
import * as RadioGroup from "@radix-ui/react-radio-group";
import type { ModelCatalog, RunOptions } from "@swarmx/swarm";
import { Command } from "cmdk";
import { useEffect, useState } from "react";
import { z } from "zod";
import { projectFetch as fetch } from "./api.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

const ModelCatalogSchema = z.strictObject({
  models: z.array(
    z.strictObject({
      id: z.string(),
      name: z.string(),
      description: z.string().optional(),
      efforts: z.array(z.strictObject({ id: z.string(), name: z.string() })),
      defaultEffort: z.string().optional(),
    }),
  ),
  modes: z
    .array(z.strictObject({ id: z.string(), name: z.string(), description: z.string().optional() }))
    .optional(),
  current: z.strictObject({
    model: z.string().optional(),
    effort: z.string().optional(),
    mode: z.string().optional(),
  }),
}) satisfies z.ZodType<ModelCatalog>;

const harnessNames: Record<string, string> = {
  codex: "Codex",
  claude: "Claude",
  hermes: "Hermes",
  openclaw: "OpenClaw",
};

export interface HarnessProps {
  harness: string;
  harnesses: string[];
  onHarnessChange: (id: string) => void;
}

export function HarnessPicker({
  harness,
  harnesses,
  onHarnessChange,
  disabled,
}: HarnessProps & { disabled?: boolean }) {
  useTranslation();
  return (
    <Menu.Root>
      <Menu.Trigger className="composer-control" aria-label={t("选择 Harness")} disabled={disabled}>
        <Icon name="swarm" className="size-3.5 shrink-0" />
        <span className="truncate">{harnessNames[harness] ?? harness}</span>
        <Icon name="chevron" className="size-3 rotate-90 text-neutral-400" />
      </Menu.Trigger>
      <Menu.Portal>
        <Menu.Content
          className="composer-menu"
          side="top"
          align="start"
          sideOffset={8}
          collisionPadding={12}
        >
          <Menu.Label className="menu-label">{t("选择 Harness")}</Menu.Label>
          <Menu.RadioGroup value={harness} onValueChange={onHarnessChange}>
            {harnesses.map((id) => (
              <Menu.RadioItem key={id} value={id} className="composer-menu-item">
                {harnessNames[id] ?? id}
                <Menu.ItemIndicator className="ml-auto">
                  <Icon name="check" className="size-3.5" />
                </Menu.ItemIndicator>
              </Menu.RadioItem>
            ))}
          </Menu.RadioGroup>
          <p className="menu-label border-t border-neutral-100 mt-1 pt-2">
            {t("切换后显示该 Harness 的任务")}
          </p>
        </Menu.Content>
      </Menu.Portal>
    </Menu.Root>
  );
}

export function RunControls({
  agentId,
  threadId,
  disabled,
  runtime,
}: {
  agentId: string;
  threadId: string;
  disabled: boolean;
  runtime: AssistantRuntime;
}) {
  useTranslation();
  const [catalog, setCatalog] = useState<ModelCatalog>();
  const [selection, setSelection] = useState<RunOptions>({});
  const [error, setError] = useState<string>();
  const [request, setRequest] = useState({ agentId, threadId });
  const [open, setOpen] = useState(false);
  const modelId = selection.model ?? catalog?.current.model;
  const model = catalog?.models.find((row) => row.id === modelId);
  const efforts = model?.efforts ?? [];
  const requestedEffort =
    selection.effort ?? (selection.model === undefined ? catalog?.current.effort : undefined);
  const effort =
    efforts.find((row) => row.id === requestedEffort) ??
    efforts.find((row) => row.id === model?.defaultEffort);
  useEffect(
    () =>
      runtime.registerModelContextProvider({
        getModelContext: () => ({
          config: {
            ...(selection.mode === undefined ? {} : { mode: selection.mode }),
            ...(selection.model === undefined ? {} : { modelName: selection.model }),
            ...((selection.model !== undefined || selection.effort !== undefined) && effort
              ? { reasoningEffort: effort.id }
              : {}),
          },
        }),
      }),
    [runtime, selection, effort],
  );
  useEffect(() => {
    const controller = new AbortController();
    setError(undefined);
    const load = async () => {
      const response = await fetch(
        `/api/v1/models?agent=${encodeURIComponent(request.agentId)}&session=${encodeURIComponent(request.threadId)}`,
        { signal: controller.signal },
      );
      if (!response.ok) throw new Error(await response.text());
      const value = ModelCatalogSchema.parse(await response.json());
      if (!controller.signal.aborted) setCatalog(value);
    };
    void load().catch((cause: unknown) => {
      if (!controller.signal.aborted)
        setError(cause instanceof Error ? cause.message : String(cause));
    });
    return () => controller.abort();
  }, [request]);

  return (
    <div className="ml-auto flex min-w-0 items-center gap-1">
      {!!catalog?.modes?.length && (
        <select
          aria-label={t("原生模式")}
          className="composer-control max-w-48"
          value={selection.mode ?? catalog.current.mode ?? ""}
          onChange={(event) => setSelection({ ...selection, mode: event.target.value })}
          disabled={disabled}
        >
          {catalog.modes.map((mode) => (
            <option key={mode.id} value={mode.id} title={mode.description}>
              {mode.name}
            </option>
          ))}
        </select>
      )}
      <Popover.Root open={open} onOpenChange={setOpen}>
        <Popover.Trigger
          className="composer-control max-w-64 gap-2 rounded-md text-neutral-500"
          role="combobox"
          aria-haspopup="listbox"
          aria-label={t("选择模型")}
          disabled={disabled}
          title={model?.name ?? modelId}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              setOpen(true);
            }
          }}
        >
          <span className="truncate">{model?.name ?? modelId ?? t("选择模型")}</span>
          {effort && <span className="shrink-0 capitalize">{effortLabel(effort)}</span>}
          <Icon name="chevron" className="size-3 rotate-90 text-neutral-400" />
        </Popover.Trigger>
        <Popover.Portal>
          <Popover.Content
            className="model-menu"
            side="top"
            align="end"
            sideOffset={6}
            collisionPadding={12}
            aria-label={t("模型与推理强度")}
          >
            {error !== undefined ? (
              <div className="p-2">
                <p
                  role="alert"
                  className="px-2 py-2 text-xs leading-5 break-words text-neutral-600"
                >
                  {error}
                </p>
                <button
                  type="button"
                  className="composer-menu-item w-full hover:bg-neutral-100"
                  onClick={() => setRequest({ agentId, threadId })}
                >
                  {t("重新加载模型")}
                </button>
              </div>
            ) : catalog === undefined ? (
              <p role="status" className="menu-label p-3">
                {t("正在加载模型…")}
              </p>
            ) : catalog.models.length === 0 ? (
              <p className="menu-label p-3">{t("此 Harness 未提供可选模型")}</p>
            ) : (
              <>
                <Command
                  {...(modelId === undefined ? {} : { defaultValue: modelId })}
                  shouldFilter={false}
                  loop
                >
                  <Command.Input aria-label={t("模型导航")} className="sr-only" />
                  <Command.List aria-label={t("模型")} className="model-menu-list">
                    {catalog.models.map((row) => (
                      <Command.Item
                        key={row.id}
                        value={row.id}
                        disabled={disabled}
                        onSelect={() => {
                          setSelection({
                            ...selection,
                            model: row.id,
                            effort: selection.effort ?? effort?.id,
                          });
                          setOpen(false);
                        }}
                        className="flex min-h-9 cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm data-[selected=true]:bg-neutral-100"
                        title={row.description}
                      >
                        <span className="min-w-0 flex-1 truncate font-medium">{row.name}</span>
                        {row.id === modelId && (
                          <Icon name="check" className="size-3.5 text-neutral-500" />
                        )}
                      </Command.Item>
                    ))}
                  </Command.List>
                </Command>
                {efforts.length > 0 && (
                  <div className="flex items-center justify-between gap-3 border-t border-neutral-200 px-3 py-2">
                    <span className="text-xs text-neutral-500">{t("推理强度")}</span>
                    <RadioGroup.Root
                      aria-label={t("推理强度")}
                      orientation="horizontal"
                      value={effort?.id ?? ""}
                      onValueChange={(id) =>
                        setSelection({ ...selection, model: modelId, effort: id })
                      }
                      disabled={disabled}
                      className="flex min-w-0 flex-wrap justify-end gap-0.5"
                    >
                      {efforts.map((row) => (
                        <RadioGroup.Item
                          key={row.id}
                          value={row.id}
                          title={row.name}
                          className="rounded-md px-1.5 py-1 text-xs text-neutral-500 capitalize transition-colors hover:bg-neutral-100 hover:text-neutral-900 data-[state=checked]:bg-neutral-100 data-[state=checked]:font-medium data-[state=checked]:text-neutral-900 disabled:opacity-40"
                        >
                          {effortLabel(row)}
                        </RadioGroup.Item>
                      ))}
                    </RadioGroup.Root>
                  </div>
                )}
              </>
            )}
          </Popover.Content>
        </Popover.Portal>
      </Popover.Root>
    </div>
  );
}

function effortLabel(effort: { id: string; name: string }): string {
  return effort.id === "medium" ? "Med" : effort.id === "xhigh" ? "XHigh" : effort.name;
}
