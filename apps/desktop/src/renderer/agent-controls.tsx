import type { AssistantRuntime, LanguageModelConfig } from "@assistant-ui/react";
import type { ModelCatalog, RunOptions } from "@swarmx/swarm";
import { DropdownMenu as Menu } from "radix-ui";
import { useEffect, useState } from "react";
import { z } from "zod";
import { bridge } from "./bridge.js";
import {
  ModelSelectorContent,
  ModelSelectorEffort,
  ModelSelectorList,
  ModelSelectorModelContext,
  ModelSelectorRoot,
  ModelSelectorTrigger,
  ModelSelectorValue,
} from "./components/assistant-ui/elements/model-selector.aui.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
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
  pi: "Pi",
  codex: "Codex",
  claude: "Claude",
  hermes: "Hermes",
  openclaw: "OpenClaw",
  dsh: "DSH",
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
          } as LanguageModelConfig & Pick<RunOptions, "mode">,
        }),
      }),
    [runtime, selection.mode],
  );
  useEffect(() => {
    let current = true;
    setError(undefined);
    const load = async () => {
      const value = ModelCatalogSchema.parse(
        await bridge().models.read({ agent: request.agentId, session: request.threadId }),
      );
      if (current) setCatalog(value);
    };
    void load().catch((cause: unknown) => {
      if (current) setError(cause instanceof Error ? cause.message : String(cause));
    });
    return () => {
      current = false;
    };
  }, [request]);

  return (
    <div className="ml-auto flex min-w-0 items-center gap-1">
      {!!catalog?.modes?.length && (
        <NativeSelect
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
        </NativeSelect>
      )}
      <ModelSelectorRoot
        models={(catalog?.models ?? []).map((row) => ({
          id: row.id,
          name: row.name,
          ...(row.description === undefined ? {} : { description: row.description }),
          efforts: row.efforts.map((level) => ({ ...level, name: effortLabel(level) })),
        }))}
        {...(modelId === undefined ? {} : { value: modelId })}
        {...(effort === undefined ? {} : { effort: effort.id })}
        onValueChange={(id) =>
          setSelection({ ...selection, model: id, effort: selection.effort ?? effort?.id })
        }
        onEffortChange={(id) => setSelection({ ...selection, model: modelId, effort: id })}
      >
        {(selection.model !== undefined || selection.effort !== undefined) && (
          <ModelSelectorModelContext />
        )}
        <ModelSelectorTrigger
          disabled={disabled}
          variant="ghost"
          size="sm"
          aria-label={t("选择模型")}
          title={model?.name ?? modelId}
        >
          <ModelSelectorValue placeholder={modelId ?? t("选择模型")} />
        </ModelSelectorTrigger>
        <ModelSelectorContent
          searchable={false}
          side="top"
          align="end"
          aria-label={t("模型与推理强度")}
        >
          {error !== undefined ? (
            <div className="p-2">
              <p role="alert" className="px-2 py-2 text-xs leading-5 break-words text-neutral-600">
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
              <ModelSelectorList />
              <ModelSelectorEffort
                label={t("推理强度")}
                className="flex-wrap [&>[role=radiogroup]]:flex-wrap"
              />
            </>
          )}
        </ModelSelectorContent>
      </ModelSelectorRoot>
    </div>
  );
}

function effortLabel(effort: { id: string; name: string }): string {
  return effort.id === "medium" ? "Med" : effort.id === "xhigh" ? "XHigh" : effort.name;
}
