import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  type WorkAttempt,
  WorkCommandSchema,
  type WorkCycle,
  type WorkFeedback,
  type WorkItem,
  type WorkRead,
  WorkReadSchema,
  type WorkSnapshot,
} from "../work.js";
import { bridge } from "./bridge.js";
import { TooltipIconButton } from "./components/assistant-ui/elements/tooltip-icon-button.js";
import { Button } from "./components/ui/radix/button.js";
import { Input } from "./components/ui/radix/input.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
import { Textarea } from "./components/ui/radix/textarea.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { NativeInteractionForm } from "./interaction-form.js";

const STATES = {
  queued: "待执行",
  running: "运行中",
  "awaiting-acceptance": "待验收",
  accepted: "已验收",
  blocked: "暂不能执行",
} as const;
const VERDICTS = {
  passed: "通过",
  failed: "不通过",
  partial: "部分完成",
  "insufficient-evidence": "证据不足",
} as const;
const PURPOSES = {
  execution: "目标执行",
  delegation: "委派执行",
  "memory-review": "记忆复盘",
  external: "外部执行",
} as const;
type Command = (payload: unknown) => Promise<boolean>;
const text = (form: FormData, name: string) => String(form.get(name) ?? "").trim();
const configuration = (form: FormData, id: string) => ({
  id,
  harness: text(form, `harness-${id}`),
  model: text(form, `model-${id}`),
  ...(text(form, `effort-${id}`) ? { effort: text(form, `effort-${id}`) } : {}),
  ...(text(form, `profile-${id}`) ? { profile: text(form, `profile-${id}`) } : {}),
});
const money = (value: number | null) =>
  value === null ? t("未知") : `$${value.toLocaleString(undefined, { maximumFractionDigits: 6 })}`;

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    // biome-ignore lint/a11y/noLabelWithoutControl: Every caller supplies one native form control as children.
    <label className="flex min-w-0 flex-col gap-1.5 text-sm">
      <span>{label}</span>
      {children}
    </label>
  );
}

function AgentFields({ id, harnesses }: { id: string; harnesses: string[] }) {
  const [harness, setHarness] = useState(harnesses[0] ?? "");
  return (
    <>
      <Field label={t("Harness")}>
        <NativeSelect
          name={`harness-${id}`}
          value={harness}
          onChange={(event) => setHarness(event.target.value)}
          required
        >
          {harnesses.map((harness) => (
            <option key={harness}>{harness}</option>
          ))}
        </NativeSelect>
      </Field>
      <Field label={t("模型名称")}>
        <Input name={`model-${id}`} required maxLength={256} />
      </Field>
      <Field label={t("推理强度（可选）")}>
        <Input name={`effort-${id}`} maxLength={256} />
      </Field>
      {harness === "dsh" && (
        <Field label={t("工具配置（可选）")}>
          <NativeSelect name={`profile-${id}`} defaultValue="">
            <option value="">{t("默认")}</option>
            <option value="sdk">{t("标准工具")}</option>
            <option value="sdk-minimal">{t("最少工具")}</option>
          </NativeSelect>
        </Field>
      )}
    </>
  );
}

export function WorkPanel({ onClose, harnesses }: { onClose(): void; harnesses: string[] }) {
  useTranslation();
  const [data, setData] = useState<WorkRead>();
  const [cycleId, setCycleId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [stopping, setStopping] = useState<string[]>([]);
  const [starting, setStarting] = useState<string[]>([]);
  const startingRequests = useRef(new Set<string>());
  const pending = useRef(false);
  const sequence = useRef(0);
  // biome-ignore lint/correctness/useExhaustiveDependencies: refresh explicitly reloads authoritative Host state.
  useEffect(() => {
    let current = true;
    const read = async () => {
      if (pending.current) return;
      const request = ++sequence.current;
      try {
        const next = WorkReadSchema.parse(await bridge().work.read(cycleId ? { cycleId } : {}));
        if (!current || request !== sequence.current) return;
        setData(next);
        if (!cycleId && next.snapshot) setCycleId(next.snapshot.cycle.id);
        setStopping((ids) => ids.filter((id) => next.activeWorkIds.includes(id)));
      } catch (cause) {
        if (current && request === sequence.current)
          setError(cause instanceof Error ? cause.message : String(cause));
      }
    };
    void read();
    const timer = setInterval(() => void read(), 2000);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [cycleId, refresh]);
  const command: Command = async (payload) => {
    if (pending.current) return false;
    const parsed = WorkCommandSchema.safeParse(payload);
    if (!parsed.success) {
      setError(parsed.error.message);
      return false;
    }
    const input = parsed.data;
    if (input.action === "start" || input.action === "startNext") {
      const key = input.action === "start" ? input.workId : `cycle:${input.cycleId}`;
      if (startingRequests.current.has(key)) return false;
      startingRequests.current.add(key);
      setStarting([...startingRequests.current]);
      setError("");
      setRefresh((value) => value + 1);
      try {
        WorkReadSchema.parse(await bridge().work.command(input));
        return true;
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
        return false;
      } finally {
        startingRequests.current.delete(key);
        setStarting([...startingRequests.current]);
        setRefresh((value) => value + 1);
      }
    }
    pending.current = true;
    sequence.current++;
    setBusy(true);
    setError("");
    try {
      const next = WorkReadSchema.parse(await bridge().work.command(input));
      setData(next);
      if (next.snapshot) setCycleId(next.snapshot.cycle.id);
      if (input.action === "stop") setStopping((ids) => [...ids, input.workId]);
      return true;
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      return false;
    } finally {
      pending.current = false;
      setBusy(false);
    }
  };
  const snapshot = data?.snapshot;
  return (
    <aside className="flex min-h-0 flex-1 flex-col" aria-label={t("长期工作侧栏")}>
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-neutral-200 px-4">
        <Icon name="book" />
        <h2 className="mr-auto font-medium">{t("长期工作")}</h2>
        <TooltipIconButton
          tooltip={t("刷新长期工作")}
          className="size-8"
          disabled={busy}
          onClick={() => setRefresh((value) => value + 1)}
        >
          <Icon name="refresh" />
        </TooltipIconButton>
        <TooltipIconButton tooltip={t("关闭侧栏")} className="size-8" onClick={onClose}>
          <Icon name="close" />
        </TooltipIconButton>
      </header>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4">
        {error && (
          <p role="alert" className="rounded-md border border-red-200 p-3 text-sm break-words">
            {error}
          </p>
        )}
        {!data && <p role="status">{t("正在加载长期工作…")}</p>}
        {data && (
          <>
            {data.cycles.length > 0 ? (
              <Field label={t("工作周期")}>
                <NativeSelect
                  aria-label={t("工作周期")}
                  disabled={busy}
                  value={cycleId}
                  onChange={(event) => {
                    setCycleId(event.target.value);
                    setData((value) => value && { ...value, snapshot: null });
                  }}
                >
                  {data.cycles.map((cycle) => (
                    <option key={cycle.id} value={cycle.id}>
                      {cycle.project}
                    </option>
                  ))}
                </NativeSelect>
              </Field>
            ) : (
              <p className="text-sm text-neutral-500">
                {t("先创建工作周期，再添加目标。每次执行都需要你明确开始。")}
              </p>
            )}
            <details className="rounded-lg border border-neutral-200 p-3">
              <summary className="cursor-pointer font-medium">{t("新建工作周期")}</summary>
              <CycleForm harnesses={harnesses} busy={busy} command={command} />
            </details>
            {snapshot && (
              <div key={snapshot.cycle.id} className="space-y-5">
                <section
                  aria-label={t("周期预算")}
                  className="space-y-3 rounded-lg bg-neutral-50 p-3"
                >
                  <dl className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <dt className="text-neutral-500">{t("预算")}</dt>
                      <dd>{money(snapshot.balance.budgetUsd)}</dd>
                    </div>
                    <div>
                      <dt className="text-neutral-500">{t("已报告支出")}</dt>
                      <dd>{money(snapshot.balance.spentUsd)}</dd>
                    </div>
                    <div>
                      <dt className="text-neutral-500">{t("仍占用的预留")}</dt>
                      <dd>{money(snapshot.balance.heldUsd)}</dd>
                    </div>
                    <div>
                      <dt className="text-neutral-500">{t("可用余额")}</dt>
                      <dd>{money(snapshot.balance.remainingUsd)}</dd>
                    </div>
                  </dl>
                  {(snapshot.coverageGaps.length > 0 ||
                    snapshot.balance.costCoverage.reported <
                      snapshot.balance.costCoverage.total) && (
                    <p className="text-xs text-neutral-600">
                      {t("存在未报告的费用；未知费用不等于零。")}
                    </p>
                  )}
                  <p className="text-xs text-neutral-500">
                    {t("预留用于执行前检查；实际费用可能超过预留。")}
                  </p>
                  {snapshot.tools.unpricedCalls > 0 && (
                    <p className="text-xs text-neutral-500">
                      {t("{{count}} 次工具调用尚未定价，暂按零计费。", {
                        count: snapshot.tools.unpricedCalls,
                      })}
                    </p>
                  )}
                  <form
                    aria-label={t("调整周期预算")}
                    className="flex flex-wrap items-end gap-2"
                    onSubmit={(event) => {
                      event.preventDefault();
                      const form = new FormData(event.currentTarget);
                      void command({
                        action: "setBudget",
                        request: {
                          cycleId: snapshot.cycle.id,
                          expectedBudgetUsd: snapshot.cycle.budgetUsd,
                          budgetUsd: Number(form.get("budget")),
                        },
                      });
                    }}
                  >
                    <Field label={t("新预算（美元）")}>
                      <Input
                        name="budget"
                        required
                        type="number"
                        min="0"
                        step="any"
                        defaultValue={snapshot.cycle.budgetUsd}
                        disabled={busy}
                      />
                    </Field>
                    <Button type="submit" variant="outline" size="sm" disabled={busy}>
                      {t("调整预算")}
                    </Button>
                  </form>
                  <Button
                    type="button"
                    size="sm"
                    disabled={busy || starting.includes(`cycle:${snapshot.cycle.id}`)}
                    onClick={() =>
                      void command({ action: "startNext", cycleId: snapshot.cycle.id })
                    }
                  >
                    {t("运行下一项")}
                  </Button>
                </section>
                <details
                  className="rounded-lg border border-neutral-200 p-3"
                  open={snapshot.items.length === 0}
                >
                  <summary className="cursor-pointer font-medium">{t("添加工作目标")}</summary>
                  <ItemForm
                    cycle={snapshot.cycle}
                    harnesses={harnesses}
                    command={command}
                    busy={busy}
                  />
                </details>
                <div className="space-y-4">
                  {snapshot.items.map((item) => (
                    <ItemCard
                      key={item.id}
                      item={item}
                      snapshot={snapshot}
                      command={command}
                      busy={busy}
                      active={data.activeWorkIds.includes(item.id) || starting.includes(item.id)}
                      stopping={stopping.includes(item.id)}
                      interactions={data.interactions.filter(
                        (interaction) => interaction.workId === item.id,
                      )}
                    />
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}

function CycleForm({
  harnesses,
  busy,
  command,
}: {
  harnesses: string[];
  busy: boolean;
  command: Command;
}) {
  const [configurations, setConfigurations] = useState(() => [crypto.randomUUID()]);
  return (
    <form
      aria-label={t("新建工作周期")}
      className="mt-4"
      onSubmit={(event) => {
        event.preventDefault();
        const form = event.currentTarget;
        const values = new FormData(form);
        void command({
          action: "createCycle",
          request: {
            id: crypto.randomUUID(),
            project: text(values, "project"),
            budgetUsd: Number(values.get("budget")),
            concurrency: Number(values.get("concurrency")),
            reviewReserveUsd: Number(values.get("review")),
            configurations: configurations.map((id) => configuration(values, id)),
          },
        }).then((saved) => {
          if (saved) {
            form.reset();
            setConfigurations([crypto.randomUUID()]);
          }
        });
      }}
    >
      <fieldset disabled={busy} className="space-y-3">
        <Field label={t("周期名称")}>
          <Input name="project" required maxLength={256} />
        </Field>
        <div className="grid grid-cols-2 gap-3">
          <Field label={t("周期预算（美元）")}>
            <Input name="budget" required type="number" min="0" step="any" />
          </Field>
          <Field label={t("最多同时执行")}>
            <Input name="concurrency" required type="number" min="1" max="64" defaultValue="1" />
          </Field>
        </div>
        <Field label={t("每次复盘预留（美元）")}>
          <Input name="review" required type="number" min="0" step="any" defaultValue="0" />
        </Field>
        <p className="text-xs text-neutral-500">
          {t("自动复盘另受记忆设置控制；预留为零时不启动付费复盘。")}
        </p>
        {configurations.map((id, index) => (
          <fieldset key={id} className="space-y-3 rounded-md border border-neutral-200 p-3">
            <legend className="px-1 text-sm">
              {t("常用 Agent {{number}}", { number: index + 1 })}
            </legend>
            <AgentFields id={id} harnesses={harnesses} />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setConfigurations((ids) => ids.filter((value) => value !== id))}
            >
              {t("移除此配置")}
            </Button>
          </fieldset>
        ))}
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={configurations.length >= 32}
            onClick={() => setConfigurations((ids) => [...ids, crypto.randomUUID()])}
          >
            {t("添加常用 Agent")}
          </Button>
          <Button
            type="submit"
            size="sm"
            disabled={configurations.length > 0 && harnesses.length === 0}
          >
            {t("创建周期")}
          </Button>
        </div>
      </fieldset>
    </form>
  );
}

function ItemForm({
  cycle,
  harnesses,
  command,
  busy,
}: {
  cycle: WorkCycle;
  harnesses: string[];
  command: Command;
  busy: boolean;
}) {
  const [mode, setMode] = useState("manual");
  const [preset, setPreset] = useState(cycle.configurations[0]?.id ?? "");
  return (
    <form
      aria-label={t("添加工作目标")}
      className="mt-4"
      onSubmit={(event) => {
        event.preventDefault();
        const form = event.currentTarget;
        const values = new FormData(form);
        void command({
          action: "createItem",
          request: {
            id: crypto.randomUUID(),
            cycleId: cycle.id,
            goal: text(values, "goal"),
            criteria: text(values, "criteria"),
            criteriaVersion: text(values, "version"),
            taskClass: text(values, "class"),
            priority: Number(values.get("priority")),
            mode,
            [mode === "managed" ? "supervisor" : "configuration"]:
              cycle.configurations.find(({ id }) => id === preset) ??
              configuration(values, "temporary"),
            runtime: {
              ...(text(values, "budget") ? { budgetUsd: Number(values.get("budget")) } : {}),
              ...(text(values, "timeout")
                ? { timeoutMs: Math.round(Number(values.get("timeout")) * 60_000) }
                : {}),
            },
          },
        }).then((saved) => {
          if (saved) form.reset();
        });
      }}
    >
      <fieldset disabled={busy} className="space-y-3">
        <Field label={t("工作目标")}>
          <Textarea name="goal" required maxLength={16000} />
        </Field>
        <Field label={t("验收标准")}>
          <Textarea name="criteria" required maxLength={16000} />
        </Field>
        <Field label={t("标准版本")}>
          <Input name="version" required maxLength={256} defaultValue="1" />
        </Field>
        <Field label={t("任务类型")}>
          <Input name="class" required maxLength={256} />
        </Field>
        <Field label={t("优先级")}>
          <Input name="priority" type="number" step="1" defaultValue="0" />
        </Field>
        <Field label={t("运行模式")}>
          <NativeSelect value={mode} onChange={(event) => setMode(event.target.value)}>
            <option value="manual">{t("手动执行")}</option>
            <option value="managed">{t("全托管")}</option>
          </NativeSelect>
        </Field>
        <Field label={mode === "managed" ? t("监督 Agent") : t("执行 Agent")}>
          <NativeSelect value={preset} onChange={(event) => setPreset(event.target.value)}>
            {cycle.configurations.map((configuration) => (
              <option key={configuration.id} value={configuration.id}>
                {configuration.harness} · {configuration.model}
                {configuration.effort ? ` · ${configuration.effort}` : ""}
              </option>
            ))}
            <option value="">{t("临时选择")}</option>
          </NativeSelect>
        </Field>
        {!preset && <AgentFields id="temporary" harnesses={harnesses} />}
        {mode === "managed" && (
          <p className="text-xs text-neutral-500">
            {t("监督 Agent 根据下级结果继续委派；完成后仍需验收。")}
          </p>
        )}
        <Field label={t("本次预算（美元，可选）")}>
          <Input name="budget" type="number" min="0.000001" step="any" />
        </Field>
        <Field label={t("运行时限（分钟，可选）")}>
          <Input name="timeout" type="number" min="0.001" step="any" />
        </Field>
        <p className="text-xs text-neutral-500">
          {t("预算留空时使用周期可用余额；时限留空时不设超时。")}
        </p>
        <Button type="submit" size="sm">
          {t("添加目标")}
        </Button>
      </fieldset>
    </form>
  );
}

function ItemCard({
  item,
  snapshot,
  command,
  busy,
  active,
  stopping,
  interactions,
}: {
  item: WorkItem;
  snapshot: WorkSnapshot;
  command: Command;
  busy: boolean;
  active: boolean;
  stopping: boolean;
  interactions: WorkRead["interactions"];
}) {
  const attempts = snapshot.reservations.filter((attempt) => attempt.workId === item.id);
  const [attemptId, setAttemptId] = useState("");
  const attempt =
    attempts.find(({ id }) => id === attemptId) ??
    attempts.filter(({ purpose }) => purpose === "execution").at(-1) ??
    attempts.at(-1);
  const feedback = snapshot.feedback
    .filter((feedback) => feedback.attemptId === attempt?.id && feedback.layer === "user")
    .at(-1);
  const sources = snapshot.executions.filter((run) => attempt?.runIds.includes(run.runId));
  const roots = sources.filter(
    (run) => !sources.some((parent) => parent.runId === run.parentRunId),
  );
  const charges = snapshot.reservations.filter((row) => row.runtimeId === attempt?.runtimeId);
  const totalCost =
    attempt?.id === attempt?.runtimeId && charges.length > 0
      ? charges.every((row) => row.costUsd !== null)
        ? charges.reduce((sum, row) => sum + (row.costUsd ?? 0), 0)
        : null
      : roots.length > 0 && roots.every((run) => run.totalCostComplete)
        ? roots.reduce((sum, run) => sum + (run.totalCostUsd ?? 0), 0)
        : null;
  const agent = item.mode === "managed" ? item.supervisor : item.configuration;
  const openSource = (resource: string, title: string) =>
    window.dispatchEvent(
      new CustomEvent("swarmx:open-research", { detail: { source: { resource, title } } }),
    );
  return (
    <article className="space-y-3 rounded-lg border border-neutral-200 p-3" aria-label={item.goal}>
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-medium break-words">{item.goal}</h3>
        <span className="shrink-0 text-xs text-neutral-500">{t(STATES[item.state])}</span>
      </div>
      <p className="whitespace-pre-wrap text-sm text-neutral-600">{item.criteria}</p>
      <p className="text-xs text-neutral-500">
        {t("标准版本")}: {item.criteriaVersion} · {item.taskClass}
      </p>
      <p className="text-xs text-neutral-500">
        {item.mode === "managed" ? t("全托管") : t("手动执行")}
        {agent && ` · ${agent.harness} · ${agent.model}${agent.effort ? ` · ${agent.effort}` : ""}`}
        {item.runtime.timeoutMs &&
          ` · ${t("时限 {{minutes}} 分钟", { minutes: item.runtime.timeoutMs / 60_000 })}`}
      </p>
      {item.blockedReason && (
        <p role="status" className="text-sm break-words">
          {item.blockedReason}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {active ? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={busy || stopping}
            onClick={() => void command({ action: "stop", workId: item.id })}
          >
            {stopping ? t("已请求停止，等待执行结束") : t("请求停止")}
          </Button>
        ) : (
          <Button
            type="button"
            size="sm"
            disabled={
              busy ||
              item.state === "running" ||
              item.state === "accepted" ||
              item.state === "awaiting-acceptance"
            }
            onClick={() => void command({ action: "start", workId: item.id })}
          >
            {t("开始执行")}
          </Button>
        )}
      </div>
      {interactions.map((interaction) => (
        <NativeInteractionForm
          key={interaction.id}
          id={interaction.id}
          title={interaction.title}
          schema={interaction.schema}
          onRespond={async (answer) => {
            const saved = await command({
              action: "respond",
              workId: item.id,
              interactionId: interaction.id,
              ...(answer === undefined ? { cancel: true } : { answer }),
            });
            if (!saved) throw new Error(t("回应未保存，请重试。"));
          }}
        />
      ))}
      {attempt && (
        <section
          className="space-y-3 border-t border-neutral-100 pt-3"
          aria-label={t("执行与验收")}
        >
          <Field label={t("执行尝试")}>
            <NativeSelect value={attempt.id} onChange={(event) => setAttemptId(event.target.value)}>
              {attempts.map((attempt, index) => (
                <option key={attempt.id} value={attempt.id}>
                  {t("第 {{number}} 次执行", { number: index + 1 })} ·{" "}
                  {t(PURPOSES[attempt.purpose])} · {attempt.configuration?.model ?? t("未知")}
                </option>
              ))}
            </NativeSelect>
          </Field>
          <p className="text-xs">
            {t("含下级的总费用（USD）")}: <span>{money(totalCost)}</span> · {t("预留")}:{" "}
            {money(attempt.reservedUsd)}
          </p>
          {totalCost === null && <p className="text-xs text-neutral-500">{t("费用不完整")}</p>}
          {attempt.report && (
            <p className="whitespace-pre-wrap text-sm break-words">{attempt.report}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {sources.flatMap((source) =>
              source.sources.slice(0, 1).map((resource) => (
                <Button
                  key={source.runId}
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => openSource(resource, t("执行记录与输出"))}
                >
                  {t("执行记录与输出")}
                </Button>
              )),
            )}
            {attempt.artifacts.map((artifact) => (
              <Button
                key={`${artifact.id}@${artifact.revision}`}
                type="button"
                variant="outline"
                size="sm"
                className="h-auto max-w-full whitespace-normal break-all"
                onClick={() =>
                  openSource(
                    `sx:a/${encodeURIComponent(artifact.id)}@${artifact.revision}`,
                    artifact.id,
                  )
                }
              >
                {t("成果版本")}: {artifact.id} · {artifact.revision}
              </Button>
            ))}
          </div>
          {attempt.purpose === "execution" && attempt.finishedAt && (
            <FeedbackForm
              key={`${attempt.id}:${feedback?.id ?? "new"}`}
              attempt={attempt}
              feedback={feedback}
              command={command}
              busy={busy}
            />
          )}
          <details>
            <summary className="cursor-pointer text-sm">{t("核对费用与外部执行状态")}</summary>
            <ReconciliationForm attempt={attempt} command={command} busy={busy || active} />
          </details>
        </section>
      )}
      <details>
        <summary className="cursor-pointer text-sm">{t("修改目标与验收标准")}</summary>
        <form
          key={item.criteriaVersion}
          aria-label={t("修改目标与验收标准")}
          className="mt-3"
          onSubmit={(event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            void command({
              action: "revise",
              request: {
                id: item.id,
                expectedCriteriaVersion: item.criteriaVersion,
                criteriaVersion: text(form, "version"),
                goal: text(form, "goal"),
                criteria: text(form, "criteria"),
              },
            });
          }}
        >
          <fieldset disabled={busy || active || item.state === "running"} className="space-y-3">
            <Field label={t("修改后的目标")}>
              <Textarea name="goal" required maxLength={16000} defaultValue={item.goal} />
            </Field>
            <Field label={t("修改后的验收标准")}>
              <Textarea name="criteria" required maxLength={16000} defaultValue={item.criteria} />
            </Field>
            <Field label={t("新标准版本")}>
              <Input name="version" required maxLength={256} />
            </Field>
            <p className="text-xs text-neutral-500">
              {t("保留原目标身份、历史执行与验收；新标准需要重新执行和验收。")}
            </p>
            <Button type="submit" variant="outline" size="sm">
              {t("保存目标修改")}
            </Button>
          </fieldset>
        </form>
      </details>
    </article>
  );
}

function FeedbackForm({
  attempt,
  feedback,
  command,
  busy,
}: {
  attempt: WorkAttempt;
  feedback: WorkFeedback | undefined;
  command: Command;
  busy: boolean;
}) {
  const [verdict, setVerdict] = useState<WorkFeedback["verdict"]>(feedback?.verdict ?? "passed");
  const [fraction, setFraction] = useState(String((feedback?.fraction ?? 1) * 100));
  const [accepted, setAccepted] = useState(feedback?.accepted ?? false);
  return (
    <form
      aria-label={t("记录用户验收")}
      onSubmit={(event) => {
        event.preventDefault();
        const values = new FormData(event.currentTarget);
        void command({
          action: "accept",
          request: {
            id: crypto.randomUUID(),
            attemptId: attempt.id,
            criteriaVersion: attempt.criteriaVersion,
            verdict,
            accepted,
            fraction: Number(fraction) / 100,
            report: text(values, "report"),
            artifacts: attempt.artifacts,
            intervention: text(values, "intervention"),
            ...(feedback ? { supersedes: feedback.id } : {}),
          },
        });
      }}
    >
      <fieldset disabled={busy} className="space-y-3">
        {feedback && (
          <p className="text-xs text-neutral-500">{t("本次提交会更正上一条验收；原记录保留。")}</p>
        )}
        <Field label={t("验收结论")}>
          <NativeSelect
            value={verdict}
            onChange={(event) => {
              const next = event.target.value as WorkFeedback["verdict"];
              setVerdict(next);
              if (next === "failed") {
                setFraction("0");
                setAccepted(false);
              }
            }}
          >
            {Object.entries(VERDICTS).map(([value, label]) => (
              <option key={value} value={value}>
                {t(label)}
              </option>
            ))}
          </NativeSelect>
        </Field>
        <Field label={t("完成比例（%）")}>
          <Input
            type="number"
            required
            min="0"
            max="100"
            step="any"
            value={fraction}
            disabled={verdict === "failed"}
            onChange={(event) => setFraction(event.target.value)}
          />
        </Field>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={accepted}
            disabled={verdict === "failed"}
            onChange={(event) => setAccepted(event.target.checked)}
          />
          {t("接受本次结果")}
        </label>
        <Field label={t("人工介入")}>
          <NativeSelect name="intervention" defaultValue={feedback?.intervention ?? "none"}>
            <option value="none">{t("无")}</option>
            <option value="revision">{t("人工修订")}</option>
            <option value="takeover">{t("人工接管")}</option>
          </NativeSelect>
        </Field>
        <Field label={t("验收报告")}>
          <Textarea
            name="report"
            required
            maxLength={16000}
            defaultValue={feedback?.report ?? ""}
          />
        </Field>
        <Button type="submit" variant="outline" size="sm">
          {feedback ? t("更正验收") : t("保存验收")}
        </Button>
      </fieldset>
    </form>
  );
}

function ReconciliationForm({
  attempt,
  command,
  busy,
}: {
  attempt: WorkAttempt;
  command: Command;
  busy: boolean;
}) {
  return (
    <div className="mt-3 space-y-4">
      <form
        aria-label={t("登记核对后的费用")}
        onSubmit={(event) => {
          event.preventDefault();
          const form = new FormData(event.currentTarget);
          void command({
            action: "reconcileCharge",
            request: {
              id: crypto.randomUUID(),
              reservationId: attempt.id,
              costUsd: Number(form.get("cost")),
              source: "invoice",
              reference: text(form, "reference"),
            },
          });
        }}
      >
        <fieldset
          disabled={busy || attempt.state === "running" || attempt.state === "reserved"}
          className="space-y-3"
        >
          <Field label={t("账单费用（美元）")}>
            <Input name="cost" type="number" required min="0" step="any" />
          </Field>
          <Field label={t("费用凭据")}>
            <Textarea name="reference" required maxLength={16000} />
          </Field>
          <Button type="submit" variant="outline" size="sm">
            {t("登记费用")}
          </Button>
        </fieldset>
      </form>
      {attempt.finishedAt === null &&
        attempt.state !== "reserved" &&
        attempt.state !== "running" && (
          <form
            aria-label={t("确认外部执行状态")}
            onSubmit={(event) => {
              event.preventDefault();
              const form = new FormData(event.currentTarget);
              void command({
                action: "reconcileOutcome",
                request: {
                  reservationId: attempt.id,
                  outcome: text(form, "outcome"),
                  reference: text(form, "reference"),
                },
              });
            }}
          >
            <fieldset disabled={busy} className="space-y-3">
              <p className="text-xs text-neutral-500">
                {t("请先在执行环境确认任务已结束。登记费用不会证明执行完成。")}
              </p>
              <Field label={t("已核实的执行结果")}>
                <NativeSelect name="outcome">
                  <option value="cancelled">{t("已取消")}</option>
                  <option value="failed">{t("失败")}</option>
                  <option value="completed">{t("已完成")}</option>
                </NativeSelect>
              </Field>
              <Field label={t("执行状态依据")}>
                <Textarea name="reference" required maxLength={16000} />
              </Field>
              <Button type="submit" variant="outline" size="sm">
                {t("确认执行已结束")}
              </Button>
            </fieldset>
          </form>
        )}
    </div>
  );
}
