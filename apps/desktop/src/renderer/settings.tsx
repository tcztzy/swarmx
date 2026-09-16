import { lazy, Suspense, useEffect, useState } from "react";
import type { z } from "zod";
import { EnvironmentResponseSchema, SettingsResponseSchema } from "../bridge-contract.js";
import { ToolGrantSchema } from "../permissions.js";
import { LanguageSchema } from "../settings.js";
import { bridge, download } from "./bridge.js";
import { Badge } from "./components/ui/radix/badge.js";
import { Button } from "./components/ui/radix/button.js";
import { CodeBlock } from "./components/ui/radix/code-block.js";
import { Input } from "./components/ui/radix/input.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

const MemorySettings = lazy(() =>
  import("./memory.js").then(({ MemorySettings }) => ({ default: MemorySettings })),
);

export function SettingsPage({ sessionId }: { sessionId?: string | undefined }) {
  useTranslation();
  const [settings, setSettings] = useState<z.infer<typeof SettingsResponseSchema>>();
  const [status, setStatus] = useState<z.infer<typeof EnvironmentResponseSchema>>();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [action, setAction] = useState("");
  useEffect(() => {
    let current = true;
    void Promise.all([
      bridge()
        .settings.read()
        .then((value) => SettingsResponseSchema.parse(value)),
      bridge()
        .environment.read()
        .then((value) => EnvironmentResponseSchema.parse(value)),
    ])
      .then(([settings, status]) => {
        if (!current) return;
        setSettings(settings);
        setStatus(status);
      })
      .catch((cause: Error) => {
        if (current) setError(cause.message);
      });
    return () => {
      current = false;
    };
  }, []);
  useEffect(() => {
    if (action !== "setup" && status?.state !== "building") return;
    let current = true;
    const timer = setInterval(() => {
      void bridge()
        .environment.read()
        .then((value) => {
          if (current) setStatus(EnvironmentResponseSchema.parse(value));
        })
        .catch((cause: Error) => {
          if (current) setError(cause.message);
        });
    }, 1000);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [action, status?.state]);

  const perform = async (name: string, task: () => Promise<unknown>) => {
    setError("");
    setNotice("");
    setAction(name);
    try {
      await task();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setAction("");
    }
  };
  const environmentAction = (name: "setup" | "inspect" | "cancel") =>
    perform(name, async () => {
      await bridge().environment.act({ action: name });
      setStatus(EnvironmentResponseSchema.parse(await bridge().environment.read()));
      setNotice(
        name === "inspect"
          ? t("镜像已验证，可以运行。")
          : name === "setup"
            ? t("运行环境已就绪。")
            : t("已取消环境构建。"),
      );
    });

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-8 px-6 py-8 md:px-10">
        <div>
          <h2 className="mt-2 text-2xl font-semibold">{t("设置")}</h2>
          <p className="mt-2 text-neutral-500">{t("管理语言、执行权限、运行环境和记忆。")}</p>
        </div>
        {error && (
          <p role="alert" className="workbench-alert">
            {error}
          </p>
        )}
        {notice && (
          <p role="status" className="rounded-lg bg-neutral-100 p-3">
            {t(notice)}
          </p>
        )}
        <section className="settings-section">
          <div>
            <h3>{t("语言")}</h3>
            <p>{t("语言更改立即生效，不改变对话、代码和科研数据。")}</p>
          </div>
          <label className="field-label">
            {t("界面语言")}
            <NativeSelect
              value={i18n.language}
              disabled={action === "language"}
              onChange={(event) => {
                const language = LanguageSchema.parse(event.target.value);
                void perform("language", async () => {
                  await bridge().language.write({ language });
                  await i18n.changeLanguage(language);
                });
              }}
            >
              <option value="zh">简体中文</option>
              <option value="en">English</option>
            </NativeSelect>
          </label>
        </section>
        {!settings ? (
          <p role="status">{t("正在读取配置…")}</p>
        ) : (
          <>
            <section className="settings-section">
              <div>
                <h3>{t("工作目录")}</h3>
              </div>
              <p className="break-all rounded-lg border border-neutral-200 bg-neutral-50 p-3 font-mono text-xs">
                {settings.cwd}
              </p>
            </section>
            <section className="settings-section">
              <div>
                <h3>{t("执行与权限")}</h3>
                <p>
                  {t(
                    "子任务只能继承或收紧 Swarm 工具授权。命令审批和原生模式由各 Harness 管理，在任务中选择。以下资源限制只作用于科研容器。",
                  )}
                </p>
              </div>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  const data = new FormData(event.currentTarget);
                  void perform("policy", async () => {
                    const next = SettingsResponseSchema.parse(
                      await bridge().settings.update({
                        ...settings.policy,
                        filesystem: data.get("filesystem"),
                        tools: data.getAll("tools"),
                        delegation: data.has("delegation"),
                        cpus: Number(data.get("cpus")),
                        memoryMb: Number(data.get("memoryMb")),
                        timeoutSeconds: Number(data.get("timeoutSeconds")),
                      }),
                    );
                    setSettings(next);
                    setNotice("权限已保存，对下一次执行生效。");
                  });
                }}
              >
                <fieldset
                  disabled={!!action || status?.state === "building"}
                  className="grid grid-cols-2 gap-4 disabled:opacity-50"
                >
                  <div className="col-span-2 space-y-2">
                    <p className="field-label">{t("Swarm 工具授权")}</p>
                    {ToolGrantSchema.options.map((grant) => (
                      <label key={grant} className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          name="tools"
                          value={grant}
                          defaultChecked={settings.policy.tools.includes(grant)}
                        />
                        {
                          {
                            "memory.read": t("读取记忆"),
                            "memory.write": t("修改记忆"),
                            "science.read": t("查询科研数据"),
                            "science.write": t("修改科研数据与执行计算"),
                          }[grant]
                        }
                      </label>
                    ))}
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        name="delegation"
                        defaultChecked={settings.policy.delegation ?? true}
                      />
                      {t("允许通过 Swarm 委派任务")}
                    </label>
                  </div>
                  <label className="field-label">
                    {t("科研容器文件访问")}
                    <NativeSelect name="filesystem" defaultValue={settings.policy.filesystem}>
                      <option value="workspace-write">{t("可写工作目录")}</option>
                      <option value="read-only">{t("只读")}</option>
                    </NativeSelect>
                  </label>
                  <label className="field-label">
                    {t("CPU 核数")}
                    <Input
                      name="cpus"
                      type="number"
                      min={1}
                      max={32}
                      required
                      defaultValue={settings.policy.cpus}
                    />
                  </label>
                  <label className="field-label">
                    {t("内存上限（MiB）")}
                    <Input
                      name="memoryMb"
                      type="number"
                      min={256}
                      max={65536}
                      required
                      defaultValue={settings.policy.memoryMb}
                    />
                  </label>
                  <label className="field-label">
                    {t("运行时限（秒）")}
                    <Input
                      name="timeoutSeconds"
                      type="number"
                      min={5}
                      max={3600}
                      required
                      defaultValue={settings.policy.timeoutSeconds}
                    />
                  </label>
                  <Button
                    type="submit"
                    variant="default"
                    size="default"
                    className="self-end justify-self-end"
                  >
                    {t("保存权限")}
                  </Button>
                </fieldset>
              </form>
            </section>
            <section className="settings-section">
              <div>
                <h3>{t("运行环境")}</h3>
                <p>
                  {t(
                    "Jupyter Data Science 提供 Python、R 和 Julia 依赖；当前科研执行使用 Python。首次安装需 Docker 和网络，执行时断网。",
                  )}
                </p>
              </div>
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <Icon name="code" />
                  <strong>Jupyter Data Science</strong>
                  <Badge variant="secondary">
                    {status?.state === "ready"
                      ? t("已就绪")
                      : status?.state === "building"
                        ? t("构建中")
                        : status?.state === "failed"
                          ? t("构建失败")
                          : t("尚未配置")}
                  </Badge>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    variant="default"
                    size="default"
                    disabled={!!action || status?.state === "building"}
                    onClick={() => void environmentAction("setup")}
                  >
                    {status?.environment ? t("重新构建") : t("设置运行环境")}
                  </Button>
                  {status?.state === "building" && (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => void environmentAction("cancel")}
                    >
                      {t("取消构建")}
                    </Button>
                  )}
                  {status?.environment && (
                    <>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={!!action}
                        onClick={() => void environmentAction("inspect")}
                      >
                        {t("验证镜像")}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          download(
                            "swarmx-environment.json",
                            JSON.stringify(status.environment, null, 2),
                          )
                        }
                      >
                        {t("导出环境清单")}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() =>
                          download(
                            "python-packages.txt",
                            `${status.environment?.packages.join("\n")}\n`,
                            "text/plain",
                          )
                        }
                      >
                        {t("导出 Python 依赖清单")}
                      </Button>
                    </>
                  )}
                </div>
                {status?.environment && (
                  <>
                    <dl className="metadata-list">
                      <dt>{t("镜像")}</dt>
                      <dd className="font-mono text-xs break-all">{status.environment.imageId}</dd>
                      <dt>Python</dt>
                      <dd>{status.environment.pythonVersion}</dd>
                      <dt>{t("架构")}</dt>
                      <dd>{status.environment.platform}</dd>
                      <dt>{t("构建时间")}</dt>
                      <dd>
                        {new Date(status.environment.createdAt).toLocaleString(i18n.language)}
                      </dd>
                    </dl>
                    <details>
                      <summary className="cursor-pointer text-neutral-500">
                        {t("已安装 {{count}} 个 Python 包", {
                          count: status.environment.packages.length,
                        })}
                      </summary>
                      <CodeBlock viewportClassName="max-h-72 overflow-auto" className="mt-2">
                        <pre className="whitespace-pre-wrap break-words">
                          {status.environment.packages.join("\n")}
                        </pre>
                      </CodeBlock>
                    </details>
                  </>
                )}
                {status?.log && (
                  <details open={status.state !== "ready"}>
                    <summary className="cursor-pointer text-neutral-500">{t("构建日志")}</summary>
                    <CodeBlock viewportClassName="max-h-72 overflow-auto" className="mt-2">
                      <pre
                        className="whitespace-pre-wrap break-words"
                        aria-label={t("环境构建日志")}
                      >
                        {status.log}
                      </pre>
                    </CodeBlock>
                  </details>
                )}
              </div>
            </section>
          </>
        )}
        <Suspense fallback={<p role="status">{t("正在读取记忆…")}</p>}>
          <MemorySettings sessionId={sessionId} />
        </Suspense>
      </div>
    </div>
  );
}
