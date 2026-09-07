import { lazy, Suspense, useEffect, useState } from "react";
import { z } from "zod";
import { ToolGrantSchema } from "../permissions.js";
import {
  EnvironmentStatusSchema,
  LanguageSchema,
  ProjectSchema,
  WorkspaceSettingsSchema,
} from "../settings.js";
import { api, download, jsonRequest } from "./api.js";
import { i18n, t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

const MemorySettings = lazy(() =>
  import("./memory.js").then(({ MemorySettings }) => ({ default: MemorySettings })),
);

const Settings = WorkspaceSettingsSchema.extend({
  workspace: ProjectSchema,
});

export function SettingsPage({
  sessionId,
  project,
}: {
  sessionId?: string | undefined;
  project?: z.infer<typeof ProjectSchema> | undefined;
}) {
  useTranslation();
  const [settings, setSettings] = useState<z.infer<typeof Settings>>();
  const [status, setStatus] = useState<z.infer<typeof EnvironmentStatusSchema>>();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [action, setAction] = useState("");
  useEffect(() => {
    if (!project) return;
    const abort = new AbortController();
    void Promise.all([
      api("/api/v1/settings", Settings, { signal: abort.signal }),
      api("/api/v1/environment", EnvironmentStatusSchema, { signal: abort.signal }),
    ])
      .then(([settings, status]) => {
        setSettings(settings);
        setStatus(status);
      })
      .catch((cause: Error) => {
        if (!abort.signal.aborted) setError(cause.message);
      });
    return () => abort.abort();
  }, [project]);
  useEffect(() => {
    if (action !== "setup" && status?.state !== "building") return;
    const abort = new AbortController();
    const timer = setInterval(() => {
      void api("/api/v1/environment", EnvironmentStatusSchema, { signal: abort.signal })
        .then(setStatus)
        .catch((cause: Error) => {
          if (!abort.signal.aborted) setError(cause.message);
        });
    }, 1000);
    return () => {
      clearInterval(timer);
      abort.abort();
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
      await api("/api/v1/environment", z.unknown(), jsonRequest({ action: name }));
      setStatus(await api("/api/v1/environment", EnvironmentStatusSchema));
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
          <h2 className="mt-2 text-2xl font-semibold">
            {project ? t("项目设置 · {{name}}", { name: project.label }) : t("通用设置")}
          </h2>
          <p className="mt-2 text-neutral-500">
            {project
              ? t("此项目独立管理任务、执行权限、运行环境和工作区记忆。")
              : t("语言和用户偏好在所有项目中共享。")}
          </p>
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
        {!project && (
          <section className="settings-section">
            <div>
              <h3>{t("语言")}</h3>
              <p>{t("语言更改立即生效，不改变对话、代码和科研数据。")}</p>
            </div>
            <label className="field-label">
              {t("界面语言")}
              <select
                className="interaction-input"
                value={i18n.language}
                disabled={action === "language"}
                onChange={(event) => {
                  const language = LanguageSchema.parse(event.target.value);
                  void perform("language", async () => {
                    await api(
                      "/api/v1/language",
                      z.strictObject({ language: LanguageSchema }),
                      jsonRequest({ language }, "PUT"),
                    );
                    await i18n.changeLanguage(language);
                  });
                }}
              >
                <option value="zh">简体中文</option>
                <option value="en">English</option>
              </select>
            </label>
          </section>
        )}
        {project &&
          (!settings ? (
            <p role="status">{t("正在读取配置…")}</p>
          ) : (
            <>
              <section className="settings-section">
                <div>
                  <h3>{t("项目目录")}</h3>
                  <p>{t("该目录属于此项目。要使用其他目录，请在侧栏添加或打开另一个项目。")}</p>
                </div>
                <p className="break-all rounded-lg border border-neutral-200 bg-neutral-50 p-3 font-mono text-xs">
                  {settings.workspace.root}
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
                      const next = await api(
                        "/api/v1/settings",
                        WorkspaceSettingsSchema,
                        jsonRequest(
                          {
                            ...settings.policy,
                            filesystem: data.get("filesystem"),
                            tools: data.getAll("tools"),
                            delegation: data.has("delegation"),
                            cpus: Number(data.get("cpus")),
                            memoryMb: Number(data.get("memoryMb")),
                            timeoutSeconds: Number(data.get("timeoutSeconds")),
                          },
                          "PUT",
                        ),
                      );
                      setSettings({ ...next, workspace: settings.workspace });
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
                      <select
                        name="filesystem"
                        className="interaction-input"
                        defaultValue={settings.policy.filesystem}
                      >
                        <option value="workspace-write">{t("可写工作目录")}</option>
                        <option value="read-only">{t("只读")}</option>
                      </select>
                    </label>
                    <label className="field-label">
                      {t("CPU 核数")}
                      <input
                        name="cpus"
                        type="number"
                        min={1}
                        max={32}
                        required
                        defaultValue={settings.policy.cpus}
                        className="interaction-input"
                      />
                    </label>
                    <label className="field-label">
                      {t("内存上限（MiB）")}
                      <input
                        name="memoryMb"
                        type="number"
                        min={256}
                        max={65536}
                        required
                        defaultValue={settings.policy.memoryMb}
                        className="interaction-input"
                      />
                    </label>
                    <label className="field-label">
                      {t("运行时限（秒）")}
                      <input
                        name="timeoutSeconds"
                        type="number"
                        min={5}
                        max={3600}
                        required
                        defaultValue={settings.policy.timeoutSeconds}
                        className="interaction-input"
                      />
                    </label>
                    <button type="submit" className="primary-button self-end justify-self-end">
                      {t("保存权限")}
                    </button>
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
                    <span className="status-pill">
                      {status?.state === "ready"
                        ? t("已就绪")
                        : status?.state === "building"
                          ? t("构建中")
                          : status?.state === "failed"
                            ? t("构建失败")
                            : t("尚未配置")}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      className="primary-button"
                      disabled={!!action || status?.state === "building"}
                      onClick={() => void environmentAction("setup")}
                    >
                      {status?.environment ? t("重新构建") : t("设置运行环境")}
                    </button>
                    {status?.state === "building" && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => void environmentAction("cancel")}
                      >
                        {t("取消构建")}
                      </button>
                    )}
                    {status?.environment && (
                      <>
                        <button
                          type="button"
                          className="secondary-button"
                          disabled={!!action}
                          onClick={() => void environmentAction("inspect")}
                        >
                          {t("验证镜像")}
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() =>
                            download(
                              "swarmx-environment.json",
                              JSON.stringify(status.environment, null, 2),
                            )
                          }
                        >
                          {t("导出环境清单")}
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() =>
                            download(
                              "python-packages.txt",
                              `${status.environment?.packages.join("\n")}\n`,
                              "text/plain",
                            )
                          }
                        >
                          {t("导出 Python 依赖清单")}
                        </button>
                      </>
                    )}
                  </div>
                  {status?.environment && (
                    <>
                      <dl className="metadata-list">
                        <dt>{t("镜像")}</dt>
                        <dd className="font-mono text-xs break-all">
                          {status.environment.imageId}
                        </dd>
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
                        <pre className="code-panel mt-2">
                          {status.environment.packages.join("\n")}
                        </pre>
                      </details>
                    </>
                  )}
                  {status?.log && (
                    <details open={status.state !== "ready"}>
                      <summary className="cursor-pointer text-neutral-500">{t("构建日志")}</summary>
                      <pre className="code-panel mt-2" aria-label={t("环境构建日志")}>
                        {status.log}
                      </pre>
                    </details>
                  )}
                </div>
              </section>
            </>
          ))}
        <Suspense fallback={<p role="status">{t("正在读取记忆…")}</p>}>
          <MemorySettings scope={project ? "project" : "user"} sessionId={sessionId} />
        </Suspense>
      </div>
    </div>
  );
}
