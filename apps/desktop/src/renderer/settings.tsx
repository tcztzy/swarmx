import { lazy, Suspense, useEffect, useState } from "react";
import type { z } from "zod";
import { SettingsResponseSchema } from "../bridge-contract.js";
import { ToolGrantSchema } from "../permissions.js";
import { LanguageSchema } from "../settings.js";
import { bridge } from "./bridge.js";
import { Button } from "./components/ui/radix/button.js";
import { NativeSelect } from "./components/ui/radix/native-select.js";
import { i18n, t, useTranslation } from "./i18n.js";

const MemorySettings = lazy(() =>
  import("./memory.js").then(({ MemorySettings }) => ({ default: MemorySettings })),
);

export function SettingsPage({ sessionId }: { sessionId?: string | undefined }) {
  useTranslation();
  const [settings, setSettings] = useState<z.infer<typeof SettingsResponseSchema>>();
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [action, setAction] = useState("");
  useEffect(() => {
    let current = true;
    void bridge()
      .settings.read()
      .then((value) => {
        if (current) setSettings(SettingsResponseSchema.parse(value));
      })
      .catch((cause: Error) => {
        if (current) setError(cause.message);
      });
    return () => {
      current = false;
    };
  }, []);
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

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl space-y-8 px-6 py-8 md:px-10">
        <div>
          <h2 className="mt-2 text-2xl font-semibold">{t("设置")}</h2>
          <p className="mt-2 text-neutral-500">{t("管理语言、执行权限和记忆。")}</p>
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
            <p>{t("语言更改立即生效，不改变对话、代码和记录。")}</p>
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
                    "子任务只能继承或收紧 Swarm 工具授权。命令审批和原生模式由各 Harness 管理，在任务中选择。",
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
                      }),
                    );
                    setSettings(next);
                    setNotice("权限已保存，对下一次执行生效。");
                  });
                }}
              >
                <fieldset
                  disabled={!!action}
                  className="grid grid-cols-2 gap-4 disabled:opacity-50"
                >
                  <div className="col-span-2 space-y-2">
                    <label className="field-label">
                      {t("项目提示词与技能文件权限")}
                      <NativeSelect name="filesystem" defaultValue={settings.policy.filesystem}>
                        <option value="read-only">{t("只读，不更新项目资源")}</option>
                        <option value="workspace-write">
                          {t("允许经审核更新已注册的项目资源")}
                        </option>
                      </NativeSelect>
                    </label>
                    <p className="text-sm text-neutral-500">
                      {t("此设置控制提示词与技能学习，不改变原生 Agent 的文件权限。")}
                    </p>
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
                            "science.read": t("读取领域引用"),
                            "science.write": t("旧版领域写入授权"),
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
          </>
        )}
        <Suspense fallback={<p role="status">{t("正在读取记忆…")}</p>}>
          <MemorySettings sessionId={sessionId} />
        </Suspense>
      </div>
    </div>
  );
}
