import { useState } from "react";
import type { z } from "zod";
import { ProjectSchema } from "../settings.js";
import { api, jsonRequest } from "./api.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

export function ProjectNav({
  projects,
  current,
  onSettings,
}: {
  projects: z.infer<typeof ProjectSchema>[];
  current: string;
  onSettings(): void;
}) {
  useTranslation();
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function open(id: string) {
    await api(`/api/v1/projects/${encodeURIComponent(id)}/open`, ProjectSchema, jsonRequest({}));
    window.location.assign(`/projects/${encodeURIComponent(id)}/`);
  }
  async function perform(task: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await task();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="mt-5 space-y-1" aria-label={t("项目列表")}>
      <div className="flex items-center justify-between px-2 text-xs text-neutral-500">
        <span>{t("项目")}</span>
        <button
          type="button"
          className="icon-button size-7"
          aria-label={t("添加项目")}
          onClick={() => setAdding(!adding)}
        >
          <Icon name="plus" />
        </button>
      </div>
      {error && (
        <p role="alert" className="workbench-alert break-words text-xs">
          {error}
        </p>
      )}
      {adding && (
        <form
          className="space-y-2 rounded-lg border border-neutral-200 bg-white p-3"
          onSubmit={(event) => {
            event.preventDefault();
            const form = new FormData(event.currentTarget);
            void perform(async () => {
              const project = await api(
                "/api/v1/projects",
                ProjectSchema,
                jsonRequest({ label: form.get("label"), root: form.get("root") }),
              );
              await open(project.id);
            });
          }}
        >
          <label className="field-label">
            {t("项目名称")}
            <input name="label" className="interaction-input" required maxLength={120} />
          </label>
          <label className="field-label">
            {t("项目目录")}
            <input name="root" className="interaction-input" required />
          </label>
          <p className="text-xs text-neutral-500">
            {t("选择已有文件夹。任务、环境和科研记录归属于这个项目。")}
          </p>
          <button type="submit" className="secondary-button" disabled={busy}>
            {t("添加并打开")}
          </button>
        </form>
      )}
      <nav className="max-h-48 overflow-y-auto" aria-label={t("切换项目")}>
        {projects.map((project) => (
          <div
            key={project.id}
            className={`flex items-center rounded-lg ${current === project.id ? "bg-neutral-200/70" : "hover:bg-neutral-200/50"}`}
          >
            <button
              type="button"
              className="flex min-w-0 flex-1 items-center gap-2 px-2.5 py-2 text-left"
              title={project.root}
              aria-current={current === project.id ? "true" : undefined}
              aria-label={t("打开项目 {{name}}", { name: project.label })}
              disabled={busy}
              onClick={() => {
                if (current !== project.id) void perform(() => open(project.id));
              }}
            >
              <Icon name="folder" className="size-4 shrink-0" />
              <span className="truncate">{project.label}</span>
            </button>
            {current === project.id && (
              <button
                type="button"
                className="icon-button size-8 shrink-0"
                aria-label={t("项目设置")}
                onClick={onSettings}
              >
                <Icon name="settings" className="size-3.5" />
              </button>
            )}
          </div>
        ))}
      </nav>
    </section>
  );
}
