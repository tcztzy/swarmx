import {
  notebookExecutionSchema,
  type ScienceArtifact,
  type ScienceNotebook,
  scienceNotebookSchema,
} from "@swarmx/science/types";
import { useEffect, useRef, useState } from "react";
import { scienceTool } from "./api.js";
import { ArtifactPreview } from "./artifact-preview.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

const EXAMPLE = `# Illustrative data only; replace with your measured data.
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 7)
y = np.array([2.1, 2.8, 3.5, 3.9, 4.8, 5.2])
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.plot(x, y, "o-", color="#303f56", linewidth=2)
ax.set(xlabel="Time (days)", ylabel="Response (a.u.)",
       title="Illustrative response — example data")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig("figure.png", dpi=200)
plt.close(fig)
print("Saved figure.png")`;

export function FigureStudio({
  projectId,
  notebook,
  source,
  inputs = [],
  previewArtifactId,
  outputPath,
  artifacts,
  onClose,
  onResult,
}: {
  projectId: string;
  notebook?: ScienceNotebook | undefined;
  source?: string | undefined;
  inputs?: string[] | undefined;
  previewArtifactId?: string | undefined;
  outputPath?: string | undefined;
  artifacts: ScienceArtifact[];
  onClose(): void;
  onResult(artifactId?: string): void;
}) {
  useTranslation();
  const [code, setCode] = useState(source ?? EXAMPLE);
  const [title, setTitle] = useState(notebook?.title ?? t("科研图像"));
  const [path, setPath] = useState(outputPath ?? (source ? "" : "figure.png"));
  const [selectedInputs, setSelectedInputs] = useState(inputs);
  const [notebookId, setNotebookId] = useState(notebook?.id);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");
  const [previewId, setPreviewId] = useState(previewArtifactId);
  const controller = useRef<AbortController>();
  useEffect(() => () => controller.current?.abort(), []);
  const run = async () => {
    controller.current = new AbortController();
    setRunning(true);
    setError("");
    setOutput("");
    try {
      const id =
        notebookId ??
        (
          await scienceTool(
            "science_notebook",
            "create",
            { requestId: crypto.randomUUID(), projectId, title },
            scienceNotebookSchema,
            controller.current.signal,
          )
        ).id;
      setNotebookId(id);
      const mime = path.endsWith(".svg")
        ? "image/svg+xml"
        : path.endsWith(".pdf")
          ? "application/pdf"
          : "image/png";
      const plotting = /matplotlib|seaborn|plotly/u.test(code);
      const result = await scienceTool(
        "science_notebook",
        "execute",
        {
          requestId: crypto.randomUUID(),
          notebookId: id,
          source: code,
          inputArtifactIds: selectedInputs,
          outputArtifact: {
            relativePath: path,
            kind: "figure",
            title,
            mime,
            license: null,
            ...(!plotting ? { reproducibilityMetadata: false } : {}),
          },
        },
        notebookExecutionSchema,
        controller.current.signal,
      );
      setOutput(
        [
          t("运行{{status}} · {{duration}} ms", {
            status: result.status === "succeeded" ? t("成功") : t("失败"),
            duration: Math.round(result.durationMs),
          }),
          result.stdout.text,
          result.stderr.text,
        ]
          .filter(Boolean)
          .join("\n"),
      );
      if (result.status === "failed")
        setError("代码执行失败。请检查输出；此次运行没有登记图像成果。");
      onResult(result.artifact?.id);
      setPreviewId(result.artifact?.id);
    } catch (cause) {
      setError(
        controller.current.signal.aborted
          ? t("执行已取消。")
          : cause instanceof Error
            ? cause.message
            : String(cause),
      );
    } finally {
      setRunning(false);
    }
  };
  return (
    <section aria-label={t("图像工作台")} className="flex min-h-0 flex-1 flex-col overflow-hidden">
      <header className="flex flex-wrap items-center gap-3 border-b border-neutral-200 px-5 py-3">
        <Icon name="image" />
        <h2 className="font-medium">{t("图像工作台")}</h2>
        <span className="text-xs text-neutral-500">{t("代码 → 隔离运行 → 版本化成果")}</span>
        <button
          type="button"
          className="icon-button ml-auto"
          disabled={running}
          onClick={onClose}
          aria-label={t("关闭图像工作台")}
        >
          <Icon name="close" />
        </button>
      </header>
      <div className="grid min-h-0 flex-1 overflow-y-auto">
        {previewId && (
          <div className="border-b border-neutral-200 p-4">
            <ArtifactPreview id={previewId} />
          </div>
        )}
        <div className="flex min-h-[420px] flex-col border-neutral-200 lg:border-r">
          <label className="sr-only" htmlFor="figure-source">
            {t("Python 图像代码")}
          </label>
          <textarea
            id="figure-source"
            className="min-h-[340px] flex-1 resize-none bg-neutral-50 p-5 font-mono text-[13px] leading-6 outline-none"
            spellCheck={false}
            value={code}
            onChange={(event) => setCode(event.target.value)}
            disabled={running}
          />
          {output && (
            <pre
              aria-label={t("代码执行输出")}
              className="max-h-52 overflow-auto border-t border-neutral-200 bg-white p-4 text-xs leading-6 whitespace-pre-wrap"
            >
              {output}
            </pre>
          )}
        </div>
        <div className="space-y-5 p-5">
          <label className="field-label">
            {t("成果名称")}
            <input
              className="interaction-input"
              value={title}
              maxLength={240}
              disabled={running}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>
          <label className="field-label">
            {t("输出文件")}
            <input
              className="interaction-input font-mono text-xs"
              aria-label={t("输出文件")}
              value={path}
              disabled={running}
              onChange={(event) => setPath(event.target.value)}
            />
            <span className="text-xs font-normal leading-5 text-neutral-500">
              {t("填写代码实际保存的相对路径，支持 PNG、SVG、PDF。")}
            </span>
          </label>
          <fieldset disabled={running}>
            <legend className="mb-2 font-medium">
              {t("输入成果")}
              <span className="text-xs font-normal text-neutral-400">{t("最多 4 个")}</span>
            </legend>
            <div className="max-h-44 space-y-2 overflow-auto">
              {artifacts.length === 0 ? (
                <p className="text-xs text-neutral-500">{t("导入数据或图片后，可在这里选择。")}</p>
              ) : (
                artifacts.map((artifact) => (
                  <label key={artifact.id} className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={selectedInputs.includes(artifact.id)}
                      disabled={!selectedInputs.includes(artifact.id) && selectedInputs.length >= 4}
                      onChange={(event) =>
                        setSelectedInputs((ids) =>
                          event.target.checked
                            ? [...ids, artifact.id]
                            : ids.filter((id) => id !== artifact.id),
                        )
                      }
                    />
                    <span className="truncate">{artifact.title}</span>
                  </label>
                ))
              )}
            </div>
          </fieldset>
          {selectedInputs.length > 0 && (
            <p className="text-xs leading-6 text-neutral-500">
              {t("按选择顺序，通过")}
              <code>os.environ["SWARMX_SCIENCE_INPUT_0"]</code> {t("读取输入文件，编号从 0 开始。")}
            </p>
          )}
          <p className="text-xs leading-6 text-neutral-500">
            {t(
              "每次执行追加代码、环境和输出记录，原有成果保留。首次运行前，请在配置中设置运行环境。",
            )}
          </p>
          {error && (
            <p role="alert" className="workbench-alert">
              {t(error)}
            </p>
          )}
          {running ? (
            <button
              className="secondary-button w-full"
              type="button"
              onClick={() => controller.current?.abort()}
            >
              <Icon name="stop" />
              {t("停止执行")}
            </button>
          ) : (
            <button
              className="primary-button w-full"
              type="button"
              disabled={!code.trim() || !title.trim() || !/\.(png|svg|pdf)$/u.test(path)}
              onClick={() => void run()}
            >
              <Icon name="play" />
              {t("运行并生成图像")}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
