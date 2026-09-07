import { scienceArtifactPreviewSchema } from "@swarmx/science/types";
import { useEffect, useState } from "react";
import type { z } from "zod";
import { api } from "./api.js";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

export function ArtifactPreview({ id, compact = false }: { id: string; compact?: boolean }) {
  useTranslation();
  const [preview, setPreview] = useState<z.infer<typeof scienceArtifactPreviewSchema>>();
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    setPreview(undefined);
    setError("");
    void api(
      `/api/v1/artifact-preview?id=${encodeURIComponent(id)}`,
      scienceArtifactPreviewSchema,
      { signal: controller.signal },
    )
      .then(setPreview)
      .catch((cause: Error) => {
        if (!controller.signal.aborted) setError(cause.message);
      });
    return () => controller.abort();
  }, [id]);
  if (error)
    return (
      <p role="alert" className="p-4 text-xs text-neutral-500">
        {error}
      </p>
    );
  if (!preview)
    return (
      <div className="grid h-32 place-items-center text-xs text-neutral-400" role="status">
        {t("加载预览…")}
      </div>
    );
  if (preview.kind === "image")
    return (
      <div
        className={`overflow-auto ${compact ? "h-40" : "max-h-[55vh] rounded-lg border border-neutral-200 bg-neutral-50"}`}
      >
        <img
          src={preview.dataUrl}
          alt={t("科研图像预览")}
          className={
            zoom && !compact ? "max-w-none" : "mx-auto h-full max-h-[55vh] w-full object-contain"
          }
        />
        {!compact && (
          <button
            className="secondary-button m-2"
            type="button"
            onClick={() => setZoom((value) => !value)}
          >
            {zoom ? t("适应宽度") : t("原始大小")}
          </button>
        )}
      </div>
    );
  if (preview.kind === "text")
    return (
      <pre
        className={`${compact ? "h-40 overflow-hidden text-[9px]" : "max-h-96 overflow-auto text-xs"} bg-neutral-50 p-4 leading-6 whitespace-pre-wrap`}
      >
        {preview.text}
      </pre>
    );
  if (preview.kind === "table")
    return (
      <div
        className={`${compact ? "h-40 overflow-hidden text-[9px]" : "max-h-96 overflow-auto text-xs"}`}
      >
        <table className="w-full text-left">
          <thead className="bg-neutral-100">
            <tr>
              {preview.columns.map((column) => (
                <th className="p-2 font-medium" key={column.id}>
                  {column.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.slice(0, compact ? 5 : 500).map((row, index) => (
              <tr className="border-b border-neutral-100" key={JSON.stringify([index, row])}>
                {row.map((value, column) => (
                  <td className="max-w-40 truncate p-2" key={preview.columns[column]?.id}>
                    {String(value ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {preview.truncated && !compact && (
          <p className="p-2 text-neutral-500">
            {t("显示前 {{shown}} 行，共 {{total}} 行。", {
              shown: preview.rows.length,
              total: preview.rowCount,
            })}
          </p>
        )}
      </div>
    );
  return (
    <div className="grid h-32 place-content-center gap-2 text-center text-xs text-neutral-400">
      <Icon name="book" className="mx-auto size-7" />
      <span>{preview.reason === "too-large" ? t("文件较大，请下载查看") : preview.mime}</span>
    </div>
  );
}
