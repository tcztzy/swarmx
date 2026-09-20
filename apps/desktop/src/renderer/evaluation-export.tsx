import { strToU8, zipSync } from "fflate";
import { useState } from "react";
import { EvaluationCrateSchema } from "../evaluation-crate.js";
import { download, tool } from "./bridge.js";
import { Button } from "./components/ui/radix/button.js";
import { useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";

export function ExportEvaluationButton({
  request,
  filename,
}: {
  request: { id: string; expectedRevision: string } | { source: string };
  filename: string;
}) {
  const { t } = useTranslation();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  return (
    <div className="space-y-2">
      <Button
        type="button"
        variant="outline"
        size="sm"
        disabled={pending}
        onClick={async () => {
          setPending(true);
          setError("");
          try {
            const crate = await tool(
              "memory",
              { action: "export_evaluation", request },
              EvaluationCrateSchema,
            );
            const files = Object.fromEntries([
              ["ro-crate-metadata.json", strToU8(JSON.stringify(crate.metadata, null, 2))],
              ...crate.files.map(({ path, content }) => [path, strToU8(content)]),
            ]);
            download(filename, zipSync(files, { mtime: new Date(1980, 0, 1) }), "application/zip");
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : String(cause));
          } finally {
            setPending(false);
          }
        }}
      >
        <Icon name="download" />
        {t("导出 RO-Crate 证据包")}
      </Button>
      <p className="text-xs text-neutral-500">{t("包含选定的私有原文，仅下载到本地。")}</p>
      {error && (
        <p role="alert" className="text-sm text-red-700">
          {error}
        </p>
      )}
    </div>
  );
}
