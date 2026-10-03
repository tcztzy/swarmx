import type { ReactNode } from "react";
import { TooltipIconButton } from "./components/assistant-ui/elements/tooltip-icon-button.js";
import { useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { SourceInspection, type SourceReference } from "./source-inspection.js";

export function ObservePanel({
  trace,
  source,
  onClose,
}: {
  trace?: ReactNode;
  source?: SourceReference | undefined;
  onClose(): void;
}) {
  const { t } = useTranslation();
  if (source) return <SourceInspection key={source.resource} source={source} onClose={onClose} />;
  return (
    <aside className="flex min-h-0 flex-1 flex-col" aria-label={t("观测侧栏")}>
      <header className="flex shrink-0 items-center justify-between border-b border-neutral-200 px-5 py-4">
        <h2 className="font-medium">{t("执行轨迹")}</h2>
        <TooltipIconButton tooltip={t("关闭侧栏")} className="size-8" onClick={onClose}>
          <Icon name="close" />
        </TooltipIconButton>
      </header>
      <div className="min-h-0 flex-1 overflow-auto">{trace}</div>
    </aside>
  );
}
