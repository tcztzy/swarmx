import { useState } from "react";
import { z } from "zod";
import { t, useTranslation } from "./i18n.js";
import { Icon } from "./icon.js";
import { CopyButton } from "./source-inspection.js";

const Concept = z.object({
  id: z.string(),
  revision: z.string(),
  metadata: z.object({
    title: z.string(),
    status: z.enum(["draft", "stable", "deprecated"]),
    sources: z.array(z.object({ resource: z.string(), title: z.string().optional() })),
  }),
});
const ReadConcept = z.discriminatedUnion("action", [
  z.object({ action: z.literal("read_memory"), data: Concept }),
  z.object({ action: z.literal("load_memory"), data: z.object({ concepts: z.array(Concept) }) }),
]);
const NativeReadConcept = z.object({
  result: z.object({ structuredContent: ReadConcept, isError: z.literal(false).optional() }),
  error: z.null(),
});

export function readConceptResult(value: unknown) {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return undefined;
    }
  }
  const result = ReadConcept.or(
    NativeReadConcept.transform((value) => value.result.structuredContent),
  ).safeParse(value);
  if (!result.success) return undefined;
  return result.data.action === "read_memory" ? result.data.data : result.data.data.concepts.at(-1);
}

export function SavedConcept({ concept }: { concept: z.infer<typeof Concept> }) {
  useTranslation();
  const [selected, setSelected] = useState("");
  return (
    <section className="saved-concept" aria-label={t("已保存的概念")}>
      <h4>{t("已保存的概念")}</h4>
      <div className="concept-metadata">
        <span className="concept-status">{concept.metadata.status}</span>
        <span className="concept-revision">
          {t("版本")}{" "}
          <code title={concept.revision}>
            {concept.revision.replace(/^sha256:/u, "").slice(0, 12)}…
          </code>
        </span>
        <CopyButton compact value={concept.revision} label={t("复制概念版本")} />
      </div>
      {concept.metadata.sources.map((source) => (
        <div
          className={`concept-source ${selected === source.resource ? "is-selected" : ""}`}
          key={source.resource}
        >
          <button
            className="concept-source-open"
            type="button"
            aria-pressed={selected === source.resource}
            onClick={() => {
              setSelected(source.resource);
              window.dispatchEvent(new CustomEvent("swarmx:open-research", { detail: { source } }));
            }}
          >
            <Icon name="image" className="size-5" />
            <span>
              <strong>
                {source.title ?? t("来源引用")}
                {source.resource.includes("@") && ` · @${source.resource.split("@").at(-1)}`}
              </strong>
              <code>{source.resource}</code>
            </span>
          </button>
          <CopyButton compact value={source.resource} label={t("复制来源引用")} />
        </div>
      ))}
    </section>
  );
}
