import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import {
  type AnnotationOccurrence,
  AnnotationTray,
  composerAnnotations,
} from "../src/client/annotation-composer.js";
import { en } from "../src/client/annotation-locales.js";
import {
  annotationReferenceInsert,
  messageQuoteAnnotation,
} from "../src/client/annotation-reference.js";

vi.mock("@deepseek-ai/dsh-client-ui-primitives", () => ({
  IconCloseOutline16: () => <span data-icon="close" />,
  IconEditOutline16: () => <span data-icon="edit" />,
  Tooltip: ({ children }: { readonly children: React.ReactNode }) => <>{children}</>,
}));

function occurrence(id: number, text: string): AnnotationOccurrence {
  const insert = annotationReferenceInsert(
    messageQuoteAnnotation({
      id: `quote-${id}`,
      createdAt: id,
      sourceSessionId: "session-1",
      messageSeq: id,
      role: "assistant",
      text,
    }),
  );
  return {
    occurrenceId: id,
    source: insert.source,
    ref: insert.ref,
    placement: "detached",
  };
}

const t = (key: keyof typeof en, values?: Record<string, unknown>) =>
  Object.entries(values ?? {}).reduce(
    (text, [name, value]) => text.replace(`{${name}}`, String(value)),
    en[key],
  );

describe("V128 composer annotation tray", () => {
  it("renders ordered selections and working edit/remove controls", () => {
    const annotations = composerAnnotations([
      occurrence(1, "annotation"),
      occurrence(2, "composer"),
    ]);
    const markup = renderToStaticMarkup(
      <AnnotationTray
        annotations={annotations}
        open
        editingId={null}
        editValue=""
        onToggle={vi.fn()}
        onBeginEdit={vi.fn()}
        onEditValue={vi.fn()}
        onCommitEdit={vi.fn()}
        onCancelEdit={vi.fn()}
        onRemove={vi.fn()}
        t={t as never}
      />,
    );

    expect(markup).toContain("2 annotations");
    expect(markup.indexOf("annotation")).toBeLessThan(markup.indexOf("composer"));
    expect(markup).toContain('aria-label="Edit annotation 1"');
    expect(markup).toContain('aria-label="Remove annotation 2"');
  });

  it("reveals row actions on hover and keyboard focus", () => {
    const stylesheet = readFileSync(
      new URL("../src/client/annotation-composer.module.css", import.meta.url),
      "utf8",
    ).replace(/\s+/g, " ");
    expect(stylesheet).toMatch(/\.itemActions\s*\{[^}]*opacity:\s*0/);
    expect(stylesheet).toMatch(/\.item:hover\s+\.itemActions[^}]*opacity:\s*1/);
    expect(stylesheet).toMatch(/\.item:focus-within\s+\.itemActions[^}]*opacity:\s*1/);
  });
});
