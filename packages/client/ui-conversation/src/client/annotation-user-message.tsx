import type { ChatNodeViewProps } from "@deepseek-ai/dsh-client-ui-conversation/client";
import {
  IconCheckOutline16,
  IconCopyOutline16,
  IconLinkOutline16,
  JsonBlock,
  MarkdownText,
  Tooltip,
  writeClipboard,
} from "@deepseek-ai/dsh-client-ui-primitives";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import {
  annotationCopyText,
  annotationPresentation,
  projectAnnotatedText,
} from "./annotation-projection.js";
import css from "./annotation-user-message.module.css";

type UserNodeProps = ChatNodeViewProps<"user"> | ChatNodeViewProps<"steering">;

function keyed<T>(
  values: readonly T[],
): ReadonlyArray<{ readonly key: string; readonly value: T }> {
  const counts = new Map<string, number>();
  return values.map((value) => {
    const serialized = JSON.stringify(value);
    const count = counts.get(serialized) ?? 0;
    counts.set(serialized, count + 1);
    return { key: `${serialized}:${count}`, value };
  });
}

function contentParts(content: readonly unknown[]) {
  const texts: string[] = [];
  const images: Array<{ attachment: unknown }> = [];
  const rest: unknown[] = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null || Array.isArray(block)) {
      rest.push(block);
      continue;
    }
    const value = block as Record<string, unknown>;
    if (value.type === "text" && typeof value.text === "string") texts.push(value.text);
    else if (value.type === "image" && value.attachment !== undefined) {
      images.push({ attachment: value.attachment });
    } else rest.push(block);
  }
  return { text: texts.join(""), images, rest };
}

function AnnotationCard({
  annotation,
}: {
  readonly annotation: Parameters<typeof annotationPresentation>[0];
}) {
  const view = annotationPresentation(annotation);
  const title =
    view.href === null ? (
      view.title
    ) : (
      <a href={view.href} target="_blank" rel="noreferrer">
        {view.title}
        <IconLinkOutline16 size={14} />
      </a>
    );
  return (
    <article className={css.annotationCard} data-annotation-card>
      <div className={css.annotationHeading}>
        <strong>{title}</strong>
        {view.meta !== null && <span>{view.meta}</span>}
      </div>
      {view.quote !== null && <blockquote>{view.quote}</blockquote>}
      {view.comment !== null && <p>{view.comment}</p>}
    </article>
  );
}

function UserActions({ text, time }: { readonly text: string; readonly time: number }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  async function copy() {
    if (!(await writeClipboard(text))) return;
    setCopied(true);
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied(false), 1_000);
  }
  const clock = useMemo(
    () => new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(time),
    [time],
  );
  return (
    <div className={css.actions}>
      <span className={css.time}>{clock}</span>
      <Tooltip label={copied ? "Copied" : "Copy"} side="bottom">
        <button type="button" aria-label={copied ? "Copied" : "Copy"} onClick={copy}>
          {copied ? <IconCheckOutline16 /> : <IconCopyOutline16 />}
        </button>
      </Tooltip>
    </div>
  );
}

const AnnotationUserMessage = memo(function AnnotationUserMessage({
  node,
  renderMessageImages,
  t,
}: UserNodeProps) {
  const { text, images, rest } = contentParts(node.data.content);
  const projection = projectAnnotatedText(text);
  const copyText = annotationCopyText(projection);
  const annotationCards = keyed(projection.annotations);
  const fallbackBlocks = keyed(rest);
  const showBubble =
    projection.body !== "" ||
    projection.annotations.length > 0 ||
    projection.invalidCount > 0 ||
    rest.length > 0;
  return (
    <div className={css.userRow} data-time-hover-root>
      <div className={css.userStack}>
        {renderMessageImages({ images: images as never, align: "end" })}
        {showBubble && (
          <div className={css.bubble}>
            {annotationCards.map(({ key, value }) => (
              <AnnotationCard key={key} annotation={value} />
            ))}
            {projection.invalidCount > 0 && (
              <p className={css.invalid} role="status">
                {projection.invalidCount} invalid annotation hidden
              </p>
            )}
            {projection.body !== "" && (
              <div className={css.body}>
                <MarkdownText text={projection.body} />
              </div>
            )}
            {fallbackBlocks.map(({ key, value }) => (
              <JsonBlock
                key={key}
                label={t("message.extraBlock")}
                payload={value}
                truncatedLabel={(total) => t("json.truncated", { total })}
              />
            ))}
          </div>
        )}
        <UserActions text={copyText} time={node.data.time} />
      </div>
    </div>
  );
});

export function AnnotationUserMessageView(props: ChatNodeViewProps<"user">) {
  return <AnnotationUserMessage {...props} />;
}

export function AnnotationSteeringMessageView(props: ChatNodeViewProps<"steering">) {
  return <AnnotationUserMessage {...props} />;
}
