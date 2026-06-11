import type React from "react";
import { Fragment, createElement, useEffect, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components, UrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

export type MessageContentKind = "message" | "thinking" | "tool_call" | "tool_result";

interface MessageContentProps {
  kind: MessageContentKind;
  content: string;
}

const MARKDOWN_KINDS = new Set<MessageContentKind>(["message", "thinking"]);
const imageSourceCache = new Map<string, Promise<string | null> | string | null>();
const markdownComponents = {
  img: MarkdownImage,
} satisfies Components;
const markdownUrlTransform: UrlTransform = (url, key) => {
  if (key === "src" && isLocalImageSource(url)) {
    return url;
  }

  return defaultUrlTransform(url);
};

export function MessageContent({ kind, content }: MessageContentProps) {
  if (!MARKDOWN_KINDS.has(kind)) {
    return createElement(Fragment, null, content);
  }

  return createElement(
    "div",
    { className: "run-event__markdown" },
    createElement(
      ReactMarkdown,
      {
        components: markdownComponents,
        remarkPlugins: [remarkGfm],
        urlTransform: markdownUrlTransform,
      },
      content,
    ),
  );
}

function MarkdownImage({ alt, src, title }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const source = typeof src === "string" ? src : "";
  const isLocal = isLocalImageSource(source);
  const [resolvedSrc, setResolvedSrc] = useState(isLocal ? "" : source);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);

    if (!source) {
      setResolvedSrc("");
      return;
    }

    if (!isLocalImageSource(source)) {
      setResolvedSrc(source);
      return;
    }

    setResolvedSrc("");
    void loadLocalImageSource(source).then((dataUrl) => {
      if (cancelled) return;
      if (dataUrl) {
        setResolvedSrc(dataUrl);
      } else {
        setFailed(true);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [source]);

  if (!source) return null;

  if (isLocal && !resolvedSrc) {
    return createElement(
      "span",
      {
        className: "run-event__image-placeholder",
        title: source,
      },
      failed ? "Image unavailable" : "Loading image",
    );
  }

  return createElement("img", {
    alt: alt ?? "",
    loading: "lazy",
    src: resolvedSrc,
    title,
  });
}

function isLocalImageSource(source: string): boolean {
  if (source.startsWith("file://")) return true;
  if (/^[A-Za-z]:[\\/]/.test(source)) return true;
  return source.startsWith("/") && !source.startsWith("//");
}

async function loadLocalImageSource(source: string): Promise<string | null> {
  const cached = imageSourceCache.get(source);
  if (typeof cached === "string" || cached === null) return cached;
  if (cached) return cached;

  const api = typeof window === "undefined" ? undefined : window.swarmxAPI;
  if (!api?.loadImageDataUrl) return null;

  const pending = api
    .loadImageDataUrl(source)
    .then((dataUrl) => {
      imageSourceCache.set(source, dataUrl);
      return dataUrl;
    })
    .catch(() => {
      imageSourceCache.set(source, null);
      return null;
    });

  imageSourceCache.set(source, pending);
  return pending;
}
