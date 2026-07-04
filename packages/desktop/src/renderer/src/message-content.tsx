import { Check, Copy } from "lucide-react";
import type React from "react";
import { Fragment, createElement, isValidElement, useEffect, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components, UrlTransform } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  type HighlightedCodeBlock,
  type HighlightedCodeToken,
  highlightCodeBlock,
} from "./code-highlighter.js";
import "katex/dist/katex.min.css";

export type MessageContentKind = "message" | "thinking" | "tool_call" | "tool_result";

interface MessageContentProps {
  kind: MessageContentKind;
  content: string;
}

const MARKDOWN_KINDS = new Set<MessageContentKind>(["message", "thinking"]);
const imageSourceCache = new Map<string, Promise<string | null> | string | null>();
type ReactMarkdownProps = React.ComponentProps<typeof ReactMarkdown>;
const mathRemarkPlugins: ReactMarkdownProps["remarkPlugins"] = [remarkGfm, remarkMath];
const mathRehypePlugins: ReactMarkdownProps["rehypePlugins"] = [
  [rehypeKatex, { strict: false, throwOnError: false, trust: false }],
];
const markdownComponents = {
  code: MarkdownCode,
  img: MarkdownImage,
  pre: MarkdownPre,
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
        rehypePlugins: mathRehypePlugins,
        remarkPlugins: mathRemarkPlugins,
        urlTransform: markdownUrlTransform,
      },
      prepareMathMarkdown(content),
    ),
  );
}

function MarkdownImage({ alt, src, title }: React.ImgHTMLAttributes<HTMLImageElement>) {
  const source = typeof src === "string" ? src : "";
  const isLocal = isLocalImageSource(source);
  const isRemote = isRemoteImageSource(source);
  const [resolvedSrc, setResolvedSrc] = useState(isLocal ? "" : source);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);

    if (!source) {
      setResolvedSrc("");
      return;
    }

    if (isRemoteImageSource(source)) {
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

  if (isRemote) {
    return createElement(
      "span",
      {
        className: "run-event__image-placeholder",
        title: source,
      },
      `Remote image blocked${alt ? `: ${alt}` : ""}`,
    );
  }

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

function MarkdownPre({ children }: React.HTMLAttributes<HTMLPreElement>) {
  const language = languageFromCodeChild(children);
  const codeText = textFromReactNode(children);
  const [copied, setCopied] = useState(false);

  async function copyCode() {
    if (!codeText || typeof navigator === "undefined" || !navigator.clipboard?.writeText) {
      return;
    }
    try {
      await navigator.clipboard.writeText(codeText);
    } catch {
      return;
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  return createElement(
    "figure",
    {
      className: "run-event__code-block",
      "data-language": language || undefined,
    },
    createElement(
      "figcaption",
      { className: "run-event__code-header" },
      createElement("span", { className: "run-event__code-language" }, language || "text"),
      createElement(
        "button",
        {
          "aria-label": copied ? "Code copied" : "Copy code",
          className: "run-event__code-copy",
          disabled: !codeText,
          onClick: copyCode,
          title: copied ? "Code copied" : "Copy code",
          type: "button",
        },
        createElement(copied ? Check : Copy, { "aria-hidden": true, size: 14 }),
      ),
    ),
    createElement(CodeBlockBody, { children, codeText, language }),
  );
}

function CodeBlockBody({
  children,
  codeText,
  language,
}: {
  children: React.ReactNode;
  codeText: string;
  language: string;
}) {
  const [highlightedCode, setHighlightedCode] = useState<HighlightedCodeBlock | null>(null);

  useEffect(() => {
    let cancelled = false;
    setHighlightedCode(null);

    if (!language || !codeText.trim()) return;

    void highlightCodeBlock(codeText, language).then((block) => {
      if (!cancelled) setHighlightedCode(block);
    });

    return () => {
      cancelled = true;
    };
  }, [codeText, language]);

  if (highlightedCode) {
    return createElement(
      "pre",
      { className: "run-event__code-highlight shiki github-dark", tabIndex: 0 },
      createElement(
        "code",
        null,
        highlightedCode.lines.map((line, lineIndex) =>
          createElement(
            "span",
            { className: "line", key: `line-${lineIndex}` },
            line.tokens.map((token, tokenIndex) =>
              createElement(
                "span",
                {
                  key: `${lineIndex}-${tokenIndex}`,
                  style: styleFromHighlightedToken(token),
                },
                token.content,
              ),
            ),
            lineIndex < highlightedCode.lines.length - 1 ? "\n" : null,
          ),
        ),
      ),
    );
  }

  return createElement("pre", { className: "run-event__code-fallback" }, children);
}

function styleFromHighlightedToken(token: HighlightedCodeToken): React.CSSProperties {
  const style: React.CSSProperties = {};
  if (token.color) style.color = token.color;
  if (token.fontStyle) {
    if (token.fontStyle & 1) style.fontStyle = "italic";
    if (token.fontStyle & 2) style.fontWeight = 700;
    if (token.fontStyle & 4) style.textDecoration = "underline";
  }
  return style;
}

function MarkdownCode({
  children,
  className,
  node: _node,
  ...props
}: React.HTMLAttributes<HTMLElement> & { children?: React.ReactNode; node?: unknown }) {
  const language = languageFromClassName(className);
  return createElement("code", {
    ...props,
    className,
    "data-language": language || undefined,
    children,
  });
}

function languageFromCodeChild(children: React.ReactNode): string {
  if (!isValidElement(children)) return "";
  const props = children.props as { className?: string };
  return languageFromClassName(props.className);
}

function languageFromClassName(className: string | undefined): string {
  const match = className?.match(/(?:^|\s)language-([^\s]+)/);
  return match?.[1] ?? "";
}

function textFromReactNode(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textFromReactNode).join("");
  if (isValidElement(node)) {
    const props = node.props as { children?: React.ReactNode };
    return textFromReactNode(props.children);
  }
  return "";
}

function isLocalImageSource(source: string): boolean {
  if (source.startsWith("file://")) return true;
  if (/^[A-Za-z]:[\\/]/.test(source)) return true;
  return source.startsWith("/") && !source.startsWith("//");
}

function isRemoteImageSource(source: string): boolean {
  return /^https?:\/\//i.test(source);
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

function prepareMathMarkdown(content: string): string {
  return splitProtectedMarkdown(content)
    .map((segment) => (segment.protected ? segment.text : normalizeMathDelimiters(segment.text)))
    .join("");
}

function splitProtectedMarkdown(content: string): Array<{ text: string; protected: boolean }> {
  const segments: Array<{ text: string; protected: boolean }> = [];
  const protectedPattern = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*(?:`|$))/g;
  let cursor = 0;
  for (const match of content.matchAll(protectedPattern)) {
    const index = match.index ?? 0;
    if (index > cursor) segments.push({ text: content.slice(cursor, index), protected: false });
    segments.push({ text: match[0], protected: true });
    cursor = index + match[0].length;
  }
  if (cursor < content.length) {
    segments.push({ text: content.slice(cursor), protected: false });
  }
  return segments;
}

function normalizeMathDelimiters(segment: string): string {
  const inlineMath: string[] = [];
  const displayMath: string[] = [];
  let normalized = segment
    .replace(/\\\[([\s\S]+?)\\\]/g, (_match, source: string) => {
      const index = displayMath.push(source) - 1;
      return `\uE110${index}\uE111`;
    })
    .replace(/\\\(([\s\S]+?)\\\)/g, (_match, source: string) => {
      const index = inlineMath.push(source) - 1;
      return `\uE100${index}\uE101`;
    });

  normalized = escapeUnsafeDollarMath(normalized);

  return normalized
    .replace(/\uE110(\d+)\uE111/g, (_match, index: string) => {
      return `$$\n${displayMath[Number(index)] ?? ""}\n$$`;
    })
    .replace(/\uE100(\d+)\uE101/g, (_match, index: string) => {
      return `$${inlineMath[Number(index)] ?? ""}$`;
    });
}

function escapeUnsafeDollarMath(segment: string): string {
  const dollarIndexes = singleDollarIndexes(segment);
  if (dollarIndexes.length === 0) return segment;

  const escaped = new Set<number>();
  const paired = new Set<number>();
  let cursor = 0;

  while (cursor < dollarIndexes.length) {
    const opener = dollarIndexes[cursor];
    if (!canOpenDollarMath(segment, opener)) {
      escaped.add(opener);
      cursor += 1;
      continue;
    }

    let closerCursor = cursor + 1;
    let closer: number | undefined;
    while (closerCursor < dollarIndexes.length) {
      const candidate = dollarIndexes[closerCursor];
      if (canCloseDollarMath(segment, opener, candidate)) {
        closer = candidate;
        break;
      }
      escaped.add(candidate);
      closerCursor += 1;
    }

    if (closer === undefined) {
      escaped.add(opener);
      cursor += 1;
      continue;
    }

    paired.add(opener);
    paired.add(closer);
    cursor = closerCursor + 1;
  }

  const output: string[] = [];
  for (let index = 0; index < segment.length; index += 1) {
    if (segment[index] === "$" && escaped.has(index) && !paired.has(index)) {
      output.push("\\$");
    } else {
      output.push(segment[index] ?? "");
    }
  }
  return output.join("");
}

function singleDollarIndexes(segment: string): number[] {
  const indexes: number[] = [];
  for (let index = 0; index < segment.length; index += 1) {
    if (segment[index] !== "$") continue;
    if (segment[index - 1] === "\\" || segment[index - 1] === "$" || segment[index + 1] === "$") {
      continue;
    }
    indexes.push(index);
  }
  return indexes;
}

function canOpenDollarMath(segment: string, index: number): boolean {
  const next = segment[index + 1] ?? "";
  if (!next || /\s/.test(next) || /\d/.test(next)) return false;
  return true;
}

function canCloseDollarMath(segment: string, opener: number, closer: number): boolean {
  if (closer <= opener + 1) return false;
  const previous = segment[closer - 1] ?? "";
  const next = segment[closer + 1] ?? "";
  if (!previous || /\s/.test(previous)) return false;
  if (next && /[A-Za-z0-9_]/.test(next)) return false;
  return segment.slice(opener + 1, closer).trim().length > 0;
}
