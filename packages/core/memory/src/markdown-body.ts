import type { Nodes } from "mdast";
import { fromMarkdown } from "mdast-util-from-markdown";
import { gfmFootnoteFromMarkdown } from "mdast-util-gfm-footnote";
import { gfmFootnote } from "micromark-extension-gfm-footnote";
import { visit } from "unist-util-visit";
import type { MemoryIssue } from "./errors.js";

export function positionAt(source: string, offset: number) {
  const prefix = source.slice(0, offset);
  return { line: prefix.split("\n").length, column: offset - prefix.lastIndexOf("\n") };
}

export function markdownText(node: Nodes): string {
  if ("children" in node) return node.children.map(markdownText).join("");
  return "value" in node ? node.value : "";
}

export function inspectMarkdown(source: string) {
  const tree = fromMarkdown(source, {
    extensions: [gfmFootnote()],
    mdastExtensions: [gfmFootnoteFromMarkdown()],
  });
  const issues: MemoryIssue[] = [];
  const footnotes = new Set<string>();
  const definitions = new Set<string>();
  const links: { url: string; offset: number; title: string }[] = [];
  const references = new Map<string, string>();
  visit(tree, "definition", (node) => {
    references.set(node.identifier, node.url);
  });
  const issue = (ruleId: string, message: string, offset: number) => {
    issues.push({ ruleId, severity: "error", message, ...positionAt(source, offset) });
  };
  visit(tree, (node) => {
    const offset = node.position?.start.offset ?? 0;
    if (node.type === "footnoteDefinition") {
      if (definitions.has(node.identifier)) {
        issue("footnote.duplicate", `Duplicate footnote definition '${node.identifier}'.`, offset);
      }
      definitions.add(node.identifier);
    } else if (node.type === "footnoteReference") {
      footnotes.add(node.identifier);
    } else if (node.type === "link" || node.type === "image") {
      links.push({ url: node.url, offset, title: markdownText(node) });
    } else if (node.type === "linkReference" || node.type === "imageReference") {
      const url = references.get(node.identifier);
      if (url !== undefined) links.push({ url, offset, title: markdownText(node) });
    } else if (node.type === "html" && /<\/?(?:script|iframe|object|embed)\b/iu.test(node.value)) {
      issue("markdown.executable", "Executable embedded HTML is not allowed.", offset);
    } else if (node.type === "text") {
      // Undefined footnotes are plain text in CommonMark; inspect only their original text span.
      const raw = source.slice(offset, node.position?.end.offset);
      for (const match of raw.matchAll(/(?<!\\)(?:\\\\)*\[\^([^\]\n]+)\]/gu)) {
        issue("footnote.undefined", `Undefined footnote '${match[1]}'.`, offset + match.index);
      }
      if (/(?<!\\)\[\[[^\]]+\]\]/u.test(raw)) {
        issue("markdown.wikilink", "Use standard Markdown links instead of Wikilinks.", offset);
      }
      if (/^\s*\^[a-z0-9-]+\s*$/imu.test(raw)) {
        issue("markdown.block-reference", "Obsidian block references are not supported.", offset);
      }
    }
  });
  return { tree, issues, footnotes, links };
}
