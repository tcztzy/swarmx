import { posix } from "node:path";
import type { Root } from "mdast";
import { isMap, parseDocument } from "yaml";
import { z } from "zod";
import { MemoryError, type MemoryIssue } from "./errors.js";
import {
  conceptRevision,
  DEFAULT_MAX_CONCEPT_BYTES,
  decodeConcept,
  memoryDateTimeSchema,
  type ParsedConcept,
  parseConcept,
} from "./markdown.js";
import { inspectMarkdown, markdownText, positionAt } from "./markdown-body.js";

export interface MemoryLintDiagnostic extends MemoryIssue {
  readonly path: string;
  readonly revision: string | null;
}

export type MemoryResourceCheck = (
  resource: string,
) => Pick<MemoryIssue, "ruleId" | "severity" | "message"> | undefined;

export interface MemoryLintOptions {
  readonly workspaceDirectory: string;
  readonly now: string;
  readonly maxBytes?: number;
  readonly checkResource?: MemoryResourceCheck;
}

export function memoryPathIsVisible(path: string, workspace: string): boolean {
  return (
    path === "index.md" ||
    path === "log.md" ||
    ["global", workspace].some(
      (directory) =>
        path === `${directory}/index.md` ||
        path === `${directory}/log.md` ||
        (path.startsWith(`${directory}/concepts/`) &&
          !path.slice(`${directory}/concepts/`.length).includes("/") &&
          path.endsWith(".md")),
    )
  );
}

function localTarget(path: string, url: string): string | null {
  if (/^[a-z][a-z0-9+.-]*:/iu.test(url) || url.startsWith("//")) return null;
  const pathname = decodeURIComponent(url.split(/[?#]/u)[0] ?? "");
  if (pathname === "") return path;
  return posix.normalize(
    pathname.startsWith("/") ? pathname.slice(1) : posix.join(posix.dirname(path), pathname),
  );
}

function localReferenceIssue(
  path: string,
  url: string,
): Pick<MemoryIssue, "ruleId" | "message"> | undefined {
  if (/^(?:javascript|data|vbscript):/iu.test(url)) {
    return { ruleId: "link.unsafe", message: "Executable link schemes are not allowed." };
  }
  let target: string | null;
  try {
    target = localTarget(path, url);
  } catch (error) {
    if (!(error instanceof URIError)) throw error;
    return { ruleId: "link.invalid", message: "Link contains invalid URL encoding." };
  }
  if (target === null) return;
  if (
    target.includes("\\") ||
    target.includes("\0") ||
    target === ".." ||
    target.startsWith("../") ||
    target === ".swarmx" ||
    target.startsWith(".swarmx/")
  ) {
    return { ruleId: "link.scope", message: "Local reference escapes live memory." };
  }
  if (
    path.includes("/concepts/") &&
    target.startsWith("workspaces/") &&
    (path.startsWith("global/") || !target.startsWith(`${posix.dirname(posix.dirname(path))}/`))
  ) {
    return { ruleId: "link.scope", message: "Local reference belongs to a different scope." };
  }
}

export function parseScopedConcept(
  path: string,
  bytes: string | Uint8Array,
  maxBytes?: number,
  checkResource?: MemoryResourceCheck,
): ParsedConcept {
  const concept = parseConcept(bytes, maxBytes);
  const source = decodeConcept(bytes);
  const global = path.startsWith("global/");
  const issues: MemoryIssue[] = [];
  if (
    concept.metadata.swarmx_scope !== (global ? "global" : "workspace") ||
    (!global && concept.metadata.swarmx_workspace !== posix.dirname(posix.dirname(path)).slice(-12))
  ) {
    issues.push({
      ruleId: "scope.mismatch",
      severity: "error",
      line: 1,
      column: 1,
      message: "Concept scope does not match its directory.",
    });
  }
  const bodyOffset = source.length - concept.body.length;
  const references = inspectMarkdown(concept.body).links.map((link) => ({
    ...link,
    offset: bodyOffset + link.offset,
  }));
  for (const entry of concept.metadata.sources) {
    references.push({ url: entry.resource, offset: 0, title: "" });
  }
  for (const { url, offset } of references) {
    const issue = /^sx:/iu.test(url) ? checkResource?.(url) : localReferenceIssue(path, url);
    if (issue && (!("severity" in issue) || issue.severity === "error")) {
      issues.push({ ...issue, severity: "error", ...positionAt(source, offset) });
    }
  }
  if (issues.length > 0)
    throw new MemoryError("Invalid memory scope or reference.", "INVALID_CONCEPT", {
      issues,
    });
  return concept;
}

/** Checks an authorized file snapshot with an explicit clock and optional Science resolver. */
export function lintMemory(
  files: ReadonlyMap<string, string | Uint8Array>,
  options: MemoryLintOptions,
): MemoryLintDiagnostic[] {
  const now = Date.parse(memoryDateTimeSchema.parse(options.now));
  const workspace = z
    .string()
    .regex(/^workspaces\/[^/]+--[a-f0-9]{12}$/u)
    .parse(options.workspaceDirectory);
  const diagnostics: MemoryLintDiagnostic[] = [];
  const documents = new Map<
    string,
    { source: string; body: string; tree: Root; concept?: ParsedConcept }
  >();
  const emit = (
    path: string,
    ruleId: string,
    severity: "error" | "warning",
    message: string,
    offset = 0,
  ) => {
    const bytes = files.get(path);
    if (bytes === undefined) throw new Error("Diagnostic has no inspected document.");
    const source = documents.get(path)?.source;
    diagnostics.push({
      path,
      revision: conceptRevision(bytes),
      ruleId,
      severity,
      message,
      ...(source === undefined ? { line: 1, column: 1 } : positionAt(source, offset)),
    });
  };
  const warn = (path: string, rule: string, message: string, offset = 0) =>
    emit(path, rule, "warning", message, offset);

  for (const [path, bytes] of [...files].sort(([a], [b]) => a.localeCompare(b))) {
    if (!memoryPathIsVisible(path, workspace)) continue;
    try {
      if (Buffer.byteLength(bytes) > (options.maxBytes ?? DEFAULT_MAX_CONCEPT_BYTES)) {
        emit(path, "document.size", "error", "Memory document exceeds its byte limit.");
        continue;
      }
      const source = decodeConcept(bytes);
      const reserved = ["index.md", "log.md"].includes(posix.basename(path));
      if (!reserved) {
        const concept = parseScopedConcept(path, bytes, options.maxBytes, options.checkResource);
        const { tree } = inspectMarkdown(concept.body);
        documents.set(path, { source, body: concept.body, tree, concept });
      } else {
        let body = source;
        if (source.startsWith("---")) {
          const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n([\s\S]*)$/u.exec(source);
          const header =
            match?.[1] === undefined ? undefined : parseDocument(match[1], { uniqueKeys: true });
          if (
            path !== "index.md" ||
            !header ||
            header.errors.length > 0 ||
            !isMap(header.contents) ||
            header.contents.items.length !== 1 ||
            header.get("okf_version") !== "0.2"
          ) {
            emit(
              path,
              "reserved.frontmatter",
              "error",
              "Only the root index may declare OKF 0.2 frontmatter.",
            );
            continue;
          }
          body = match?.[2] ?? "";
        }
        const { tree, issues } = inspectMarkdown(body);
        documents.set(path, { source, body, tree });
        const bodyLine = positionAt(source, source.length - body.length).line;
        for (const issue of issues)
          diagnostics.push({
            ...issue,
            path,
            line: issue.line + bodyLine - 1,
            revision: conceptRevision(bytes),
          });
      }
    } catch (error) {
      if (!(error instanceof MemoryError) || error.code !== "INVALID_CONCEPT") throw error;
      if (error.issues.length === 0) emit(path, "document.invalid", "error", error.message);
      for (const issue of error.issues)
        diagnostics.push({ ...issue, path, revision: conceptRevision(bytes) });
    }
  }

  const linked = new Map<string, Set<string>>();
  for (const [path, { source, body, tree, concept }] of documents) {
    const bodyOffset = source.length - body.length;
    const { links, footnotes } = inspectMarkdown(body);
    const targets = new Set<string>();
    linked.set(path, targets);
    const checkLink = (url: string, offset: number) => {
      const issue = localReferenceIssue(path, url);
      if (issue) {
        emit(path, issue.ruleId, "error", issue.message, offset);
        return;
      }
      const target = localTarget(path, url);
      if (target === null) return;
      if (
        target.startsWith("workspaces/") &&
        target !== "workspaces/" &&
        (!target.startsWith(`${workspace}/`) || path.startsWith("global/"))
      ) {
        if (concept)
          emit(
            path,
            "link.scope",
            "error",
            "Local reference belongs to a different scope.",
            offset,
          );
        return;
      }
      targets.add(target);
      const exists =
        files.has(target) ||
        (url.endsWith("/") &&
          [...files.keys()].some((file) =>
            file.startsWith(target.endsWith("/") ? target : `${target}/`),
          ));
      if (!exists) warn(path, "link.broken", `Local link target '${target}' is missing.`, offset);
    };
    for (const link of links) checkLink(link.url, bodyOffset + link.offset);

    if (concept) {
      const ids = new Set(
        concept.metadata.sources.flatMap((entry) =>
          entry.id === undefined ? [] : [entry.id.toLowerCase()],
        ),
      );
      for (const id of footnotes) {
        if (!ids.has(id))
          warn(
            path,
            "source.unassociated",
            `Footnote '${id}' has no associated source.`,
            bodyOffset,
          );
      }
      for (const entry of concept.metadata.sources) {
        if (entry.id !== undefined && !footnotes.has(entry.id.toLowerCase())) {
          warn(path, "source.unused", `Source '${entry.id}' is not cited by a footnote.`);
        }
        if (/^sx:/iu.test(entry.resource)) {
          const issue = options.checkResource?.(entry.resource);
          if (issue) emit(path, issue.ruleId, issue.severity, issue.message);
          if (!options.checkResource)
            warn(
              path,
              "source.unchecked",
              "Science reference requires the workspace Science resolver.",
            );
        } else if (/^(?:\.{0,2}\/|[^\s]+\.md(?:[?#]|$))/u.test(entry.resource)) {
          checkLink(entry.resource, 0);
        }
      }
      if (concept.metadata.type === "Finding" && concept.metadata.sources.length === 0) {
        warn(path, "source.missing", "Finding has no recorded evidence source.");
      }
      if (
        concept.metadata.stale_after !== undefined &&
        now >= Date.parse(concept.metadata.stale_after)
      ) {
        warn(
          path,
          "lifecycle.stale",
          "Memory is past stale_after; review it before use.",
          Math.max(0, source.indexOf("\nstale_after:") + 1),
        );
      }
      continue;
    }

    const index = posix.basename(path) === "index.md";
    let dated = false;
    if (!tree.children.some((node) => node.type === "heading")) {
      emit(
        path,
        index ? "index.structure" : "log.structure",
        "error",
        "Reserved document needs a heading.",
      );
    }
    for (const node of tree.children) {
      const offset = bodyOffset + (node.position?.start.offset ?? 0);
      if (node.type === "heading") {
        if (!index && node.depth > 1) {
          dated = z.iso.date().safeParse(markdownText(node)).success;
          if (!dated)
            emit(
              path,
              "log.date",
              "error",
              "Log date headings must be valid YYYY-MM-DD dates.",
              offset,
            );
        }
      } else if (node.type === "list") {
        if (!index && !dated)
          emit(
            path,
            "log.structure",
            "error",
            "Log entries need a preceding date heading.",
            offset,
          );
        if (!index) continue;
        for (const item of node.children) {
          const paragraph = item.children[0];
          const link = paragraph?.type === "paragraph" ? paragraph.children[0] : undefined;
          if (link?.type !== "link") {
            emit(
              path,
              "index.structure",
              "error",
              "Index entries must begin with a Markdown link.",
              offset,
            );
            continue;
          }
          let target: string | null;
          try {
            target = localTarget(path, link.url);
          } catch (error) {
            if (error instanceof URIError) continue;
            throw error;
          }
          const destination = target === null ? undefined : documents.get(target)?.concept;
          if (!destination) continue;
          const description = markdownText(item)
            .slice(markdownText(link).length)
            .trim()
            .replace(/^-\s*/u, "");
          const expected = `${destination.metadata.description}${destination.metadata.status === "deprecated" ? " (deprecated)" : ""}`;
          if (markdownText(link) !== destination.metadata.title || description !== expected) {
            warn(
              path,
              "index.stale",
              "Index title or description differs from the current concept.",
              offset,
            );
          }
        }
      } else {
        emit(
          path,
          index ? "index.structure" : "log.structure",
          "error",
          "Reserved documents contain headings and list entries.",
          offset,
        );
      }
    }
  }
  for (const [path, { concept }] of documents) {
    if (!concept) continue;
    const index = `${posix.dirname(posix.dirname(path))}/index.md`;
    if (!linked.get(index)?.has(path))
      warn(path, "index.missing", "Concept is absent from its scope index.");
  }
  return diagnostics.sort(
    (a, b) =>
      a.path.localeCompare(b.path) ||
      a.line - b.line ||
      a.column - b.column ||
      a.ruleId.localeCompare(b.ruleId),
  );
}
