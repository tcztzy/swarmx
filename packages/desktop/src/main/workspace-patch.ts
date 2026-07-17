export type CodexPatchOperation =
  | { type: "add"; path: string; content: string }
  | { type: "delete"; path: string }
  | { type: "update"; path: string; moveTo?: string; hunks: CodexPatchHunk[] };

export interface CodexPatchHunk {
  lines: Array<{ kind: "context" | "add" | "remove"; text: string }>;
  endOfFile: boolean;
}

export interface AppliedCodexUpdate {
  content: string;
  replacements: number;
}

export function parseCodexPatch(input: string): CodexPatchOperation[] {
  if (typeof input !== "string" || input.length === 0) {
    throw new Error("apply_patch requires a non-empty patch.");
  }
  const lines = input.replace(/\r\n?/g, "\n").split("\n");
  if (lines[0] !== "*** Begin Patch") {
    throw new Error('apply_patch must start with "*** Begin Patch".');
  }

  const operations: CodexPatchOperation[] = [];
  let index = 1;
  while (index < lines.length) {
    const line = lines[index];
    if (line === "*** End Patch") {
      if (lines.slice(index + 1).some((candidate) => candidate.length > 0)) {
        throw new Error("apply_patch contains content after the end marker.");
      }
      if (operations.length === 0) throw new Error("apply_patch contains no file operations.");
      return operations;
    }
    if (line?.startsWith("*** Add File: ")) {
      const path = requiredHeaderPath(line, "*** Add File: ");
      index += 1;
      const content: string[] = [];
      while (index < lines.length && !isOperationBoundary(lines[index])) {
        const contentLine = lines[index] ?? "";
        if (!contentLine.startsWith("+")) {
          throw new Error(`Add File ${path} contains a line without a + prefix.`);
        }
        content.push(contentLine.slice(1));
        index += 1;
      }
      if (content.length === 0) throw new Error(`Add File ${path} has no content lines.`);
      operations.push({ type: "add", path, content: `${content.join("\n")}\n` });
      continue;
    }
    if (line?.startsWith("*** Delete File: ")) {
      operations.push({
        type: "delete",
        path: requiredHeaderPath(line, "*** Delete File: "),
      });
      index += 1;
      continue;
    }
    if (line?.startsWith("*** Update File: ")) {
      const path = requiredHeaderPath(line, "*** Update File: ");
      index += 1;
      let moveTo: string | undefined;
      if (lines[index]?.startsWith("*** Move to: ")) {
        moveTo = requiredHeaderPath(lines[index] ?? "", "*** Move to: ");
        index += 1;
      }
      const hunks: CodexPatchHunk[] = [];
      let current: CodexPatchHunk | undefined;
      while (index < lines.length && !isOperationBoundary(lines[index])) {
        const changeLine = lines[index] ?? "";
        if (changeLine === "*** End of File") {
          if (!current) throw new Error(`Update File ${path} has no hunk before End of File.`);
          current.endOfFile = true;
          index += 1;
          continue;
        }
        if (changeLine === "@@" || changeLine.startsWith("@@ ")) {
          current = { lines: [], endOfFile: false };
          hunks.push(current);
          index += 1;
          continue;
        }
        current ??= { lines: [], endOfFile: false };
        if (hunks[hunks.length - 1] !== current) hunks.push(current);
        const prefix = changeLine[0];
        if (prefix !== " " && prefix !== "+" && prefix !== "-") {
          throw new Error(`Update File ${path} contains an invalid hunk line.`);
        }
        current.lines.push({
          kind: prefix === "+" ? "add" : prefix === "-" ? "remove" : "context",
          text: changeLine.slice(1),
        });
        index += 1;
      }
      if (hunks.length === 0 && !moveTo) throw new Error(`Update File ${path} has no changes.`);
      operations.push({ type: "update", path, ...(moveTo ? { moveTo } : {}), hunks });
      continue;
    }
    throw new Error(`Unexpected apply_patch line: ${line ?? "<end>"}`);
  }
  throw new Error('apply_patch is missing "*** End Patch".');
}

export function applyCodexUpdate(content: string, hunks: CodexPatchHunk[]): AppliedCodexUpdate {
  let updated = content;
  let replacements = 0;
  const newline = content.includes("\r\n") ? "\r\n" : "\n";

  for (const hunk of hunks) {
    const oldLines = hunk.lines.filter((line) => line.kind !== "add").map((line) => line.text);
    const newLines = hunk.lines.filter((line) => line.kind !== "remove").map((line) => line.text);
    if (oldLines.length === 0) {
      if (!hunk.endOfFile) {
        throw new Error("An add-only update hunk requires *** End of File.");
      }
      const addition = newLines.join(newline);
      updated = `${updated}${updated.endsWith(newline) || updated.length === 0 ? "" : newline}${addition}${newline}`;
      replacements += 1;
      continue;
    }

    const oldBody = oldLines.join(newline);
    const candidates = [`${oldBody}${newline}`, oldBody];
    const match = candidates
      .map((candidate) => ({ candidate, indexes: occurrenceIndexes(updated, candidate) }))
      .find(({ indexes }) => indexes.length === 1);
    if (!match) {
      const occurrences = occurrenceIndexes(updated, oldBody).length;
      throw new Error(
        occurrences > 1
          ? "apply_patch hunk context is ambiguous in the current file."
          : "apply_patch hunk context was not found in the current file.",
      );
    }
    const replacement = `${newLines.join(newline)}${match.candidate.endsWith(newline) ? newline : ""}`;
    const at = match.indexes[0] ?? 0;
    updated = `${updated.slice(0, at)}${replacement}${updated.slice(at + match.candidate.length)}`;
    replacements += 1;
  }
  return { content: updated, replacements };
}

function requiredHeaderPath(line: string, prefix: string): string {
  const value = line.slice(prefix.length);
  if (!value.trim()) throw new Error(`${prefix.trim()} requires a path.`);
  return value;
}

function isOperationBoundary(line: string | undefined): boolean {
  return (
    line === "*** End Patch" ||
    line?.startsWith("*** Add File: ") === true ||
    line?.startsWith("*** Delete File: ") === true ||
    line?.startsWith("*** Update File: ") === true
  );
}

function occurrenceIndexes(content: string, search: string): number[] {
  if (!search) return [];
  const indexes: number[] = [];
  let from = 0;
  while (from <= content.length - search.length) {
    const index = content.indexOf(search, from);
    if (index < 0) break;
    indexes.push(index);
    from = index + Math.max(1, search.length);
  }
  return indexes;
}
