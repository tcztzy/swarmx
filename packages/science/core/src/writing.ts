import { createHash } from "node:crypto";
import type { ScienceDocumentFormat, WritingDiagnostic } from "./contracts.js";

const MAX_DIAGNOSTICS = 200;
const CLAIM = /\b(?:significantly|improves?|outperforms?|better|increases?|decreases?)\b/giu;
const EVIDENCE =
  /(?:\b(?:CI|confidence interval|p\s*[<=>]|run\s*#)\b|\d+(?:\.\d+)?\s*%|\\cite|@(?:run|exp|cite):)/iu;
const DELIMITER_PAIRS = new Map([
  [")", "("],
  ["]", "["],
  ["}", "{"],
]);

export function sourceHash(source: string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(source).digest("hex")}`;
}

export function documentFormat(name: string): ScienceDocumentFormat {
  if (name.endsWith(".typ")) return "typst";
  if (name.endsWith(".tex")) return "latex";
  if (name.endsWith(".md")) return "markdown";
  return "bibtex";
}

function delimiterDiagnostics(source: string): WritingDiagnostic[] {
  const stack: Array<{ character: string; index: number }> = [];
  const diagnostics: WritingDiagnostic[] = [];
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (!character || (index > 0 && source[index - 1] === "\\")) continue;
    if (character === "(" || character === "[" || character === "{") {
      stack.push({ character, index });
      continue;
    }
    const opening = DELIMITER_PAIRS.get(character);
    if (!opening) continue;
    const current = stack.at(-1);
    if (current?.character === opening) {
      stack.pop();
      continue;
    }
    diagnostics.push({
      code: "unbalanced-delimiter",
      scope: "structural",
      severity: "error",
      message: `Closing '${character}' has no matching '${opening}'.`,
      start: index,
      end: index + 1,
    });
  }
  for (const opening of stack) {
    diagnostics.push({
      code: "unbalanced-delimiter",
      scope: "structural",
      severity: "error",
      message: `Opening '${opening.character}' is not closed.`,
      start: opening.index,
      end: opening.index + 1,
    });
  }
  return diagnostics;
}

function latexEnvironmentDiagnostics(source: string): WritingDiagnostic[] {
  const tokens = [...source.matchAll(/\\(begin|end)\{([^{}]+)\}/gu)];
  const stack: Array<{ environment: string; index: number; length: number }> = [];
  const diagnostics: WritingDiagnostic[] = [];
  for (const token of tokens) {
    const operation = token[1];
    const environment = token[2];
    if (!operation || !environment || token.index === undefined) continue;
    if (operation === "begin") {
      stack.push({ environment, index: token.index, length: token[0].length });
      continue;
    }
    const opening = stack.at(-1);
    if (opening?.environment === environment) {
      stack.pop();
      continue;
    }
    diagnostics.push({
      code: "unbalanced-environment",
      scope: "structural",
      severity: "error",
      message: `Environment '${environment}' ends without a matching begin.`,
      start: token.index,
      end: token.index + token[0].length,
    });
  }
  for (const opening of stack) {
    diagnostics.push({
      code: "unbalanced-environment",
      scope: "structural",
      severity: "error",
      message: `Environment '${opening.environment}' is not closed.`,
      start: opening.index,
      end: opening.index + opening.length,
    });
  }
  return diagnostics;
}

function figureDiagnostics(source: string, format: ScienceDocumentFormat): WritingDiagnostic[] {
  const references =
    format === "latex"
      ? [...source.matchAll(/\\(?:ref|autoref|cref)\{(fig:[^{}]+)\}/gu)]
      : [...source.matchAll(/@(fig:[A-Za-z0-9_-]+)/gu)];
  const labels = new Set(
    (format === "latex"
      ? [...source.matchAll(/\\label\{(fig:[^{}]+)\}/gu)]
      : [...source.matchAll(/<(fig:[A-Za-z0-9_-]+)>/gu)]
    )
      .map((match) => match[1])
      .filter((label): label is string => label !== undefined),
  );
  return references.flatMap((reference) => {
    const label = reference[1];
    if (!label || labels.has(label) || reference.index === undefined) return [];
    return [
      {
        code: "figure-reference-missing" as const,
        scope: "scientific" as const,
        severity: "warning" as const,
        message: `Figure reference '${label}' has no matching source label.`,
        start: reference.index,
        end: reference.index + reference[0].length,
      },
    ];
  });
}

function claimDiagnostics(source: string): WritingDiagnostic[] {
  const diagnostics: WritingDiagnostic[] = [];
  for (const line of source.matchAll(/[^\n]+/gu)) {
    if (line.index === undefined || EVIDENCE.test(line[0])) continue;
    CLAIM.lastIndex = 0;
    const claim = CLAIM.exec(line[0]);
    if (!claim) continue;
    diagnostics.push({
      code: "claim-needs-evidence",
      scope: "scientific",
      severity: "warning",
      message: "Scientific claim lacks a citation, run link, or quantitative evidence.",
      start: line.index + claim.index,
      end: line.index + claim.index + claim[0].length,
    });
  }
  return diagnostics;
}

export function analyzeDocument(
  source: string,
  format: ScienceDocumentFormat,
): WritingDiagnostic[] {
  const diagnostics = [
    ...delimiterDiagnostics(source),
    ...(format === "latex" ? latexEnvironmentDiagnostics(source) : []),
    ...figureDiagnostics(source, format),
    ...claimDiagnostics(source),
  ];
  return diagnostics
    .sort((left, right) => left.start - right.start || left.code.localeCompare(right.code))
    .slice(0, MAX_DIAGNOSTICS);
}
