import Papa from "papaparse";
import type { ScienceArtifactPreview } from "./contracts.js";

const MAX_COLUMNS = 256;
const MAX_ROWS = 500;
const NUMBER = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?$/iu;

type TablePreview = Omit<
  Extract<ScienceArtifactPreview, { kind: "table" }>,
  "artifactId" | "digest" | "mime" | "size"
>;
type Scalar = TablePreview["rows"][number][number];

function csvScalar(value: string): Scalar {
  const trimmed = value.trim();
  if (trimmed.length === 0) return null;
  if (trimmed === "true") return true;
  if (trimmed === "false") return false;
  if (NUMBER.test(trimmed)) {
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return numeric;
  }
  return value;
}

function jsonScalar(value: unknown): Scalar {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
  return JSON.stringify(value);
}

function columnType(
  rows: readonly (readonly Scalar[])[],
  index: number,
): "boolean" | "number" | "string" {
  const values = rows.map((row) => row[index]).filter((value) => value !== null);
  if (values.length > 0 && values.every((value) => typeof value === "number")) return "number";
  if (values.length > 0 && values.every((value) => typeof value === "boolean")) return "boolean";
  return "string";
}

function csvPreview(text: string, delimiter: "," | "\t"): TablePreview | undefined {
  const parsed = Papa.parse<string[]>(text, { delimiter, skipEmptyLines: "greedy" });
  if (parsed.errors.length > 0 || parsed.data.length === 0) return undefined;
  const [header = [], ...records] = parsed.data;
  const sourceColumnCount = Math.max(header.length, ...records.map((row) => row.length));
  const columnCount = Math.min(sourceColumnCount, MAX_COLUMNS);
  const rows = records
    .slice(0, MAX_ROWS)
    .map((record) =>
      Array.from({ length: columnCount }, (_, index) => csvScalar(record[index] ?? "")),
    );
  return {
    kind: "table",
    columns: Array.from({ length: columnCount }, (_, index) => ({
      id: `column-${index}`,
      name: header[index] ?? "",
      type: columnType(rows, index),
    })),
    rows,
    rowCount: records.length,
    truncated: records.length > MAX_ROWS || sourceColumnCount > MAX_COLUMNS,
  };
}

function jsonPreview(text: string): TablePreview | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (
    !Array.isArray(parsed) ||
    !parsed.every((value) => typeof value === "object" && value !== null && !Array.isArray(value))
  ) {
    return undefined;
  }
  const records = parsed as Record<string, unknown>[];
  const names: string[] = [];
  const seen = new Set<string>();
  for (const record of records) {
    for (const name of Object.keys(record)) {
      if (seen.has(name)) continue;
      seen.add(name);
      names.push(name);
    }
  }
  const visibleNames = names.slice(0, MAX_COLUMNS);
  const rows = records
    .slice(0, MAX_ROWS)
    .map((record) => visibleNames.map((name) => jsonScalar(record[name] ?? null)));
  return {
    kind: "table",
    columns: visibleNames.map((name, index) => ({
      id: `column-${index}`,
      name,
      type: columnType(rows, index),
    })),
    rows,
    rowCount: records.length,
    truncated: records.length > MAX_ROWS || names.length > MAX_COLUMNS,
  };
}

/** Parse only bounded text already verified by the Artifact Store. */
export function tabularPreview(mime: string, text: string): TablePreview | undefined {
  if (mime === "text/csv") return csvPreview(text, ",");
  if (mime === "text/tab-separated-values") return csvPreview(text, "\t");
  if (mime === "application/json") return jsonPreview(text);
  return undefined;
}
