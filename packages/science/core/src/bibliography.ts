import { ScienceError } from "./errors.js";

export interface BibEntry {
  readonly type: string;
  readonly key: string;
  readonly fields: Readonly<Record<string, string>>;
}

const MAX_BIB_BYTES = 8 * 1024 * 1024;
const MAX_BIB_ENTRIES = 10_000;
const MAX_BIB_FIELD_BYTES = 64 * 1024;
const PRIVATE_FIELDS = new Set([
  "attachment",
  "attachments",
  "file",
  "files",
  "fulltext",
  "full-text",
  "local-url",
  "pdf",
]);
const PREFERRED_FIELDS = [
  "title",
  "author",
  "editor",
  "year",
  "date",
  "journal",
  "booktitle",
  "publisher",
  "institution",
  "school",
  "volume",
  "number",
  "pages",
  "edition",
  "language",
  "abstract",
  "keywords",
  "doi",
  "isbn",
  "issn",
  "url",
  "x-source",
  "x-source-id",
  "x-source-version",
  "x-search-hits",
] as const;

function invalid(message: string, cause?: unknown): never {
  throw new ScienceError(`Invalid BibTeX: ${message}`, "BIBLIOGRAPHY_INVALID", {
    ...(cause instanceof Error ? { cause } : {}),
  });
}

class BibParser {
  private index = 0;

  constructor(private readonly input: string) {}

  parse(): BibEntry[] {
    const entries: BibEntry[] = [];
    const keys = new Set<string>();
    while (true) {
      this.skipTrivia();
      if (this.index >= this.input.length) return entries;
      this.expect("@");
      const type = this.identifier("entry type").toLowerCase();
      this.skipTrivia();
      const open = this.peek();
      if (open !== "{" && open !== "(") invalid(`expected entry body at offset ${this.index}`);
      this.index += 1;
      const close = open === "{" ? "}" : ")";
      if (type === "comment" || type === "preamble" || type === "string") {
        this.skipBalanced(open, close);
        continue;
      }
      const entry = this.entry(type, close);
      if (keys.has(entry.key)) invalid(`duplicate citation key '${entry.key}'`);
      keys.add(entry.key);
      entries.push(entry);
      if (entries.length > MAX_BIB_ENTRIES) invalid(`more than ${MAX_BIB_ENTRIES} entries`);
    }
  }

  private entry(type: string, close: string): BibEntry {
    this.skipTrivia();
    const keyStart = this.index;
    while (this.index < this.input.length && this.peek() !== "," && this.peek() !== close) {
      this.index += 1;
    }
    const key = this.input.slice(keyStart, this.index).trim();
    if (!/^[^\s,{}()]{1,200}$/u.test(key)) invalid(`invalid citation key '${key}'`);
    if (this.peek() !== ",") invalid(`entry '${key}' has no fields`);
    this.index += 1;
    const fields: Record<string, string> = {};
    while (true) {
      this.skipTrivia();
      if (this.peek() === close) {
        this.index += 1;
        return { type, key, fields };
      }
      const name = this.identifier("field name").toLowerCase();
      this.skipTrivia();
      this.expect("=");
      this.skipTrivia();
      const value = this.value(close);
      if (!PRIVATE_FIELDS.has(name)) fields[name] = normalizeField(value, name);
      this.skipTrivia();
      if (this.peek() === ",") {
        this.index += 1;
        continue;
      }
      if (this.peek() === close) continue;
      invalid(`expected ',' or '${close}' after field '${name}'`);
    }
  }

  private value(close: string): string {
    let value = this.valuePart(close);
    while (true) {
      this.skipTrivia();
      if (this.peek() !== "#") return value;
      this.index += 1;
      this.skipTrivia();
      value += this.valuePart(close);
    }
  }

  private valuePart(close: string): string {
    const next = this.peek();
    if (next === "{") return this.bracedValue();
    if (next === '"') return this.quotedValue();
    const start = this.index;
    while (this.index < this.input.length) {
      const character = this.peek();
      if (character === "," || character === "#" || character === close) break;
      this.index += 1;
    }
    const value = this.input.slice(start, this.index).trim();
    if (!value) invalid(`empty field value at offset ${start}`);
    return value;
  }

  private bracedValue(): string {
    this.expect("{");
    let depth = 1;
    let value = "";
    while (this.index < this.input.length) {
      const character = this.input[this.index++];
      if (character === "\\") {
        value += character;
        if (this.index < this.input.length) value += this.input[this.index++];
        continue;
      }
      if (character === "{") {
        depth += 1;
        value += character;
        continue;
      }
      if (character === "}") {
        depth -= 1;
        if (depth === 0) return value;
        value += character;
        continue;
      }
      value += character;
    }
    return invalid("unterminated braced value");
  }

  private quotedValue(): string {
    this.expect('"');
    let depth = 0;
    let value = "";
    while (this.index < this.input.length) {
      const character = this.input[this.index++];
      if (character === "\\") {
        value += character;
        if (this.index < this.input.length) value += this.input[this.index++];
        continue;
      }
      if (character === "{") depth += 1;
      if (character === "}" && depth > 0) depth -= 1;
      if (character === '"' && depth === 0) return value;
      value += character;
    }
    return invalid("unterminated quoted value");
  }

  private skipBalanced(open: string, close: string): void {
    let depth = 1;
    while (this.index < this.input.length) {
      const character = this.input[this.index++];
      if (character === "\\") {
        this.index += this.index < this.input.length ? 1 : 0;
      } else if (character === open) {
        depth += 1;
      } else if (character === close) {
        depth -= 1;
        if (depth === 0) return;
      }
    }
    invalid(`unterminated @${open === "{" ? "comment" : "entry"}`);
  }

  private identifier(label: string): string {
    const start = this.index;
    while (/[A-Za-z0-9_:-]/u.test(this.peek())) this.index += 1;
    const value = this.input.slice(start, this.index);
    if (!/^[A-Za-z][A-Za-z0-9_:-]{0,63}$/u.test(value)) {
      invalid(`invalid ${label} at offset ${start}`);
    }
    return value;
  }

  private skipTrivia(): void {
    while (this.index < this.input.length) {
      if (/\s/u.test(this.peek())) {
        this.index += 1;
        continue;
      }
      if (this.peek() !== "%") return;
      while (this.index < this.input.length && this.peek() !== "\n") this.index += 1;
    }
  }

  private expect(value: string): void {
    if (this.peek() !== value) invalid(`expected '${value}' at offset ${this.index}`);
    this.index += 1;
  }

  private peek(): string {
    return this.input[this.index] ?? "";
  }
}

function normalizeField(value: string, name: string): string {
  const normalized = value.replaceAll("\r\n", "\n").replaceAll("\r", "\n").trim().normalize();
  if (!normalized || normalized.includes("\0")) invalid(`invalid value for field '${name}'`);
  if (Buffer.byteLength(normalized) > MAX_BIB_FIELD_BYTES) {
    invalid(`field '${name}' exceeds ${MAX_BIB_FIELD_BYTES} bytes`);
  }
  return normalized;
}

function balancedBraces(value: string): boolean {
  let depth = 0;
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "\\") {
      index += 1;
    } else if (value[index] === "{") {
      depth += 1;
    } else if (value[index] === "}") {
      depth -= 1;
      if (depth < 0) return false;
    }
  }
  return depth === 0;
}

function serializableValue(value: string, name: string): string {
  const normalized = normalizeField(value, name);
  if (balancedBraces(normalized)) return normalized;
  return normalized.replaceAll(/(?<!\\)([{}])/gu, "\\$1");
}

function fieldOrder(name: string): [number, string] {
  const preferred = PREFERRED_FIELDS.indexOf(name as (typeof PREFERRED_FIELDS)[number]);
  return [preferred === -1 ? PREFERRED_FIELDS.length : preferred, name];
}

export function parseBibFile(input: string): BibEntry[] {
  if (Buffer.byteLength(input) > MAX_BIB_BYTES) {
    throw new ScienceError("BibTeX snapshot is too large", "BIBLIOGRAPHY_TOO_LARGE");
  }
  return new BibParser(input).parse();
}

export function serializeBibFile(entries: readonly BibEntry[]): string {
  if (entries.length > MAX_BIB_ENTRIES) {
    throw new ScienceError("BibTeX snapshot has too many entries", "BIBLIOGRAPHY_TOO_LARGE");
  }
  const seen = new Set<string>();
  const blocks = entries.map((entry) => {
    const type = entry.type.toLowerCase();
    if (!/^[a-z][a-z0-9_-]{0,39}$/u.test(type)) invalid(`invalid entry type '${entry.type}'`);
    if (!/^[^\s,{}()]{1,200}$/u.test(entry.key)) invalid(`invalid citation key '${entry.key}'`);
    if (seen.has(entry.key)) invalid(`duplicate citation key '${entry.key}'`);
    seen.add(entry.key);
    const fields = Object.entries(entry.fields)
      .filter(([name]) => !PRIVATE_FIELDS.has(name.toLowerCase()))
      .sort(([left], [right]) => {
        const [leftIndex, leftName] = fieldOrder(left);
        const [rightIndex, rightName] = fieldOrder(right);
        return leftIndex - rightIndex || leftName.localeCompare(rightName, "en");
      })
      .map(([name, value]) => {
        const normalizedName = name.toLowerCase();
        if (!/^[a-z][a-z0-9_:-]{0,63}$/u.test(normalizedName)) {
          invalid(`invalid field name '${name}'`);
        }
        return `  ${normalizedName} = {${serializableValue(value, normalizedName)}},`;
      });
    return [`@${type}{${entry.key},`, ...fields, "}"].join("\n");
  });
  const serialized = blocks.length === 0 ? "" : `${blocks.join("\n\n")}\n`;
  if (Buffer.byteLength(serialized) > MAX_BIB_BYTES) {
    throw new ScienceError("BibTeX snapshot is too large", "BIBLIOGRAPHY_TOO_LARGE");
  }
  return serialized;
}
