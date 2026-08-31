import { createHash, randomUUID } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { z } from "zod";
import { type BibEntry, parseBibFile, serializeBibFile } from "./bibliography.js";
import {
  type LiteratureMatchField,
  type LiteratureSearchRequest,
  type LiteratureSearchResult,
  literatureSearchRequestSchema,
  literatureSearchResultSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";

const DEFAULT_ZOTERO_BASE_URL = "http://127.0.0.1:23119";
const DEFAULT_TIMEOUT_MS = 10_000;
const MAX_CANDIDATES = 500;
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_QUERY_PARTS = 6;
const MAX_ZOTERO_RESULTS_PER_QUERY = 100;
const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "for",
  "from",
  "how",
  "in",
  "is",
  "of",
  "on",
  "or",
  "the",
  "to",
  "using",
  "what",
  "with",
]);
const BIB_TYPES: Readonly<Record<string, string>> = {
  artwork: "misc",
  audioRecording: "misc",
  blogPost: "misc",
  book: "book",
  bookSection: "incollection",
  conferencePaper: "inproceedings",
  dictionaryEntry: "incollection",
  document: "misc",
  encyclopediaArticle: "incollection",
  film: "misc",
  forumPost: "misc",
  hearing: "misc",
  instantMessage: "misc",
  interview: "misc",
  journalArticle: "article",
  letter: "misc",
  magazineArticle: "article",
  manuscript: "unpublished",
  map: "misc",
  newspaperArticle: "article",
  patent: "misc",
  podcast: "misc",
  presentation: "misc",
  radioBroadcast: "misc",
  report: "techreport",
  statute: "misc",
  thesis: "phdthesis",
  tvBroadcast: "misc",
  videoRecording: "misc",
  webpage: "misc",
};

const zoteroCreatorSchema = z
  .object({
    creatorType: z.string().max(100),
    firstName: z.string().max(500).optional(),
    lastName: z.string().max(500).optional(),
    name: z.string().max(1_000).optional(),
  })
  .passthrough();

const zoteroItemSchema = z
  .object({
    key: z.string().regex(/^[A-Z0-9]{8}$/u),
    version: z.number().int().nonnegative(),
    data: z
      .object({
        key: z
          .string()
          .regex(/^[A-Z0-9]{8}$/u)
          .optional(),
        itemType: z.string().min(1).max(100),
        title: z.string().max(10_000).optional(),
        date: z.string().max(500).optional(),
        citationKey: z.string().max(500).optional(),
        creators: z.array(zoteroCreatorSchema).max(500).optional(),
        tags: z
          .array(z.object({ tag: z.string().max(500) }).passthrough())
          .max(500)
          .optional(),
        abstractNote: z.string().max(100_000).optional(),
        publicationTitle: z.string().max(2_000).optional(),
        proceedingsTitle: z.string().max(2_000).optional(),
        bookTitle: z.string().max(2_000).optional(),
        publisher: z.string().max(2_000).optional(),
        institution: z.string().max(2_000).optional(),
        university: z.string().max(2_000).optional(),
        volume: z.string().max(200).optional(),
        issue: z.string().max(200).optional(),
        pages: z.string().max(200).optional(),
        edition: z.string().max(200).optional(),
        language: z.string().max(200).optional(),
        DOI: z.string().max(500).optional(),
        ISBN: z.string().max(500).optional(),
        ISSN: z.string().max(500).optional(),
        url: z.string().max(5_000).optional(),
      })
      .passthrough(),
  })
  .passthrough();

const zoteroItemsSchema = z.array(zoteroItemSchema).max(MAX_ZOTERO_RESULTS_PER_QUERY);
type ZoteroItem = z.infer<typeof zoteroItemSchema>;

interface BibliographyExport {
  readonly entries: readonly BibEntry[];
  readonly source: "zotero";
  readonly sourceVersion: string | null;
}

export interface BibliographySource {
  readonly id: "zotero";
  search(request: LiteratureSearchRequest, signal?: AbortSignal): Promise<BibliographyExport>;
}

interface ZoteroSourceOptions {
  readonly baseUrl?: string;
  readonly fetch?: typeof globalThis.fetch;
  readonly maxResponseBytes?: number;
  readonly timeoutMs?: number;
}

function abortIfRequested(signal?: AbortSignal): void {
  signal?.throwIfAborted();
}

function parseRequest(request: LiteratureSearchRequest): LiteratureSearchRequest {
  try {
    return literatureSearchRequestSchema.parse(request);
  } catch (error) {
    throw new ScienceError("Invalid literature search request", "INVALID_REQUEST", {
      cause: error,
    });
  }
}

function loopbackBaseUrl(value: string): URL {
  let url: URL;
  try {
    url = new URL(value);
  } catch (error) {
    throw new ScienceError("Zotero base URL is invalid", "INVALID_REQUEST", { cause: error });
  }
  const loopback =
    url.hostname === "127.0.0.1" || url.hostname === "localhost" || url.hostname === "[::1]";
  if (url.protocol !== "http:" || !loopback || (url.pathname !== "/" && url.pathname !== "")) {
    throw new ScienceError("Zotero source must be an HTTP loopback origin", "INVALID_REQUEST");
  }
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url;
}

function queryParts(query: string): string[] {
  const normalized = query.trim().replaceAll(/\s+/gu, " ");
  const parts = [normalized];
  const tokens = normalized.match(/[\p{L}\p{N}][\p{L}\p{N}:+./-]*/gu) ?? [];
  for (const token of tokens) {
    const lowered = token.toLocaleLowerCase("en-US");
    if (lowered.length < 2 || STOPWORDS.has(lowered)) continue;
    if (!parts.some((part) => part.toLocaleLowerCase("en-US") === lowered)) parts.push(token);
    if (parts.length >= MAX_QUERY_PARTS) break;
  }
  return parts;
}

async function boundedText(response: Response, maxBytes: number): Promise<string> {
  const length = response.headers.get("content-length");
  if (length && Number(length) > maxBytes) {
    throw new ScienceError("Zotero response is too large", "BIBLIOGRAPHY_TOO_LARGE");
  }
  if (!response.body) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    bytes += chunk.value.byteLength;
    if (bytes > maxBytes) {
      await reader.cancel();
      throw new ScienceError("Zotero response is too large", "BIBLIOGRAPHY_TOO_LARGE");
    }
    text += decoder.decode(chunk.value, { stream: true });
  }
  return text + decoder.decode();
}

function newestVersion(values: readonly (string | null)[]): string | null {
  const versions = values.filter((value): value is string => value !== null);
  if (versions.length === 0) return null;
  return (
    versions.sort((left, right) => Number(right) - Number(left) || right.localeCompare(left))[0] ??
    null
  );
}

function bibText(value: string, maxLength = 4_000): string {
  return value
    .trim()
    .slice(0, maxLength)
    .normalize()
    .replaceAll(/([&%#$])/gu, "\\$1")
    .replaceAll("_", "\\_")
    .replaceAll("{", "\\{")
    .replaceAll("}", "\\}");
}

function safeUrl(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : undefined;
  } catch {
    return undefined;
  }
}

function safeCitationKey(value: string | undefined, itemKey: string): string {
  const normalized = value
    ?.trim()
    .replaceAll(/[^A-Za-z0-9_:.+/@-]+/gu, "_")
    .replaceAll(/^_+|_+$/gu, "")
    .slice(0, 180);
  return normalized && /^[^\s,{}()]+$/u.test(normalized)
    ? normalized
    : `zotero_${itemKey.toLocaleLowerCase("en-US")}`;
}

function creators(item: ZoteroItem, kind: "author" | "editor"): string | undefined {
  const values = (item.data.creators ?? [])
    .filter((creator) => creator.creatorType === kind)
    .map((creator) => {
      if (creator.name?.trim()) return `{${bibText(creator.name, 500)}}`;
      const last = creator.lastName?.trim();
      const first = creator.firstName?.trim();
      if (!last && !first) return null;
      return first
        ? `${bibText(last ?? "", 500)}, ${bibText(first, 500)}`
        : bibText(last ?? "", 500);
    })
    .filter((value): value is string => value !== null);
  return values.length === 0 ? undefined : values.join(" and ");
}

function year(value: string | undefined): string | undefined {
  return value?.match(/(?:^|\D)(\d{4})(?:\D|$)/u)?.[1];
}

function itemEntry(
  item: ZoteroItem,
  sourceVersion: string | null,
  hits: number,
  usedKeys: Set<string>,
): BibEntry | null {
  const type = BIB_TYPES[item.data.itemType] ?? null;
  const title = item.data.title?.trim();
  if (!type || !title) return null;
  const baseKey = safeCitationKey(item.data.citationKey, item.key);
  let key = baseKey;
  if (usedKeys.has(key)) key = `${baseKey}_${item.key.toLocaleLowerCase("en-US")}`;
  usedKeys.add(key);
  const fields: Record<string, string> = {
    title: bibText(title, 1_000),
    "x-search-hits": String(hits),
    "x-source": "zotero",
    "x-source-id": item.key,
  };
  const put = (name: string, value: string | undefined) => {
    if (value?.trim()) fields[name] = value;
  };
  put("author", creators(item, "author"));
  put("editor", creators(item, "editor"));
  put("year", year(item.data.date));
  put("date", item.data.date ? bibText(item.data.date, 500) : undefined);
  put(
    "journal",
    item.data.publicationTitle ? bibText(item.data.publicationTitle, 1_000) : undefined,
  );
  put(
    "booktitle",
    item.data.proceedingsTitle
      ? bibText(item.data.proceedingsTitle, 1_000)
      : item.data.bookTitle
        ? bibText(item.data.bookTitle, 1_000)
        : undefined,
  );
  put("publisher", item.data.publisher ? bibText(item.data.publisher, 1_000) : undefined);
  put("institution", item.data.institution ? bibText(item.data.institution, 1_000) : undefined);
  put("school", item.data.university ? bibText(item.data.university, 1_000) : undefined);
  put("volume", item.data.volume ? bibText(item.data.volume, 200) : undefined);
  put("number", item.data.issue ? bibText(item.data.issue, 200) : undefined);
  put("pages", item.data.pages ? bibText(item.data.pages, 200) : undefined);
  put("edition", item.data.edition ? bibText(item.data.edition, 200) : undefined);
  put("language", item.data.language ? bibText(item.data.language, 200) : undefined);
  put("abstract", item.data.abstractNote ? bibText(item.data.abstractNote, 4_000) : undefined);
  put(
    "keywords",
    item.data.tags?.length
      ? item.data.tags
          .map((tag) => bibText(tag.tag, 500))
          .filter(Boolean)
          .join(", ")
      : undefined,
  );
  put("doi", item.data.DOI?.trim().toLocaleLowerCase("en-US"));
  put("isbn", item.data.ISBN ? bibText(item.data.ISBN, 500) : undefined);
  put("issn", item.data.ISSN ? bibText(item.data.ISSN, 500) : undefined);
  put("url", safeUrl(item.data.url));
  put("x-source-version", sourceVersion ?? undefined);
  return { type, key, fields };
}

export class ZoteroBibliographySource implements BibliographySource {
  readonly id = "zotero" as const;
  private readonly baseUrl: URL;
  private readonly fetch: typeof globalThis.fetch;
  private readonly maxResponseBytes: number;
  private readonly timeoutMs: number;

  constructor(options: ZoteroSourceOptions = {}) {
    this.baseUrl = loopbackBaseUrl(options.baseUrl ?? DEFAULT_ZOTERO_BASE_URL);
    this.fetch = options.fetch ?? globalThis.fetch;
    this.maxResponseBytes = options.maxResponseBytes ?? MAX_RESPONSE_BYTES;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  async search(
    request: LiteratureSearchRequest,
    signal?: AbortSignal,
  ): Promise<BibliographyExport> {
    abortIfRequested(signal);
    const parts = queryParts(request.query);
    const responses = await Promise.all(parts.map((part) => this.query(part, signal)));
    const sourceVersion = newestVersion(responses.map((response) => response.version));
    const candidates = new Map<string, { hits: number; item: ZoteroItem }>();
    for (const response of responses) {
      for (const item of response.items) {
        const current = candidates.get(item.key);
        if (current) {
          candidates.set(item.key, { item: current.item, hits: current.hits + 1 });
        } else if (candidates.size < MAX_CANDIDATES) {
          candidates.set(item.key, { item, hits: 1 });
        }
      }
    }
    const usedKeys = new Set<string>();
    const entries = [...candidates.values()]
      .sort((left, right) => left.item.key.localeCompare(right.item.key, "en"))
      .map(({ hits, item }) => itemEntry(item, sourceVersion, hits, usedKeys))
      .filter((entry): entry is BibEntry => entry !== null);
    return { source: "zotero", sourceVersion, entries };
  }

  private async query(
    query: string,
    signal?: AbortSignal,
  ): Promise<{ readonly items: readonly ZoteroItem[]; readonly version: string | null }> {
    const url = new URL("api/users/0/items/top", this.baseUrl);
    url.searchParams.set("q", query);
    url.searchParams.set("qmode", "everything");
    url.searchParams.set("limit", String(MAX_ZOTERO_RESULTS_PER_QUERY));
    url.searchParams.set("sort", "title");
    url.searchParams.set("direction", "asc");
    const timeout = AbortSignal.timeout(this.timeoutMs);
    const requestSignal = signal ? AbortSignal.any([signal, timeout]) : timeout;
    let response: Response;
    try {
      response = await this.fetch(url, {
        headers: { Accept: "application/json", "Zotero-API-Version": "3" },
        signal: requestSignal,
      });
    } catch (error) {
      abortIfRequested(signal);
      throw new ScienceError(
        "Local Zotero is unavailable; open Zotero and enable 'Allow other applications on this computer to communicate with Zotero'",
        "LITERATURE_SOURCE_UNAVAILABLE",
        { cause: error },
      );
    }
    if (!response.ok) {
      throw new ScienceError(
        response.status === 403
          ? "Zotero local API is disabled; enable 'Allow other applications on this computer to communicate with Zotero'"
          : `Local Zotero returned HTTP ${response.status}`,
        "LITERATURE_SOURCE_UNAVAILABLE",
      );
    }
    try {
      const text = await boundedText(response, this.maxResponseBytes);
      const items = zoteroItemsSchema.parse(JSON.parse(text));
      return { items, version: response.headers.get("last-modified-version") };
    } catch (error) {
      if (error instanceof ScienceError) throw error;
      throw new ScienceError(
        "Local Zotero returned invalid bibliography data",
        "LITERATURE_SOURCE_UNAVAILABLE",
        {
          cause: error,
        },
      );
    }
  }
}

function displayText(value: string | undefined): string | null {
  if (!value) return null;
  return value
    .replaceAll(/\\([&%#_$}{])/gu, "$1")
    .replaceAll(/[{}]/gu, "")
    .replaceAll(/\s+/gu, " ")
    .trim();
}

function displayAuthors(value: string | undefined): string[] {
  if (!value) return [];
  return value
    .split(/\s+and\s+/iu)
    .map((author) => displayText(author) ?? "")
    .map((author) => {
      const comma = author.indexOf(",");
      return comma === -1
        ? author
        : `${author.slice(comma + 1).trim()} ${author.slice(0, comma).trim()}`.trim();
    })
    .filter(Boolean)
    .slice(0, 100);
}

function normalizedSearch(value: string): string {
  return (displayText(value) ?? "").toLocaleLowerCase("en-US");
}

function resultFor(entry: BibEntry, query: string) {
  const title = displayText(entry.fields.title) ?? entry.key;
  const authors = displayAuthors(entry.fields.author ?? entry.fields.editor);
  const abstract = displayText(entry.fields.abstract)?.slice(0, 4_000) ?? null;
  const keywords = (displayText(entry.fields.keywords) ?? "")
    .split(/[,;]/u)
    .map((keyword) => keyword.trim())
    .filter(Boolean)
    .slice(0, 100);
  const venue =
    displayText(
      entry.fields.journal ??
        entry.fields.booktitle ??
        entry.fields.publisher ??
        entry.fields.institution ??
        entry.fields.school,
    ) ?? null;
  const doi = displayText(entry.fields.doi)?.toLocaleLowerCase("en-US") ?? null;
  const url = safeUrl(displayText(entry.fields.url) ?? undefined) ?? null;
  const parsedYear = Number(entry.fields.year?.match(/\d{4}/u)?.[0]);
  const yearValue =
    Number.isInteger(parsedYear) && parsedYear >= 1000 && parsedYear <= 3000 ? parsedYear : null;
  const fields: Array<[LiteratureMatchField, string, number]> = [
    ["title", title, 20],
    ["authors", authors.join(" "), 8],
    ["abstract", abstract ?? "", 4],
    ["keywords", keywords.join(" "), 10],
    ["venue", venue ?? "", 6],
    ["identifier", `${entry.key} ${doi ?? ""}`, 12],
  ];
  const normalizedQuery = normalizedSearch(query);
  const tokens = queryParts(query).slice(1).map(normalizedSearch).filter(Boolean);
  if (tokens.length === 0) tokens.push(normalizedQuery);
  const matchedFields: LiteratureMatchField[] = [];
  let score = Math.max(0, Number(entry.fields["x-search-hits"] ?? 0)) * 3;
  for (const [name, text, weight] of fields) {
    const normalized = normalizedSearch(text);
    let matched = false;
    if (normalizedQuery && normalized.includes(normalizedQuery)) {
      score += name === "title" ? 100 : weight * 2;
      matched = true;
    }
    for (const token of tokens) {
      if (!token || !normalized.includes(token)) continue;
      score += weight;
      matched = true;
    }
    if (matched) matchedFields.push(name);
  }
  return {
    citationKey: entry.key,
    sourceItemKey: entry.fields["x-source-id"] ?? "",
    entryType: entry.type,
    title,
    authors,
    year: yearValue,
    venue,
    doi,
    url,
    abstract,
    keywords,
    score: Math.round(score),
    matchedFields,
    bibtex: serializeBibFile([entry]),
  };
}

function matchesFilters(
  work: ReturnType<typeof resultFor>,
  filters: LiteratureSearchRequest["filters"],
): boolean {
  if (filters?.entryTypes && !filters.entryTypes.includes(work.entryType)) return false;
  if (
    filters?.years?.from !== undefined &&
    (work.year === null || work.year < filters.years.from)
  ) {
    return false;
  }
  if (filters?.years?.to !== undefined && (work.year === null || work.year > filters.years.to)) {
    return false;
  }
  return true;
}

export class LiteratureSearchRuntime {
  private readonly root: string;

  constructor(
    root: string,
    private readonly source: BibliographySource,
  ) {
    this.root = resolve(root);
  }

  async search(
    workspaceKey: string,
    request: LiteratureSearchRequest,
    signal?: AbortSignal,
  ): Promise<LiteratureSearchResult> {
    abortIfRequested(signal);
    if (!/^[A-Za-z0-9_-]{1,128}$/u.test(workspaceKey)) {
      throw new ScienceError("Literature workspace identity is invalid", "WORKSPACE_UNAVAILABLE");
    }
    const parsedRequest = parseRequest(request);
    const exported = await this.source.search(parsedRequest, signal);
    abortIfRequested(signal);
    const serialized = serializeBibFile(exported.entries);
    const entries = this.writeThenRead(workspaceKey, serialized);
    abortIfRequested(signal);
    const works = entries
      .map((entry) => resultFor(entry, parsedRequest.query))
      .filter((work) => matchesFilters(work, parsedRequest.filters))
      .sort(
        (left, right) =>
          right.score - left.score ||
          (right.year ?? 0) - (left.year ?? 0) ||
          left.title.localeCompare(right.title, "en"),
      );
    const result = {
      source: this.source.id,
      ranking: "zotero-local-v1" as const,
      query: parsedRequest.query,
      totalCandidates: works.length,
      snapshot: {
        source: this.source.id,
        format: "bibtex" as const,
        digest: `sha256:${createHash("sha256").update(serialized).digest("hex")}` as const,
        entryCount: entries.length,
        sourceVersion: exported.sourceVersion,
      },
      results: works.slice(0, parsedRequest.limit),
    };
    return literatureSearchResultSchema.parse(result);
  }

  private writeThenRead(workspaceKey: string, serialized: string): BibEntry[] {
    const directory = join(this.root, "literature", workspaceKey);
    mkdirSync(directory, { recursive: true, mode: 0o700 });
    chmodSync(directory, 0o700);
    const temporary = join(directory, `.zotero-${randomUUID()}.bib.tmp`);
    const stable = join(directory, "zotero.bib");
    try {
      writeFileSync(temporary, serialized, { encoding: "utf8", flag: "wx", mode: 0o600 });
      const entries = parseBibFile(readFileSync(temporary, "utf8"));
      renameSync(temporary, stable);
      return entries;
    } finally {
      rmSync(temporary, { force: true });
    }
  }
}
