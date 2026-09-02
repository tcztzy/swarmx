import { SessionId, SessionSeq } from "@deepseek-ai/dsh-session";
import {
  extractSessionEventText,
  type SessionEventReadRequest,
  type SessionEventResultFilter,
  type SessionEventSearchDocument,
  type SessionEventWindow,
  SessionQueryError,
  type SessionRecord,
  type SessionSearchExecContext,
  type SessionSearchHit,
  type SessionSearchPage,
  type SessionSearchRequest,
} from "@deepseek-ai/dsh-session-query";
import { z } from "zod";
import { PkbError } from "./errors.js";
import type { ConversationExcerpt, PkbVault } from "./vault.js";

const searchSchema = z.object({
  allAuthorized: z.boolean().optional(),
  limit: z.number().int().min(1).max(20).optional(),
  query: z.string().trim().min(1).max(200),
  scope: z.enum(["all", "workspace"]).optional(),
});

const locatorSchema = z.object({
  allAuthorized: z.boolean().optional(),
  seq: z.number().int().nonnegative(),
  sessionId: z.string().trim().min(1).max(1_024),
});

export interface ConversationSessionQuery {
  filterEvents(
    sessionId: SessionId,
    filters: readonly SessionEventResultFilter[],
  ): Promise<SessionEventSearchDocument[]>;
  listSessions(signal?: AbortSignal): Promise<SessionRecord[]>;
  readEvent(request: SessionEventReadRequest, signal?: AbortSignal): Promise<SessionEventWindow>;
  searchSessions(
    request: SessionSearchRequest,
    exec?: SessionSearchExecContext,
  ): Promise<SessionSearchPage<SessionSearchHit>>;
}

export interface ConversationSearchRequest {
  readonly allAuthorized?: boolean;
  readonly limit?: number;
  readonly query: string;
  readonly scope?: "all" | "workspace";
}

export interface ConversationLocator {
  readonly allAuthorized?: boolean;
  readonly seq: number;
  readonly sessionId: string;
}

export interface ConversationSearchItem {
  readonly eventTime: string;
  readonly eventType: string;
  readonly locator: {
    readonly seq: number;
    readonly sessionId: string;
  };
  readonly snippet: string;
  readonly trust: "untrusted-evidence";
}

export interface ConversationSearchResult {
  readonly diagnostics: readonly string[];
  readonly items: readonly ConversationSearchItem[];
}

export interface ConversationEventEvidence extends Omit<ConversationSearchItem, "snippet"> {
  readonly text: string;
}

export interface ConversationArchiveConfig {
  readonly maxEvidenceCharacters?: number;
  readonly maxSessions?: number;
  readonly snippetCharacters?: number;
}

function requestError(message: string, cause?: unknown): PkbError {
  return new PkbError(message, "INVALID_REQUEST", cause === undefined ? undefined : { cause });
}

function parseRequest<T>(schema: { parse(value: unknown): T }, value: unknown): T {
  try {
    return schema.parse(value);
  } catch (error) {
    throw requestError("Invalid conversation archive request", error);
  }
}

function hitKey(hit: { sessionId: SessionId; seq: number }): string {
  return `${hit.sessionId}:${String(hit.seq)}`;
}

export class ConversationArchive {
  private readonly maxEvidenceCharacters: number;
  private readonly maxSessions: number;
  private readonly snippetCharacters: number;

  constructor(
    private readonly vault: PkbVault,
    private readonly query: ConversationSessionQuery,
    config: ConversationArchiveConfig = {},
  ) {
    this.maxEvidenceCharacters = config.maxEvidenceCharacters ?? 16_000;
    this.maxSessions = config.maxSessions ?? 50;
    this.snippetCharacters = config.snippetCharacters ?? 800;
    if (this.maxEvidenceCharacters < 1 || this.maxSessions < 1 || this.snippetCharacters < 1) {
      throw requestError("Conversation archive limits must be positive");
    }
  }

  async search(
    cwd: string,
    rawRequest: ConversationSearchRequest,
    signal?: AbortSignal,
  ): Promise<ConversationSearchResult> {
    const request = parseRequest(searchSchema, rawRequest);
    const scope = request.scope ?? "workspace";
    if (scope === "all" && request.allAuthorized !== true) {
      throw new PkbError(
        "Searching conversations from every workspace requires authorization",
        "AUTHORIZATION_REQUIRED",
      );
    }
    const candidates = await this.candidates(cwd, scope, signal);
    if (candidates.length === 0) return { diagnostics: [], items: [] };

    const diagnostics: string[] = [];
    const hits = new Map<string, SessionEventSearchDocument | SessionSearchHit["bestMatch"]>();
    try {
      const page = await this.query.searchSessions(
        {
          eventFilters: [{ kind: "surface", values: ["current"] }],
          limit: request.limit ?? 20,
          query: request.query,
          sessionFilters: [{ kind: "id", values: candidates.map(({ header }) => header.id) }],
        },
        signal === undefined ? {} : { signal },
      );
      for (const item of page.items) hits.set(hitKey(item.bestMatch), item.bestMatch);
    } catch (error) {
      if (!(error instanceof SessionQueryError) || error.code !== "SESSION_QUERY_SEARCH_DISABLED") {
        throw error;
      }
      diagnostics.push("Full-text index unavailable; used literal scan.");
    }

    const scanned = await Promise.all(
      candidates.map(({ header }) =>
        this.query.filterEvents(header.id, [
          { kind: "surface", values: ["current"] },
          { kind: "text", text: request.query },
        ]),
      ),
    );
    for (const document of scanned.flat()) hits.set(hitKey(document), document);

    const items = Array.from(hits.values())
      .sort((left, right) => right.time - left.time || right.seq - left.seq)
      .slice(0, request.limit ?? 20)
      .map((hit) => this.searchItem(hit));
    return { diagnostics, items };
  }

  async read(
    cwd: string,
    rawLocator: ConversationLocator,
    signal?: AbortSignal,
  ): Promise<ConversationEventEvidence> {
    const locator = parseRequest(locatorSchema, rawLocator);
    const window = await this.authorizedEvent(cwd, locator, signal);
    const text = extractSessionEventText(window.target).trim();
    if (text.length === 0) {
      throw new PkbError("Conversation event has no semantic text", "INVALID_REQUEST");
    }
    return {
      eventTime: new Date(window.target.time).toISOString(),
      eventType: window.target.type,
      locator: { seq: window.target.seq, sessionId: window.session.id },
      text: Array.from(text).slice(0, this.maxEvidenceCharacters).join(""),
      trust: "untrusted-evidence",
    };
  }

  async capture(
    cwd: string,
    rawLocator: ConversationLocator,
    signal?: AbortSignal,
  ): Promise<ConversationExcerpt> {
    const evidence = await this.read(cwd, rawLocator, signal);
    return this.vault.saveConversationExcerpt(
      cwd,
      {
        eventTime: Date.parse(evidence.eventTime),
        eventType: evidence.eventType,
        seq: evidence.locator.seq,
        sessionId: evidence.locator.sessionId,
        text: evidence.text,
      },
      signal,
    );
  }

  private async candidates(
    cwd: string,
    scope: "all" | "workspace",
    signal?: AbortSignal,
  ): Promise<SessionRecord[]> {
    const records = (await this.query.listSessions(signal))
      .slice()
      .sort((left, right) => right.header.createdAt - left.header.createdAt);
    if (scope === "all") return records.slice(0, this.maxSessions);
    const current = await this.vault.resolveWorkspace(cwd);
    const accepted: SessionRecord[] = [];
    for (const record of records) {
      if (record.header.cwd === undefined) continue;
      try {
        const workspace = await this.vault.resolveWorkspace(record.header.cwd);
        if (workspace.key === current.key) accepted.push(record);
      } catch (error) {
        if (error instanceof PkbError && error.code === "WORKSPACE_UNAVAILABLE") continue;
        throw error;
      }
      if (accepted.length >= this.maxSessions) break;
    }
    return accepted;
  }

  private async authorizedEvent(
    cwd: string,
    locator: z.infer<typeof locatorSchema>,
    signal?: AbortSignal,
  ): Promise<SessionEventWindow> {
    const records = await this.query.listSessions(signal);
    const record = records.find(({ header }) => header.id === locator.sessionId);
    if (record === undefined) {
      throw new PkbError("Conversation session not found", "CONCEPT_NOT_FOUND");
    }
    if (locator.allAuthorized !== true) {
      if (record.header.cwd === undefined) {
        throw new PkbError("Conversation has no workspace identity", "UNSAFE_PATH");
      }
      const [current, source] = await Promise.all([
        this.vault.resolveWorkspace(cwd),
        this.vault.resolveWorkspace(record.header.cwd),
      ]);
      if (current.key !== source.key) {
        throw new PkbError("Conversation belongs to another workspace", "UNSAFE_PATH");
      }
    }
    const window = await this.query.readEvent(
      {
        after: 0,
        before: 0,
        seq: SessionSeq(locator.seq),
        sessionId: SessionId(locator.sessionId),
      },
      signal,
    );
    if (window.session.id !== locator.sessionId || window.target.seq !== locator.seq) {
      throw new PkbError("Conversation event locator changed during capture", "IO_ERROR");
    }
    return window;
  }

  private searchItem(
    hit: SessionEventSearchDocument | SessionSearchHit["bestMatch"],
  ): ConversationSearchItem {
    const snippet = "snippet" in hit ? hit.snippet : hit.text;
    return {
      eventTime: new Date(hit.time).toISOString(),
      eventType: hit.type,
      locator: { seq: hit.seq, sessionId: hit.sessionId },
      snippet: Array.from(snippet.trim()).slice(0, this.snippetCharacters).join(""),
      trust: "untrusted-evidence",
    };
  }
}
