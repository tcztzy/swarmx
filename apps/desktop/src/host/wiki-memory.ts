import { randomUUID } from "node:crypto";
import { realpath } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { WikiMemoryClient, type WikiMemorySearchResult } from "@swarmx/memory";
import { z } from "zod";
import manifest from "../../package.json" with { type: "json" };

const AbsolutePath = z
  .string()
  .min(1)
  .max(4_096)
  .refine((value) => isAbsolute(value) && !value.includes("\0"));
const OptionsSchema = z.strictObject({
  command: AbsolutePath,
  brain: AbsolutePath,
  scopes: z.array(AbsolutePath).min(1).max(64),
  roots: z.array(AbsolutePath).min(1).max(64),
  env: z.record(z.string().regex(/^[A-Za-z_][A-Za-z0-9_]*$/u), z.string().max(32_768)).optional(),
});
export type HostWikiMemoryOptions = z.input<typeof OptionsSchema>;
export const WikiSearchCallSchema = z.strictObject({
  action: z.literal("search_wiki_memory"),
  request: z.strictObject({
    query: z.string().trim().min(1).max(1_000),
    maxResults: z.number().int().min(1).max(20).default(8),
    maxChars: z.number().int().min(80).max(2_000).default(600),
    sections: z
      .array(z.enum(["frontmatter", "body"]))
      .min(1)
      .max(2)
      .optional(),
  }),
});
type SearchRequest = z.output<typeof WikiSearchCallSchema>["request"];
type Diagnostic = { code: string; message: string; sourceId?: string };
const DEADLINE_MS = 15_000;

function excerpt(text: string, limit: number): string {
  const clipped = text.slice(0, limit);
  return /[\uD800-\uDBFF]$/u.test(clipped) ? clipped.slice(0, -1) : clipped;
}

function privateText(text: string, limit: number): string {
  return excerpt(
    text
      .replace(/(["'`])((?:file:\/\/|[A-Za-z]:[\\/]|[\\/])[\s\S]*?)\1/giu, "$1[path withheld]$1")
      .replace(/(?:file:\/\/|[A-Za-z]:[\\/]|[\\/])[^\r\n"'<>`]*/giu, "[path withheld]"),
    limit,
  );
}

function relativeId(value: string): boolean {
  return (
    value.length > 0 &&
    value.length <= 1_024 &&
    !/[\\:%]/u.test(value) &&
    [...value].every(
      (character) => character.charCodeAt(0) >= 32 && character.charCodeAt(0) !== 127,
    ) &&
    !isAbsolute(value) &&
    value.split("/").every((part) => part.length > 0 && part !== "." && part !== "..")
  );
}

export function unavailableWikiSearch(status: "disabled" | "unconfigured") {
  return {
    status,
    records: [],
    totalRecords: 0,
    truncated: false,
    partial: false,
    diagnostics: [],
  };
}

/** The trusted Host owns this connection; Agent requests contain no launch or root choices. */
export class HostWikiMemory {
  private readonly sources: Map<string, string>;
  private readonly shutdown = new AbortController();
  private connection:
    | { sdk: Client; transport: StdioClientTransport; ready: Promise<void> }
    | undefined;
  private closing?: Promise<void>;

  private constructor(
    private readonly options: z.output<typeof OptionsSchema>,
    private readonly cwd: string,
    private readonly brainRoot: string,
  ) {
    this.sources = new Map(
      options.roots.map((root) => [root, `urn:swarmx:wiki-source:${randomUUID()}`]),
    );
  }

  static async create(raw: HostWikiMemoryOptions, cwd: string): Promise<HostWikiMemory> {
    const options = OptionsSchema.parse(raw);
    try {
      const [command, brain, scopes, roots] = await Promise.all([
        realpath(options.command),
        realpath(options.brain),
        Promise.all(options.scopes.map((path) => realpath(path))),
        Promise.all(options.roots.map((path) => realpath(path))),
      ]);
      const brainRoot = await realpath(join(brain, "wiki"));
      if (!roots.includes(brainRoot)) throw new Error("Missing approved brain root.");
      return new HostWikiMemory(
        { ...options, command, brain, scopes, roots: [...new Set(roots)] },
        cwd,
        brainRoot,
      );
    } catch {
      throw new Error(
        "Wiki memory configuration requires existing paths and an approved brain/wiki root.",
      );
    }
  }

  async search(request: SearchRequest, signal: AbortSignal, enabled: () => boolean) {
    try {
      signal.throwIfAborted();
      const connection = this.connect();
      await this.waitForConnection(connection.ready, signal);
      signal.throwIfAborted();
      this.shutdown.signal.throwIfAborted();
      if (!enabled()) return unavailableWikiSearch("disabled");
      const client = new WikiMemoryClient(
        { scopes: this.options.scopes },
        {
          callTool: (call, operationSignal) => {
            operationSignal.throwIfAborted();
            this.shutdown.signal.throwIfAborted();
            if (call.name !== "search_memory")
              throw new Error("Wiki Host connection is read-only.");
            return connection.sdk.callTool(call, undefined, {
              signal: operationSignal,
              timeout: DEADLINE_MS,
            });
          },
        },
      );
      const result = await client.search(request, signal);
      signal.throwIfAborted();
      return this.project(result, request);
    } catch {
      throw new Error(
        signal.aborted || this.shutdown.signal.aborted
          ? "Wiki memory search was cancelled."
          : "Wiki memory search is unavailable or returned an invalid response.",
      );
    }
  }

  private connect() {
    this.shutdown.signal.throwIfAborted();
    if (this.connection) return this.connection;
    const sdk = new Client(
      { name: "swarmx-wiki-search", version: manifest.version },
      { capabilities: {} },
    );
    const transport = new StdioClientTransport({
      command: this.options.command,
      args: ["mcp"],
      cwd: this.cwd,
      env: {
        ...this.options.env,
        MEMORY_DATA_DIR: this.options.brain,
        MEMORY_WORKSPACE_DIR: this.cwd,
      },
      stderr: "ignore",
      maxBufferSize: 2 * 1024 * 1024,
    });
    const start = transport.start.bind(transport);
    transport.start = async () => {
      this.shutdown.signal.throwIfAborted();
      await start();
      if (this.shutdown.signal.aborted) {
        await transport.close();
        this.shutdown.signal.throwIfAborted();
      }
    };
    const send = transport.send.bind(transport);
    transport.send = async (message) => {
      this.shutdown.signal.throwIfAborted();
      return send(message);
    };
    sdk.onclose = () => {
      if (this.connection?.sdk === sdk) this.connection = undefined;
    };
    const ready = sdk.connect(transport, { signal: this.shutdown.signal, timeout: DEADLINE_MS });
    this.connection = { sdk, transport, ready };
    // Initialization can outlive a cancelled waiter. Keep rejection observed until shutdown.
    void ready
      .catch(async () => {
        await transport.close();
        await sdk.close();
        if (this.connection?.sdk === sdk) this.connection = undefined;
      })
      .catch(() => {});
    return this.connection;
  }

  private async waitForConnection(ready: Promise<void>, signal: AbortSignal) {
    signal.throwIfAborted();
    let abort: (() => void) | undefined;
    const cancelled = new Promise<never>((_resolve, reject) => {
      abort = () => reject(new Error("Wiki memory search was cancelled."));
      signal.addEventListener("abort", abort, { once: true });
    });
    try {
      await Promise.race([ready, cancelled]);
    } finally {
      if (abort) signal.removeEventListener("abort", abort);
    }
  }

  private project(result: WikiMemorySearchResult, request: SearchRequest) {
    const diagnostics: Diagnostic[] = [];
    let diagnosticsOmitted = false;
    const diagnose = (code: string, message: string, sourceId?: string) => {
      const entry: Diagnostic = {
        code,
        message: privateText(message, 180),
        ...(sourceId ? { sourceId } : {}),
      };
      if (
        diagnostics.length >= 15 ||
        Buffer.byteLength(JSON.stringify([...diagnostics, entry])) > 3_000
      )
        diagnosticsOmitted = true;
      else diagnostics.push(entry);
    };
    let contentBudget = 16_000;
    let omitted = 0;
    let truncated = result.truncated === true;
    const records: {
      sourceId: string;
      datasetId: string;
      documentId: string;
      documentName: string;
      score: number;
      priority: string;
      content?: string;
      truncated: boolean;
      fullChars?: number;
    }[] = [];
    for (const record of result.records) {
      const origin =
        typeof record.resolvedRoot === "string" ? resolve(record.resolvedRoot) : this.brainRoot;
      const sourceId = this.sources.get(origin);
      if (
        !sourceId ||
        !relativeId(record.documentId) ||
        (record.resolvedRoot !== undefined &&
          (typeof record.resolvedRoot !== "string" || !isAbsolute(record.resolvedRoot)))
      ) {
        omitted++;
        diagnose(
          "origin-omitted",
          "A result with an unapproved origin or invalid document identity was omitted.",
        );
        continue;
      }
      const content =
        record.content === undefined
          ? undefined
          : privateText(record.content, Math.min(request.maxChars, contentBudget));
      const hitTruncated =
        record.truncated === true ||
        (content !== undefined && content.length < (record.content?.length ?? 0));
      const projected = {
        sourceId,
        documentId: record.documentId,
        datasetId: privateText(record.datasetId, 128),
        documentName: privateText(record.documentName, 180),
        score: record.score,
        priority: privateText(record.priority, 32),
        ...(content === undefined ? {} : { content }),
        truncated: hitTruncated,
        ...(Number.isSafeInteger(record.fullChars) && (record.fullChars as number) >= 0
          ? { fullChars: record.fullChars as number }
          : {}),
      };
      if (
        records.length >= request.maxResults ||
        Buffer.byteLength(JSON.stringify([...records, projected])) > 27_000
      ) {
        omitted++;
        continue;
      }
      records.push(projected);
      contentBudget -= content?.length ?? 0;
      truncated ||= hitTruncated;
      if (
        projected.datasetId !== record.datasetId ||
        projected.documentName !== record.documentName ||
        projected.priority !== record.priority
      )
        diagnose(
          "metadata-projected",
          "Result metadata was bounded or had absolute paths withheld.",
          sourceId,
        );
      if (content !== undefined && content !== record.content)
        diagnose(
          "excerpt-projected",
          "An excerpt was additionally bounded or had absolute paths withheld.",
          sourceId,
        );
    }
    const nativeErrors = Array.isArray(result.errors) ? result.errors : [];
    for (const error of nativeErrors) {
      const message =
        typeof error === "string"
          ? error
          : typeof error === "object" &&
              error !== null &&
              "message" in error &&
              typeof error.message === "string"
            ? error.message
            : "A native search source reported an error.";
      diagnose("native-search", message);
    }
    const partial =
      result.partial !== undefined && result.partial !== null && result.partial !== false;
    if (partial) diagnose("native-partial", "Native search reported partial retrieval.");
    if (omitted > 0)
      diagnose(
        "records-omitted",
        `${omitted} results were omitted by Host origin, identity or response limits.`,
      );
    if (diagnosticsOmitted)
      diagnostics.push({
        code: "diagnostics-omitted",
        message: "Additional search diagnostics were omitted by Host limits.",
      });
    return {
      status: "available" as const,
      records,
      totalRecords: result.totalRecords,
      truncated: truncated || omitted > 0,
      partial: partial || nativeErrors.length > 0 || omitted > 0 || diagnosticsOmitted,
      diagnostics,
    };
  }

  close(): Promise<void> {
    this.closing ??= this.closeConnection();
    return this.closing;
  }

  private async closeConnection(): Promise<void> {
    this.shutdown.abort(new Error("Wiki memory connection is closing."));
    const connection = this.connection;
    if (!connection) return;
    await connection.transport.close();
    await Promise.allSettled([connection.ready]);
    await connection.sdk.close();
    await connection.transport.close();
  }
}
