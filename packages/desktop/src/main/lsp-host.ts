import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import type { Dirent } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  type ExtensionInventory,
  type LspCapability,
  SWARMX_LOCAL_FILES_LSP_ID,
  SWARMX_SKILLS_LSP_ID,
  type SkillCapability,
} from "@swarmx/core";

const DEFAULT_LSP_TIMEOUT_MS = 15_000;
const SHUTDOWN_TIMEOUT_MS = 3_000;
const EXIT_TIMEOUT_MS = 1_000;
const STDERR_LIMIT_BYTES = 8 * 1024;
const LOCAL_FILE_COMPLETION_LIMIT = 100;
const SKILL_COMPLETION_LIMIT = 100;
const FILE_COMPLETION_ITEM_KIND = 17;
const FOLDER_COMPLETION_ITEM_KIND = 19;
const SKILL_COMPLETION_ITEM_KIND = 18;

export interface LspTextPosition {
  line: number;
  character: number;
}

export interface LspCompletionRequest {
  serverId: string;
  workspaceRoot: string;
  text: string;
  position: LspTextPosition;
  documentUri?: string;
  languageId?: string;
  triggerCharacter?: string;
  timeoutMs?: number;
}

export interface LspCompletionResponse {
  serverId: string;
  status: "ok";
  result: unknown;
}

export interface LspStopRequest {
  serverId: string;
  workspaceRoot?: string;
}

export interface LspStopResponse {
  serverId: string;
  stopped: boolean;
}

interface LspCommand {
  program: string;
  args: string[];
  cwd: string;
}

type JsonRpcId = number | string | null;

interface JsonRpcMessage {
  jsonrpc: "2.0";
  id?: JsonRpcId;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

interface PendingRequest {
  method: string;
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
}

export class LspHost {
  private readonly sessions = new Map<string, LspSession>();

  async complete(
    inventory: ExtensionInventory,
    request: LspCompletionRequest,
  ): Promise<LspCompletionResponse> {
    validateCompletionRequest(request);
    const workspaceRoot = await normalizeWorkspaceRoot(request.workspaceRoot);
    const server = resolveLspServer(inventory, request.serverId);
    if (server.id === SWARMX_LOCAL_FILES_LSP_ID) {
      const result = await completeLocalFiles(workspaceRoot, request);
      return { serverId: server.id, status: "ok", result };
    }
    if (server.id === SWARMX_SKILLS_LSP_ID) {
      const result = completeSkills(inventory, request);
      return { serverId: server.id, status: "ok", result };
    }

    const key = sessionKey(server.id, workspaceRoot);
    const existing = this.sessions.get(key);
    const session = existing?.isAlive() ? existing : this.startSession(server, workspaceRoot, key);

    const result = await session.complete(request);
    return { serverId: server.id, status: "ok", result };
  }

  async stop(request: LspStopRequest): Promise<LspStopResponse> {
    if (!request.serverId?.trim()) {
      throw new Error("LSP stop requires serverId.");
    }

    const workspaceRoot = request.workspaceRoot
      ? await normalizeWorkspaceRoot(request.workspaceRoot)
      : undefined;
    const keys = workspaceRoot
      ? [sessionKey(request.serverId, workspaceRoot)]
      : [...this.sessions.keys()].filter((key) => key.startsWith(`${request.serverId}\0`));

    let stopped = false;
    for (const key of keys) {
      const session = this.sessions.get(key);
      if (!session) continue;
      stopped = true;
      this.sessions.delete(key);
      await session.stop();
    }

    return { serverId: request.serverId, stopped };
  }

  private startSession(server: LspCapability, workspaceRoot: string, key: string): LspSession {
    const command = resolveLspCommand(server, workspaceRoot);
    const child = spawn(command.program, command.args, {
      cwd: command.cwd,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const connection = new JsonRpcConnection(server.id, child);
    const session = new LspSession(server, workspaceRoot, connection);
    this.sessions.set(key, session);
    connection.onClosed(() => {
      if (this.sessions.get(key) === session) {
        this.sessions.delete(key);
      }
    });
    return session;
  }
}

class LspSession {
  private initialized = false;
  private version = 0;
  private readonly openDocuments = new Set<string>();

  constructor(
    private readonly server: LspCapability,
    private readonly workspaceRoot: string,
    private readonly connection: JsonRpcConnection,
  ) {}

  isAlive(): boolean {
    return this.connection.isAlive();
  }

  async complete(request: LspCompletionRequest): Promise<unknown> {
    const timeoutMs = timeoutForRequest(request.timeoutMs);
    await this.initialize(timeoutMs);

    const uri = documentUriFromRequest(request, this.workspaceRoot);
    const languageId =
      request.languageId ??
      this.server.languageIds?.[0] ??
      this.server.languages?.[0] ??
      "plaintext";
    this.version += 1;

    if (this.openDocuments.has(uri)) {
      this.connection.sendNotification("textDocument/didChange", {
        textDocument: { uri, version: this.version },
        contentChanges: [{ text: request.text }],
      });
    } else {
      this.connection.sendNotification("textDocument/didOpen", {
        textDocument: {
          uri,
          languageId,
          version: this.version,
          text: request.text,
        },
      });
      this.openDocuments.add(uri);
    }

    const context = request.triggerCharacter
      ? { triggerKind: 2, triggerCharacter: request.triggerCharacter }
      : { triggerKind: 1 };

    return this.connection.sendRequest(
      "textDocument/completion",
      {
        textDocument: { uri },
        position: request.position,
        context,
      },
      timeoutMs,
    );
  }

  async stop(): Promise<void> {
    if (!this.connection.isAlive()) return;
    try {
      await this.connection.sendRequest("shutdown", null, SHUTDOWN_TIMEOUT_MS);
    } catch {
      // A broken server still needs an exit signal or process kill below.
    }
    if (this.connection.isAlive()) {
      this.connection.sendNotification("exit", {});
    }
    await this.connection.waitForExit(EXIT_TIMEOUT_MS);
    if (this.connection.isAlive()) {
      this.connection.kill();
      await this.connection.waitForExit(EXIT_TIMEOUT_MS);
    }
  }

  private async initialize(timeoutMs: number): Promise<void> {
    if (this.initialized) return;
    const workspaceUri = pathToFileURL(this.workspaceRoot).href;
    await this.connection.sendRequest(
      "initialize",
      {
        processId: process.pid,
        clientInfo: { name: "SwarmX Desktop" },
        rootUri: workspaceUri,
        workspaceFolders: [
          {
            uri: workspaceUri,
            name: path.basename(this.workspaceRoot) || "workspace",
          },
        ],
        capabilities: {
          textDocument: {
            completion: {
              completionItem: { snippetSupport: false },
            },
            synchronization: {
              didSave: false,
              dynamicRegistration: false,
            },
          },
          workspace: {
            configuration: true,
          },
          window: {
            workDoneProgress: false,
          },
        },
      },
      timeoutMs,
    );
    this.connection.sendNotification("initialized", {});
    this.initialized = true;
  }
}

class JsonRpcConnection {
  private buffer = Buffer.alloc(0);
  private nextId = 1;
  private stderr = "";
  private closed = false;
  private closedCallbacks: Array<() => void> = [];
  private readonly pending = new Map<JsonRpcId, PendingRequest>();
  private readonly exitPromise: Promise<void>;
  private resolveExit!: () => void;

  constructor(
    private readonly serverId: string,
    private readonly child: ChildProcessWithoutNullStreams,
  ) {
    this.exitPromise = new Promise((resolve) => {
      this.resolveExit = resolve;
    });
    child.stdout.on("data", (chunk: Buffer) => this.handleStdout(chunk));
    child.stderr.on("data", (chunk: Buffer) => this.appendStderr(chunk));
    child.once("error", (error) => this.close(error));
    child.once("exit", (code, signal) => {
      const reason = new Error(
        `LSP server "${this.serverId}" exited with code ${code ?? "null"} and signal ${
          signal ?? "null"
        }.${this.stderrSummary()}`,
      );
      this.close(reason);
    });
  }

  onClosed(callback: () => void): void {
    this.closedCallbacks.push(callback);
  }

  isAlive(): boolean {
    return !this.closed && this.child.exitCode === null && !this.child.killed;
  }

  sendRequest(method: string, params: unknown, timeoutMs: number): Promise<unknown> {
    if (!this.isAlive()) {
      return Promise.reject(new Error(`LSP server "${this.serverId}" is not running.`));
    }

    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(
          new Error(
            `LSP request "${method}" to server "${this.serverId}" timed out after ${timeoutMs} ms.${this.stderrSummary()}`,
          ),
        );
      }, timeoutMs);
      this.pending.set(id, { method, resolve, reject, timeout });
      this.send({ jsonrpc: "2.0", id, method, params });
    });
  }

  sendNotification(method: string, params: unknown): void {
    if (!this.isAlive()) return;
    this.send({ jsonrpc: "2.0", method, params });
  }

  async waitForExit(timeoutMs: number): Promise<void> {
    if (!this.isAlive()) return;
    await Promise.race([
      this.exitPromise,
      new Promise<void>((resolve) => {
        setTimeout(resolve, timeoutMs);
      }),
    ]);
  }

  kill(): void {
    if (this.isAlive()) {
      this.child.kill();
    }
  }

  private handleStdout(chunk: Buffer): void {
    this.buffer = Buffer.concat([this.buffer, chunk]);

    while (true) {
      const headerEnd = this.buffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) return;

      const header = this.buffer.subarray(0, headerEnd).toString("ascii");
      const lengthMatch = /Content-Length:\s*(\d+)/i.exec(header);
      if (!lengthMatch) {
        this.close(
          new Error(`LSP server "${this.serverId}" sent a message without Content-Length.`),
        );
        return;
      }

      const contentLength = Number.parseInt(lengthMatch[1], 10);
      const bodyStart = headerEnd + 4;
      const bodyEnd = bodyStart + contentLength;
      if (this.buffer.length < bodyEnd) return;

      const body = this.buffer.subarray(bodyStart, bodyEnd).toString("utf8");
      this.buffer = this.buffer.subarray(bodyEnd);
      try {
        this.handleMessage(JSON.parse(body) as JsonRpcMessage);
      } catch (error) {
        this.close(
          new Error(
            `LSP server "${this.serverId}" sent invalid JSON-RPC: ${
              error instanceof Error ? error.message : String(error)
            }.`,
          ),
        );
        return;
      }
    }
  }

  private handleMessage(message: JsonRpcMessage): void {
    if (message.method && hasOwn(message, "id")) {
      this.handleServerRequest(message);
      return;
    }

    if (!hasOwn(message, "id")) return;
    const pending = this.pending.get(message.id ?? null);
    if (!pending) return;

    clearTimeout(pending.timeout);
    this.pending.delete(message.id ?? null);
    if (message.error) {
      pending.reject(
        new Error(
          `LSP request "${pending.method}" to server "${this.serverId}" failed: ${message.error.message}.${this.stderrSummary()}`,
        ),
      );
      return;
    }
    pending.resolve(message.result);
  }

  private handleServerRequest(message: JsonRpcMessage): void {
    let result: unknown = null;
    if (message.method === "workspace/configuration") {
      const params = message.params as { items?: unknown[] } | undefined;
      result = Array.isArray(params?.items) ? params.items.map(() => null) : [];
    }
    this.send({ jsonrpc: "2.0", id: message.id ?? null, result });
  }

  private send(message: JsonRpcMessage): void {
    const body = JSON.stringify(message);
    const header = `Content-Length: ${Buffer.byteLength(body, "utf8")}\r\n\r\n`;
    this.child.stdin.write(`${header}${body}`);
  }

  private appendStderr(chunk: Buffer): void {
    this.stderr = `${this.stderr}${chunk.toString("utf8")}`;
    if (Buffer.byteLength(this.stderr, "utf8") > STDERR_LIMIT_BYTES) {
      this.stderr = this.stderr.slice(-STDERR_LIMIT_BYTES);
    }
  }

  private close(reason: Error): void {
    if (this.closed) return;
    this.closed = true;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timeout);
      pending.reject(reason);
    }
    this.pending.clear();
    for (const callback of this.closedCallbacks) {
      callback();
    }
    this.resolveExit();
  }

  private stderrSummary(): string {
    const trimmed = this.stderr.trim();
    return trimmed ? ` Stderr: ${trimmed}` : "";
  }
}

function resolveLspServer(inventory: ExtensionInventory, serverId: string): LspCapability {
  const id = serverId.trim();
  if (!id) throw new Error("LSP completion requires serverId.");
  const matches = inventory.lspServers.filter((server) => server.id === id);
  if (matches.length === 0) throw new Error(`Unknown LSP server "${id}".`);
  if (matches.length > 1) throw new Error(`Ambiguous LSP server "${id}".`);
  return matches[0];
}

function resolveLspCommand(server: LspCapability, workspaceRoot: string): LspCommand {
  if (!server.command) {
    throw new Error(`LSP server "${server.id}" does not declare a command.`);
  }

  const parts = Array.isArray(server.command) ? server.command : [server.command, ...server.args];
  const [program, ...args] = parts;
  if (!program) {
    throw new Error(`LSP server "${server.id}" declares an empty command.`);
  }

  return {
    program,
    args,
    cwd: server.cwd ? path.resolve(workspaceRoot, server.cwd) : workspaceRoot,
  };
}

async function completeLocalFiles(
  workspaceRoot: string,
  request: LspCompletionRequest,
): Promise<unknown> {
  const context = localFileCompletionContext(request.text, request.position);
  if (!context) return { isIncomplete: false, items: [] };

  const listing = localFileListingPath(workspaceRoot, context.pathPrefix);
  if (!listing) return { isIncomplete: false, items: [] };

  let entries: Dirent<string>[];
  try {
    entries = await readdir(listing.directoryPath, { withFileTypes: true });
  } catch {
    return { isIncomplete: false, items: [] };
  }

  const items = entries
    .filter((entry) => entry.isDirectory() || entry.isFile())
    .filter((entry) => entry.name.startsWith(listing.basenamePrefix))
    .filter((entry) => listing.basenamePrefix.startsWith(".") || !entry.name.startsWith("."))
    .sort((left, right) => {
      if (left.isDirectory() !== right.isDirectory()) return left.isDirectory() ? -1 : 1;
      return left.name.localeCompare(right.name);
    })
    .slice(0, LOCAL_FILE_COMPLETION_LIMIT)
    .map((entry) => {
      const relativePath = joinPosixPath(listing.relativeDirectory, entry.name);
      const pathText = entry.isDirectory() ? `${relativePath}/` : relativePath;
      const insertText = `@${pathText}`;
      return {
        label: insertText,
        kind: entry.isDirectory() ? FOLDER_COMPLETION_ITEM_KIND : FILE_COMPLETION_ITEM_KIND,
        detail: entry.isDirectory() ? "Workspace folder" : "Workspace file",
        sortText: `${entry.isDirectory() ? "0" : "1"}:${pathText}`,
        filterText: insertText,
        insertText,
        textEdit: {
          range: {
            start: context.rangeStart,
            end: request.position,
          },
          newText: insertText,
        },
        data: {
          kind: "local_file",
          path: pathText,
          workspaceRoot,
          syntax: "file",
        },
      };
    });

  return {
    isIncomplete: items.length >= LOCAL_FILE_COMPLETION_LIMIT,
    items,
  };
}

function completeSkills(inventory: ExtensionInventory, request: LspCompletionRequest): unknown {
  const context = prefixCompletionContext(request.text, request.position, "$");
  if (!context) return { isIncomplete: false, items: [] };

  const query = context.query.toLowerCase();
  const items = inventory.skills
    .filter((skill) => skillMatchesQuery(skill, query))
    .sort((left, right) => left.id.localeCompare(right.id))
    .slice(0, SKILL_COMPLETION_LIMIT)
    .map((skill) => {
      const insertText = `$${skill.id}`;
      return {
        label: insertText,
        kind: SKILL_COMPLETION_ITEM_KIND,
        detail: skill.name ? `Skill ${skill.name}` : "Skill",
        documentation: skill.description,
        sortText: skill.id,
        filterText: [insertText, skill.name].filter(Boolean).join(" "),
        insertText,
        textEdit: {
          range: {
            start: context.rangeStart,
            end: request.position,
          },
          newText: insertText,
        },
        data: {
          kind: "skill",
          skillId: skill.id,
          name: skill.name,
          path: skill.path,
          canonicalPath: skill.canonicalPath,
          governanceRef: skill.governanceRef,
          readOnly: skill.readOnly,
          sourcePluginId: skill.sourcePluginId,
        },
      };
    });

  return {
    isIncomplete: items.length >= SKILL_COMPLETION_LIMIT,
    items,
  };
}

function skillMatchesQuery(skill: SkillCapability, query: string): boolean {
  if (!query) return true;
  return skillSearchTerms(skill).some((term) => term.toLowerCase().includes(query));
}

function skillSearchTerms(skill: SkillCapability): string[] {
  return [
    skill.id,
    skill.id.split(".").at(-1) ?? skill.id,
    skill.name,
    skill.path,
    skill.canonicalPath,
  ].filter((term): term is string => Boolean(term));
}

interface LocalFileCompletionContext {
  pathPrefix: string;
  rangeStart: LspTextPosition;
}

function localFileCompletionContext(
  text: string,
  position: LspTextPosition,
): LocalFileCompletionContext | null {
  const context = prefixCompletionContext(text, position, "@");
  if (!context) return null;

  const rawPrefix = context.query;
  if (!rawPrefix) return null;
  if (/^[a-z][a-z0-9+.-]*:/i.test(rawPrefix)) return null;
  return {
    pathPrefix: rawPrefix,
    rangeStart: context.rangeStart,
  };
}

interface PrefixCompletionContext {
  query: string;
  rangeStart: LspTextPosition;
}

function prefixCompletionContext(
  text: string,
  position: LspTextPosition,
  prefix: "@" | "$",
): PrefixCompletionContext | null {
  const cursorOffset = offsetAtPosition(text, position);
  const beforeCursor = text.slice(0, cursorOffset);
  const tokenStart =
    Math.max(
      beforeCursor.lastIndexOf(" "),
      beforeCursor.lastIndexOf("\n"),
      beforeCursor.lastIndexOf("\t"),
      beforeCursor.lastIndexOf("("),
      beforeCursor.lastIndexOf("["),
      beforeCursor.lastIndexOf("{"),
      beforeCursor.lastIndexOf(","),
    ) + 1;
  const token = beforeCursor.slice(tokenStart);
  if (!token.startsWith(prefix)) return null;

  return {
    query: token.slice(prefix.length),
    rangeStart: positionAtOffset(text, tokenStart),
  };
}

interface LocalFileListingPath {
  directoryPath: string;
  relativeDirectory: string;
  basenamePrefix: string;
}

function localFileListingPath(
  workspaceRoot: string,
  pathPrefix: string,
): LocalFileListingPath | null {
  const normalizedPrefix = normalizeLocalFilePrefix(pathPrefix);
  if (normalizedPrefix === null) return null;

  const slashIndex = normalizedPrefix.lastIndexOf("/");
  const relativeDirectory = slashIndex >= 0 ? normalizedPrefix.slice(0, slashIndex) : "";
  const basenamePrefix =
    slashIndex >= 0 ? normalizedPrefix.slice(slashIndex + 1) : normalizedPrefix;
  const directoryPath = path.resolve(workspaceRoot, relativeDirectory);

  if (!isPathInsideWorkspace(workspaceRoot, directoryPath)) return null;
  return { directoryPath, relativeDirectory, basenamePrefix };
}

function normalizeLocalFilePrefix(pathPrefix: string): string | null {
  const withoutDotSlash = pathPrefix.startsWith("./") ? pathPrefix.slice(2) : pathPrefix;
  if (withoutDotSlash.startsWith("/") || withoutDotSlash.includes("\\")) return null;
  const parts = withoutDotSlash.split("/").filter((part) => part.length > 0);
  if (parts.some((part) => part === "." || part === "..")) return null;
  return withoutDotSlash;
}

function isPathInsideWorkspace(workspaceRoot: string, candidate: string): boolean {
  const relative = path.relative(workspaceRoot, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function joinPosixPath(directory: string, basename: string): string {
  return directory ? `${directory}/${basename}` : basename;
}

function offsetAtPosition(text: string, position: LspTextPosition): number {
  let offset = 0;
  for (let line = 0; line < position.line; line += 1) {
    const nextLine = text.indexOf("\n", offset);
    if (nextLine < 0) {
      throw new Error("LSP completion position.line is outside the document.");
    }
    offset = nextLine + 1;
  }

  const lineEnd = text.indexOf("\n", offset);
  const end = lineEnd < 0 ? text.length : lineEnd;
  if (offset + position.character > end) {
    throw new Error("LSP completion position.character is outside the document line.");
  }
  return offset + position.character;
}

function positionAtOffset(text: string, targetOffset: number): LspTextPosition {
  let line = 0;
  let lineStart = 0;
  while (lineStart < targetOffset) {
    const nextLine = text.indexOf("\n", lineStart);
    if (nextLine < 0 || nextLine >= targetOffset) break;
    line += 1;
    lineStart = nextLine + 1;
  }
  return { line, character: targetOffset - lineStart };
}

async function normalizeWorkspaceRoot(workspaceRoot: string): Promise<string> {
  if (!workspaceRoot?.trim()) {
    throw new Error("LSP request requires workspaceRoot.");
  }
  const resolved = path.resolve(workspaceRoot);
  const workspaceStat = await stat(resolved);
  if (!workspaceStat.isDirectory()) {
    throw new Error(`LSP workspaceRoot must be a directory: ${resolved}`);
  }
  return resolved;
}

function validateCompletionRequest(request: LspCompletionRequest): void {
  if (!request.serverId?.trim()) throw new Error("LSP completion requires serverId.");
  if (typeof request.text !== "string") throw new Error("LSP completion requires document text.");
  if (!request.position || !Number.isInteger(request.position.line) || request.position.line < 0) {
    throw new Error("LSP completion requires a non-negative integer position.line.");
  }
  if (!Number.isInteger(request.position.character) || request.position.character < 0) {
    throw new Error("LSP completion requires a non-negative integer position.character.");
  }
}

function timeoutForRequest(timeoutMs: number | undefined): number {
  if (timeoutMs === undefined) return DEFAULT_LSP_TIMEOUT_MS;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    throw new Error("LSP timeoutMs must be a positive number.");
  }
  return Math.min(timeoutMs, DEFAULT_LSP_TIMEOUT_MS);
}

function documentUriFromRequest(request: LspCompletionRequest, workspaceRoot: string): string {
  const requested = request.documentUri?.trim();
  if (!requested) {
    return pathToFileURL(path.join(workspaceRoot, ".swarmx-lsp-buffer")).href;
  }
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(requested)) {
    return requested;
  }
  const filePath = path.isAbsolute(requested) ? requested : path.resolve(workspaceRoot, requested);
  return pathToFileURL(filePath).href;
}

function sessionKey(serverId: string, workspaceRoot: string): string {
  return `${serverId}\0${workspaceRoot}`;
}

function hasOwn(value: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}
