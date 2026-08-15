import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import type { Dirent } from "node:fs";
import { readdir, readFile, realpath, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  type ExtensionInventory,
  type LspCapability,
  preflightExtensionComposition,
  type SkillCapability,
  SWARMX_LOCAL_FILES_LSP_ID,
  SWARMX_SKILLS_LSP_ID,
} from "@swarmx/core";
import { currentRequestSignal } from "@swarmx/core/request-scope";
import {
  CancellationTokenSource,
  createMessageConnection,
  type MessageConnection,
  StreamMessageReader,
  StreamMessageWriter,
} from "vscode-jsonrpc/node";

const DEFAULT_LSP_TIMEOUT_MS = 15_000;
const SHUTDOWN_TIMEOUT_MS = 3_000;
const EXIT_TIMEOUT_MS = 1_000;
const STDERR_LIMIT_BYTES = 8 * 1024;
const LOCAL_FILE_COMPLETION_LIMIT = 100;
const SKILL_COMPLETION_LIMIT = 100;
const LSP_DOCUMENT_LIMIT_BYTES = 4 * 1024 * 1024;
const FILE_COMPLETION_ITEM_KIND = 17;
const FOLDER_COMPLETION_ITEM_KIND = 19;
const SKILL_COMPLETION_ITEM_KIND = 18;
const LSP_ENVIRONMENT_KEYS = [
  "PATH",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "TMPDIR",
  "TMP",
  "TEMP",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "LC_MESSAGES",
  "SystemRoot",
  "WINDIR",
  "ComSpec",
  "PATHEXT",
] as const;

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

export type ClaudeLspOperation =
  | "goToDefinition"
  | "findReferences"
  | "hover"
  | "documentSymbol"
  | "workspaceSymbol"
  | "goToImplementation"
  | "prepareCallHierarchy"
  | "incomingCalls"
  | "outgoingCalls";

export interface ClaudeLspRequest {
  operation: ClaudeLspOperation;
  filePath: string;
  line: number;
  character: number;
  query?: string;
}

export interface ClaudeLspResponse {
  operation: ClaudeLspOperation;
  result: string;
  filePath: string;
  resultCount?: number;
  fileCount?: number;
}

interface LspCommand {
  program: string;
  args: string[];
  cwd: string;
}

export class LspHost {
  private readonly sessions = new Map<string, LspSession>();

  supportsClaudeOperations(inventory: ExtensionInventory): boolean {
    return inventory.lspServers.some(
      (server) => isCommandBackedLspServer(server) && isLspServerAdmitted(inventory, server),
    );
  }

  async complete(
    inventory: ExtensionInventory,
    request: LspCompletionRequest,
  ): Promise<LspCompletionResponse> {
    validateCompletionRequest(request);
    const workspaceRoot = await normalizeWorkspaceRoot(request.workspaceRoot);
    const server = resolveLspServer(inventory, request.serverId);
    assertLspServerAdmitted(inventory, server);
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

  async operate(
    inventory: ExtensionInventory,
    workspaceRoot: string,
    request: ClaudeLspRequest,
  ): Promise<ClaudeLspResponse> {
    validateClaudeLspRequest(request);
    const root = await normalizeWorkspaceRoot(workspaceRoot);
    const file = await readContainedLspFile(root, request.filePath);
    const server = resolveLspServerForFile(inventory, file.path);
    assertLspServerAdmitted(inventory, server);
    const key = sessionKey(server.id, root);
    const existing = this.sessions.get(key);
    const session = existing?.isAlive() ? existing : this.startSession(server, root, key);
    const raw = await session.operate(
      request,
      file.path,
      file.text,
      languageIdForPath(file.path),
      currentRequestSignal(),
    );
    return claudeLspResponse(request, raw, root);
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
      env: lspEnvironment(process.env),
      stdio: ["pipe", "pipe", "pipe"],
    });
    const connection = new LspRpcConnection(server.id, child);
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
    private readonly connection: LspRpcConnection,
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
    this.syncDocument(uri, request.text, languageId);

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

  async operate(
    request: ClaudeLspRequest,
    filePath: string,
    text: string,
    languageId: string,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const timeoutMs = DEFAULT_LSP_TIMEOUT_MS;
    await this.initialize(timeoutMs, signal);
    const uri = pathToFileURL(filePath).href;
    this.syncDocument(uri, text, languageId);
    const position = { line: request.line - 1, character: request.character - 1 };
    const textDocument = { uri };
    const positionParams = { textDocument, position };

    if (request.operation === "workspaceSymbol") {
      return this.connection.sendRequest(
        "workspace/symbol",
        { query: request.query },
        timeoutMs,
        signal,
      );
    }
    if (request.operation === "documentSymbol") {
      return this.connection.sendRequest(
        "textDocument/documentSymbol",
        { textDocument },
        timeoutMs,
        signal,
      );
    }
    if (request.operation === "findReferences") {
      return this.connection.sendRequest(
        "textDocument/references",
        { ...positionParams, context: { includeDeclaration: true } },
        timeoutMs,
        signal,
      );
    }
    if (request.operation === "goToDefinition") {
      return this.connection.sendRequest(
        "textDocument/definition",
        positionParams,
        timeoutMs,
        signal,
      );
    }
    if (request.operation === "goToImplementation") {
      return this.connection.sendRequest(
        "textDocument/implementation",
        positionParams,
        timeoutMs,
        signal,
      );
    }
    if (request.operation === "hover") {
      return this.connection.sendRequest("textDocument/hover", positionParams, timeoutMs, signal);
    }

    const prepared = await this.connection.sendRequest(
      "textDocument/prepareCallHierarchy",
      positionParams,
      timeoutMs,
      signal,
    );
    if (request.operation === "prepareCallHierarchy") return prepared;
    const item = Array.isArray(prepared) ? prepared[0] : prepared;
    if (!item) return [];
    return this.connection.sendRequest(
      request.operation === "incomingCalls"
        ? "callHierarchy/incomingCalls"
        : "callHierarchy/outgoingCalls",
      { item },
      timeoutMs,
      signal,
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

  private syncDocument(uri: string, text: string, languageId: string): void {
    this.version += 1;
    if (this.openDocuments.has(uri)) {
      this.connection.sendNotification("textDocument/didChange", {
        textDocument: { uri, version: this.version },
        contentChanges: [{ text }],
      });
      return;
    }
    this.connection.sendNotification("textDocument/didOpen", {
      textDocument: { uri, languageId, version: this.version, text },
    });
    this.openDocuments.add(uri);
  }

  private async initialize(timeoutMs: number, signal?: AbortSignal): Promise<void> {
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
      signal,
    );
    this.connection.sendNotification("initialized", {});
    this.initialized = true;
  }
}

class LspRpcConnection {
  private readonly rpc: MessageConnection;
  private stderr = Buffer.alloc(0);
  private closed = false;
  private failure?: Error;
  private closedCallbacks: Array<() => void> = [];
  private readonly exitPromise: Promise<void>;
  private resolveExit!: () => void;

  constructor(
    private readonly serverId: string,
    private readonly child: ChildProcessWithoutNullStreams,
  ) {
    this.exitPromise = new Promise((resolve) => {
      this.resolveExit = resolve;
    });
    this.rpc = createMessageConnection(
      new StreamMessageReader(child.stdout),
      new StreamMessageWriter(child.stdin),
    );
    this.rpc.onRequest("workspace/configuration", (params: unknown) => {
      const items = isRecord(params) && Array.isArray(params.items) ? params.items : [];
      return items.map(() => null);
    });
    this.rpc.onRequest(() => null);
    this.rpc.onError(([error]) => this.terminate(error));
    this.rpc.onClose(() => {
      this.terminate(new Error(`LSP server "${this.serverId}" closed its JSON-RPC transport.`));
    });
    this.rpc.listen();

    child.stderr.on("data", (chunk: Buffer) => this.appendStderr(chunk));
    child.once("error", (error) => {
      this.terminate(error);
      this.resolveExit();
    });
    child.once("exit", (code, signal) => {
      const reason = new Error(
        `LSP server "${this.serverId}" exited with code ${code ?? "null"} and signal ${
          signal ?? "null"
        }.${this.stderrSummary()}`,
      );
      this.close(reason);
      this.resolveExit();
    });
  }

  onClosed(callback: () => void): void {
    if (this.closed) {
      callback();
      return;
    }
    this.closedCallbacks.push(callback);
  }

  isAlive(): boolean {
    return !this.closed && this.child.exitCode === null && !this.child.killed;
  }

  async sendRequest(
    method: string,
    params: unknown,
    timeoutMs: number,
    signal?: AbortSignal,
  ): Promise<unknown> {
    if (!this.isAlive()) {
      throw new Error(`LSP server "${this.serverId}" is not running.`);
    }
    if (signal?.aborted) {
      throw lspAbortReason(signal);
    }

    const cancellation = new CancellationTokenSource();
    let timeoutError: Error | undefined;
    let timedOut = false;
    let rejectInterruption!: (reason: Error) => void;
    const interrupted = new Promise<never>((_resolve, reject) => {
      rejectInterruption = reject;
    });
    const abort = (): void => {
      const reason = lspAbortReason(signal);
      cancellation.cancel();
      rejectInterruption(reason);
      this.terminate(reason);
    };
    const timeout = setTimeout(() => {
      timedOut = true;
      timeoutError = new Error(
        `LSP request "${method}" to server "${this.serverId}" timed out after ${timeoutMs} ms.${this.stderrSummary()}`,
      );
      cancellation.cancel();
      rejectInterruption(timeoutError);
      this.terminate(timeoutError);
    }, timeoutMs);
    signal?.addEventListener("abort", abort, { once: true });

    try {
      const response =
        params === null || params === undefined
          ? this.rpc.sendRequest<unknown>(method, cancellation.token)
          : this.rpc.sendRequest<unknown>(method, params, cancellation.token);
      return await Promise.race([response, interrupted]);
    } catch (error) {
      if (signal?.aborted) throw lspAbortReason(signal);
      if (timedOut && timeoutError) throw timeoutError;
      const failure = this.failure ?? error;
      throw new Error(
        `LSP request "${method}" to server "${this.serverId}" failed: ${
          failure instanceof Error ? failure.message : String(failure)
        }.${this.stderrSummary()}`,
        { cause: failure },
      );
    } finally {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
      cancellation.dispose();
    }
  }

  sendNotification(method: string, params: unknown): void {
    if (!this.isAlive()) return;
    try {
      void this.rpc.sendNotification(method, params).catch((error) => {
        this.terminate(error instanceof Error ? error : new Error(String(error)));
      });
    } catch (error) {
      this.terminate(error instanceof Error ? error : new Error(String(error)));
    }
  }

  async waitForExit(timeoutMs: number): Promise<void> {
    if (this.child.exitCode !== null) return;
    await new Promise<void>((resolve) => {
      const timeout = setTimeout(resolve, timeoutMs);
      void this.exitPromise.then(() => {
        clearTimeout(timeout);
        resolve();
      });
    });
  }

  kill(): void {
    if (this.isAlive()) {
      this.child.kill();
    }
  }

  private appendStderr(chunk: Buffer): void {
    this.stderr = Buffer.concat([this.stderr, chunk]);
    if (this.stderr.byteLength > STDERR_LIMIT_BYTES) {
      this.stderr = this.stderr.subarray(this.stderr.byteLength - STDERR_LIMIT_BYTES);
    }
  }

  private close(reason: Error): void {
    this.failure = reason;
    if (this.closed) return;
    this.closed = true;
    this.rpc.dispose();
    for (const callback of this.closedCallbacks) {
      callback();
    }
    this.closedCallbacks = [];
  }

  private terminate(reason: Error): void {
    this.close(reason);
    if (this.child.exitCode === null && !this.child.killed) {
      this.child.kill();
    }
  }

  private stderrSummary(): string {
    const trimmed = this.stderr.toString("utf8").trim();
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

function assertLspServerAdmitted(inventory: ExtensionInventory, server: LspCapability): void {
  const bundleIds = owningBundleIds(inventory, server);
  if (bundleIds.length === 0) {
    throw new Error(`LSP server "${server.id}" has no host-owned Extension inventory record.`);
  }
  const preflight = preflightExtensionComposition({
    bundles: inventory.bundles,
    selectedExtensionIds: bundleIds,
    installedExtensions: inventory.installedExtensions ?? [],
  });
  if (preflight.status === "ready") return;
  const issues = preflight.issues
    .filter((issue) => issue.severity === "error")
    .map((issue) => issue.message)
    .join(" ");
  throw new Error(`LSP server "${server.id}" is blocked by Extension preflight. ${issues}`);
}

function isLspServerAdmitted(inventory: ExtensionInventory, server: LspCapability): boolean {
  try {
    assertLspServerAdmitted(inventory, server);
    return true;
  } catch {
    return false;
  }
}

function owningBundleIds(inventory: ExtensionInventory, server: LspCapability): string[] {
  return inventory.bundles
    .filter((bundle) =>
      bundle.capabilities.lspServers.some((candidate) => candidate.id === server.id),
    )
    .map((bundle) => bundle.id)
    .sort((left, right) => left.localeCompare(right));
}

function lspEnvironment(parent: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const environment: NodeJS.ProcessEnv = {};
  for (const key of LSP_ENVIRONMENT_KEYS) {
    const value = parent[key];
    if (value !== undefined) environment[key] = value;
  }
  return environment;
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

const CLAUDE_LSP_OPERATIONS = new Set<ClaudeLspOperation>([
  "goToDefinition",
  "findReferences",
  "hover",
  "documentSymbol",
  "workspaceSymbol",
  "goToImplementation",
  "prepareCallHierarchy",
  "incomingCalls",
  "outgoingCalls",
]);

function validateClaudeLspRequest(request: ClaudeLspRequest): void {
  if (!CLAUDE_LSP_OPERATIONS.has(request.operation)) {
    throw new Error(`Unsupported LSP operation: ${String(request.operation)}.`);
  }
  if (!request.filePath?.trim()) throw new Error("LSP requires filePath.");
  if (!Number.isInteger(request.line) || request.line < 1) {
    throw new Error("LSP line must be a positive one-based integer.");
  }
  if (!Number.isInteger(request.character) || request.character < 1) {
    throw new Error("LSP character must be a positive one-based integer.");
  }
  if (
    request.operation === "workspaceSymbol" &&
    (typeof request.query !== "string" || !request.query.trim())
  ) {
    throw new Error("LSP workspaceSymbol requires a non-empty query.");
  }
}

async function readContainedLspFile(
  workspaceRoot: string,
  requestedPath: string,
): Promise<{ path: string; text: string }> {
  const resolved = path.isAbsolute(requestedPath)
    ? path.resolve(requestedPath)
    : path.resolve(workspaceRoot, requestedPath);
  const canonical = await realpath(resolved);
  const relative = path.relative(workspaceRoot, canonical);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`LSP file escapes the Project: ${requestedPath}`);
  }
  const fileStat = await stat(canonical);
  if (!fileStat.isFile()) throw new Error(`LSP file must be a regular file: ${requestedPath}`);
  if (fileStat.size > LSP_DOCUMENT_LIMIT_BYTES) {
    throw new Error(
      `LSP file exceeds the ${LSP_DOCUMENT_LIMIT_BYTES}-byte read limit: ${requestedPath}`,
    );
  }
  const bytes = await readFile(canonical);
  let text: string;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`LSP file must contain valid UTF-8 text: ${requestedPath}`);
  }
  return { path: canonical, text };
}

function isCommandBackedLspServer(server: LspCapability): boolean {
  return server.command !== undefined;
}

function resolveLspServerForFile(inventory: ExtensionInventory, filePath: string): LspCapability {
  const candidates = inventory.lspServers.filter(isCommandBackedLspServer);
  const extension = path.extname(filePath).toLowerCase();
  const languageId = languageIdForPath(filePath);
  const matches = candidates.filter((server) =>
    lspServerMatchesFile(server, extension, languageId),
  );
  if (matches.length === 1) return matches[0];
  if (matches.length === 0) {
    throw new Error(
      `No configured LSP server supports ${extension || "this file type"} (${languageId}).`,
    );
  }
  throw new Error(
    `Multiple configured LSP servers support ${extension || filePath}: ${matches
      .map((server) => server.id)
      .join(", ")}.`,
  );
}

function lspServerMatchesFile(
  server: LspCapability,
  extension: string,
  languageId: string,
): boolean {
  const record = server as LspCapability & {
    extensionToLanguage?: Record<string, string>;
    extensions?: string[];
  };
  const declaredExtensionLanguage = record.extensionToLanguage?.[extension];
  if (declaredExtensionLanguage) return declaredExtensionLanguage.toLowerCase() === languageId;
  if (record.extensions?.some((item) => normalizeExtension(item) === extension)) return true;
  const languages = [...server.languages, ...server.languageIds].map((item) => item.toLowerCase());
  if (languages.length === 0) return true;
  return languages.includes(languageId) || languages.includes(extension.slice(1));
}

function normalizeExtension(value: string): string {
  const normalized = value.trim().toLowerCase();
  return normalized.startsWith(".") ? normalized : `.${normalized}`;
}

function languageIdForPath(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  return (
    {
      ".c": "c",
      ".cc": "cpp",
      ".cpp": "cpp",
      ".cs": "csharp",
      ".css": "css",
      ".go": "go",
      ".html": "html",
      ".java": "java",
      ".js": "javascript",
      ".jsx": "javascriptreact",
      ".json": "json",
      ".lua": "lua",
      ".md": "markdown",
      ".php": "php",
      ".py": "python",
      ".rb": "ruby",
      ".rs": "rust",
      ".sh": "shellscript",
      ".swift": "swift",
      ".toml": "toml",
      ".ts": "typescript",
      ".tsx": "typescriptreact",
      ".txt": "plaintext",
      ".vue": "vue",
      ".yaml": "yaml",
      ".yml": "yaml",
    }[extension] ??
    (extension.slice(1) || "plaintext")
  );
}

function claudeLspResponse(
  request: ClaudeLspRequest,
  raw: unknown,
  workspaceRoot: string,
): ClaudeLspResponse {
  const metrics = lspResultMetrics(request.operation, raw);
  return {
    operation: request.operation,
    result: formatClaudeLspResult(request.operation, raw, workspaceRoot),
    filePath: request.filePath,
    ...(metrics.resultCount === undefined ? {} : { resultCount: metrics.resultCount }),
    ...(metrics.fileCount === undefined ? {} : { fileCount: metrics.fileCount }),
  };
}

function lspResultMetrics(
  operation: ClaudeLspOperation,
  raw: unknown,
): { resultCount?: number; fileCount?: number } {
  if (operation === "hover") {
    return raw == null ? { resultCount: 0, fileCount: 0 } : { resultCount: 1, fileCount: 1 };
  }
  const values = Array.isArray(raw) ? raw : raw == null ? [] : [raw];
  if (operation === "documentSymbol") {
    return { resultCount: countDocumentSymbols(values), fileCount: values.length > 0 ? 1 : 0 };
  }
  const uris = values.flatMap((value) => lspResultUris(operation, value));
  return { resultCount: values.length, fileCount: new Set(uris).size };
}

function countDocumentSymbols(values: unknown[]): number {
  let count = 0;
  const visit = (value: unknown): void => {
    if (!isRecord(value)) return;
    count += 1;
    if (Array.isArray(value.children)) for (const child of value.children) visit(child);
  };
  for (const value of values) visit(value);
  return count;
}

function lspResultUris(operation: ClaudeLspOperation, value: unknown): string[] {
  if (!isRecord(value)) return [];
  if (operation === "incomingCalls")
    return isRecord(value.from) ? stringValues(value.from.uri) : [];
  if (operation === "outgoingCalls") return isRecord(value.to) ? stringValues(value.to.uri) : [];
  if (operation === "workspaceSymbol") {
    return isRecord(value.location) ? stringValues(value.location.uri) : [];
  }
  return stringValues(
    value.uri ?? value.targetUri ?? (isRecord(value.location) ? value.location.uri : undefined),
  );
}

function stringValues(value: unknown): string[] {
  return typeof value === "string" && value ? [value] : [];
}

function formatClaudeLspResult(
  operation: ClaudeLspOperation,
  raw: unknown,
  workspaceRoot: string,
): string {
  if (raw == null || (Array.isArray(raw) && raw.length === 0)) {
    return `No results found for ${operation}.`;
  }
  if (operation === "hover") return formatHover(raw);
  const values = Array.isArray(raw) ? raw : [raw];
  return values
    .map((value) => formatLspValue(operation, value, workspaceRoot))
    .filter(Boolean)
    .join("\n");
}

function formatHover(raw: unknown): string {
  if (!isRecord(raw)) return stringifyLspValue(raw);
  const contents = raw.contents;
  if (typeof contents === "string") return contents;
  if (isRecord(contents) && typeof contents.value === "string") return contents.value;
  if (Array.isArray(contents)) {
    return contents
      .map((item) =>
        typeof item === "string"
          ? item
          : isRecord(item) && typeof item.value === "string"
            ? item.value
            : stringifyLspValue(item),
      )
      .join("\n");
  }
  return stringifyLspValue(raw);
}

function formatLspValue(
  operation: ClaudeLspOperation,
  value: unknown,
  workspaceRoot: string,
): string {
  if (!isRecord(value)) return stringifyLspValue(value);
  const endpoint =
    operation === "incomingCalls" ? value.from : operation === "outgoingCalls" ? value.to : value;
  const record = isRecord(endpoint) ? endpoint : value;
  const name = typeof record.name === "string" ? `${record.name} - ` : "";
  const location = isRecord(record.location) ? record.location : record;
  const uri =
    typeof location.uri === "string"
      ? location.uri
      : typeof location.targetUri === "string"
        ? location.targetUri
        : undefined;
  const range = isRecord(location.range)
    ? location.range
    : isRecord(location.targetRange)
      ? location.targetRange
      : isRecord(record.selectionRange)
        ? record.selectionRange
        : undefined;
  if (!uri) return `${name}${stringifyLspValue(value)}`;
  return `${name}${displayLspUri(uri, workspaceRoot)}${formatLspRange(range)}`;
}

function displayLspUri(uri: string, workspaceRoot: string): string {
  if (!uri.startsWith("file://")) return uri;
  try {
    const filePath = fileURLToPath(uri);
    const relative = path.relative(workspaceRoot, filePath);
    return relative.startsWith("..") || path.isAbsolute(relative) ? filePath : relative || ".";
  } catch {
    return uri;
  }
}

function formatLspRange(value: unknown): string {
  if (!isRecord(value) || !isRecord(value.start)) return "";
  const line = typeof value.start.line === "number" ? value.start.line + 1 : undefined;
  const character =
    typeof value.start.character === "number" ? value.start.character + 1 : undefined;
  return line === undefined ? "" : `:${line}${character === undefined ? "" : `:${character}`}`;
}

function stringifyLspValue(value: unknown): string {
  return typeof value === "string" ? value : (JSON.stringify(value) ?? String(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function lspAbortReason(signal?: AbortSignal): Error {
  return signal?.reason instanceof Error ? signal.reason : new Error("LSP request cancelled.");
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
  return realpath(resolved);
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
