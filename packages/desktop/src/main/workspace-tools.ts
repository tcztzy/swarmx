import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import type { Dirent } from "node:fs";
import {
  link,
  lstat,
  mkdir,
  mkdtemp,
  open,
  opendir,
  readFile,
  realpath,
  rename,
  rm,
  stat,
  unlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  type HarnessPermissionPolicy,
  HarnessPermissionPolicySchema,
  type HarnessToolAccess,
  type LocalMcpTool,
  type LocalTextTool,
  type LocalTool,
  type ModelApi,
  localToolResult,
  resolveHarnessToolPermission,
} from "@swarmx/core";
import type {
  ClaudeInteractionRequest,
  ClaudeInteractionResponse,
  ClaudeQuestion,
} from "./agent-interactions.js";
import type { ClaudeLspRequest, ClaudeLspResponse } from "./lsp-host.js";
import { applyCodexUpdate, parseCodexPatch } from "./workspace-patch.js";
import {
  WORKSPACE_SHELL_DEFAULTS,
  WorkspaceShell,
  type WorkspaceShellExecResult,
  type WorkspaceShellResult,
  type WorkspaceShellSessionSnapshot,
} from "./workspace-shell.js";

const MAX_RELATIVE_PATH_BYTES = 4 * 1024;
const GIT_COMMAND_SLACK_BYTES = 64 * 1024;
const MAX_GIT_STATUS_BYTES = 4 * 1024 * 1024;
const MAX_SKILL_BYTES = 512 * 1024;
const MAX_PLAN_BYTES = 1024 * 1024;

const CLAUDE_GREP_INPUT_SCHEMA = {
  type: "object",
  properties: {
    pattern: { type: "string", description: "The regular expression pattern to search for." },
    path: { type: "string", description: "File or directory to search in." },
    glob: { type: "string", description: "Glob pattern to filter files." },
    output_mode: {
      type: "string",
      enum: ["content", "files_with_matches", "count"],
      description: "Output content, matching paths, or match counts.",
    },
    "-B": { type: "number" },
    "-A": { type: "number" },
    "-C": { type: "number" },
    context: { type: "number" },
    "-n": { type: "boolean" },
    "-i": { type: "boolean" },
    "-o": { type: "boolean" },
    type: { type: "string" },
    head_limit: { type: "number" },
    offset: { type: "number" },
    multiline: { type: "boolean" },
  },
  required: ["pattern"],
} as const;

// Compatible with OpenAI Codex's Apache-2.0 apply_patch grammar.
const CODEX_APPLY_PATCH_GRAMMAR = String.raw`start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?
hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?
filename: /(.+)/
add_line: "+" /(.*)/ LF -> line
change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF
%import common.LF`;

export const WORKSPACE_TOOLS_DEFAULTS = {
  maxFileBytes: 1024 * 1024,
  maxWriteFileBytes: 1024 * 1024,
  maxDirectoryEntries: 500,
  maxReviewFiles: 200,
  maxPatchBytes: 2 * 1024 * 1024,
  maxPatchBytesPerFile: 256 * 1024,
  gitTimeoutMs: 10_000,
} as const;

const WORKSPACE_TOOLS_HARD_LIMITS = {
  maxFileBytes: 64 * 1024 * 1024,
  maxWriteFileBytes: 64 * 1024 * 1024,
  maxDirectoryEntries: 10_000,
  maxReviewFiles: 1_000,
  maxPatchBytes: 32 * 1024 * 1024,
  maxPatchBytesPerFile: 8 * 1024 * 1024,
  gitTimeoutMs: 60_000,
} as const;

export interface WorkspaceToolsOptions {
  maxFileBytes?: number;
  maxWriteFileBytes?: number;
  maxDirectoryEntries?: number;
  maxReviewFiles?: number;
  maxPatchBytes?: number;
  maxPatchBytesPerFile?: number;
  gitTimeoutMs?: number;
}

export type WorkspaceEntryKind = "directory" | "file" | "symlink" | "other";

export interface WorkspaceDirectoryEntry {
  name: string;
  path: string;
  kind: WorkspaceEntryKind;
}

export interface WorkspaceDirectoryListing {
  path: string;
  entries: WorkspaceDirectoryEntry[];
  truncated: boolean;
}

export interface WorkspaceTextFile {
  path: string;
  content: string;
  size: number;
  truncated: boolean;
  sha256?: string;
}

export interface WorkspaceWriteResult {
  path: string;
  size: number;
  sha256: string;
  created: boolean;
  content: string;
  originalContent: string | null;
}

export interface WorkspaceEditResult extends WorkspaceWriteResult {
  originalContent: string;
  replacements: number;
}

export interface WorkspaceDeleteResult {
  path: string;
  deleted: true;
}

export interface WorkspacePatchResult {
  operations: Array<{
    type: "add" | "delete" | "update" | "move";
    path: string;
    destination?: string;
  }>;
}

export interface WorkspaceSearchResult {
  output: string;
  truncated: boolean;
  countIsComplete?: boolean;
  durationMs?: number;
  mode?: "content" | "files_with_matches" | "count";
  filenames?: string[];
  totalFiles?: number;
  totalLines?: number;
  appliedLimit?: number;
  appliedOffset?: number;
}

export type WorkspaceToolProfile = "claude_code" | "codex";

export interface WorkspaceAgentToolOptions {
  model?: string;
  apiProtocol?: ModelApi;
  skills?: readonly WorkspaceAgentSkill[];
  effort?: string;
  sessionId?: string;
  interact?: (request: ClaudeInteractionRequest) => Promise<ClaudeInteractionResponse>;
  closeInteractions?: () => void;
  lsp?: (request: ClaudeLspRequest) => Promise<ClaudeLspResponse>;
  agent?: (request: ClaudeAgentInvocation) => Promise<ClaudeAgentResult>;
  sessionTools?: ClaudeSessionToolBridge;
  borrowShell?: boolean;
  permissionPolicy?: HarnessPermissionPolicy;
}

export interface ClaudeSessionActivation {
  source: "monitor" | "cron";
  prompt: string;
  taskId?: string;
  jobId?: string;
}

export interface ClaudeMonitorInvocation {
  command: string;
  description: string;
  timeoutMs: number;
  persistent: boolean;
}

export interface ClaudeMonitorResult {
  taskId: string;
  timeoutMs: number;
  persistent?: boolean;
}

export interface ClaudeCronCreateInvocation {
  cron: string;
  prompt: string;
  recurring: boolean;
  durable: boolean;
}

export interface ClaudeCronJob {
  id: string;
  cron: string;
  humanSchedule: string;
  prompt: string;
  recurring?: boolean;
  durable?: boolean;
}

export interface ClaudeSessionToolBridge {
  monitor(request: ClaudeMonitorInvocation): Promise<ClaudeMonitorResult>;
  createCron(request: ClaudeCronCreateInvocation): Promise<{
    id: string;
    humanSchedule: string;
    recurring: boolean;
    durable?: boolean;
  }>;
  deleteCron(id: string): Promise<{ id: string }>;
  listCrons(): Promise<{ jobs: ClaudeCronJob[] }>;
}

export interface ClaudeAgentInvocation {
  description: string;
  prompt: string;
  subagentType?: string;
  model?: "sonnet" | "opus" | "haiku";
  resume?: string;
}

export interface ClaudeAgentResult {
  status: "completed";
  prompt: string;
  agentId: string;
  content: Array<{ type: "text"; text: string }>;
  totalToolUseCount: number;
  totalDurationMs: number;
  totalTokens: number;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number | null;
    cache_read_input_tokens: number | null;
    server_tool_use: {
      web_search_requests: number;
      web_fetch_requests: number;
    } | null;
    service_tier: "standard" | "priority" | "batch" | null;
    cache_creation: {
      ephemeral_1h_input_tokens: number;
      ephemeral_5m_input_tokens: number;
    } | null;
  };
}

export interface WorkspaceAgentSkill {
  id: string;
  name?: string;
  filePath: string;
  description?: string;
}

export interface WorkspaceReviewFile {
  path: string;
  previousPath?: string;
  status: string;
  patch: string;
  binary: boolean;
  additions: number;
  deletions: number;
  truncated: boolean;
  error?: string;
}

export interface WorkspaceReviewSnapshot {
  root: string;
  branch: string | null;
  isRepository: boolean;
  files: WorkspaceReviewFile[];
  truncated: boolean;
  error?: string;
}

interface ResolvedWorkspaceToolsOptions {
  maxFileBytes: number;
  maxWriteFileBytes: number;
  maxDirectoryEntries: number;
  maxReviewFiles: number;
  maxPatchBytes: number;
  maxPatchBytesPerFile: number;
  gitTimeoutMs: number;
}

interface GitCommandResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  truncated: boolean;
  missingExecutable: boolean;
  error?: Error;
}

interface SearchCommandResult {
  stdout: string;
  stderr: string;
  truncated: boolean;
  exitCode: number | null;
  missingExecutable: boolean;
}

interface ClaudeWorktreeState {
  originalCwd: string;
  gitRoot: string;
  worktreePath: string;
  worktreeBranch: string;
  originalHeadCommit: string;
}

interface ClaudeWorktreeStatus {
  changedFiles: number;
  commits: number;
}

interface ClaudeGitResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

class ClaudePlanMode {
  private active = false;
  private directory: string | undefined;
  private filePath: string | undefined;

  get isActive(): boolean {
    return this.active;
  }

  async enter(): Promise<string> {
    if (!this.directory) {
      this.directory = await mkdtemp(path.join(tmpdir(), "swarmx-claude-plan-"));
      this.filePath = path.join(this.directory, "plan.md");
    }
    if (!this.filePath) throw new Error("Plan file could not be created.");
    if (!this.active) await writeFile(this.filePath, "", { encoding: "utf8", mode: 0o600 });
    this.active = true;
    return this.filePath;
  }

  matches(candidate: string): boolean {
    return this.active && this.filePath !== undefined && path.resolve(candidate) === this.filePath;
  }

  assertCanMutate(toolName: string): void {
    if (this.active) {
      throw new Error(
        `${toolName} is unavailable in plan mode. Inspect the Project, write only the plan file, or call ExitPlanMode.`,
      );
    }
  }

  async write(content: string): Promise<{ filePath: string; originalContent: string }> {
    if (!this.active || !this.filePath) throw new Error("Plan mode is not active.");
    if (Buffer.byteLength(content) > MAX_PLAN_BYTES) {
      throw new Error(`Plan exceeds the ${MAX_PLAN_BYTES}-byte limit.`);
    }
    const originalContent = await readFile(this.filePath, "utf8");
    await writeFile(this.filePath, content, { encoding: "utf8", mode: 0o600 });
    return { filePath: this.filePath, originalContent };
  }

  async read(): Promise<{ filePath: string; content: string }> {
    if (!this.active || !this.filePath) throw new Error("Plan mode is not active.");
    const content = await readFile(this.filePath, "utf8");
    if (Buffer.byteLength(content) > MAX_PLAN_BYTES) {
      throw new Error(`Plan exceeds the ${MAX_PLAN_BYTES}-byte limit.`);
    }
    return { filePath: this.filePath, content };
  }

  async planForApproval(): Promise<{ filePath: string; plan: string }> {
    const { filePath, content } = await this.read();
    if (!content.trim()) {
      throw new Error(`Write the plan to ${filePath} before calling ExitPlanMode.`);
    }
    return { filePath, plan: content };
  }

  approve(): void {
    this.active = false;
  }

  async close(): Promise<void> {
    this.active = false;
    const directory = this.directory;
    this.directory = undefined;
    this.filePath = undefined;
    if (directory) await rm(directory, { recursive: true, force: true });
  }
}

interface GitStatusEntry {
  path: string;
  previousPath?: string;
  status: string;
}

interface ParsedGitStatus {
  entries: GitStatusEntry[];
  truncated: boolean;
}

interface DecodedFile {
  content: string;
  size: number;
  truncated: boolean;
  sha256?: string;
}

interface WritableTarget {
  lexicalPath: string;
  exists: boolean;
  mode: number;
  sha256?: string;
}

type WorkspaceAtomicWriteResult = Omit<WorkspaceWriteResult, "content" | "originalContent">;

type ClaudeTaskStatus = "pending" | "in_progress" | "completed";

interface ClaudeTaskRecord {
  id: string;
  subject: string;
  description: string;
  activeForm?: string;
  status: ClaudeTaskStatus;
  blocks: Set<string>;
  blockedBy: Set<string>;
  owner?: string;
  metadata: Record<string, unknown>;
}

interface ClaudeTodoRecord {
  content: string;
  status: ClaudeTaskStatus;
  activeForm: string;
}

interface ClaudeNotebookCell extends Record<string, unknown> {
  cell_type: string;
  id?: string;
  source: string | string[];
}

class ClaudeTaskStore {
  #nextId = 1;
  readonly #tasks = new Map<string, ClaudeTaskRecord>();

  create(input: Record<string, unknown>) {
    const subject = requiredNonemptyToolText(input.subject, "subject");
    const description = requiredNonemptyToolText(input.description, "description");
    const activeForm = optionalToolText(input.activeForm, "activeForm");
    const metadata = optionalMetadata(input.metadata);
    const id = String(this.#nextId++);
    this.#tasks.set(id, {
      id,
      subject,
      description,
      ...(activeForm === undefined ? {} : { activeForm }),
      status: "pending",
      blocks: new Set(),
      blockedBy: new Set(),
      metadata,
    });
    return { task: { id, subject } };
  }

  get(taskIdValue: unknown) {
    const taskId = requiredNonemptyToolText(taskIdValue, "taskId");
    const task = this.#tasks.get(taskId);
    return {
      task: task
        ? {
            id: task.id,
            subject: task.subject,
            description: task.description,
            status: task.status,
            blocks: [...task.blocks],
            blockedBy: [...task.blockedBy],
          }
        : null,
    };
  }

  list() {
    return {
      tasks: [...this.#tasks.values()].map((task) => ({
        id: task.id,
        subject: task.subject,
        status: task.status,
        ...(task.owner === undefined ? {} : { owner: task.owner }),
        blockedBy: [...task.blockedBy],
      })),
    };
  }

  update(input: Record<string, unknown>) {
    const taskId = requiredNonemptyToolText(input.taskId, "taskId");
    const task = this.#tasks.get(taskId);
    if (!task) {
      return {
        success: false,
        taskId,
        updatedFields: [] as string[],
        error: `Task ${taskId} was not found.`,
      };
    }

    const subject = optionalNonemptyToolText(input.subject, "subject");
    const description = optionalNonemptyToolText(input.description, "description");
    const activeForm = optionalNonemptyToolText(input.activeForm, "activeForm");
    const owner = optionalNonemptyToolText(input.owner, "owner");
    const metadata = input.metadata === undefined ? undefined : optionalMetadata(input.metadata);
    const status = optionalClaudeTaskStatus(input.status, true);
    const addBlocks = optionalTaskIds(input.addBlocks, "addBlocks");
    const addBlockedBy = optionalTaskIds(input.addBlockedBy, "addBlockedBy");
    this.#validateLinks(taskId, [...addBlocks, ...addBlockedBy]);

    const updatedFields: string[] = [];
    const previousStatus = task.status;
    if (subject !== undefined) {
      task.subject = subject;
      updatedFields.push("subject");
    }
    if (description !== undefined) {
      task.description = description;
      updatedFields.push("description");
    }
    if (activeForm !== undefined) {
      task.activeForm = activeForm;
      updatedFields.push("activeForm");
    }
    if (owner !== undefined) {
      task.owner = owner;
      updatedFields.push("owner");
    }
    if (metadata !== undefined) {
      for (const [key, value] of Object.entries(metadata)) {
        if (value === null) Reflect.deleteProperty(task.metadata, key);
        else task.metadata[key] = value;
      }
      updatedFields.push("metadata");
    }
    for (const blockedId of addBlocks) {
      task.blocks.add(blockedId);
      this.#tasks.get(blockedId)?.blockedBy.add(taskId);
    }
    if (addBlocks.length > 0) updatedFields.push("blocks");
    for (const blockerId of addBlockedBy) {
      task.blockedBy.add(blockerId);
      this.#tasks.get(blockerId)?.blocks.add(taskId);
    }
    if (addBlockedBy.length > 0) updatedFields.push("blockedBy");

    if (status === "deleted") {
      for (const other of this.#tasks.values()) {
        other.blocks.delete(taskId);
        other.blockedBy.delete(taskId);
      }
      this.#tasks.delete(taskId);
      updatedFields.push("status");
      return {
        success: true,
        taskId,
        updatedFields,
        statusChange: { from: previousStatus, to: "deleted" },
      };
    }
    if (status !== undefined) {
      task.status = status;
      updatedFields.push("status");
    }
    return {
      success: true,
      taskId,
      updatedFields,
      ...(status === undefined || status === previousStatus
        ? {}
        : { statusChange: { from: previousStatus, to: status } }),
    };
  }

  #validateLinks(taskId: string, linkedIds: string[]): void {
    for (const linkedId of linkedIds) {
      if (linkedId === taskId) throw new Error("A task cannot block itself.");
      if (!this.#tasks.has(linkedId)) throw new Error(`Task ${linkedId} was not found.`);
    }
  }
}

class WorkspaceBinaryFileError extends Error {
  constructor() {
    super("Only UTF-8 text files can be read.");
    this.name = "WorkspaceBinaryFileError";
  }
}

/** Bounded filesystem and Git operations confined to one workspace root. */
export class WorkspaceTools {
  #root: string;
  readonly #options: ResolvedWorkspaceToolsOptions;
  readonly #readVersions = new Map<string, string>();

  constructor(root: string, options: WorkspaceToolsOptions = {}) {
    if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
      throw new Error("A valid workspace root is required.");
    }

    this.#root = path.resolve(root);
    this.#options = resolveOptions(options);
  }

  get root(): string {
    return this.#root;
  }

  rebindRoot(root: string): void {
    if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
      throw new Error("A valid workspace root is required.");
    }
    this.#root = path.resolve(root);
    this.#readVersions.clear();
  }

  /** Returns a single directory level. Call again with an entry path to expand it. */
  async listDirectory(relativePath = ""): Promise<WorkspaceDirectoryListing> {
    const normalizedPath = normalizeRelativePath(relativePath, true);
    const root = await this.#realRoot();
    const target = await resolveExistingPath(root, normalizedPath, true);
    const targetStat = await stat(target);
    if (!targetStat.isDirectory()) throw new Error("Workspace path is not a directory.");

    const entries: WorkspaceDirectoryEntry[] = [];
    const directory = await opendir(target);
    for await (const entry of directory) {
      entries.push({
        name: entry.name,
        path: joinRelativePath(normalizedPath, entry.name),
        kind: directoryEntryKind(entry),
      });
      if (entries.length > this.#options.maxDirectoryEntries) break;
    }

    const truncated = entries.length > this.#options.maxDirectoryEntries;
    if (truncated) entries.length = this.#options.maxDirectoryEntries;
    entries.sort(compareDirectoryEntries);

    return { path: normalizedPath, entries, truncated };
  }

  /** Reads at most maxFileBytes from a contained UTF-8 regular file. */
  async readFile(relativePath: string): Promise<WorkspaceTextFile> {
    const normalizedPath = normalizeRelativePath(relativePath, false);
    const root = await this.#realRoot();
    const target = await resolveExistingPath(root, normalizedPath, false);
    const decoded = await readUtf8File(target, this.#options.maxFileBytes);
    if (decoded.sha256) this.#readVersions.set(normalizedPath, decoded.sha256);
    else this.#readVersions.delete(normalizedPath);
    return { path: normalizedPath, ...decoded };
  }

  /** Atomically creates a text file or replaces a file that was read without intervening changes. */
  async writeFile(relativePath: string, content: string): Promise<WorkspaceWriteResult> {
    const normalizedPath = normalizeRelativePath(relativePath, false);
    const encoded = validateWriteContent(content, this.#options.maxWriteFileBytes);
    const root = await this.#realRoot();
    const expectedSha256 = this.#readVersions.get(normalizedPath);
    const target = await this.#writableTarget(root, normalizedPath, expectedSha256 !== undefined);
    if (target.exists && !expectedSha256) {
      throw new Error("Existing files must be read completely before writing.");
    }
    if (target.exists ? target.sha256 !== expectedSha256 : expectedSha256 !== undefined) {
      throw staleWorkspaceFileError();
    }
    const originalContent = target.exists
      ? await completeOriginalContent(
          target.lexicalPath,
          this.#options.maxFileBytes,
          expectedSha256,
        )
      : null;
    const result = await this.#atomicWrite(root, normalizedPath, encoded, target, expectedSha256);
    return { ...result, content, originalContent };
  }

  /** Replaces an exact text occurrence in a previously read, unchanged file. */
  async editFile(
    relativePath: string,
    oldText: string,
    newText: string,
    replaceAll = false,
  ): Promise<WorkspaceEditResult> {
    if (typeof oldText !== "string" || oldText.length === 0) {
      throw new Error("oldText must be a non-empty exact text match.");
    }
    if (typeof newText !== "string") throw new Error("newText must be text.");

    const normalizedPath = normalizeRelativePath(relativePath, false);
    const expectedSha256 = this.#readVersions.get(normalizedPath);
    if (!expectedSha256) {
      throw new Error("Files must be read completely before editing.");
    }

    const root = await this.#realRoot();
    const target = await this.#writableTarget(root, normalizedPath, true);
    if (!target.exists || target.sha256 !== expectedSha256) throw staleWorkspaceFileError();
    const decoded = await readUtf8File(target.lexicalPath, this.#options.maxFileBytes);
    if (decoded.truncated || decoded.sha256 !== expectedSha256) throw staleWorkspaceFileError();

    const replacements = exactOccurrenceCount(decoded.content, oldText);
    if (replacements === 0) throw new Error("oldText was not found in the file.");
    if (!replaceAll && replacements !== 1) {
      throw new Error(
        `oldText matched ${replacements} occurrences; provide a unique match or set replaceAll.`,
      );
    }

    const content = replaceAll
      ? decoded.content.split(oldText).join(newText)
      : decoded.content.replace(oldText, newText);
    const encoded = validateWriteContent(content, this.#options.maxWriteFileBytes);
    const result = await this.#atomicWrite(root, normalizedPath, encoded, target, expectedSha256);
    return {
      ...result,
      content,
      originalContent: decoded.content,
      replacements: replaceAll ? replacements : 1,
    };
  }

  /** Deletes a previously read, unchanged regular file. */
  async deleteFile(relativePath: string): Promise<WorkspaceDeleteResult> {
    const normalizedPath = normalizeRelativePath(relativePath, false);
    const expectedSha256 = this.#readVersions.get(normalizedPath);
    if (!expectedSha256) {
      throw new Error("Files must be read completely before deletion.");
    }
    const root = await this.#realRoot();
    const target = await this.#writableTarget(root, normalizedPath, true);
    if (!target.exists || target.sha256 !== expectedSha256) throw staleWorkspaceFileError();
    await unlink(target.lexicalPath);
    this.#readVersions.delete(normalizedPath);
    return { path: normalizedPath, deleted: true };
  }

  /** Applies the Codex apply_patch envelope through the same guarded file operations. */
  async applyPatch(patchText: string): Promise<WorkspacePatchResult> {
    if (Buffer.byteLength(patchText) > this.#options.maxPatchBytes) {
      throw new Error(`apply_patch exceeds the ${this.#options.maxPatchBytes}-byte limit.`);
    }
    const operations = parseCodexPatch(patchText);
    const applied: WorkspacePatchResult["operations"] = [];

    for (const operation of operations) {
      const source = workspaceRelativePath(this.root, operation.path, false);
      if (operation.type === "add") {
        await this.writeFile(source, operation.content);
        applied.push({ type: "add", path: source });
        continue;
      }

      const current = await this.readFile(source);
      if (current.truncated) throw new Error(`apply_patch cannot modify truncated file ${source}.`);
      if (operation.type === "delete") {
        await this.deleteFile(source);
        applied.push({ type: "delete", path: source });
        continue;
      }

      const updated = applyCodexUpdate(current.content, operation.hunks);
      if (operation.moveTo) {
        const destination = workspaceRelativePath(this.root, operation.moveTo, false);
        await this.writeFile(destination, updated.content);
        await this.deleteFile(source);
        applied.push({ type: "move", path: source, destination });
      } else {
        await this.writeFile(source, updated.content);
        applied.push({ type: "update", path: source });
      }
    }
    return { operations: applied };
  }

  /** Lists files matching a Claude Code Glob pattern without invoking a shell. */
  async glob(pattern: string, relativePath = ""): Promise<WorkspaceSearchResult> {
    const startedAt = Date.now();
    const requiredPattern = requiredToolText(pattern, "pattern");
    if (!requiredPattern) throw new Error("pattern must be non-empty text.");
    const normalizedPath = normalizeRelativePath(relativePath, true);
    const root = await this.#realRoot();
    await resolveExistingPath(root, normalizedPath, true);
    const result = await this.#search(root, [
      "--files",
      "--hidden",
      "--no-ignore",
      "--glob",
      requiredPattern,
      "--glob",
      "!.git/**",
      "--",
      normalizedPath || ".",
    ]);
    const searched = searchResult(result);
    const allFilenames = outputLines(searched.output);
    const timestamped = await Promise.all(
      allFilenames.map(async (filename) => {
        try {
          const fileStat = await stat(path.resolve(root, normalizedSearchPath(filename)));
          return { filename, modifiedAt: fileStat.mtimeMs };
        } catch {
          return { filename, modifiedAt: 0 };
        }
      }),
    );
    timestamped.sort(
      (left, right) =>
        right.modifiedAt - left.modifiedAt || left.filename.localeCompare(right.filename),
    );
    const filenames = timestamped.slice(0, 100).map((entry) => entry.filename);
    const truncated = searched.truncated || allFilenames.length > filenames.length;
    return {
      ...searched,
      output: filenames.join("\n"),
      truncated,
      countIsComplete: !searched.truncated,
      durationMs: Math.max(0, Date.now() - startedAt),
      filenames,
      totalFiles: allFilenames.length,
      totalLines: allFilenames.length,
    };
  }

  /** Runs a bounded Claude Code-compatible Grep query without invoking a shell. */
  async grep(input: Record<string, unknown>): Promise<WorkspaceSearchResult> {
    const startedAt = Date.now();
    const pattern = requiredToolText(input.pattern, "pattern");
    if (!pattern) throw new Error("pattern must be non-empty text.");
    const normalizedPath = normalizeRelativePath(stringToolPath(input.path), true);
    const root = await this.#realRoot();
    await resolveExistingPath(root, normalizedPath, true);
    const outputMode = optionalEnum(
      input.output_mode,
      ["content", "files_with_matches", "count"] as const,
      "output_mode",
      "files_with_matches",
    );
    const arguments_ = ["--hidden", "--glob", "!.git"];
    if (outputMode === "files_with_matches") arguments_.push("--files-with-matches");
    else if (outputMode === "count") arguments_.push("--count");
    else arguments_.push("--line-number");
    appendBooleanFlag(arguments_, input["-i"], "--ignore-case", "-i");
    appendBooleanFlag(arguments_, input["-n"], "--line-number", "-n");
    appendBooleanFlag(arguments_, input["-o"], "--only-matching", "-o");
    appendBooleanFlag(arguments_, input.multiline, "--multiline", "multiline");
    appendIntegerFlag(arguments_, input["-B"], "--before-context", "-B");
    appendIntegerFlag(arguments_, input["-A"], "--after-context", "-A");
    appendIntegerFlag(arguments_, input["-C"] ?? input.context, "--context", "context");
    if (input.glob !== undefined) arguments_.push("--glob", requiredToolText(input.glob, "glob"));
    if (input.type !== undefined) arguments_.push("--type", requiredToolText(input.type, "type"));
    const headLimit = optionalNonnegativeInteger(input.head_limit, "head_limit") ?? 250;
    const offset = optionalNonnegativeInteger(input.offset, "offset") ?? 0;
    arguments_.push("--", pattern, normalizedPath || ".");
    const result = await this.#search(root, arguments_);
    const searched = searchResult(result);
    const allLines = outputLines(searched.output);
    const selectedLines = allLines.slice(offset, headLimit === 0 ? undefined : offset + headLimit);
    const filenames = searchFilenames(selectedLines, outputMode);
    const totalFilenames = searchFilenames(allLines, outputMode);
    return {
      output: selectedLines.length > 0 ? `${selectedLines.join("\n")}\n` : "",
      truncated: searched.truncated || offset > 0 || selectedLines.length < allLines.length,
      durationMs: Math.max(0, Date.now() - startedAt),
      mode: outputMode,
      filenames,
      totalFiles: totalFilenames.length,
      totalLines: allLines.length,
      appliedLimit: headLimit,
      appliedOffset: offset,
    };
  }

  /** Captures the current branch and bounded staged, unstaged, and untracked patches. */
  async review(): Promise<WorkspaceReviewSnapshot> {
    const root = await this.#realRoot();
    const repository = await this.#git(root, ["rev-parse", "--is-inside-work-tree"], 4 * 1024);
    if (!repository.ok || repository.stdout.trim() !== "true") {
      const error = repository.missingExecutable
        ? "Git is not available."
        : isNotGitRepository(repository.stderr)
          ? undefined
          : gitErrorMessage(repository);
      return {
        root: this.root,
        branch: null,
        isRepository: false,
        files: [],
        truncated: repository.truncated,
        ...(error ? { error } : {}),
      };
    }

    const [branch, status] = await Promise.all([
      this.#branch(root),
      this.#git(
        root,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."],
        gitStatusOutputLimit(this.#options.maxReviewFiles),
      ),
    ]);

    if (!status.ok) {
      return {
        root: this.root,
        branch,
        isRepository: true,
        files: [],
        truncated: status.truncated,
        error: gitErrorMessage(status),
      };
    }

    const parsedStatus = parseGitStatus(status.stdout, this.#options.maxReviewFiles);
    const files: WorkspaceReviewFile[] = [];
    let remainingPatchBytes = this.#options.maxPatchBytes;

    for (const entry of parsedStatus.entries) {
      const fileBudget = Math.min(this.#options.maxPatchBytesPerFile, remainingPatchBytes);
      const file = await this.#reviewFile(root, entry, fileBudget);
      files.push(file);
      remainingPatchBytes -= Buffer.byteLength(file.patch);
    }

    return {
      root: this.root,
      branch,
      isRepository: true,
      files,
      truncated: status.truncated || parsedStatus.truncated || files.some((file) => file.truncated),
    };
  }

  async #reviewFile(
    root: string,
    entry: GitStatusEntry,
    patchBudget: number,
  ): Promise<WorkspaceReviewFile> {
    if (entry.status === "??") return this.#untrackedReviewFile(root, entry, patchBudget);

    let patch = "";
    let truncated = patchBudget === 0;
    let error: string | undefined;
    const phases: string[][] = [];
    const pathspecs = [entry.path, ...(entry.previousPath ? [entry.previousPath] : [])];
    const indexStatus = entry.status[0];
    const worktreeStatus = entry.status[1];

    if (indexStatus && indexStatus !== " " && indexStatus !== "?") {
      phases.push(["diff", "--cached", ...safeDiffArguments(), "--", ...pathspecs]);
    }
    if (worktreeStatus && worktreeStatus !== " " && worktreeStatus !== "?") {
      phases.push(["diff", ...safeDiffArguments(), "--", ...pathspecs]);
    }

    for (const arguments_ of phases) {
      const usedBytes = Buffer.byteLength(patch);
      const availableBytes = patchBudget - usedBytes;
      if (availableBytes <= 0) {
        truncated = true;
        break;
      }

      const result = await this.#git(root, arguments_, availableBytes);
      if (!result.ok) {
        error ??= gitErrorMessage(result);
        continue;
      }

      const separator = patch && result.stdout ? "\n" : "";
      const appended = truncateUtf8(`${patch}${separator}${result.stdout}`, patchBudget);
      patch = appended.value;
      truncated ||= result.truncated || appended.truncated;
    }

    const counts = patchLineCounts(patch);
    return {
      ...entry,
      patch,
      binary: isBinaryPatch(patch),
      ...counts,
      truncated,
      ...(error ? { error } : {}),
    };
  }

  async #untrackedReviewFile(
    root: string,
    entry: GitStatusEntry,
    patchBudget: number,
  ): Promise<WorkspaceReviewFile> {
    let patch = "";
    let binary = false;
    let truncated = patchBudget === 0;
    let error: string | undefined;

    if (patchBudget > 0) {
      try {
        const target = await resolveExistingPath(root, entry.path, false);
        const decoded = await readUtf8File(
          target,
          Math.min(this.#options.maxFileBytes, patchBudget),
        );
        const boundedPatch = truncateUtf8(untrackedPatch(entry.path, decoded.content), patchBudget);
        patch = boundedPatch.value;
        truncated ||= decoded.truncated || boundedPatch.truncated;
      } catch (cause) {
        if (cause instanceof WorkspaceBinaryFileError) {
          binary = true;
        } else {
          error = errorMessage(cause);
        }
      }
    }

    const counts = patchLineCounts(patch);
    return {
      ...entry,
      patch,
      binary,
      ...counts,
      truncated,
      ...(error ? { error } : {}),
    };
  }

  async #branch(root: string): Promise<string | null> {
    const symbolic = await this.#git(root, ["symbolic-ref", "--quiet", "--short", "HEAD"], 4096);
    if (symbolic.ok && symbolic.stdout.trim()) return symbolic.stdout.trim();

    const detached = await this.#git(root, ["rev-parse", "--short", "HEAD"], 4096);
    return detached.ok && detached.stdout.trim() ? detached.stdout.trim() : null;
  }

  async #realRoot(): Promise<string> {
    const resolved = await realpath(this.root);
    const rootStat = await stat(resolved);
    if (!rootStat.isDirectory()) throw new Error("Workspace root is not a directory.");
    return resolved;
  }

  async #writableTarget(
    root: string,
    normalizedPath: string,
    calculateDigest: boolean,
  ): Promise<WritableTarget> {
    const lexicalPath = path.resolve(root, normalizedPath);
    if (!isContainedPath(root, lexicalPath)) throw new Error("Workspace path escapes the root.");

    const resolvedParent = await realpath(path.dirname(lexicalPath));
    if (!isContainedPath(root, resolvedParent)) {
      throw new Error("Workspace path resolves outside the root.");
    }

    try {
      const targetStat = await lstat(lexicalPath);
      if (targetStat.isSymbolicLink()) {
        throw new Error("Workspace writes through symbolic links are not allowed.");
      }
      if (!targetStat.isFile()) throw new Error("Workspace path is not a regular file.");
      const resolvedTarget = await realpath(lexicalPath);
      if (!isContainedPath(root, resolvedTarget)) {
        throw new Error("Workspace path resolves outside the root.");
      }
      return {
        lexicalPath,
        exists: true,
        mode: targetStat.mode & 0o777,
        ...(calculateDigest ? { sha256: await fileSha256(lexicalPath) } : {}),
      };
    } catch (cause) {
      if (!isFileNotFoundError(cause)) throw cause;
      return { lexicalPath, exists: false, mode: 0o644 };
    }
  }

  async #atomicWrite(
    root: string,
    normalizedPath: string,
    content: Buffer,
    originalTarget: WritableTarget,
    expectedSha256: string | undefined,
  ): Promise<WorkspaceAtomicWriteResult> {
    const temporaryPath = path.join(
      path.dirname(originalTarget.lexicalPath),
      `.${path.basename(originalTarget.lexicalPath)}.swarmx-${randomUUID()}.tmp`,
    );
    let temporaryExists = false;
    try {
      const handle = await open(temporaryPath, "wx", originalTarget.mode);
      temporaryExists = true;
      try {
        await handle.writeFile(content);
        await handle.sync();
      } finally {
        await handle.close();
      }

      const currentTarget = await this.#writableTarget(
        root,
        normalizedPath,
        expectedSha256 !== undefined,
      );
      if (
        currentTarget.exists !== originalTarget.exists ||
        (currentTarget.exists && currentTarget.sha256 !== expectedSha256)
      ) {
        throw staleWorkspaceFileError();
      }

      if (originalTarget.exists) {
        await rename(temporaryPath, originalTarget.lexicalPath);
      } else {
        try {
          await link(temporaryPath, originalTarget.lexicalPath);
        } catch (cause) {
          if (isAlreadyExistsError(cause)) throw staleWorkspaceFileError();
          throw cause;
        }
        await unlink(temporaryPath);
      }
      temporaryExists = false;

      const sha256 = sha256Buffer(content);
      this.#readVersions.set(normalizedPath, sha256);
      return {
        path: normalizedPath,
        size: content.length,
        sha256,
        created: !originalTarget.exists,
      };
    } finally {
      if (temporaryExists) await unlink(temporaryPath).catch(() => undefined);
    }
  }

  #git(root: string, arguments_: string[], outputLimit: number): Promise<GitCommandResult> {
    const captureLimit = Math.max(1, outputLimit);
    return new Promise((resolve) => {
      execFile(
        "git",
        [
          "-C",
          root,
          "-c",
          "core.quotepath=false",
          "-c",
          "core.fsmonitor=false",
          "-c",
          "status.relativePaths=true",
          "-c",
          "diff.external=",
          ...arguments_,
        ],
        {
          encoding: "utf8",
          env: gitEnvironment(),
          maxBuffer: captureLimit + GIT_COMMAND_SLACK_BYTES,
          timeout: this.#options.gitTimeoutMs,
          windowsHide: true,
        },
        (cause, stdout, stderr) => {
          const boundedStdout = truncateUtf8(stdout, outputLimit);
          const boundedStderr = truncateUtf8(stderr, 8 * 1024);
          const overflow = isMaxBufferError(cause);
          const error = cause instanceof Error ? cause : undefined;
          resolve({
            ok: !cause || overflow,
            stdout: boundedStdout.value,
            stderr: boundedStderr.value,
            truncated: overflow || boundedStdout.truncated || boundedStderr.truncated,
            missingExecutable: isMissingExecutable(cause),
            ...(error ? { error } : {}),
          });
        },
      );
    });
  }

  #search(root: string, arguments_: string[]): Promise<SearchCommandResult> {
    const outputLimit = this.#options.maxPatchBytes;
    return new Promise((resolve) => {
      execFile(
        "rg",
        arguments_,
        {
          cwd: root,
          encoding: "utf8",
          maxBuffer: outputLimit + GIT_COMMAND_SLACK_BYTES,
          timeout: this.#options.gitTimeoutMs,
          windowsHide: true,
        },
        (cause, stdout, stderr) => {
          const boundedStdout = truncateUtf8(stdout, outputLimit);
          const boundedStderr = truncateUtf8(stderr, 8 * 1024);
          const overflow = isMaxBufferError(cause);
          const exitCode =
            cause instanceof Error && "code" in cause && typeof cause.code === "number"
              ? cause.code
              : cause
                ? null
                : 0;
          resolve({
            stdout: boundedStdout.value,
            stderr: boundedStderr.value,
            truncated: overflow || boundedStdout.truncated || boundedStderr.truncated,
            exitCode,
            missingExecutable: isMissingExecutable(cause),
          });
        },
      );
    });
  }
}

class ClaudeWorktreeManager {
  readonly #tools: WorkspaceTools;
  readonly #shell: WorkspaceShell;
  readonly #originalRoot: string;
  #state: ClaudeWorktreeState | undefined;
  #transitioning = false;

  constructor(tools: WorkspaceTools, shell: WorkspaceShell) {
    this.#tools = tools;
    this.#shell = shell;
    this.#originalRoot = tools.root;
  }

  async enter(nameValue: unknown) {
    return this.#transition(async () => {
      if (this.#state) throw new Error("Already in a worktree session.");
      const name = claudeWorktreeName(nameValue);
      const originalCwd = await realpath(this.#originalRoot);
      const repository = await claudeGit(originalCwd, ["rev-parse", "--show-toplevel"]);
      if (repository.code !== 0 || !repository.stdout.trim()) {
        throw new Error("Cannot create a worktree: the Project root is not a Git repository.");
      }
      const gitRoot = await realpath(repository.stdout.trim());
      if (gitRoot !== originalCwd) {
        throw new Error(
          "Cannot create a worktree: the Project root must be the Git repository root.",
        );
      }

      const lexicalParent = path.join(gitRoot, ".claude", "worktrees");
      await mkdir(lexicalParent, { recursive: true });
      const worktreeParent = await realpath(lexicalParent);
      if (!isContainedPath(gitRoot, worktreeParent)) {
        throw new Error("The .claude/worktrees directory escapes the Project root.");
      }
      const requestedPath = path.join(worktreeParent, name);
      const worktreeBranch = `worktree-${name}`;
      let worktreePath: string;
      let originalHeadCommit: string;

      if (await pathExists(requestedPath)) {
        worktreePath = await realpath(requestedPath);
        if (
          !isContainedPath(worktreeParent, worktreePath) ||
          !(await isRegisteredWorktree(gitRoot, worktreePath))
        ) {
          throw new Error(
            `Refusing to resume ${requestedPath}: it is not a registered Project worktree.`,
          );
        }
        const branch = await claudeGit(worktreePath, ["symbolic-ref", "--short", "HEAD"]);
        if (branch.code !== 0 || branch.stdout.trim() !== worktreeBranch) {
          throw new Error(`Refusing to resume ${worktreePath}: expected branch ${worktreeBranch}.`);
        }
        originalHeadCommit = await requiredGitOutput(worktreePath, ["rev-parse", "HEAD"]);
      } else {
        const existingBranch = await claudeGit(gitRoot, [
          "show-ref",
          "--verify",
          "--quiet",
          `refs/heads/${worktreeBranch}`,
        ]);
        if (existingBranch.code === 0) {
          throw new Error(`Cannot create worktree: branch ${worktreeBranch} already exists.`);
        }
        if (existingBranch.code !== 1) {
          throw new Error(existingBranch.stderr.trim() || "Could not inspect worktree branch.");
        }
        const created = await claudeGit(gitRoot, [
          "worktree",
          "add",
          "-b",
          worktreeBranch,
          requestedPath,
          "HEAD",
        ]);
        if (created.code !== 0) {
          throw new Error(created.stderr.trim() || "Failed to create worktree.");
        }
        worktreePath = await realpath(requestedPath);
        if (!isContainedPath(worktreeParent, worktreePath)) {
          throw new Error("Created worktree escapes the Project worktree directory.");
        }
        originalHeadCommit = await requiredGitOutput(worktreePath, ["rev-parse", "HEAD"]);
      }

      const commonDirectoryOutput = await requiredGitOutput(worktreePath, [
        "rev-parse",
        "--git-common-dir",
      ]);
      const commonDirectory = await realpath(
        path.isAbsolute(commonDirectoryOutput)
          ? commonDirectoryOutput
          : path.resolve(worktreePath, commonDirectoryOutput),
      );
      if (!isContainedPath(gitRoot, commonDirectory)) {
        throw new Error("Linked worktree Git metadata escapes the Project root.");
      }

      this.#tools.rebindRoot(worktreePath);
      this.#shell.rebindRoot(worktreePath, [commonDirectory]);
      this.#state = {
        originalCwd,
        gitRoot,
        worktreePath,
        worktreeBranch,
        originalHeadCommit,
      };
      const message = `Created worktree at ${worktreePath} on branch ${worktreeBranch}. The session is now working in the worktree. Use ExitWorktree to leave mid-session, or exit the session to preserve it.`;
      return { worktreePath, worktreeBranch, message };
    });
  }

  async exit(action: "keep" | "remove", discardChanges: boolean) {
    return this.#transition(async () => {
      const state = this.#state;
      if (!state) {
        throw new Error(
          "No-op: there is no active EnterWorktree session to exit. This tool only operates on worktrees created by EnterWorktree in the current request. No filesystem changes were made.",
        );
      }

      const status = await this.#status(state);
      if (action === "remove" && !discardChanges) {
        if (!status) {
          throw new Error(
            `Could not verify worktree state at ${state.worktreePath}. Refusing to remove without explicit confirmation. Re-invoke with discard_changes: true to proceed, or use action: "keep".`,
          );
        }
        if (status.changedFiles > 0 || status.commits > 0) {
          const changes = [
            ...(status.changedFiles > 0
              ? [
                  `${status.changedFiles} uncommitted ${status.changedFiles === 1 ? "file" : "files"}`,
                ]
              : []),
            ...(status.commits > 0
              ? [
                  `${status.commits} ${status.commits === 1 ? "commit" : "commits"} on ${state.worktreeBranch}`,
                ]
              : []),
          ];
          throw new Error(
            `Worktree has ${changes.join(" and ")}. Removing will discard this work permanently. Confirm with the user, then re-invoke with discard_changes: true, or use action: "keep".`,
          );
        }
      }

      if (action === "keep") {
        this.#restoreOriginalRoot();
        this.#state = undefined;
        const message = `Exited worktree. Your work is preserved at ${state.worktreePath} on branch ${state.worktreeBranch}. Session is now back in ${state.originalCwd}.`;
        return {
          action,
          originalCwd: state.originalCwd,
          worktreePath: state.worktreePath,
          worktreeBranch: state.worktreeBranch,
          message,
        };
      }

      await this.#shell.stopSessionsWithin(state.worktreePath);
      this.#restoreOriginalRoot();
      const removed = await claudeGit(state.gitRoot, [
        "worktree",
        "remove",
        "--force",
        state.worktreePath,
      ]);
      if (removed.code !== 0) {
        throw new Error(removed.stderr.trim() || "Failed to remove worktree.");
      }
      const deletedBranch = await claudeGit(state.gitRoot, ["branch", "-D", state.worktreeBranch]);
      this.#state = undefined;
      if (deletedBranch.code !== 0) {
        throw new Error(
          `Worktree was removed, but branch ${state.worktreeBranch} could not be deleted: ${deletedBranch.stderr.trim()}`,
        );
      }
      const discardedFiles = status?.changedFiles ?? 0;
      const discardedCommits = status?.commits ?? 0;
      const discarded = [
        ...(discardedCommits > 0
          ? [`${discardedCommits} ${discardedCommits === 1 ? "commit" : "commits"}`]
          : []),
        ...(discardedFiles > 0
          ? [`${discardedFiles} uncommitted ${discardedFiles === 1 ? "file" : "files"}`]
          : []),
      ];
      const message = `Exited and removed worktree at ${state.worktreePath}.${discarded.length > 0 ? ` Discarded ${discarded.join(" and ")}.` : ""} Session is now back in ${state.originalCwd}.`;
      return {
        action,
        originalCwd: state.originalCwd,
        worktreePath: state.worktreePath,
        worktreeBranch: state.worktreeBranch,
        discardedFiles,
        discardedCommits,
        message,
      };
    });
  }

  close(): void {
    if (!this.#state) return;
    this.#restoreOriginalRoot();
    this.#state = undefined;
  }

  async #status(state: ClaudeWorktreeState): Promise<ClaudeWorktreeStatus | null> {
    const changed = await claudeGit(state.worktreePath, ["status", "--porcelain"]);
    if (changed.code !== 0) return null;
    const commits = await claudeGit(state.worktreePath, [
      "rev-list",
      "--count",
      `${state.originalHeadCommit}..HEAD`,
    ]);
    if (commits.code !== 0) return null;
    return {
      changedFiles: outputLines(changed.stdout).filter((line) => line.trim()).length,
      commits: Number.parseInt(commits.stdout.trim(), 10) || 0,
    };
  }

  #restoreOriginalRoot(): void {
    this.#tools.rebindRoot(this.#originalRoot);
    this.#shell.rebindRoot(this.#originalRoot);
  }

  async #transition<T>(operation: () => Promise<T>): Promise<T> {
    if (this.#transitioning) throw new Error("A worktree transition is already in progress.");
    this.#transitioning = true;
    try {
      return await operation();
    } finally {
      this.#transitioning = false;
    }
  }
}

export function workspaceAgentTools(
  tools: WorkspaceTools,
  shell = new WorkspaceShell(tools.root),
  options: WorkspaceAgentToolOptions = {},
): LocalTool[] {
  const profileTools =
    workspaceToolProfile(options) === "claude_code"
      ? claudeCodeWorkspaceTools(tools, shell, options)
      : codexWorkspaceTools(tools, shell, options.apiProtocol);
  if (!options.permissionPolicy) return profileTools;
  const policy = HarnessPermissionPolicySchema.parse(options.permissionPolicy);
  return profileTools.map((tool) => permissionGuardedTool(tool, policy, options.interact));
}

const READ_ONLY_PERMISSION_TOOLS = new Set([
  "AskUserQuestion",
  "CronList",
  "EnterPlanMode",
  "ExitPlanMode",
  "Glob",
  "Grep",
  "LSP",
  "Read",
  "ReportFindings",
  "TaskCreate",
  "TaskGet",
  "TaskList",
  "TaskOutput",
  "TaskUpdate",
  "TodoWrite",
]);

const WRITE_PERMISSION_TOOLS = new Set(["Edit", "NotebookEdit", "Write", "apply_patch"]);

function permissionGuardedTool(
  tool: LocalTool,
  policy: HarnessPermissionPolicy,
  interact: WorkspaceAgentToolOptions["interact"],
): LocalTool {
  const authorize = (input: Record<string, unknown> | string) =>
    authorizeWorkspaceTool(tool.name, workspaceToolAccess(tool.name), input, policy, interact);
  if (tool.kind === "text") {
    const textTool = tool as LocalTextTool;
    return {
      ...textTool,
      call: async (input: string) => {
        await authorize(input);
        return textTool.call(input);
      },
    };
  }
  const functionTool = tool as LocalMcpTool;
  return {
    ...functionTool,
    call: async (input: Record<string, unknown>) => {
      await authorize(input);
      return functionTool.call(input);
    },
  };
}

async function authorizeWorkspaceTool(
  toolName: string,
  access: HarnessToolAccess,
  input: Record<string, unknown> | string,
  policy: HarnessPermissionPolicy,
  interact: WorkspaceAgentToolOptions["interact"],
): Promise<void> {
  const resolved = resolveHarnessToolPermission(policy, { toolName, access });
  if (resolved.decision === "allow") return;
  if (resolved.decision === "deny") {
    throw new Error(
      `Tool "${toolName}" is denied by Harness permission policy (${resolved.reason}).`,
    );
  }
  if (!interact) {
    throw new Error(
      `Tool "${toolName}" requires approval, but no interaction bridge is available.`,
    );
  }
  const response = await interact({
    kind: "tool_approval",
    title: `Allow ${toolName}?`,
    toolKind: access,
    summary: workspaceToolApprovalSummary(toolName, input),
    options: [
      { optionId: "reject_once", name: "Reject", kind: "reject_once" },
      { optionId: "allow_once", name: "Allow once", kind: "allow_once" },
    ],
  });
  if (response.kind !== "tool_approval" || response.optionId !== "allow_once") {
    throw new Error(`Tool "${toolName}" was rejected by the user.`);
  }
}

function workspaceToolAccess(toolName: string): HarnessToolAccess {
  if (READ_ONLY_PERMISSION_TOOLS.has(toolName)) return "read";
  if (WRITE_PERMISSION_TOOLS.has(toolName)) return "write";
  return "execute";
}

function workspaceToolApprovalSummary(
  toolName: string,
  input: Record<string, unknown> | string,
): string {
  if (typeof input === "string") return `${toolName} requested a bounded Project patch.`;
  const safeFields = ["file_path", "path", "workdir", "name", "action", "description", "cron"];
  const details = safeFields.flatMap((field) => {
    const value = input[field];
    return typeof value === "string" && value.trim()
      ? [`${field}: ${boundedApprovalText(value)}`]
      : [];
  });
  if ("command" in input || "cmd" in input) {
    details.push("command: Project-sandboxed shell command");
  }
  return details.length > 0
    ? `${toolName}\n${details.join("\n")}`
    : `${toolName} requested a ${workspaceToolAccess(toolName)} operation in the active Project.`;
}

function boundedApprovalText(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length <= 240 ? compact : `${compact.slice(0, 239)}…`;
}

export function workspaceToolProfile(
  options: WorkspaceAgentToolOptions = {},
): WorkspaceToolProfile {
  return /(?:claude|sonnet|opus|haiku|fable)/i.test(options.model ?? "") ? "claude_code" : "codex";
}

function claudeCodeWorkspaceTools(
  tools: WorkspaceTools,
  shell: WorkspaceShell,
  options: WorkspaceAgentToolOptions,
): LocalMcpTool[] {
  const planMode = new ClaudePlanMode();
  const worktreeManager = new ClaudeWorktreeManager(tools, shell);
  const dispose = async (): Promise<void> => {
    options.closeInteractions?.();
    worktreeManager.close();
    await Promise.allSettled([...(options.borrowShell ? [] : [shell.close()]), planMode.close()]);
  };
  const taskStore = new ClaudeTaskStore();
  const skillTool = claudeSkillTool(options);
  const lspTool = claudeLspTool(options);
  const agentTool = claudeAgentTool(options);
  const interactionTools = claudeInteractionTools(planMode, options);
  const sessionTools = claudeSessionTools(planMode, options.sessionTools);
  const worktreeTools = claudeWorktreeTools(planMode, worktreeManager);
  let todos: ClaudeTodoRecord[] = [];
  return [
    ...(agentTool ? [agentTool] : []),
    {
      name: "Bash",
      description:
        "Executes a shell command in the Project sandbox. Network access and writes outside the Project are denied.",
      inputSchema: {
        type: "object",
        properties: {
          command: { type: "string", description: "The command to execute." },
          timeout: { type: "number", description: "Optional timeout in milliseconds." },
          description: { type: "string", description: "A concise description of the command." },
          run_in_background: { type: "boolean" },
          dangerouslyDisableSandbox: { type: "boolean" },
        },
        required: ["command"],
      },
      dispose,
      call: async (input) => {
        planMode.assertCanMutate("Bash");
        if (input.dangerouslyDisableSandbox === true) {
          throw new Error("The Project sandbox cannot be disabled.");
        }
        const command = requiredToolText(input.command, "command");
        const timeoutMs =
          input.timeout === undefined
            ? undefined
            : requiredBoundedInteger(input.timeout, "timeout", 1, 600_000);
        if (input.run_in_background === true) {
          const result = await shell.startBackground(command, {
            ...(timeoutMs === undefined ? {} : { timeoutMs }),
          });
          const structuredContent = claudeBashOutput(result, result.sessionId);
          return localToolResult(
            `Command running in background with ID: ${result.sessionId}`,
            structuredContent,
          );
        }
        if (startsWithSleepCommand(command)) {
          const result = await shell.run(command, {
            ...(input.timeout === undefined ? {} : { timeoutMs }),
          });
          return localToolResult(
            formatClaudeBashResult(result),
            claudeBashOutput(result, undefined, timeoutMs),
          );
        }
        const result = await shell.runWithBackgroundFallback(
          command,
          timeoutMs ?? WORKSPACE_SHELL_DEFAULTS.timeoutMs,
        );
        if (result.status === "running") {
          const structuredContent = claudeBashOutput(result, result.sessionId);
          return localToolResult(
            formatClaudeBackgroundFallback(result, timeoutMs ?? WORKSPACE_SHELL_DEFAULTS.timeoutMs),
            structuredContent,
          );
        }
        return localToolResult(
          formatClaudeBashResult(result),
          claudeBashOutput(result, undefined, timeoutMs),
        );
      },
    },
    ...worktreeTools,
    {
      name: "Read",
      description:
        "Reads a bounded UTF-8 text file from the Project. Absolute Project paths and Project-relative paths are accepted.",
      inputSchema: {
        type: "object",
        properties: {
          file_path: { type: "string", description: "The absolute path to the file to read." },
          offset: { type: "number", description: "The line number to start reading from." },
          limit: { type: "number", description: "The number of lines to read." },
          pages: { type: "string", description: "Page range for PDF files." },
        },
        required: ["file_path"],
      },
      call: async (input) => {
        const requestedPath = requiredToolPath(input.file_path);
        if (planMode.matches(requestedPath)) {
          const result = await planMode.read();
          const totalLines = fileLines(result.content).length;
          return localToolResult(numberedFileContent(result.content, 1), {
            type: "text",
            file: {
              filePath: result.filePath,
              content: result.content,
              numLines: totalLines,
              startLine: 1,
              totalLines,
            },
          });
        }
        const result = await readForClaude(tools, input);
        const structuredContent = {
          type: "text" as const,
          file: {
            filePath: path.resolve(tools.root, result.file.path),
            content: result.file.content,
            numLines: result.numLines,
            startLine: result.startLine,
            totalLines: result.totalLines,
            ...(result.file.truncated ? { truncatedByTokenCap: true } : {}),
          },
        };
        return localToolResult(
          numberedFileContent(result.file.content, result.startLine),
          structuredContent,
        );
      },
    },
    {
      name: "Edit",
      description:
        "Performs exact string replacements in a previously read, unchanged Project file.",
      inputSchema: {
        type: "object",
        properties: {
          file_path: { type: "string", description: "The absolute path to the file to modify." },
          old_string: { type: "string", description: "The exact text to replace." },
          new_string: { type: "string", description: "The replacement text." },
          replace_all: { type: "boolean", description: "Replace all occurrences." },
        },
        required: ["file_path", "old_string", "new_string"],
      },
      call: async (input) => {
        planMode.assertCanMutate("Edit");
        const filePath = workspaceRelativePath(
          tools.root,
          requiredToolPath(input.file_path),
          false,
        );
        const oldString = requiredToolText(input.old_string, "old_string");
        const newString = requiredToolText(input.new_string, "new_string");
        const replaceAll = optionalToolBoolean(input.replace_all, "replace_all");
        const result = await tools.editFile(filePath, oldString, newString, replaceAll);
        const absolutePath = path.resolve(tools.root, result.path);
        return localToolResult(`The file ${absolutePath} has been updated successfully.`, {
          filePath: absolutePath,
          oldString,
          newString,
          originalFile: result.originalContent,
          structuredPatch: wholeFileStructuredPatch(result.originalContent, result.content),
          userModified: false,
          replaceAll,
        });
      },
    },
    {
      name: "Write",
      description:
        "Writes a UTF-8 file in the Project. Existing files must be read completely before replacement.",
      inputSchema: {
        type: "object",
        properties: {
          file_path: { type: "string", description: "The absolute path to the file to write." },
          content: { type: "string", description: "The content to write." },
        },
        required: ["file_path", "content"],
      },
      call: async (input) => {
        const requestedPath = requiredToolPath(input.file_path);
        const content = requiredToolText(input.content, "content");
        if (planMode.matches(requestedPath)) {
          const result = await planMode.write(content);
          return localToolResult(
            `The plan file ${result.filePath} has been updated successfully.`,
            {
              type: "update",
              filePath: result.filePath,
              content,
              structuredPatch: wholeFileStructuredPatch(result.originalContent, content),
              originalFile: result.originalContent,
              userModified: false,
            },
          );
        }
        planMode.assertCanMutate("Write");
        const filePath = workspaceRelativePath(tools.root, requestedPath, false);
        const result = await tools.writeFile(filePath, content);
        const absolutePath = path.resolve(tools.root, result.path);
        return localToolResult(
          result.created
            ? `File created successfully at: ${absolutePath}`
            : `The file ${absolutePath} has been updated successfully.`,
          {
            type: result.created ? "create" : "update",
            filePath: absolutePath,
            content,
            structuredPatch: wholeFileStructuredPatch(result.originalContent ?? "", content),
            originalFile: result.originalContent,
            userModified: false,
          },
        );
      },
    },
    {
      name: "Glob",
      description: "Finds Project files matching a glob pattern, sorted by ripgrep.",
      inputSchema: {
        type: "object",
        properties: {
          pattern: { type: "string", description: "The glob pattern to match files against." },
          path: { type: "string", description: "The directory to search in." },
        },
        required: ["pattern"],
      },
      call: async (input) => {
        const result = await tools.glob(
          requiredToolText(input.pattern, "pattern"),
          input.path === undefined
            ? ""
            : workspaceRelativePath(tools.root, requiredToolText(input.path, "path"), true),
        );
        const filenames = (result.filenames ?? []).map((file) =>
          path.resolve(tools.root, normalizedSearchPath(file)),
        );
        return localToolResult(filenames.join("\n") || "No files found", {
          durationMs: result.durationMs ?? 0,
          numFiles: filenames.length,
          filenames,
          truncated: result.truncated,
          totalMatches: result.totalFiles ?? filenames.length,
          countIsComplete: result.countIsComplete ?? !result.truncated,
        });
      },
    },
    {
      name: "Grep",
      description: "Searches Project files using ripgrep with bounded output.",
      inputSchema: CLAUDE_GREP_INPUT_SCHEMA,
      call: async (input) => {
        const result = await tools.grep({
          ...input,
          ...(input.path === undefined
            ? {}
            : {
                path: workspaceRelativePath(tools.root, requiredToolText(input.path, "path"), true),
              }),
        });
        const filenames = (result.filenames ?? []).map((file) =>
          path.resolve(tools.root, normalizedSearchPath(file)),
        );
        const lines = outputLines(result.output);
        const structuredContent = {
          mode: result.mode,
          numFiles: filenames.length,
          filenames,
          content: result.output,
          numLines: lines.length,
          numMatches: grepMatchCount(lines, result.mode),
          totalFiles: result.totalFiles,
          totalLines: result.totalLines,
          appliedLimit: result.appliedLimit,
          appliedOffset: result.appliedOffset,
        };
        return localToolResult(result.output || "No matches found", structuredContent);
      },
    },
    {
      name: "NotebookEdit",
      description:
        "Replaces, inserts, or deletes a Jupyter notebook cell in the Project by cell ID.",
      inputSchema: {
        type: "object",
        properties: {
          notebook_path: { type: "string", description: "The absolute path to the notebook." },
          cell_id: { type: "string", description: "The ID of the cell to edit or insert after." },
          new_source: { type: "string", description: "The new source for the cell." },
          cell_type: { type: "string", enum: ["code", "markdown"] },
          edit_mode: { type: "string", enum: ["replace", "insert", "delete"] },
        },
        required: ["notebook_path", "new_source"],
      },
      call: async (input) => {
        planMode.assertCanMutate("NotebookEdit");
        const result = await editClaudeNotebook(tools, input);
        const action =
          result.edit_mode === "insert"
            ? "Inserted"
            : result.edit_mode === "delete"
              ? "Deleted"
              : "Updated";
        return localToolResult(
          `${action} ${result.cell_type} cell ${result.cell_id ?? ""} in ${result.notebook_path}.`,
          result,
        );
      },
    },
    {
      name: "ReportFindings",
      description: "Reports verified code-review findings using repo-relative file locations.",
      inputSchema: {
        type: "object",
        properties: {
          level: { type: "string", enum: ["low", "medium", "high", "xhigh", "max"] },
          findings: {
            type: "array",
            maxItems: 32,
            items: {
              type: "object",
              properties: {
                file: { type: "string" },
                line: { type: "number" },
                summary: { type: "string" },
                failure_scenario: { type: "string" },
                category: { type: "string" },
                verdict: { type: "string", enum: ["CONFIRMED", "PLAUSIBLE"] },
                outcome: {
                  type: "string",
                  enum: ["fixed", "skipped", "no_change_needed"],
                },
              },
              required: ["file", "summary", "failure_scenario"],
            },
          },
        },
        required: ["findings"],
      },
      call: async (input) => {
        const findings = parseClaudeFindings(input.findings);
        const level = optionalStringEnum(
          input.level,
          ["low", "medium", "high", "xhigh", "max"] as const,
          "level",
        );
        const structuredContent = {
          count: findings.length,
          ...(level === undefined ? {} : { level }),
          findings,
        };
        return localToolResult(
          findings.length === 0
            ? "No findings reported."
            : `Reported ${findings.length} finding${findings.length === 1 ? "" : "s"}.`,
          structuredContent,
        );
      },
    },
    ...interactionTools,
    ...sessionTools,
    ...(skillTool ? [skillTool] : []),
    ...(lspTool ? [lspTool] : []),
    {
      name: "TaskCreate",
      description: "Creates a task in the current request's task list.",
      inputSchema: {
        type: "object",
        properties: {
          subject: { type: "string", description: "A brief title for the task." },
          description: { type: "string", description: "What needs to be done." },
          activeForm: { type: "string", description: "Present continuous progress text." },
          metadata: { type: "object" },
        },
        required: ["subject", "description"],
      },
      call: async (input) => {
        const result = taskStore.create(input);
        return localToolResult(
          `Task #${result.task.id} created successfully: ${result.task.subject}`,
          result,
        );
      },
    },
    {
      name: "TaskGet",
      description: "Retrieves a task from the current request's task list.",
      inputSchema: {
        type: "object",
        properties: { taskId: { type: "string", description: "The task ID to retrieve." } },
        required: ["taskId"],
      },
      call: async (input) => {
        const result = taskStore.get(input.taskId);
        return localToolResult(formatClaudeTask(result.task), result);
      },
    },
    {
      name: "TaskList",
      description: "Lists tasks in the current request's task list.",
      inputSchema: { type: "object", properties: {} },
      call: async () => {
        const result = taskStore.list();
        return localToolResult(formatClaudeTaskList(result.tasks), result);
      },
    },
    {
      name: "TaskUpdate",
      description: "Updates task fields, status, ownership, metadata, or dependencies.",
      inputSchema: {
        type: "object",
        properties: {
          taskId: { type: "string" },
          subject: { type: "string" },
          description: { type: "string" },
          activeForm: { type: "string" },
          status: {
            type: "string",
            enum: ["pending", "in_progress", "completed", "deleted"],
          },
          addBlocks: { type: "array", items: { type: "string" } },
          addBlockedBy: { type: "array", items: { type: "string" } },
          owner: { type: "string" },
          metadata: { type: "object" },
        },
        required: ["taskId"],
      },
      call: async (input) => {
        const result = taskStore.update(input);
        return localToolResult(
          result.success
            ? `Updated task #${result.taskId}: ${result.updatedFields.join(", ") || "no fields changed"}`
            : (result.error ?? `Task ${result.taskId} could not be updated.`),
          result,
          { isError: !result.success },
        );
      },
    },
    {
      name: "TodoWrite",
      description: "Replaces the current request's todo list with a validated todo list.",
      inputSchema: {
        type: "object",
        properties: {
          todos: {
            type: "array",
            items: {
              type: "object",
              properties: {
                content: { type: "string" },
                status: { type: "string", enum: ["pending", "in_progress", "completed"] },
                activeForm: { type: "string" },
              },
              required: ["content", "status", "activeForm"],
            },
          },
        },
        required: ["todos"],
      },
      call: async (input) => {
        const nextTodos = parseClaudeTodos(input.todos);
        const structuredContent = {
          oldTodos: todos.map((todo) => ({ ...todo })),
          newTodos: nextTodos.map((todo) => ({ ...todo })),
        };
        todos = nextTodos;
        return localToolResult("Updated todo list successfully.", structuredContent);
      },
    },
    {
      name: "TaskOutput",
      description: "Retrieves output from a running or completed background task.",
      inputSchema: {
        type: "object",
        properties: {
          task_id: { type: "string", description: "The task ID to get output from." },
          block: { type: "boolean", description: "Whether to wait for completion." },
          timeout: { type: "number", description: "Max wait time in milliseconds." },
        },
        required: ["task_id", "block", "timeout"],
      },
      call: async (input) => {
        const taskId = requiredTaskId(input.task_id);
        const block = requiredBoolean(input.block, "block");
        const timeoutMs = requiredBoundedInteger(input.timeout, "timeout", 1, 600_000);
        const result = await shell.taskOutput(taskId, { block, timeoutMs });
        return localToolResult(formatClaudeTaskOutput(result), {
          ...claudeBashOutput(result, taskId),
          status: result.status,
          exitCode: result.exitCode,
        });
      },
    },
    {
      name: "TaskStop",
      description: "Stops a running background task by ID.",
      inputSchema: {
        type: "object",
        properties: {
          task_id: { type: "string" },
          shell_id: { type: "string", description: "Deprecated; use task_id." },
        },
      },
      call: async (input) => {
        const taskId = requiredTaskId(input.task_id ?? input.shell_id);
        const result = await shell.stop(taskId);
        const structuredContent = {
          message:
            result.status === "stopped"
              ? `Task ${taskId} was stopped successfully.`
              : `Task ${taskId} had already ${result.status}.`,
          task_id: String(taskId),
          task_type: "local_bash",
          command: result.command,
        };
        return localToolResult(structuredContent.message, structuredContent);
      },
    },
  ];
}

function claudeSessionTools(
  planMode: ClaudePlanMode,
  bridge: ClaudeSessionToolBridge | undefined,
): LocalMcpTool[] {
  if (!bridge) return [];
  return [
    {
      name: "Monitor",
      description:
        "Starts a sandboxed background command and delivers bounded stdout events back to this session.",
      inputSchema: {
        type: "object",
        additionalProperties: false,
        properties: {
          command: { type: "string" },
          description: { type: "string" },
          timeout_ms: { type: "number", minimum: 1_000, default: 300_000 },
          persistent: { type: "boolean", default: false },
        },
        required: ["command", "description"],
      },
      call: async (input) => {
        planMode.assertCanMutate("Monitor");
        const persistent = optionalToolBoolean(input.persistent, "persistent");
        const timeoutMs = requiredBoundedNumber(
          input.timeout_ms ?? 300_000,
          "timeout_ms",
          1_000,
          persistent ? Number.MAX_SAFE_INTEGER : 3_600_000,
        );
        const result = await bridge.monitor({
          command: requiredToolText(input.command, "command"),
          description: requiredToolText(input.description, "description"),
          timeoutMs,
          persistent,
        });
        return localToolResult(`Monitor started with task ID: ${result.taskId}`, result);
      },
    },
    {
      name: "CronCreate",
      description: "Schedules a prompt in this session using a five-field cron expression.",
      inputSchema: {
        type: "object",
        additionalProperties: false,
        properties: {
          cron: { type: "string" },
          prompt: { type: "string" },
          recurring: { type: "boolean", default: true },
          durable: { type: "boolean", default: false },
        },
        required: ["cron", "prompt"],
      },
      call: async (input) => {
        planMode.assertCanMutate("CronCreate");
        const result = await bridge.createCron({
          cron: requiredToolText(input.cron, "cron"),
          prompt: requiredToolText(input.prompt, "prompt"),
          recurring:
            input.recurring === undefined ? true : requiredBoolean(input.recurring, "recurring"),
          durable: optionalToolBoolean(input.durable, "durable"),
        });
        return localToolResult(
          `Created scheduled job ${result.id}: ${result.humanSchedule}`,
          result,
        );
      },
    },
    {
      name: "CronDelete",
      description: "Deletes a scheduled prompt from this session.",
      inputSchema: {
        type: "object",
        additionalProperties: false,
        properties: { id: { type: "string" } },
        required: ["id"],
      },
      call: async (input) => {
        planMode.assertCanMutate("CronDelete");
        const result = await bridge.deleteCron(requiredToolText(input.id, "id"));
        return localToolResult(`Deleted scheduled job ${result.id}.`, result);
      },
    },
    {
      name: "CronList",
      description: "Lists scheduled prompts in this session.",
      inputSchema: {
        type: "object",
        additionalProperties: false,
        properties: {},
      },
      call: async () => {
        const result = await bridge.listCrons();
        return localToolResult(
          result.jobs.length === 0
            ? "No scheduled jobs."
            : result.jobs
                .map((job) => `${job.id}: ${job.humanSchedule} — ${job.prompt}`)
                .join("\n"),
          result,
        );
      },
    },
  ];
}

function claudeWorktreeTools(
  planMode: ClaudePlanMode,
  manager: ClaudeWorktreeManager,
): LocalMcpTool[] {
  return [
    {
      name: "EnterWorktree",
      description:
        "Creates an isolated Git worktree in .claude/worktrees and switches this request's Project tools into it. Use only when the user explicitly asks to work in a worktree.",
      inputSchema: {
        type: "object",
        properties: {
          name: {
            type: "string",
            description: "Optional name for the worktree. A random name is generated if omitted.",
          },
        },
      },
      call: async (input) => {
        planMode.assertCanMutate("EnterWorktree");
        const result = await manager.enter(input.name);
        return localToolResult(result.message, result);
      },
    },
    {
      name: "ExitWorktree",
      description:
        "Exits a worktree created by EnterWorktree and restores the original Project root. It can preserve the worktree or safely remove it.",
      inputSchema: {
        type: "object",
        properties: {
          action: {
            type: "string",
            enum: ["keep", "remove"],
            description: '"keep" preserves the worktree and branch; "remove" deletes both.',
          },
          discard_changes: {
            type: "boolean",
            description:
              'Required true when action is "remove" and the worktree has uncommitted files or commits after entry.',
          },
        },
        required: ["action"],
      },
      call: async (input) => {
        planMode.assertCanMutate("ExitWorktree");
        const action = optionalStringEnum(input.action, ["keep", "remove"] as const, "action");
        if (!action) throw new Error("action is required.");
        const discardChanges =
          input.discard_changes === undefined
            ? false
            : requiredBoolean(input.discard_changes, "discard_changes");
        const result = await manager.exit(action, discardChanges);
        return localToolResult(result.message, result);
      },
    },
  ];
}

function claudeAgentTool(options: WorkspaceAgentToolOptions): LocalMcpTool | undefined {
  const runAgent = options.agent;
  if (!runAgent) return undefined;
  return {
    name: "Agent",
    description: "Launches a synchronous child agent to handle a complex, multi-step task.",
    inputSchema: {
      type: "object",
      properties: {
        description: {
          type: "string",
          description: "A short (3-5 word) description of the task.",
        },
        prompt: { type: "string", description: "The task for the agent to perform." },
        subagent_type: {
          type: "string",
          description: "The type of specialized agent to use for this task.",
        },
        model: {
          type: "string",
          enum: ["sonnet", "opus", "haiku"],
          description: "Optional model override for this agent.",
        },
        resume: {
          type: "string",
          description: "Optional agent ID to resume from the current request.",
        },
      },
      required: ["description", "prompt"],
    },
    call: async (input) => {
      for (const field of ["name", "team_name", "mode", "cwd", "isolation"] as const) {
        if (input[field] !== undefined) {
          throw new Error(`${field} is unavailable for synchronous SwarmX child agents.`);
        }
      }
      if (input.run_in_background !== undefined) {
        if (input.run_in_background !== false) {
          throw new Error("Background child agents are not supported in this request lifecycle.");
        }
      }
      const model = optionalStringEnum(input.model, ["sonnet", "opus", "haiku"] as const, "model");
      const result = await runAgent({
        description: requiredNonemptyToolText(input.description, "description"),
        prompt: requiredNonemptyToolText(input.prompt, "prompt"),
        ...(input.subagent_type === undefined
          ? {}
          : {
              subagentType: requiredNonemptyToolText(input.subagent_type, "subagent_type"),
            }),
        ...(model === undefined ? {} : { model }),
        ...(input.resume === undefined
          ? {}
          : { resume: requiredNonemptyToolText(input.resume, "resume") }),
      });
      const content = result.content.map((block) => block.text).join("\n");
      if (!content.trim()) throw new Error("Child agent returned no text response.");
      return localToolResult(content, result);
    },
  };
}

function claudeInteractionTools(
  planMode: ClaudePlanMode,
  options: WorkspaceAgentToolOptions,
): LocalMcpTool[] {
  const interact = options.interact;
  if (!interact) return [];
  return [
    {
      name: "AskUserQuestion",
      description:
        "Asks the user 1-4 multiple-choice questions. Each question automatically includes a free-text Other option and waits for a human response.",
      inputSchema: {
        type: "object",
        properties: {
          questions: {
            type: "array",
            minItems: 1,
            maxItems: 4,
            items: {
              type: "object",
              properties: {
                question: { type: "string" },
                header: { type: "string" },
                options: {
                  type: "array",
                  minItems: 2,
                  maxItems: 4,
                  items: {
                    type: "object",
                    properties: {
                      label: { type: "string" },
                      description: { type: "string" },
                      preview: { type: "string" },
                    },
                    required: ["label", "description"],
                  },
                },
                multiSelect: { type: "boolean" },
              },
              required: ["question", "header", "options", "multiSelect"],
            },
          },
        },
        required: ["questions"],
      },
      call: async (input) => {
        const questions = parseClaudeQuestions(input.questions);
        const response = await interact({ kind: "questions", questions });
        if (response.kind !== "questions") {
          throw new Error("Question interaction returned an invalid response kind.");
        }
        const answers = validateClaudeAnswers(questions, response.answers);
        return localToolResult(formatClaudeAnswers(questions, answers), { questions, answers });
      },
    },
    {
      name: "EnterPlanMode",
      description:
        "Switches to read-only plan mode. Project changes and shell execution stay blocked until the user approves ExitPlanMode.",
      inputSchema: { type: "object", properties: {} },
      call: async () => {
        const filePath = await planMode.enter();
        const message = `Plan mode enabled. Inspect the Project without changing it. Write the implementation plan to ${filePath} using Write, then call ExitPlanMode for user approval.`;
        return localToolResult(message, { message });
      },
    },
    {
      name: "ExitPlanMode",
      description:
        "Presents the completed plan file to the user for approval. Use only after writing the plan in plan mode.",
      inputSchema: {
        type: "object",
        properties: {
          allowedPrompts: {
            type: "array",
            deprecated: true,
            items: {
              type: "object",
              properties: { tool: { type: "string", enum: ["Bash"] }, prompt: { type: "string" } },
              required: ["tool", "prompt"],
            },
          },
        },
      },
      call: async () => {
        const { filePath, plan } = await planMode.planForApproval();
        const response = await interact({ kind: "plan_approval", plan, filePath });
        if (response.kind !== "plan_approval") {
          throw new Error("Plan approval interaction returned an invalid response kind.");
        }
        const structuredContent = { plan, isAgent: false, filePath };
        if (!response.approved) {
          const feedback = response.feedback?.trim();
          return localToolResult(
            feedback
              ? `User rejected the plan and stayed in plan mode. Feedback: ${feedback}`
              : "User rejected the plan and stayed in plan mode.",
            structuredContent,
            { isError: true },
          );
        }
        planMode.approve();
        return localToolResult("User approved Claude's plan. Plan mode is now inactive.", {
          ...structuredContent,
          planWasEdited: false,
        });
      },
    },
  ];
}

function codexWorkspaceTools(
  tools: WorkspaceTools,
  shell: WorkspaceShell,
  apiProtocol: ModelApi | undefined,
): LocalTool[] {
  const dispose = (): Promise<void> => shell.close();
  const applyPatchCall = async (input: string) => {
    const result = await tools.applyPatch(input);
    return localToolResult(formatCodexPatchResult(result), result);
  };
  const applyPatch: LocalTool =
    apiProtocol === "openai_responses"
      ? {
          kind: "text",
          name: "apply_patch",
          description:
            "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
          format: { type: "grammar", syntax: "lark", definition: CODEX_APPLY_PATCH_GRAMMAR },
          call: applyPatchCall,
        }
      : {
          name: "apply_patch",
          description: "Apply a Codex patch envelope to files in the Project.",
          inputSchema: {
            type: "object",
            properties: { patch: { type: "string", description: "The complete patch envelope." } },
            required: ["patch"],
          },
          call: async (input) => applyPatchCall(requiredToolText(input.patch, "patch")),
        };
  return [
    {
      name: "exec_command",
      description:
        "Runs a command in a pipe or PTY session, returning output or a session ID for ongoing interaction.",
      inputSchema: {
        type: "object",
        properties: {
          cmd: { type: "string", description: "Shell command to execute." },
          workdir: {
            type: "string",
            description: "Working directory for the command. Defaults to the turn cwd.",
          },
          tty: {
            type: "boolean",
            description: "True allocates a PTY; false or omitted uses plain pipes.",
          },
          yield_time_ms: {
            type: "number",
            description:
              "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms.",
          },
          max_output_tokens: {
            type: "number",
            description:
              "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy.",
          },
          shell: {
            type: "string",
            description: "Shell binary to launch. Defaults to the user's default shell.",
          },
          login: {
            type: "boolean",
            description:
              "True runs the shell with -l/-i semantics; false disables them. Defaults to true.",
          },
          sandbox_permissions: { type: "string" },
          justification: { type: "string" },
          prefix_rule: { type: "array", items: { type: "string" } },
        },
        required: ["cmd"],
      },
      dispose,
      call: async (input) => {
        if (input.sandbox_permissions === "require_escalated") {
          throw new Error("The Project sandbox cannot be escalated.");
        }
        const result = await shell.exec(requiredToolText(input.cmd, "cmd"), {
          ...(input.tty === undefined ? {} : { tty: requiredBoolean(input.tty, "tty") }),
          ...(input.workdir === undefined
            ? {}
            : { workdir: requiredToolText(input.workdir, "workdir") }),
          ...(input.yield_time_ms === undefined
            ? {}
            : {
                yieldTimeMs: requiredBoundedInteger(
                  input.yield_time_ms,
                  "yield_time_ms",
                  250,
                  30_000,
                ),
              }),
          ...(input.max_output_tokens === undefined
            ? {}
            : {
                maxOutputTokens: requiredBoundedInteger(
                  input.max_output_tokens,
                  "max_output_tokens",
                  1,
                  50_000,
                ),
              }),
        });
        return codexExecToolResult(result);
      },
    },
    {
      name: "write_stdin",
      description: "Writes characters to an existing exec session and returns recent output.",
      inputSchema: {
        type: "object",
        properties: {
          session_id: {
            type: "number",
            description: "Identifier of the running unified exec session.",
          },
          chars: {
            type: "string",
            description: "Bytes to write to stdin. Defaults to empty, which polls without writing.",
          },
          yield_time_ms: {
            type: "number",
            description:
              "Wait before yielding output. Non-empty writes default to 250 ms; empty polls default to 5000 ms.",
          },
          max_output_tokens: { type: "number" },
        },
        required: ["session_id"],
      },
      call: async (input) => {
        const chars = input.chars === undefined ? "" : requiredToolText(input.chars, "chars");
        const result = await shell.writeStdin(
          requiredBoundedInteger(input.session_id, "session_id", 1, Number.MAX_SAFE_INTEGER),
          chars,
          {
            ...(input.yield_time_ms === undefined
              ? {}
              : {
                  yieldTimeMs: requiredBoundedInteger(
                    input.yield_time_ms,
                    "yield_time_ms",
                    0,
                    chars ? 30_000 : 300_000,
                  ),
                }),
            ...(input.max_output_tokens === undefined
              ? {}
              : {
                  maxOutputTokens: requiredBoundedInteger(
                    input.max_output_tokens,
                    "max_output_tokens",
                    1,
                    50_000,
                  ),
                }),
          },
        );
        return codexExecToolResult(result);
      },
    },
    applyPatch,
  ];
}

export function projectAgentContextMessage(
  root: string,
  options: WorkspaceAgentToolOptions = {},
): string {
  const projectName = path.basename(path.resolve(root)) || "Project";
  const profile = workspaceToolProfile(options);
  const claudeToolNames = [
    ...(options.agent ? ["Agent"] : []),
    "Bash",
    "EnterWorktree",
    "ExitWorktree",
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "NotebookEdit",
    "ReportFindings",
    ...(options.interact ? ["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"] : []),
    ...(options.sessionTools ? ["Monitor", "CronCreate", "CronDelete", "CronList"] : []),
    ...(options.skills?.length ? ["Skill"] : []),
    ...(options.lsp ? ["LSP"] : []),
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskUpdate",
    "TodoWrite",
    "TaskOutput",
    "TaskStop",
  ];
  const toolNames =
    profile === "claude_code"
      ? `${claudeToolNames.slice(0, -1).join(", ")}, and ${claudeToolNames.at(-1)}`
      : "exec_command, write_stdin, and apply_patch";
  return [
    `Active Project: ${projectName}`,
    `Project root: ${path.resolve(root)}`,
    `You have ${toolNames} tools rooted at this Project.`,
    "For questions about this Project, inspect relevant files before answering. Start with README or a package manifest when appropriate.",
    profile === "claude_code"
      ? "Read every existing file completely before replacing or editing it. Prefer Edit for focused changes, then use Bash for bounded builds or tests."
      : "Use apply_patch for focused file changes and exec_command for bounded inspection, builds, or tests.",
    "The command sandbox denies network access and writes outside this Project.",
    "Never claim that Project files are unavailable before attempting the workspace tools.",
  ].join("\n");
}

const CLAUDE_LSP_OPERATIONS = [
  "goToDefinition",
  "findReferences",
  "hover",
  "documentSymbol",
  "workspaceSymbol",
  "goToImplementation",
  "prepareCallHierarchy",
  "incomingCalls",
  "outgoingCalls",
] as const;

function claudeLspTool(options: WorkspaceAgentToolOptions): LocalMcpTool | undefined {
  if (!options.lsp) return undefined;
  return {
    name: "LSP",
    description: `Interact with Language Server Protocol (LSP) servers to get code intelligence features.
Supported operations:
- goToDefinition: Find where a symbol is defined
- findReferences: Find all references to a symbol
- hover: Get hover information (documentation, type info) for a symbol
- documentSymbol: Get all symbols (functions, classes, variables) in a document
- workspaceSymbol: Search for symbols matching a query across the entire workspace
- goToImplementation: Find implementations of an interface or abstract method
- prepareCallHierarchy: Get call hierarchy item at a position (functions/methods)
- incomingCalls: Find all functions/methods that call the function at a position
- outgoingCalls: Find all functions/methods called by the function at a position
All operations require filePath, line, and character. Lines and character offsets are 1-based. workspaceSymbol also requires query.`,
    inputSchema: {
      type: "object",
      properties: {
        operation: {
          type: "string",
          enum: CLAUDE_LSP_OPERATIONS,
          description: "The LSP operation to perform",
        },
        filePath: {
          type: "string",
          description: "The absolute or relative path to the file",
        },
        line: {
          type: "number",
          description: "The line number (1-based, as shown in editors)",
        },
        character: {
          type: "number",
          description: "The character offset (1-based, as shown in editors)",
        },
        query: {
          type: "string",
          description: "The symbol name or partial name to search for (workspaceSymbol only).",
        },
      },
      required: ["operation", "filePath", "line", "character"],
    },
    call: async (input) => {
      const operation = optionalStringEnum(input.operation, CLAUDE_LSP_OPERATIONS, "operation");
      if (!operation) throw new Error("operation is required.");
      const request: ClaudeLspRequest = {
        operation,
        filePath: requiredToolText(input.filePath, "filePath"),
        line: requiredBoundedInteger(input.line, "line", 1, 10_000_000),
        character: requiredBoundedInteger(input.character, "character", 1, 10_000_000),
        ...(input.query === undefined ? {} : { query: requiredToolText(input.query, "query") }),
      };
      const result = await options.lsp?.(request);
      if (!result) throw new Error("LSP backend is unavailable.");
      return localToolResult(result.result, result);
    },
  };
}

function claudeSkillTool(options: WorkspaceAgentToolOptions): LocalMcpTool | undefined {
  const skills = (options.skills ?? []).filter(
    (skill) =>
      Boolean(skill.id.trim()) && Boolean(skill.filePath.trim()) && path.isAbsolute(skill.filePath),
  );
  if (skills.length === 0) return undefined;
  const available = skills
    .map(
      (skill) =>
        `- ${skill.name && skill.name !== skill.id ? `${skill.name} (${skill.id})` : skill.id}${skill.description ? `: ${skill.description}` : ""}`,
    )
    .join("\n");
  return {
    name: "Skill",
    description: `Executes one selected skill by loading its instructions on demand.\nAvailable skills:\n${available}`,
    inputSchema: {
      type: "object",
      properties: {
        skill: { type: "string", description: "The configured skill name or ID." },
        args: { type: "string", description: "Optional arguments passed to the skill." },
      },
      required: ["skill"],
    },
    call: async (input) => {
      const requested = requiredNonemptyToolText(input.skill, "skill");
      const matches = skills.filter((skill) => skill.id === requested || skill.name === requested);
      if (matches.length === 0) throw new Error(`Skill ${requested} is not available.`);
      if (matches.length > 1) throw new Error(`Skill ${requested} is ambiguous.`);
      const skill = matches[0];
      if (!skill) throw new Error(`Skill ${requested} is not available.`);
      const args = optionalToolText(input.args, "args") ?? "";
      const loaded = await loadClaudeSkill(skill, args, options);
      return localToolResult(loaded.content, {
        skill: skill.id,
        args,
        content: loaded.content,
        sourcePath: loaded.sourcePath,
      });
    },
  };
}

async function loadClaudeSkill(
  skill: WorkspaceAgentSkill,
  args: string,
  options: WorkspaceAgentToolOptions,
): Promise<{ content: string; sourcePath: string }> {
  let candidate = path.resolve(skill.filePath);
  const candidateStat = await stat(candidate);
  if (candidateStat.isDirectory()) candidate = path.join(candidate, "SKILL.md");
  const sourcePath = await realpath(candidate);
  const sourceStat = await stat(sourcePath);
  if (!sourceStat.isFile()) throw new Error(`Skill ${skill.id} does not resolve to a file.`);
  if (sourceStat.size > MAX_SKILL_BYTES) {
    throw new Error(`Skill ${skill.id} exceeds the ${MAX_SKILL_BYTES}-byte limit.`);
  }
  const buffer = await readFile(sourcePath);
  if (buffer.includes(0))
    throw new Error(`Skill ${skill.id} must be UTF-8 text without NUL bytes.`);
  let document: string;
  try {
    document = new TextDecoder("utf-8", { fatal: true }).decode(buffer);
  } catch {
    throw new Error(`Skill ${skill.id} must be UTF-8 text without NUL bytes.`);
  }
  const parsed = parseClaudeSkillDocument(document);
  return {
    content: expandClaudeSkill(parsed.body, parsed.argumentNames, args, {
      skillDirectory: path.dirname(sourcePath),
      effort: options.effort,
      sessionId: options.sessionId,
    }),
    sourcePath,
  };
}

function parseClaudeSkillDocument(content: string): {
  body: string;
  argumentNames: string[];
} {
  if (!content.startsWith("---\n") && !content.startsWith("---\r\n")) {
    return { body: content, argumentNames: [] };
  }
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(content);
  if (!match) return { body: content, argumentNames: [] };
  const frontmatter = match[1] ?? "";
  const body = content.slice(match[0].length);
  const lines = frontmatter.split(/\r?\n/);
  const argumentLine = lines.findIndex((line) => /^arguments\s*:/.test(line));
  if (argumentLine < 0) return { body, argumentNames: [] };
  const inline = lines[argumentLine]?.replace(/^arguments\s*:\s*/, "").trim() ?? "";
  if (inline) {
    const value = inline.startsWith("[") && inline.endsWith("]") ? inline.slice(1, -1) : inline;
    return {
      body,
      argumentNames: value
        .split(inline.startsWith("[") ? "," : /\s+/)
        .map((name) => name.trim().replace(/^['"]|['"]$/g, ""))
        .filter((name) => /^[A-Za-z_][A-Za-z0-9_-]*$/.test(name)),
    };
  }
  const argumentNames: string[] = [];
  for (const line of lines.slice(argumentLine + 1)) {
    const item = /^\s+-\s*([A-Za-z_][A-Za-z0-9_-]*)\s*$/.exec(line);
    if (!item) break;
    argumentNames.push(item[1] ?? "");
  }
  return { body, argumentNames: argumentNames.filter(Boolean) };
}

function expandClaudeSkill(
  body: string,
  argumentNames: string[],
  rawArguments: string,
  variables: { skillDirectory: string; effort?: string; sessionId?: string },
): string {
  const arguments_ = parseClaudeSkillArguments(rawArguments);
  const usesArguments =
    /\$ARGUMENTS(?:\[\d+\])?|\$\d+/.test(body) ||
    argumentNames.some((name) =>
      new RegExp(`\\$${escapeRegExp(name)}(?![A-Za-z0-9_-])`).test(body),
    );
  let expanded = body.replace(/\$ARGUMENTS\[(\d+)\]/g, (_match, index: string) => {
    return arguments_[Number(index)] ?? "";
  });
  expanded = expanded.replace(/\$(\d+)/g, (_match, index: string) => {
    return arguments_[Number(index)] ?? "";
  });
  for (const [index, name] of argumentNames.entries()) {
    expanded = expanded.replace(
      new RegExp(`\\$${escapeRegExp(name)}(?![A-Za-z0-9_-])`, "g"),
      arguments_[index] ?? "",
    );
  }
  expanded = expanded.replaceAll("$ARGUMENTS", rawArguments);
  expanded = expanded.replaceAll("${CLAUDE_SKILL_DIR}", variables.skillDirectory);
  if (variables.effort !== undefined) {
    expanded = expanded.replaceAll("${CLAUDE_EFFORT}", variables.effort);
  }
  if (variables.sessionId !== undefined) {
    expanded = expanded.replaceAll("${CLAUDE_SESSION_ID}", variables.sessionId);
  }
  if (rawArguments && !usesArguments)
    expanded = `${expanded.trimEnd()}\n\nARGUMENTS: ${rawArguments}\n`;
  return expanded;
}

function parseClaudeSkillArguments(input: string): string[] {
  const values: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  let escaped = false;
  const flush = (): void => {
    if (current) values.push(current);
    current = "";
  };
  for (const character of input) {
    if (escaped) {
      current += character;
      escaped = false;
    } else if (character === "\\" && quote !== "'") {
      escaped = true;
    } else if (quote) {
      if (character === quote) quote = undefined;
      else current += character;
    } else if (character === "'" || character === '"') {
      quote = character;
    } else if (/\s/.test(character)) {
      flush();
    } else {
      current += character;
    }
  }
  if (escaped) current += "\\";
  if (quote) throw new Error("Skill args contain an unterminated quote.");
  flush();
  return values;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

async function editClaudeNotebook(
  tools: WorkspaceTools,
  input: Record<string, unknown>,
): Promise<{
  new_source: string;
  old_source?: string;
  cell_id?: string;
  cell_type: "code" | "markdown";
  language: string;
  edit_mode: "replace" | "insert" | "delete";
  notebook_path: string;
  original_file: string;
  updated_file: string;
}> {
  const notebookPath = workspaceRelativePath(
    tools.root,
    requiredToolPath(input.notebook_path),
    false,
  );
  if (path.extname(notebookPath).toLowerCase() !== ".ipynb") {
    throw new Error("notebook_path must identify a .ipynb file.");
  }
  const original = await tools.readFile(notebookPath);
  if (original.truncated) throw new Error("The notebook must be read completely before editing.");
  const notebook = parseClaudeNotebook(original.content);
  const cells = notebook.cells as ClaudeNotebookCell[];
  const newSource = requiredToolText(input.new_source, "new_source");
  const editMode =
    optionalStringEnum(input.edit_mode, ["replace", "insert", "delete"] as const, "edit_mode") ??
    "replace";
  const requestedCellType = optionalStringEnum(
    input.cell_type,
    ["code", "markdown"] as const,
    "cell_type",
  );
  const cellId = optionalNonemptyToolText(input.cell_id, "cell_id");

  let outputCellId: string | undefined;
  let outputCellType: "code" | "markdown";
  let oldSource: string | undefined;
  if (editMode === "insert") {
    if (requestedCellType === undefined) {
      throw new Error("cell_type is required when edit_mode is insert.");
    }
    const anchorIndex = cellId === undefined ? -1 : cells.findIndex((cell) => cell.id === cellId);
    if (cellId !== undefined && anchorIndex < 0) throw new Error(`Cell ${cellId} was not found.`);
    outputCellId = uniqueNotebookCellId(cells);
    outputCellType = requestedCellType;
    const cell: ClaudeNotebookCell = {
      cell_type: outputCellType,
      id: outputCellId,
      metadata: {},
      source: newSource,
      ...(outputCellType === "code" ? { execution_count: null, outputs: [] } : {}),
    };
    cells.splice(anchorIndex + 1, 0, cell);
  } else {
    if (cellId === undefined) throw new Error(`cell_id is required when edit_mode is ${editMode}.`);
    const cellIndex = cells.findIndex((cell) => cell.id === cellId);
    if (cellIndex < 0) throw new Error(`Cell ${cellId} was not found.`);
    const cell = cells[cellIndex];
    if (!cell || (cell.cell_type !== "code" && cell.cell_type !== "markdown")) {
      throw new Error(`Cell ${cellId} must be a code or markdown cell.`);
    }
    outputCellId = cellId;
    outputCellType = requestedCellType ?? cell.cell_type;
    oldSource = notebookSourceText(cell.source);
    if (editMode === "delete") {
      cells.splice(cellIndex, 1);
    } else {
      cell.cell_type = outputCellType;
      cell.source = notebookSourceValue(cell.source, newSource);
      if (outputCellType === "code") {
        if (!("execution_count" in cell)) cell.execution_count = null;
        if (!Array.isArray(cell.outputs)) cell.outputs = [];
      } else {
        Reflect.deleteProperty(cell, "execution_count");
        Reflect.deleteProperty(cell, "outputs");
      }
    }
  }

  const updatedFile = serializeClaudeNotebook(notebook, original.content);
  const writeResult = await tools.writeFile(notebookPath, updatedFile);
  const absolutePath = path.resolve(tools.root, writeResult.path);
  return {
    new_source: newSource,
    ...(oldSource === undefined ? {} : { old_source: oldSource }),
    ...(outputCellId === undefined ? {} : { cell_id: outputCellId }),
    cell_type: outputCellType,
    language: claudeNotebookLanguage(notebook),
    edit_mode: editMode,
    notebook_path: absolutePath,
    original_file: original.content,
    updated_file: updatedFile,
  };
}

function parseClaudeNotebook(content: string): Record<string, unknown> & { cells: unknown[] } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    throw new Error("The notebook is not valid JSON.");
  }
  if (!isToolRecord(parsed) || !Array.isArray(parsed.cells)) {
    throw new Error("The notebook must contain a cells array.");
  }
  for (const cell of parsed.cells) {
    if (
      !isToolRecord(cell) ||
      typeof cell.cell_type !== "string" ||
      !(
        typeof cell.source === "string" ||
        (Array.isArray(cell.source) && cell.source.every((line) => typeof line === "string"))
      ) ||
      (cell.id !== undefined && typeof cell.id !== "string")
    ) {
      throw new Error("The notebook contains an invalid cell.");
    }
  }
  return parsed as Record<string, unknown> & { cells: unknown[] };
}

function uniqueNotebookCellId(cells: ClaudeNotebookCell[]): string {
  const ids = new Set(cells.map((cell) => cell.id).filter((id): id is string => Boolean(id)));
  for (;;) {
    const candidate = randomUUID().replaceAll("-", "").slice(0, 8);
    if (!ids.has(candidate)) return candidate;
  }
}

function notebookSourceText(source: string | string[]): string {
  return Array.isArray(source) ? source.join("") : source;
}

function notebookSourceValue(previous: string | string[], next: string): string | string[] {
  if (!Array.isArray(previous)) return next;
  return next.match(/[^\n]*\n|[^\n]+$/g) ?? [];
}

function serializeClaudeNotebook(notebook: Record<string, unknown>, original: string): string {
  const indentationMatch = original.match(/\n([\t ]+)\S/);
  const indentation = indentationMatch?.[1]?.includes("\t")
    ? "\t"
    : Math.min(indentationMatch?.[1]?.length ?? 2, 8);
  return `${JSON.stringify(notebook, null, indentation)}${original.endsWith("\n") ? "\n" : ""}`;
}

function claudeNotebookLanguage(notebook: Record<string, unknown>): string {
  const metadata = isToolRecord(notebook.metadata) ? notebook.metadata : {};
  const languageInfo = isToolRecord(metadata.language_info) ? metadata.language_info : {};
  const kernelspec = isToolRecord(metadata.kernelspec) ? metadata.kernelspec : {};
  return typeof languageInfo.name === "string"
    ? languageInfo.name
    : typeof kernelspec.language === "string"
      ? kernelspec.language
      : "unknown";
}

function parseClaudeTodos(value: unknown): ClaudeTodoRecord[] {
  if (!Array.isArray(value)) throw new Error("todos must be an array.");
  return value.map((item, index) => {
    if (!isToolRecord(item)) throw new Error(`todos[${index}] must be an object.`);
    return {
      content: requiredNonemptyToolText(item.content, `todos[${index}].content`),
      status: requiredClaudeTaskStatus(item.status, `todos[${index}].status`),
      activeForm: requiredNonemptyToolText(item.activeForm, `todos[${index}].activeForm`),
    };
  });
}

function parseClaudeFindings(value: unknown) {
  if (!Array.isArray(value)) throw new Error("findings must be an array.");
  if (value.length > 32) throw new Error("findings must contain at most 32 items.");
  return value.map((item, index) => {
    if (!isToolRecord(item)) throw new Error(`findings[${index}] must be an object.`);
    const line = optionalPositiveInteger(item.line, `findings[${index}].line`);
    const category = optionalNonemptyToolText(item.category, `findings[${index}].category`);
    const verdict = optionalStringEnum(
      item.verdict,
      ["CONFIRMED", "PLAUSIBLE"] as const,
      `findings[${index}].verdict`,
    );
    const outcome = optionalStringEnum(
      item.outcome,
      ["fixed", "skipped", "no_change_needed"] as const,
      `findings[${index}].outcome`,
    );
    return {
      file: normalizeRelativePath(
        requiredNonemptyToolText(item.file, `findings[${index}].file`),
        false,
      ),
      ...(line === undefined ? {} : { line }),
      summary: requiredNonemptyToolText(item.summary, `findings[${index}].summary`),
      failure_scenario: requiredNonemptyToolText(
        item.failure_scenario,
        `findings[${index}].failure_scenario`,
      ),
      ...(category === undefined ? {} : { category }),
      ...(verdict === undefined ? {} : { verdict }),
      ...(outcome === undefined ? {} : { outcome }),
    };
  });
}

function formatClaudeTask(
  task: {
    id: string;
    subject: string;
    description: string;
    status: ClaudeTaskStatus;
    blocks: string[];
    blockedBy: string[];
  } | null,
): string {
  if (!task) return "Task not found.";
  return [
    `#${task.id} [${task.status}] ${task.subject}`,
    task.description,
    ...(task.blockedBy.length > 0 ? [`Blocked by: ${task.blockedBy.join(", ")}`] : []),
    ...(task.blocks.length > 0 ? [`Blocks: ${task.blocks.join(", ")}`] : []),
  ].join("\n");
}

function formatClaudeTaskList(
  tasks: Array<{
    id: string;
    subject: string;
    status: ClaudeTaskStatus;
    owner?: string;
    blockedBy: string[];
  }>,
): string {
  if (tasks.length === 0) return "No tasks found.";
  return tasks
    .map(
      (task) =>
        `#${task.id} [${task.status}] ${task.subject}${task.owner ? ` (${task.owner})` : ""}${task.blockedBy.length > 0 ? ` [blocked by ${task.blockedBy.join(", ")}]` : ""}`,
    )
    .join("\n");
}

async function readForClaude(
  tools: WorkspaceTools,
  input: Record<string, unknown>,
): Promise<{
  file: WorkspaceTextFile;
  startLine: number;
  numLines: number;
  totalLines: number;
}> {
  if (input.pages !== undefined) {
    throw new Error("PDF page reads are not supported; Read accepts UTF-8 Project files.");
  }
  const result = await tools.readFile(
    workspaceRelativePath(tools.root, requiredToolPath(input.file_path), false),
  );
  const offset = optionalPositiveInteger(input.offset, "offset") ?? 1;
  const limit = optionalPositiveInteger(input.limit, "limit");
  const lines = fileLines(result.content);
  if (offset === 1 && limit === undefined) {
    return {
      file: result,
      startLine: 1,
      numLines: lines.length,
      totalLines: lines.length,
    };
  }
  const selected = lines.slice(offset - 1, limit === undefined ? undefined : offset - 1 + limit);
  return {
    file: { ...result, content: selected.join("\n") },
    startLine: offset,
    numLines: selected.length,
    totalLines: lines.length,
  };
}

function claudeWorktreeName(value: unknown): string {
  const name = value === undefined ? randomClaudeWorktreeName() : value;
  if (
    typeof name !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name) ||
    name === "." ||
    name === ".."
  ) {
    throw new Error(
      "Worktree name must be 1-64 portable letters, numbers, underscores, or hyphens and cannot contain path separators.",
    );
  }
  return name;
}

function randomClaudeWorktreeName(): string {
  const adjectives = ["calm", "bright", "swift", "keen", "bold"];
  const nouns = ["fox", "owl", "elm", "oak", "ray"];
  const suffix = randomUUID().replaceAll("-", "").slice(0, 4);
  const adjective = adjectives[Math.floor(Math.random() * adjectives.length)] ?? "calm";
  const noun = nouns[Math.floor(Math.random() * nouns.length)] ?? "fox";
  return `${adjective}-${noun}-${suffix}`;
}

async function pathExists(candidate: string): Promise<boolean> {
  try {
    await lstat(candidate);
    return true;
  } catch (cause) {
    if (isFileNotFoundError(cause)) return false;
    throw cause;
  }
}

async function isRegisteredWorktree(gitRoot: string, worktreePath: string): Promise<boolean> {
  const listed = await claudeGit(gitRoot, ["worktree", "list", "--porcelain"]);
  if (listed.code !== 0) return false;
  for (const line of outputLines(listed.stdout)) {
    if (!line.startsWith("worktree ")) continue;
    const listedPath = line.slice("worktree ".length);
    try {
      if ((await realpath(listedPath)) === worktreePath) return true;
    } catch {
      // Stale worktree registrations are not resumable.
    }
  }
  return false;
}

async function requiredGitOutput(cwd: string, arguments_: string[]): Promise<string> {
  const result = await claudeGit(cwd, arguments_);
  const output = result.stdout.trim();
  if (result.code !== 0 || !output) {
    throw new Error(result.stderr.trim() || `git ${arguments_.join(" ")} failed.`);
  }
  return output;
}

function claudeGit(cwd: string, arguments_: string[]): Promise<ClaudeGitResult> {
  return new Promise((resolve) => {
    execFile(
      "git",
      ["-c", "core.quotepath=false", "-c", "core.fsmonitor=false", ...arguments_],
      {
        cwd,
        encoding: "utf8",
        env: gitEnvironment(),
        maxBuffer: MAX_GIT_STATUS_BYTES + GIT_COMMAND_SLACK_BYTES,
        timeout: WORKSPACE_TOOLS_HARD_LIMITS.gitTimeoutMs,
        windowsHide: true,
      },
      (cause, stdout, stderr) => {
        resolve({
          code:
            cause instanceof Error && "code" in cause && typeof cause.code === "number"
              ? cause.code
              : cause
                ? null
                : 0,
          stdout: truncateUtf8(stdout, MAX_GIT_STATUS_BYTES).value,
          stderr: truncateUtf8(stderr, 16 * 1024).value,
        });
      },
    );
  });
}

function workspaceRelativePath(root: string, value: string, allowRoot: boolean): string {
  if (typeof value !== "string" || value.includes("\0")) {
    throw new Error("Workspace path must be valid text.");
  }
  if (!path.isAbsolute(value)) return normalizeRelativePath(value, allowRoot);
  const resolvedRoot = path.resolve(root);
  const candidate = path.resolve(value);
  if (!isContainedPath(resolvedRoot, candidate)) {
    throw new Error("Workspace path escapes the root.");
  }
  return normalizeRelativePath(toPortablePath(path.relative(resolvedRoot, candidate)), allowRoot);
}

function searchResult(result: SearchCommandResult): WorkspaceSearchResult {
  if (result.missingExecutable) throw new Error("ripgrep (rg) is required for Glob and Grep.");
  if (result.exitCode !== 0 && result.exitCode !== 1 && !result.truncated) {
    throw new Error(result.stderr.trim() || "ripgrep search failed.");
  }
  return { output: result.stdout, truncated: result.truncated };
}

function outputLines(output: string): string[] {
  if (!output) return [];
  const lines = output.split(/\r?\n/);
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

function searchFilenames(
  lines: readonly string[],
  mode: "content" | "files_with_matches" | "count",
): string[] {
  const filenames = new Set<string>();
  for (const line of lines) {
    if (!line || line === "--") continue;
    if (mode === "files_with_matches") {
      filenames.add(line);
      continue;
    }
    const match =
      mode === "count" ? line.match(/^(.+):\d+$/) : line.match(/^(.+?)(?::|-)\d+(?::|-)/);
    if (match?.[1]) filenames.add(match[1]);
  }
  return [...filenames];
}

function normalizedSearchPath(value: string): string {
  return value.replace(/^\.\//, "");
}

function grepMatchCount(
  lines: readonly string[],
  mode: "content" | "files_with_matches" | "count" | undefined,
): number {
  if (mode === "count") {
    return lines.reduce((total, line) => {
      const match = line.match(/:(\d+)$/);
      return total + (match ? Number.parseInt(match[1], 10) : 0);
    }, 0);
  }
  return lines.filter((line) => line !== "--").length;
}

function numberedFileContent(content: string, startLine: number): string {
  return fileLines(content)
    .map((line, index) => `${String(startLine + index).padStart(6)}→${line}`)
    .join("\n");
}

function fileLines(content: string): string[] {
  if (!content) return [];
  const lines = content.split(/\r?\n/);
  if (lines.at(-1) === "") lines.pop();
  return lines;
}

function wholeFileStructuredPatch(
  originalContent: string,
  content: string,
): Array<{
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: string[];
}> {
  if (originalContent === content) return [];
  const originalLines = fileLines(originalContent);
  const newLines = fileLines(content);
  return [
    {
      oldStart: 1,
      oldLines: originalLines.length,
      newStart: 1,
      newLines: newLines.length,
      lines: [...originalLines.map((line) => `-${line}`), ...newLines.map((line) => `+${line}`)],
    },
  ];
}

function claudeBashOutput(
  result: WorkspaceShellResult | WorkspaceShellSessionSnapshot,
  backgroundTaskId?: number,
  timeoutMs?: number,
): {
  stdout: string;
  stderr: string;
  interrupted: boolean;
  backgroundTaskId?: string;
  timedOutAfterMs?: number;
} {
  return {
    stdout: result.stdout,
    stderr: result.stderr,
    interrupted: result.signal !== null || result.timedOut,
    ...(backgroundTaskId === undefined ? {} : { backgroundTaskId: String(backgroundTaskId) }),
    ...(result.timedOut
      ? { timedOutAfterMs: timeoutMs ?? WORKSPACE_SHELL_DEFAULTS.timeoutMs }
      : {}),
  };
}

function formatClaudeBashResult(result: WorkspaceShellResult): string {
  const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
  const status = result.timedOut
    ? `Command timed out after ${result.durationMs}ms.`
    : result.exitCode === 0
      ? ""
      : `Command exited with code ${result.exitCode ?? "unknown"}.`;
  return [output || (status ? "" : "Command completed successfully."), status]
    .filter(Boolean)
    .join("\n");
}

function formatClaudeBackgroundFallback(
  result: WorkspaceShellSessionSnapshot,
  timeoutMs: number,
): string {
  const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
  return [
    output,
    `Command did not complete within its ${timeoutMs / 1000}s timeout and was moved to the background.`,
    `Task ID: ${result.sessionId}`,
  ]
    .filter(Boolean)
    .join("\n");
}

function startsWithSleepCommand(command: string): boolean {
  return /^\s*(?:builtin\s+)?sleep(?:\s|$)/.test(command);
}

function formatClaudeTaskOutput(result: WorkspaceShellSessionSnapshot): string {
  const retrievalStatus = result.status === "running" ? "running" : "success";
  const output = [result.stdout, result.stderr].filter(Boolean).join("\n");
  return [
    `<retrieval_status>${retrievalStatus}</retrieval_status>`,
    `<task_id>${result.sessionId}</task_id>`,
    "<task_type>local_bash</task_type>",
    `<status>${result.status}</status>`,
    ...(result.exitCode === null ? [] : [`<exit_code>${result.exitCode}</exit_code>`]),
    `<output>${output}</output>`,
  ].join("\n");
}

function codexExecToolResult(result: WorkspaceShellExecResult) {
  const structuredContent = {
    chunk_id: result.chunkId,
    wall_time_seconds: result.wallTimeSeconds,
    ...(result.exitCode === undefined ? {} : { exit_code: result.exitCode }),
    ...(result.sessionId === undefined ? {} : { session_id: result.sessionId }),
    ...(result.originalTokenCount === undefined
      ? {}
      : { original_token_count: result.originalTokenCount }),
    output: result.output,
  };
  const sections = [
    `Chunk ID: ${result.chunkId}`,
    `Wall time: ${result.wallTimeSeconds.toFixed(4)} seconds`,
    ...(result.exitCode === undefined ? [] : [`Process exited with code ${result.exitCode}`]),
    ...(result.sessionId === undefined
      ? []
      : [`Process running with session ID ${result.sessionId}`]),
    ...(result.originalTokenCount === undefined
      ? []
      : [`Original token count: ${result.originalTokenCount}`]),
    "Output:",
    result.output,
  ];
  return localToolResult(sections.join("\n"), structuredContent);
}

function formatCodexPatchResult(result: WorkspacePatchResult): string {
  return [
    "Success.",
    ...result.operations.map((operation) => {
      if (operation.type === "move") {
        return `Moved ${operation.path} to ${operation.destination}.`;
      }
      const action =
        operation.type === "add" ? "Added" : operation.type === "delete" ? "Deleted" : "Updated";
      return `${action} ${operation.path}.`;
    }),
  ].join("\n");
}

function parseClaudeQuestions(value: unknown): ClaudeQuestion[] {
  if (!Array.isArray(value) || value.length < 1 || value.length > 4) {
    throw new Error("questions must contain between 1 and 4 questions.");
  }
  const questions = value.map((candidate, questionIndex) => {
    if (!isToolRecord(candidate)) throw new Error(`questions[${questionIndex}] must be an object.`);
    const question = requiredNonemptyToolText(
      candidate.question,
      `questions[${questionIndex}].question`,
    );
    const header = requiredNonemptyToolText(candidate.header, `questions[${questionIndex}].header`);
    if ([...header].length > 12) {
      throw new Error(`questions[${questionIndex}].header must contain at most 12 characters.`);
    }
    if (
      !Array.isArray(candidate.options) ||
      candidate.options.length < 2 ||
      candidate.options.length > 4
    ) {
      throw new Error(`questions[${questionIndex}].options must contain between 2 and 4 options.`);
    }
    const labels = new Set<string>();
    const options = candidate.options.map((option, optionIndex) => {
      if (!isToolRecord(option)) {
        throw new Error(`questions[${questionIndex}].options[${optionIndex}] must be an object.`);
      }
      const label = requiredNonemptyToolText(
        option.label,
        `questions[${questionIndex}].options[${optionIndex}].label`,
      );
      if (labels.has(label))
        throw new Error(`Question "${question}" has duplicate option "${label}".`);
      if (label === "Other") throw new Error('Do not provide an "Other" option; the UI adds it.');
      labels.add(label);
      return {
        label,
        description: requiredNonemptyToolText(
          option.description,
          `questions[${questionIndex}].options[${optionIndex}].description`,
        ),
        ...(option.preview === undefined
          ? {}
          : {
              preview: requiredToolText(
                option.preview,
                `questions[${questionIndex}].options[${optionIndex}].preview`,
              ),
            }),
      };
    });
    return {
      question,
      header,
      options,
      multiSelect: requiredBoolean(
        candidate.multiSelect,
        `questions[${questionIndex}].multiSelect`,
      ),
    };
  });
  if (new Set(questions.map((question) => question.question)).size !== questions.length) {
    throw new Error("Question text must be unique because answers are keyed by question text.");
  }
  return questions;
}

function validateClaudeAnswers(
  questions: readonly ClaudeQuestion[],
  value: Record<string, string>,
): Record<string, string> {
  if (!isToolRecord(value)) throw new Error("Question interaction returned invalid answers.");
  const answers: Record<string, string> = {};
  for (const question of questions) {
    const answer = value[question.question];
    if (typeof answer !== "string" || !answer.trim()) {
      throw new Error(`Question "${question.question}" requires an answer.`);
    }
    answers[question.question] = answer.trim();
  }
  return answers;
}

function formatClaudeAnswers(
  questions: readonly ClaudeQuestion[],
  answers: Readonly<Record<string, string>>,
): string {
  return questions
    .map((question) => `${question.question}: ${answers[question.question]}`)
    .join("\n");
}

function requiredTaskId(value: unknown): number {
  if (typeof value === "number") {
    return requiredBoundedInteger(value, "task_id", 1, Number.MAX_SAFE_INTEGER);
  }
  if (typeof value !== "string" || !/^\d+$/.test(value)) {
    throw new Error("task_id must be a positive integer string.");
  }
  return requiredBoundedInteger(Number(value), "task_id", 1, Number.MAX_SAFE_INTEGER);
}

function requiredBoolean(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${name} must be a boolean.`);
  return value;
}

function requiredBoundedInteger(
  value: unknown,
  name: string,
  minimum: number,
  maximum: number,
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}.`);
  }
  return value as number;
}

function requiredBoundedNumber(
  value: unknown,
  name: string,
  minimum: number,
  maximum: number,
): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new Error(`${name} must be a number between ${minimum} and ${maximum}.`);
  }
  return value;
}

function appendBooleanFlag(arguments_: string[], value: unknown, flag: string, name: string): void {
  if (value === undefined || value === false) return;
  if (value !== true) throw new Error(`${name} must be a boolean.`);
  if (!arguments_.includes(flag)) arguments_.push(flag);
}

function appendIntegerFlag(arguments_: string[], value: unknown, flag: string, name: string): void {
  if (value === undefined) return;
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  arguments_.push(flag, String(value));
}

function optionalPositiveInteger(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return value as number;
}

function optionalNonnegativeInteger(value: unknown, name: string): number | undefined {
  if (value === undefined) return undefined;
  if (!Number.isSafeInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return value as number;
}

function requiredPositiveInteger(value: unknown, name: string): number {
  const parsed = optionalPositiveInteger(value, name);
  if (parsed === undefined) throw new Error(`${name} must be a positive integer.`);
  return parsed;
}

function optionalEnum<const T extends readonly string[]>(
  value: unknown,
  choices: T,
  name: string,
  fallback: T[number],
): T[number] {
  if (value === undefined) return fallback;
  if (typeof value !== "string" || !choices.includes(value)) {
    throw new Error(`${name} must be one of ${choices.join(", ")}.`);
  }
  return value as T[number];
}

function stringToolPath(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function requiredToolPath(value: unknown): string {
  if (typeof value !== "string" || !value.trim())
    throw new Error("A Project file path is required.");
  return value;
}

function requiredToolText(value: unknown, name: string): string {
  if (typeof value !== "string") throw new Error(`${name} must be text.`);
  return value;
}

function requiredNonemptyToolText(value: unknown, name: string): string {
  const text = requiredToolText(value, name);
  if (!text.trim()) throw new Error(`${name} must not be empty.`);
  return text;
}

function optionalToolText(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  return requiredToolText(value, name);
}

function optionalNonemptyToolText(value: unknown, name: string): string | undefined {
  if (value === undefined) return undefined;
  return requiredNonemptyToolText(value, name);
}

function optionalStringEnum<const T extends readonly string[]>(
  value: unknown,
  choices: T,
  name: string,
): T[number] | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || !choices.includes(value)) {
    throw new Error(`${name} must be one of ${choices.join(", ")}.`);
  }
  return value as T[number];
}

function optionalClaudeTaskStatus(
  value: unknown,
  allowDeleted = false,
): ClaudeTaskStatus | "deleted" | undefined {
  return optionalStringEnum(
    value,
    allowDeleted
      ? (["pending", "in_progress", "completed", "deleted"] as const)
      : (["pending", "in_progress", "completed"] as const),
    "status",
  );
}

function requiredClaudeTaskStatus(value: unknown, name: string): ClaudeTaskStatus {
  const status = optionalStringEnum(value, ["pending", "in_progress", "completed"] as const, name);
  if (status === undefined) throw new Error(`${name} is required.`);
  return status;
}

function optionalTaskIds(value: unknown, name: string): string[] {
  if (value === undefined) return [];
  if (!Array.isArray(value)) throw new Error(`${name} must be an array.`);
  return [
    ...new Set(value.map((item, index) => requiredNonemptyToolText(item, `${name}[${index}]`))),
  ];
}

function optionalMetadata(value: unknown): Record<string, unknown> {
  if (value === undefined) return {};
  if (!isToolRecord(value)) throw new Error("metadata must be an object.");
  const metadata: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (["__proto__", "constructor", "prototype"].includes(key)) {
      throw new Error(`metadata key ${key} is not allowed.`);
    }
    metadata[key] = entry;
  }
  return metadata;
}

function isToolRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function optionalToolBoolean(value: unknown, name: string): boolean {
  if (value === undefined) return false;
  if (typeof value !== "boolean") throw new Error(`${name} must be a boolean.`);
  return value;
}

function resolveOptions(options: WorkspaceToolsOptions): ResolvedWorkspaceToolsOptions {
  const maxPatchBytes = configuredLimit(
    options.maxPatchBytes,
    WORKSPACE_TOOLS_DEFAULTS.maxPatchBytes,
    WORKSPACE_TOOLS_HARD_LIMITS.maxPatchBytes,
    "maxPatchBytes",
  );
  return {
    maxFileBytes: configuredLimit(
      options.maxFileBytes,
      WORKSPACE_TOOLS_DEFAULTS.maxFileBytes,
      WORKSPACE_TOOLS_HARD_LIMITS.maxFileBytes,
      "maxFileBytes",
    ),
    maxWriteFileBytes: configuredLimit(
      options.maxWriteFileBytes,
      WORKSPACE_TOOLS_DEFAULTS.maxWriteFileBytes,
      WORKSPACE_TOOLS_HARD_LIMITS.maxWriteFileBytes,
      "maxWriteFileBytes",
    ),
    maxDirectoryEntries: configuredLimit(
      options.maxDirectoryEntries,
      WORKSPACE_TOOLS_DEFAULTS.maxDirectoryEntries,
      WORKSPACE_TOOLS_HARD_LIMITS.maxDirectoryEntries,
      "maxDirectoryEntries",
    ),
    maxReviewFiles: configuredLimit(
      options.maxReviewFiles,
      WORKSPACE_TOOLS_DEFAULTS.maxReviewFiles,
      WORKSPACE_TOOLS_HARD_LIMITS.maxReviewFiles,
      "maxReviewFiles",
    ),
    maxPatchBytes,
    maxPatchBytesPerFile: Math.min(
      maxPatchBytes,
      configuredLimit(
        options.maxPatchBytesPerFile,
        WORKSPACE_TOOLS_DEFAULTS.maxPatchBytesPerFile,
        WORKSPACE_TOOLS_HARD_LIMITS.maxPatchBytesPerFile,
        "maxPatchBytesPerFile",
      ),
    ),
    gitTimeoutMs: configuredLimit(
      options.gitTimeoutMs,
      WORKSPACE_TOOLS_DEFAULTS.gitTimeoutMs,
      WORKSPACE_TOOLS_HARD_LIMITS.gitTimeoutMs,
      "gitTimeoutMs",
    ),
  };
}

function configuredLimit(
  value: number | undefined,
  fallback: number,
  maximum: number,
  name: string,
): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value <= 0 || value > maximum) {
    throw new Error(`${name} must be an integer between 1 and ${maximum}.`);
  }
  return value;
}

async function resolveExistingPath(
  root: string,
  relativePath: string,
  allowRoot: boolean,
): Promise<string> {
  const normalizedPath = normalizeRelativePath(relativePath, allowRoot);
  const lexicalTarget = path.resolve(root, normalizedPath || ".");
  if (!isContainedPath(root, lexicalTarget)) throw new Error("Workspace path escapes the root.");

  const resolvedTarget = await realpath(lexicalTarget);
  if (!isContainedPath(root, resolvedTarget)) {
    throw new Error("Workspace path resolves outside the root.");
  }
  return resolvedTarget;
}

function normalizeRelativePath(value: string, allowRoot: boolean): string {
  if (typeof value !== "string" || value.includes("\0")) {
    throw new Error("Workspace path must be a valid relative path.");
  }
  if (
    path.isAbsolute(value) ||
    path.posix.isAbsolute(value) ||
    path.win32.isAbsolute(value) ||
    /^[A-Za-z]:/.test(value)
  ) {
    throw new Error("Absolute workspace paths are not allowed.");
  }

  const rawSegments = value.split(/[\\/]+/);
  if (rawSegments.includes("..")) throw new Error("Workspace path traversal is not allowed.");

  const normalized = path.normalize(value || ".");
  if (normalized === ".") {
    if (allowRoot) return "";
    throw new Error("A workspace file path is required.");
  }
  if (Buffer.byteLength(normalized) > MAX_RELATIVE_PATH_BYTES) {
    throw new Error("Workspace path is too long.");
  }

  const relative = toPortablePath(normalized);
  if (!allowRoot && !relative) throw new Error("A workspace file path is required.");
  return relative;
}

function isContainedPath(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return (
    relative === "" ||
    (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative))
  );
}

function toPortablePath(value: string): string {
  return path.sep === "/" ? value : value.split(path.sep).join("/");
}

function joinRelativePath(parent: string, name: string): string {
  return parent ? `${parent}/${name}` : name;
}

function directoryEntryKind(entry: Dirent): WorkspaceEntryKind {
  if (entry.isDirectory()) return "directory";
  if (entry.isFile()) return "file";
  if (entry.isSymbolicLink()) return "symlink";
  return "other";
}

function compareDirectoryEntries(left: WorkspaceDirectoryEntry, right: WorkspaceDirectoryEntry) {
  const rank = (entry: WorkspaceDirectoryEntry) => (entry.kind === "directory" ? 0 : 1);
  return rank(left) - rank(right) || left.name.localeCompare(right.name);
}

async function completeOriginalContent(
  target: string,
  maxBytes: number,
  expectedSha256: string | undefined,
): Promise<string> {
  const decoded = await readUtf8File(target, maxBytes);
  if (decoded.truncated || decoded.sha256 !== expectedSha256) throw staleWorkspaceFileError();
  return decoded.content;
}

async function readUtf8File(target: string, maxBytes: number): Promise<DecodedFile> {
  const handle = await open(target, "r");
  try {
    const fileStat = await handle.stat();
    if (!fileStat.isFile()) throw new Error("Workspace path is not a regular file.");

    const buffer = Buffer.allocUnsafe(maxBytes + 4);
    let offset = 0;
    while (offset < buffer.length) {
      const result = await handle.read(buffer, offset, buffer.length - offset, offset);
      if (result.bytesRead === 0) break;
      offset += result.bytesRead;
    }

    const sample = buffer.subarray(0, offset);
    if (sample.includes(0)) throw new WorkspaceBinaryFileError();
    const truncated = fileStat.size > maxBytes || sample.length > maxBytes;
    const content = decodeUtf8Prefix(
      sample.subarray(0, Math.min(sample.length, maxBytes)),
      truncated,
    );
    return {
      content,
      size: fileStat.size,
      truncated,
      ...(!truncated ? { sha256: sha256Buffer(sample) } : {}),
    };
  } finally {
    await handle.close();
  }
}

function validateWriteContent(value: string, maximumBytes: number): Buffer {
  if (typeof value !== "string" || value.includes("\0")) {
    throw new Error("Workspace file content must be UTF-8 text without NUL bytes.");
  }
  const encoded = Buffer.from(value, "utf8");
  if (encoded.length > maximumBytes) {
    throw new Error(`Workspace file content exceeds the ${maximumBytes}-byte write limit.`);
  }
  return encoded;
}

function exactOccurrenceCount(content: string, search: string): number {
  let count = 0;
  let offset = 0;
  while (offset <= content.length - search.length) {
    const match = content.indexOf(search, offset);
    if (match < 0) break;
    count += 1;
    offset = match + search.length;
  }
  return count;
}

async function fileSha256(target: string): Promise<string> {
  const handle = await open(target, "r");
  try {
    const targetStat = await handle.stat();
    if (!targetStat.isFile()) throw new Error("Workspace path is not a regular file.");
    const hash = createHash("sha256");
    const buffer = Buffer.allocUnsafe(64 * 1024);
    let position = 0;
    while (true) {
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, position);
      if (bytesRead === 0) break;
      hash.update(buffer.subarray(0, bytesRead));
      position += bytesRead;
    }
    return hash.digest("hex");
  } finally {
    await handle.close();
  }
}

function sha256Buffer(value: Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function staleWorkspaceFileError(): Error {
  return new Error(
    "Workspace file changed after it was read; read it again before writing or editing.",
  );
}

function isFileNotFoundError(cause: unknown): boolean {
  return cause instanceof Error && "code" in cause && cause.code === "ENOENT";
}

function isAlreadyExistsError(cause: unknown): boolean {
  return cause instanceof Error && "code" in cause && cause.code === "EEXIST";
}

function decodeUtf8Prefix(buffer: Buffer, mayEndMidCharacter: boolean): string {
  const minimumLength = mayEndMidCharacter ? Math.max(0, buffer.length - 3) : buffer.length;
  for (let length = buffer.length; length >= minimumLength; length -= 1) {
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, length));
    } catch {
      // A byte cap can split one UTF-8 code point; only the final three bytes may be removed.
    }
  }
  throw new WorkspaceBinaryFileError();
}

function safeDiffArguments(): string[] {
  return ["--relative", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=3"];
}

function parseGitStatus(stdout: string, maximumEntries: number): ParsedGitStatus {
  const records = stdout.split("\0");
  const completeOutput = records.at(-1) === "";
  records.pop();
  const entries: GitStatusEntry[] = [];
  let truncated = !completeOutput;

  for (let index = 0; index < records.length; index += 1) {
    const record = records[index];
    if (!record || record.length < 4 || record[2] !== " ") {
      truncated = true;
      continue;
    }

    const status = record.slice(0, 2);
    const candidatePath = record.slice(3);
    const renameOrCopy = status.includes("R") || status.includes("C");
    let previousCandidate: string | undefined;
    if (renameOrCopy) {
      index += 1;
      previousCandidate = records[index];
      if (previousCandidate === undefined) {
        truncated = true;
        continue;
      }
    }

    try {
      const normalizedPath = normalizeRelativePath(candidatePath, false);
      const previousPath = previousCandidate
        ? normalizeRelativePath(previousCandidate, false)
        : undefined;
      if (entries.length >= maximumEntries) {
        truncated = true;
        continue;
      }
      entries.push({
        path: normalizedPath,
        status,
        ...(previousPath ? { previousPath } : {}),
      });
    } catch {
      truncated = true;
    }
  }

  return { entries, truncated };
}

function untrackedPatch(relativePath: string, content: string): string {
  const source = quotePatchPath(`a/${relativePath}`);
  const target = quotePatchPath(`b/${relativePath}`);
  const lines = content ? content.split("\n") : [];
  const hasTrailingNewline = content.endsWith("\n");
  if (hasTrailingNewline) lines.pop();

  const header = [
    `diff --git ${source} ${target}`,
    "new file mode 100644",
    "--- /dev/null",
    `+++ ${target}`,
  ];
  if (lines.length === 0) return `${header.join("\n")}\n`;

  const body = lines.map((line) => `+${line}`).join("\n");
  return `${header.join("\n")}\n@@ -0,0 +1,${lines.length} @@\n${body}\n${
    hasTrailingNewline ? "" : "\\ No newline at end of file\n"
  }`;
}

function quotePatchPath(value: string): string {
  return /^[\w./@+-]+$/.test(value) ? value : JSON.stringify(value);
}

function patchLineCounts(patch: string): { additions: number; deletions: number } {
  let additions = 0;
  let deletions = 0;
  for (const line of patch.split("\n")) {
    if (line.startsWith("+") && !line.startsWith("+++")) additions += 1;
    if (line.startsWith("-") && !line.startsWith("---")) deletions += 1;
  }
  return { additions, deletions };
}

function isBinaryPatch(patch: string): boolean {
  return /^(?:Binary files .* differ|GIT binary patch)$/m.test(patch);
}

function truncateUtf8(value: string, maximumBytes: number): { value: string; truncated: boolean } {
  const buffer = Buffer.from(value);
  if (buffer.length <= maximumBytes) return { value, truncated: false };
  if (maximumBytes <= 0) return { value: "", truncated: true };

  const prefix = buffer.subarray(0, maximumBytes);
  for (let length = prefix.length; length >= Math.max(0, prefix.length - 3); length -= 1) {
    try {
      return {
        value: new TextDecoder("utf-8", { fatal: true }).decode(prefix.subarray(0, length)),
        truncated: true,
      };
    } catch {
      // Remove only a partial trailing UTF-8 code point.
    }
  }
  return { value: "", truncated: true };
}

function gitStatusOutputLimit(maximumFiles: number): number {
  return Math.min(MAX_GIT_STATUS_BYTES, Math.max(64 * 1024, maximumFiles * 4096));
}

function gitEnvironment(): NodeJS.ProcessEnv {
  const environment = { ...process.env };
  for (const name of [
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_WORK_TREE",
  ]) {
    delete environment[name];
  }
  for (const name of Object.keys(environment)) {
    if (/^GIT_CONFIG_(?:KEY|VALUE)_\d+$/.test(name)) delete environment[name];
  }
  return {
    ...environment,
    GIT_LITERAL_PATHSPECS: "1",
    GIT_OPTIONAL_LOCKS: "0",
    GIT_PAGER: "cat",
    GIT_TERMINAL_PROMPT: "0",
    LC_ALL: "C",
  };
}

function isMaxBufferError(cause: unknown): boolean {
  return (
    cause instanceof Error && "code" in cause && cause.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER"
  );
}

function isMissingExecutable(cause: unknown): boolean {
  return cause instanceof Error && "code" in cause && cause.code === "ENOENT";
}

function isNotGitRepository(stderr: string): boolean {
  return stderr.toLowerCase().includes("not a git repository");
}

function gitErrorMessage(result: GitCommandResult): string {
  if (result.missingExecutable) return "Git is not available.";
  const message = result.stderr.trim() || result.error?.message || "Git command failed.";
  return truncateUtf8(message, 1024).value;
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}
