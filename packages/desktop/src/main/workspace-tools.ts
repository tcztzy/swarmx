import { execFile } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import type { Dirent } from "node:fs";
import { link, lstat, open, opendir, realpath, rename, stat, unlink } from "node:fs/promises";
import path from "node:path";
import type { LocalMcpTool } from "@swarmx/core";
import { WorkspaceShell, workspaceShellAgentTool } from "./workspace-shell.js";

const MAX_RELATIVE_PATH_BYTES = 4 * 1024;
const GIT_COMMAND_SLACK_BYTES = 64 * 1024;
const MAX_GIT_STATUS_BYTES = 4 * 1024 * 1024;

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
}

export interface WorkspaceEditResult extends WorkspaceWriteResult {
  replacements: number;
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

class WorkspaceBinaryFileError extends Error {
  constructor() {
    super("Only UTF-8 text files can be read.");
    this.name = "WorkspaceBinaryFileError";
  }
}

/** Bounded filesystem and Git operations confined to one workspace root. */
export class WorkspaceTools {
  readonly root: string;
  readonly #options: ResolvedWorkspaceToolsOptions;
  readonly #readVersions = new Map<string, string>();

  constructor(root: string, options: WorkspaceToolsOptions = {}) {
    if (typeof root !== "string" || !root.trim() || root.includes("\0")) {
      throw new Error("A valid workspace root is required.");
    }

    this.root = path.resolve(root);
    this.#options = resolveOptions(options);
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
      throw new Error(
        "Existing files must be read completely with workspace_read_file before writing.",
      );
    }
    if (target.exists ? target.sha256 !== expectedSha256 : expectedSha256 !== undefined) {
      throw staleWorkspaceFileError();
    }
    return this.#atomicWrite(root, normalizedPath, encoded, target, expectedSha256);
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
      throw new Error("Files must be read completely with workspace_read_file before editing.");
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
    return { ...result, replacements: replaceAll ? replacements : 1 };
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
  ): Promise<WorkspaceWriteResult> {
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
}

export function workspaceAgentTools(
  tools: WorkspaceTools,
  shell = new WorkspaceShell(tools.root),
): LocalMcpTool[] {
  return [
    {
      name: "workspace_list_directory",
      description:
        "List one directory level inside the active Project. Paths are relative to the Project root.",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", description: "Relative directory path; empty for root." },
        },
        additionalProperties: false,
      },
      call: async (input) => tools.listDirectory(stringToolPath(input.path)),
    },
    {
      name: "workspace_read_file",
      description:
        "Read a bounded UTF-8 text file inside the active Project. Read an existing file before editing or replacing it. Paths are relative to the Project root.",
      inputSchema: {
        type: "object",
        properties: { path: { type: "string", minLength: 1, description: "Relative file path." } },
        required: ["path"],
        additionalProperties: false,
      },
      call: async (input) => tools.readFile(requiredToolPath(input.path)),
    },
    {
      name: "workspace_write_file",
      description:
        "Create a UTF-8 text file or atomically replace an unchanged file inside the active Project. Existing files must first be read completely with workspace_read_file.",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", minLength: 1, description: "Relative file path." },
          content: { type: "string", description: "Complete new UTF-8 file content." },
        },
        required: ["path", "content"],
        additionalProperties: false,
      },
      call: async (input) =>
        tools.writeFile(requiredToolPath(input.path), requiredToolText(input.content, "content")),
    },
    {
      name: "workspace_edit_file",
      description:
        "Replace exact text in an unchanged UTF-8 file inside the active Project. Read the file first; use a unique oldText match unless replaceAll is explicitly true.",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", minLength: 1, description: "Relative file path." },
          oldText: { type: "string", minLength: 1, description: "Exact text to replace." },
          newText: { type: "string", description: "Replacement text." },
          replaceAll: {
            type: "boolean",
            description: "Replace every exact occurrence. Defaults to false.",
          },
        },
        required: ["path", "oldText", "newText"],
        additionalProperties: false,
      },
      call: async (input) =>
        tools.editFile(
          requiredToolPath(input.path),
          requiredToolText(input.oldText, "oldText"),
          requiredToolText(input.newText, "newText"),
          optionalToolBoolean(input.replaceAll, "replaceAll"),
        ),
    },
    workspaceShellAgentTool(shell),
  ];
}

export function projectAgentContextMessage(root: string): string {
  const projectName = path.basename(path.resolve(root)) || "Project";
  return [
    `Active Project: ${projectName}`,
    `Project root: ${path.resolve(root)}`,
    "You have workspace_list_directory, workspace_read_file, workspace_write_file, workspace_edit_file, and workspace_shell tools rooted at this Project.",
    "For questions about this Project, inspect relevant files before answering. Start with the root listing and README or package manifest when appropriate.",
    "Read every existing file completely before replacing or editing it. Prefer workspace_edit_file for focused changes, then use workspace_shell for bounded builds or tests.",
    "Never claim that Project files are unavailable before attempting the workspace tools.",
  ].join("\n");
}

function stringToolPath(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function requiredToolPath(value: unknown): string {
  if (typeof value !== "string" || !value.trim())
    throw new Error("A relative file path is required.");
  return value;
}

function requiredToolText(value: unknown, name: string): string {
  if (typeof value !== "string") throw new Error(`${name} must be text.`);
  return value;
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
