import { execFile } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  WorkspaceTools,
  projectAgentContextMessage,
  workspaceAgentTools,
} from "./workspace-tools.js";

const temporaryDirectories = new Set<string>();

afterEach(async () => {
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("WorkspaceTools", () => {
  it("returns staged, unstaged, and untracked text patches", async () => {
    const root = await temporaryDirectory();
    await git(root, "init", "-b", "main");
    await writeFile(path.join(root, "staged.txt"), "staged before\n");
    await writeFile(path.join(root, "unstaged.txt"), "unstaged before\n");
    await git(root, "add", ".");
    await git(
      root,
      "-c",
      "user.email=workspace-tools@example.test",
      "-c",
      "user.name=Workspace Tools Test",
      "commit",
      "-m",
      "fixture",
    );

    await writeFile(path.join(root, "staged.txt"), "staged after\n");
    await git(root, "add", "staged.txt");
    await writeFile(path.join(root, "unstaged.txt"), "unstaged after\n");
    await writeFile(path.join(root, "untracked.txt"), "new file\n");

    const snapshot = await new WorkspaceTools(root).review();

    expect(snapshot).toMatchObject({
      root,
      branch: "main",
      isRepository: true,
      truncated: false,
    });
    expect(snapshot.files.map((file) => [file.path, file.status])).toEqual([
      ["staged.txt", "M "],
      ["unstaged.txt", " M"],
      ["untracked.txt", "??"],
    ]);

    const staged = snapshot.files.find((file) => file.path === "staged.txt");
    expect(staged?.patch).toContain("-staged before");
    expect(staged?.patch).toContain("+staged after");
    expect(staged).toMatchObject({ additions: 1, deletions: 1, binary: false });

    const unstaged = snapshot.files.find((file) => file.path === "unstaged.txt");
    expect(unstaged?.patch).toContain("-unstaged before");
    expect(unstaged?.patch).toContain("+unstaged after");

    const untracked = snapshot.files.find((file) => file.path === "untracked.txt");
    expect(untracked?.patch).toContain("--- /dev/null");
    expect(untracked?.patch).toContain("+new file");
    expect(untracked).toMatchObject({ additions: 1, deletions: 0, binary: false });
  }, 30_000);

  it("returns an empty, non-error snapshot outside a Git repository", async () => {
    const root = await temporaryDirectory();

    await expect(new WorkspaceTools(root).review()).resolves.toEqual({
      root,
      branch: null,
      isRepository: false,
      files: [],
      truncated: false,
    });
  });

  it("rejects absolute, traversal, and escaping symlink paths", async () => {
    const parent = await temporaryDirectory();
    const root = path.join(parent, "workspace");
    const outside = path.join(parent, "outside");
    await mkdir(root);
    await mkdir(outside);
    await writeFile(path.join(root, "inside.txt"), "inside\n");
    await writeFile(path.join(outside, "secret.txt"), "secret\n");
    await symlink(
      outside,
      path.join(root, "escape"),
      process.platform === "win32" ? "junction" : "dir",
    );
    await symlink(path.join(root, "inside.txt"), path.join(root, "inside-link.txt"), "file");
    const tools = new WorkspaceTools(root);

    await expect(tools.readFile("../outside/secret.txt")).rejects.toThrow(/traversal/i);
    await expect(tools.readFile(path.join(outside, "secret.txt"))).rejects.toThrow(/absolute/i);
    await expect(tools.readFile("escape/secret.txt")).rejects.toThrow(/outside the root/i);
    await expect(tools.listDirectory("escape")).rejects.toThrow(/outside the root/i);
    await expect(tools.readFile("inside-link.txt")).resolves.toMatchObject({
      path: "inside-link.txt",
      content: "inside\n",
    });
  });

  it("V346 exposes Project identity and bounded coding Agent tools", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "README.md"), "# Workspace fixture\n");
    const tools = workspaceAgentTools(new WorkspaceTools(root));
    const list = tools.find((tool) => tool.name === "workspace_list_directory");
    const read = tools.find((tool) => tool.name === "workspace_read_file");
    const write = tools.find((tool) => tool.name === "workspace_write_file");
    const edit = tools.find((tool) => tool.name === "workspace_edit_file");

    expect(projectAgentContextMessage(root)).toContain(`Active Project: ${path.basename(root)}`);
    expect(projectAgentContextMessage(root)).toContain("inspect relevant files before answering");
    expect(projectAgentContextMessage(root)).toContain("workspace_shell");
    expect(tools.map((tool) => tool.name)).toEqual([
      "workspace_list_directory",
      "workspace_read_file",
      "workspace_write_file",
      "workspace_edit_file",
      "workspace_shell",
    ]);
    await expect(list?.call({ path: "" })).resolves.toMatchObject({
      entries: [expect.objectContaining({ path: "README.md", kind: "file" })],
    });
    await expect(read?.call({ path: "README.md" })).resolves.toMatchObject({
      path: "README.md",
      content: "# Workspace fixture\n",
    });
    await expect(read?.call({ path: "../outside.txt" })).rejects.toThrow(/traversal/i);
    await expect(write?.call({ path: "created.txt", content: "created\n" })).resolves.toMatchObject(
      {
        path: "created.txt",
        created: true,
      },
    );
    await expect(
      edit?.call({ path: "README.md", oldText: "fixture", newText: "edited" }),
    ).resolves.toMatchObject({ replacements: 1, created: false });
  });

  it("V347 creates files atomically and protects existing files with read digests", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "existing.txt"), "before\n");
    const tools = new WorkspaceTools(root);

    const created = await tools.writeFile("nested.txt", "new\n");
    expect(created).toMatchObject({ path: "nested.txt", size: 4, created: true });
    expect(created.sha256).toMatch(/^[a-f0-9]{64}$/);
    await expect(readFile(path.join(root, "nested.txt"), "utf8")).resolves.toBe("new\n");

    await expect(tools.writeFile("existing.txt", "blocked\n")).rejects.toThrow(/read completely/i);
    const read = await tools.readFile("existing.txt");
    expect(read.sha256).toMatch(/^[a-f0-9]{64}$/);
    await expect(tools.writeFile("existing.txt", "after\n")).resolves.toMatchObject({
      created: false,
      size: 6,
    });
    await expect(readFile(path.join(root, "existing.txt"), "utf8")).resolves.toBe("after\n");

    await tools.readFile("existing.txt");
    await writeFile(path.join(root, "existing.txt"), "external\n");
    await expect(tools.writeFile("existing.txt", "stale\n")).rejects.toThrow(/changed after/i);
    await expect(readFile(path.join(root, "existing.txt"), "utf8")).resolves.toBe("external\n");
  });

  it("V348 performs exact edits and rejects ambiguous or escaping mutations", async () => {
    const parent = await temporaryDirectory();
    const root = path.join(parent, "workspace");
    const outside = path.join(parent, "outside");
    await mkdir(root);
    await mkdir(outside);
    await writeFile(path.join(root, "content.txt"), "one one two\n");
    await writeFile(path.join(outside, "secret.txt"), "secret\n");
    await symlink(outside, path.join(root, "escape"), "dir");
    await symlink(path.join(root, "content.txt"), path.join(root, "content-link.txt"), "file");
    const tools = new WorkspaceTools(root);

    await tools.readFile("content.txt");
    await expect(tools.editFile("content.txt", "one", "ONE")).rejects.toThrow(/2 occurrences/i);
    await expect(tools.editFile("content.txt", "one", "ONE", true)).resolves.toMatchObject({
      replacements: 2,
      created: false,
    });
    await expect(readFile(path.join(root, "content.txt"), "utf8")).resolves.toBe("ONE ONE two\n");

    await expect(tools.writeFile("../outside/new.txt", "no\n")).rejects.toThrow(/traversal/i);
    await expect(tools.writeFile("escape/new.txt", "no\n")).rejects.toThrow(/outside the root/i);
    await tools.readFile("content-link.txt");
    await expect(tools.writeFile("content-link.txt", "no\n")).rejects.toThrow(/symbolic links/i);
  });

  it("rejects oversized writes and incomplete reads cannot authorize replacement", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "large.txt"), "abcdef");
    const tools = new WorkspaceTools(root, { maxFileBytes: 5, maxWriteFileBytes: 5 });

    await expect(tools.readFile("large.txt")).resolves.toMatchObject({ truncated: true });
    await expect(tools.writeFile("large.txt", "small")).rejects.toThrow(/read completely/i);
    await expect(tools.writeFile("new.txt", "abcdef")).rejects.toThrow(/5-byte write limit/i);
  });

  it("caps file, directory, review-file, and patch output", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "large.txt"), "abcdef");
    await writeFile(path.join(root, "binary.dat"), Buffer.from([0x61, 0x00, 0x62]));
    await mkdir(path.join(root, "folder"));
    const tools = new WorkspaceTools(root, { maxFileBytes: 5, maxDirectoryEntries: 2 });

    await expect(tools.readFile("large.txt")).resolves.toEqual({
      path: "large.txt",
      content: "abcde",
      size: 6,
      truncated: true,
    });
    await expect(tools.readFile("binary.dat")).rejects.toThrow(/UTF-8 text/i);
    const listing = await tools.listDirectory();
    expect(listing.entries).toHaveLength(2);
    expect(listing.truncated).toBe(true);

    await git(root, "init", "-b", "main");
    await writeFile(path.join(root, "a-first-untracked.txt"), `${"first\n".repeat(40)}`);
    await writeFile(path.join(root, "a-second-untracked.txt"), `${"second\n".repeat(40)}`);
    const review = await new WorkspaceTools(root, {
      maxReviewFiles: 1,
      maxPatchBytes: 80,
      maxPatchBytesPerFile: 80,
    }).review();

    expect(review.files).toHaveLength(1);
    expect(Buffer.byteLength(review.files[0]?.patch ?? "")).toBeLessThanOrEqual(80);
    expect(review.files[0]?.truncated).toBe(true);
    expect(review.truncated).toBe(true);
  }, 15_000);
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(os.tmpdir(), "swarmx-workspace-tools-"));
  temporaryDirectories.add(directory);
  return directory;
}

function git(cwd: string, ...arguments_: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile("git", arguments_, { cwd, encoding: "utf8" }, (error, stdout) => {
      if (error) reject(error);
      else resolve(stdout);
    });
  });
}
