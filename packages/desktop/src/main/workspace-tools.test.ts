import { execFile } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  realpath,
  rm,
  symlink,
  utimes,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import type { LocalMcpTool, LocalTextTool, LocalToolResult } from "@swarmx/core";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ClaudeInteractionRequest, ClaudeInteractionResponse } from "./agent-interactions.js";
import { WorkspaceShell } from "./workspace-shell.js";
import {
  type ClaudeSessionToolBridge,
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
  it("V444-V447 enforces direct tool policy and one-call desktop approval", async () => {
    const root = await temporaryDirectory();
    const withoutBridge = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      permissionPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
    });
    const blockedWrite = withoutBridge.find((tool) => tool.name === "Write") as LocalMcpTool;
    await expect(
      blockedWrite.call({ file_path: "blocked.txt", content: "must not be written\n" }),
    ).rejects.toThrow(/requires approval.*no interaction bridge/i);

    const interact = vi
      .fn()
      .mockResolvedValueOnce({ kind: "tool_approval", optionId: "reject_once" })
      .mockResolvedValueOnce({ kind: "tool_approval", optionId: "allow_once" });
    const approvedTools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      permissionPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      interact,
    });
    const write = approvedTools.find((tool) => tool.name === "Write") as LocalMcpTool;
    await expect(
      write.call({ file_path: "rejected.txt", content: "private body\n" }),
    ).rejects.toThrow(/rejected by the user/i);
    await expect(
      write.call({ file_path: "approved.txt", content: "private body\n" }),
    ).resolves.toEqual(
      expect.objectContaining({ content: expect.stringContaining("successfully") }),
    );
    expect(interact).toHaveBeenCalledTimes(2);
    expect(interact.mock.calls[1]?.[0]).toMatchObject({
      kind: "tool_approval",
      title: "Allow Write?",
      summary: expect.stringContaining("approved.txt"),
    });
    expect(JSON.stringify(interact.mock.calls)).not.toContain("private body");

    const planTools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      permissionPolicy: { mode: "plan", allowedTools: ["Write"], deniedTools: [] },
      interact,
    });
    const planWrite = planTools.find((tool) => tool.name === "Write") as LocalMcpTool;
    await expect(planWrite.call({ file_path: "plan.txt", content: "no\n" })).rejects.toThrow(
      /plan_read_only/i,
    );

    const trustedTools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "gpt-5.4",
      permissionPolicy: { mode: "trusted", allowedTools: [], deniedTools: ["exec_command"] },
    });
    const exec = trustedTools.find((tool) => tool.name === "exec_command") as LocalMcpTool;
    await expect(exec.call({ cmd: "pwd" })).rejects.toThrow(/explicit_deny/i);
  });

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

  it("V356-V357 selects model-trained Project tool profiles", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "README.md"), "# Workspace fixture\n");
    const workspace = new WorkspaceTools(root);
    const claude = workspaceAgentTools(workspace, undefined, { model: "claude-sonnet-4-6" });
    const codex = workspaceAgentTools(workspace, undefined, {
      model: "gpt-5.4",
      apiProtocol: "openai_responses",
    });
    const codexJson = workspaceAgentTools(workspace, undefined, {
      model: "gpt-5.4",
      apiProtocol: "openai_chat",
    });
    const bash = claude.find((tool) => tool.name === "Bash") as LocalMcpTool | undefined;
    const read = claude.find((tool) => tool.name === "Read") as LocalMcpTool | undefined;
    const edit = claude.find((tool) => tool.name === "Edit") as LocalMcpTool | undefined;
    const write = claude.find((tool) => tool.name === "Write") as LocalMcpTool | undefined;
    const execCommand = codex.find((tool) => tool.name === "exec_command") as
      | LocalMcpTool
      | undefined;

    expect(projectAgentContextMessage(root, { model: "claude-opus-4-6" })).toContain(
      `Active Project: ${path.basename(root)}`,
    );
    expect(projectAgentContextMessage(root, { model: "gpt-5.4" })).toContain(
      "exec_command, write_stdin, and apply_patch",
    );
    expect(claude.map((tool) => tool.name)).toEqual([
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
      "TaskCreate",
      "TaskGet",
      "TaskList",
      "TaskUpdate",
      "TodoWrite",
      "TaskOutput",
      "TaskStop",
    ]);
    expect(codex.map((tool) => tool.name)).toEqual(["exec_command", "write_stdin", "apply_patch"]);
    expect(codex[2]?.kind).toBe("text");
    expect(codexJson[2]?.kind).not.toBe("text");
    expect(Object.keys(bash?.inputSchema.properties as object)).toEqual(
      expect.arrayContaining([
        "command",
        "timeout",
        "description",
        "run_in_background",
        "dangerouslyDisableSandbox",
      ]),
    );
    expect(Object.keys(execCommand?.inputSchema.properties as object)).toEqual(
      expect.arrayContaining([
        "cmd",
        "workdir",
        "tty",
        "yield_time_ms",
        "max_output_tokens",
        "sandbox_permissions",
      ]),
    );
    const unreadWriteError = await write
      ?.call({ file_path: path.join(root, "README.md"), content: "replacement\n" })
      .catch((error: unknown) => error);
    expect(unreadWriteError).toEqual(
      expect.objectContaining({
        message: "Existing files must be read completely before writing.",
      }),
    );
    await expect(
      read?.call({ file_path: path.join(root, "README.md"), harmless_extra: true }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("1→# Workspace fixture"),
      structuredContent: {
        type: "text",
        file: expect.objectContaining({
          filePath: path.join(root, "README.md"),
          content: "# Workspace fixture\n",
          numLines: 1,
          startLine: 1,
          totalLines: 1,
        }),
      },
    });
    await expect(
      write?.call({ file_path: path.join(root, "README.md"), content: "updated\n" }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("updated successfully"),
      structuredContent: expect.objectContaining({
        type: "update",
        filePath: path.join(root, "README.md"),
        content: "updated\n",
        originalFile: "# Workspace fixture\n",
        structuredPatch: expect.any(Array),
      }),
    });
    await expect(
      edit?.call({
        file_path: path.join(root, "README.md"),
        old_string: "updated",
        new_string: "edited",
      }),
    ).resolves.toMatchObject({
      structuredContent: expect.objectContaining({
        filePath: path.join(root, "README.md"),
        oldString: "updated",
        newString: "edited",
        originalFile: "updated\n",
        userModified: false,
        replaceAll: false,
      }),
    });
    await expect(read?.call({ file_path: "../outside.txt" })).rejects.toThrow(/traversal/i);
    await expect(bash?.call({ command: "pwd", dangerouslyDisableSandbox: true })).rejects.toThrow(
      /cannot be disabled/i,
    );
    await expect(
      execCommand?.call({ cmd: "pwd", sandbox_permissions: "require_escalated" }),
    ).rejects.toThrow(/cannot be escalated/i);
  });

  it("V408-V413 enters, rebinds, preserves, and guardedly removes Claude worktrees", async () => {
    const root = await temporaryDirectory();
    await git(root, "init", "-b", "main");
    await writeFile(path.join(root, "base.txt"), "main\n");
    await git(root, "add", "base.txt");
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
    const canonicalRoot = await realpath(root);
    const workspace = new WorkspaceTools(root);
    const shell = new WorkspaceShell(root);
    const lspRoots: string[] = [];
    const tools = workspaceAgentTools(workspace, shell, {
      model: "claude-sonnet-4-6",
      lsp: async (request) => {
        lspRoots.push(workspace.root);
        return {
          operation: request.operation,
          result: request.filePath,
          filePath: request.filePath,
        };
      },
    });
    const bash = tools.find((tool) => tool.name === "Bash") as LocalMcpTool;
    const enter = tools.find((tool) => tool.name === "EnterWorktree") as LocalMcpTool;
    const exit = tools.find((tool) => tool.name === "ExitWorktree") as LocalMcpTool;
    const read = tools.find((tool) => tool.name === "Read") as LocalMcpTool;
    const write = tools.find((tool) => tool.name === "Write") as LocalMcpTool;
    const lsp = tools.find((tool) => tool.name === "LSP") as LocalMcpTool;
    const taskOutput = tools.find((tool) => tool.name === "TaskOutput") as LocalMcpTool;

    expect(enter.inputSchema).toEqual({
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "Optional name for the worktree. A random name is generated if omitted.",
        },
      },
    });
    expect(exit.inputSchema).toMatchObject({
      type: "object",
      properties: {
        action: { type: "string", enum: ["keep", "remove"] },
        discard_changes: { type: "boolean" },
      },
      required: ["action"],
    });
    await read.call({ file_path: "base.txt" });
    await expect(enter.call({ name: "../escape" })).rejects.toThrow(/portable/i);

    const entered = (await enter.call({ name: "feature" })) as LocalToolResult;
    const worktreePath = path.join(canonicalRoot, ".claude", "worktrees", "feature");
    expect(entered).toMatchObject({
      content: expect.stringContaining(`Created worktree at ${worktreePath}`),
      structuredContent: {
        worktreePath,
        worktreeBranch: "worktree-feature",
        message: expect.stringContaining("session is now working in the worktree"),
      },
    });
    expect(workspace.root).toBe(worktreePath);
    expect(shell.root).toBe(worktreePath);
    await expect(write.call({ file_path: "base.txt", content: "worktree\n" })).rejects.toThrow(
      /read completely/i,
    );
    await read.call({ file_path: "base.txt" });
    await write.call({ file_path: "base.txt", content: "worktree\n" });
    await lsp.call({
      operation: "hover",
      filePath: "base.txt",
      line: 1,
      character: 1,
    });
    expect(lspRoots).toEqual([worktreePath]);
    await expect(readFile(path.join(root, "base.txt"), "utf8")).resolves.toBe("main\n");
    if (process.platform === "darwin") {
      await expect(bash.call({ command: "pwd" })).resolves.toMatchObject({
        content: expect.stringContaining(worktreePath),
      });
    }
    await expect(enter.call({ name: "other" })).rejects.toThrow(/already in a worktree/i);

    await expect(exit.call({ action: "keep" })).resolves.toMatchObject({
      structuredContent: {
        action: "keep",
        originalCwd: canonicalRoot,
        worktreePath,
        worktreeBranch: "worktree-feature",
        message: expect.stringContaining("Your work is preserved"),
      },
    });
    expect(workspace.root).toBe(path.resolve(root));
    expect(shell.root).toBe(path.resolve(root));
    await lsp.call({
      operation: "hover",
      filePath: "base.txt",
      line: 1,
      character: 1,
    });
    expect(lspRoots).toEqual([worktreePath, path.resolve(root)]);
    await expect(readFile(path.join(worktreePath, "base.txt"), "utf8")).resolves.toBe("worktree\n");

    await enter.call({ name: "feature" });
    await write.call({ file_path: "committed.txt", content: "commit\n" });
    await git(worktreePath, "add", ".");
    await git(
      worktreePath,
      "-c",
      "user.email=workspace-tools@example.test",
      "-c",
      "user.name=Workspace Tools Test",
      "commit",
      "-m",
      "worktree commit",
    );
    await write.call({ file_path: "dirty.txt", content: "dirty\n" });
    let backgroundTaskId: string | undefined;
    if (process.platform === "darwin") {
      const background = (await bash.call({
        command: "sleep 30",
        run_in_background: true,
      })) as LocalToolResult;
      backgroundTaskId = (background.structuredContent as { backgroundTaskId: string })
        .backgroundTaskId;
    }
    await expect(exit.call({ action: "remove" })).rejects.toThrow(
      /1 uncommitted file and 1 commit/i,
    );
    await expect(
      exit.call({ action: "remove", discard_changes: true, harmless_extra: true }),
    ).resolves.toMatchObject({
      structuredContent: {
        action: "remove",
        originalCwd: canonicalRoot,
        worktreePath,
        worktreeBranch: "worktree-feature",
        discardedFiles: 1,
        discardedCommits: 1,
        message: expect.stringContaining("Exited and removed worktree"),
      },
    });
    await expect(realpath(worktreePath)).rejects.toThrow();
    await expect(git(root, "branch", "--list", "worktree-feature")).resolves.toBe("");
    if (backgroundTaskId) {
      await expect(
        taskOutput.call({ task_id: backgroundTaskId, block: false, timeout: 1_000 }),
      ).resolves.toMatchObject({ structuredContent: { status: "stopped" } });
    }
    await expect(exit.call({ action: "keep" })).rejects.toThrow(/No-op/i);

    const generated = (await enter.call({})) as LocalToolResult;
    const generatedPath = (generated.structuredContent as { worktreePath: string }).worktreePath;
    expect(path.basename(generatedPath)).toMatch(
      /^(calm|bright|swift|keen|bold)-(fox|owl|elm|oak|ray)-[a-f0-9]{4}$/,
    );
    await exit.call({ action: "remove" });
    await bash.dispose?.();
  }, 30_000);

  it("V412 preserves an active Claude worktree when the tool manager closes", async () => {
    const root = await temporaryDirectory();
    await git(root, "init", "-b", "main");
    await writeFile(path.join(root, "base.txt"), "main\n");
    await git(root, "add", "base.txt");
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
    const workspace = new WorkspaceTools(root);
    const shell = new WorkspaceShell(root);
    const tools = workspaceAgentTools(workspace, shell, { model: "claude-sonnet-4-6" });
    const bash = tools.find((tool) => tool.name === "Bash") as LocalMcpTool;
    const enter = tools.find((tool) => tool.name === "EnterWorktree") as LocalMcpTool;
    const write = tools.find((tool) => tool.name === "Write") as LocalMcpTool;
    const entered = (await enter.call({ name: "preserve" })) as LocalToolResult;
    const worktreePath = (entered.structuredContent as { worktreePath: string }).worktreePath;
    await write.call({ file_path: "unfinished.txt", content: "keep me\n" });

    await bash.dispose?.();

    expect(workspace.root).toBe(path.resolve(root));
    await expect(readFile(path.join(worktreePath, "unfinished.txt"), "utf8")).resolves.toBe(
      "keep me\n",
    );
    await expect(git(root, "branch", "--list", "worktree-preserve")).resolves.toContain(
      "worktree-preserve",
    );
  }, 30_000);

  it("V402-V405 conditionally projects the exact Claude LSP tool contract", async () => {
    const root = await temporaryDirectory();
    const lsp = vi.fn().mockResolvedValue({
      operation: "goToDefinition",
      result: "src/index.ts:4:2",
      filePath: "src/index.ts",
      resultCount: 1,
      fileCount: 1,
    });
    const claude = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      lsp,
    });
    const codex = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "gpt-5.4",
      lsp,
    });
    const tool = claude.find((candidate) => candidate.name === "LSP") as LocalMcpTool;

    expect(codex.some((candidate) => candidate.name === "LSP")).toBe(false);
    expect(projectAgentContextMessage(root, { model: "claude-opus-4-6", lsp })).toContain("LSP");
    expect(tool.inputSchema).toMatchObject({
      required: ["operation", "filePath", "line", "character"],
      properties: {
        operation: {
          enum: [
            "goToDefinition",
            "findReferences",
            "hover",
            "documentSymbol",
            "workspaceSymbol",
            "goToImplementation",
            "prepareCallHierarchy",
            "incomingCalls",
            "outgoingCalls",
          ],
        },
      },
    });
    await expect(
      tool.call({
        operation: "goToDefinition",
        filePath: "src/index.ts",
        line: 4,
        character: 2,
        harmless_extra: true,
      }),
    ).resolves.toEqual(
      expect.objectContaining({
        content: "src/index.ts:4:2",
        structuredContent: {
          operation: "goToDefinition",
          result: "src/index.ts:4:2",
          filePath: "src/index.ts",
          resultCount: 1,
          fileCount: 1,
        },
      }),
    );
    expect(lsp).toHaveBeenCalledWith({
      operation: "goToDefinition",
      filePath: "src/index.ts",
      line: 4,
      character: 2,
    });
  });

  it("V416-V420 conditionally projects the synchronous Claude Agent contract", async () => {
    const root = await temporaryDirectory();
    const agent = vi.fn().mockResolvedValue({
      status: "completed",
      prompt: "Inspect the runtime",
      agentId: "agent-123",
      content: [{ type: "text", text: "The runtime is sound." }],
      totalToolUseCount: 3,
      totalDurationMs: 25,
      totalTokens: 42,
      usage: {
        input_tokens: 30,
        output_tokens: 12,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: 4,
        server_tool_use: null,
        service_tier: null,
        cache_creation: null,
      },
    });
    const claude = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      agent,
    });
    const codex = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "gpt-5.4",
      agent,
    });
    const tool = claude.find((candidate) => candidate.name === "Agent") as LocalMcpTool;

    expect(codex.some((candidate) => candidate.name === "Agent")).toBe(false);
    expect(projectAgentContextMessage(root, { model: "claude-opus-4-6", agent })).toContain(
      "Agent, Bash",
    );
    expect(tool.inputSchema).toMatchObject({
      required: ["description", "prompt"],
      properties: {
        description: { type: "string" },
        prompt: { type: "string" },
        subagent_type: { type: "string" },
        model: { type: "string", enum: ["sonnet", "opus", "haiku"] },
        resume: { type: "string" },
      },
    });
    expect(Object.keys(tool.inputSchema.properties as object)).not.toContain("run_in_background");
    const result = await tool.call({
      description: "Inspect runtime",
      prompt: "Inspect the runtime",
      subagent_type: "general-purpose",
      model: "sonnet",
      resume: "agent-previous",
      harmless_extra: true,
    });
    expect(result.content).toBe("The runtime is sound.");
    expect(result.structuredContent).toEqual({
      status: "completed",
      prompt: "Inspect the runtime",
      agentId: "agent-123",
      content: [{ type: "text", text: "The runtime is sound." }],
      totalToolUseCount: 3,
      totalDurationMs: 25,
      totalTokens: 42,
      usage: {
        input_tokens: 30,
        output_tokens: 12,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: 4,
        server_tool_use: null,
        service_tier: null,
        cache_creation: null,
      },
    });
    expect(agent).toHaveBeenCalledWith({
      description: "Inspect runtime",
      prompt: "Inspect the runtime",
      subagentType: "general-purpose",
      model: "sonnet",
      resume: "agent-previous",
    });
    await expect(
      tool.call({
        description: "Background work",
        prompt: "Run later",
        run_in_background: true,
      }),
    ).rejects.toThrow(/background child agents are not supported/i);
    await expect(
      tool.call({
        description: "Isolated work",
        prompt: "Run elsewhere",
        isolation: "worktree",
      }),
    ).rejects.toThrow(/isolation is unavailable/i);
  });

  it("V375-V379 shares Claude task state and returns definite failures", async () => {
    const root = await temporaryDirectory();
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
    });
    const taskCreate = tools.find((tool) => tool.name === "TaskCreate") as LocalMcpTool;
    const taskGet = tools.find((tool) => tool.name === "TaskGet") as LocalMcpTool;
    const taskList = tools.find((tool) => tool.name === "TaskList") as LocalMcpTool;
    const taskUpdate = tools.find((tool) => tool.name === "TaskUpdate") as LocalMcpTool;

    await expect(
      taskCreate.call({
        subject: "Inspect runtime",
        description: "Read runtime code",
        activeForm: "Inspecting runtime",
        metadata: { source: "review" },
        harmless_extra: true,
      }),
    ).resolves.toMatchObject({
      content: "Task #1 created successfully: Inspect runtime",
      structuredContent: { task: { id: "1", subject: "Inspect runtime" } },
    });
    await taskCreate.call({ subject: "Run tests", description: "Verify runtime behavior" });
    await expect(
      taskUpdate.call({
        taskId: "1",
        status: "in_progress",
        owner: "reviewer",
        addBlocks: ["2"],
        metadata: { source: null, phase: 2 },
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        success: true,
        taskId: "1",
        updatedFields: ["owner", "metadata", "blocks", "status"],
        statusChange: { from: "pending", to: "in_progress" },
      },
    });
    await expect(taskGet.call({ taskId: "2" })).resolves.toMatchObject({
      structuredContent: {
        task: expect.objectContaining({ id: "2", blockedBy: ["1"] }),
      },
    });
    await expect(
      taskUpdate.call({ taskId: "1", subject: "Mutated", status: "invalid" }),
    ).rejects.toThrow(/status must be one of/i);
    await expect(taskGet.call({ taskId: "1" })).resolves.toMatchObject({
      structuredContent: { task: expect.objectContaining({ subject: "Inspect runtime" }) },
    });
    await expect(taskList.call({ ignored: true })).resolves.toMatchObject({
      structuredContent: {
        tasks: [
          expect.objectContaining({ id: "1", status: "in_progress", owner: "reviewer" }),
          expect.objectContaining({ id: "2", status: "pending", blockedBy: ["1"] }),
        ],
      },
    });
    await expect(
      taskUpdate.call({ taskId: "missing", status: "completed" }),
    ).resolves.toMatchObject({
      content: "Task missing was not found.",
      isError: true,
      structuredContent: {
        success: false,
        taskId: "missing",
        updatedFields: [],
        error: "Task missing was not found.",
      },
    });
    await taskUpdate.call({ taskId: "2", status: "deleted" });
    await expect(taskGet.call({ taskId: "2" })).resolves.toMatchObject({
      structuredContent: { task: null },
    });
    await expect(taskGet.call({ taskId: "1" })).resolves.toMatchObject({
      structuredContent: { task: expect.objectContaining({ blocks: [] }) },
    });
  });

  it("V384 loads only selected Claude Skills and expands native arguments", async () => {
    const root = await temporaryDirectory();
    const skillDirectory = path.join(root, "skills", "review");
    const plainSkillDirectory = path.join(root, "skills", "plain");
    await mkdir(skillDirectory, { recursive: true });
    await mkdir(plainSkillDirectory, { recursive: true });
    await writeFile(
      path.join(skillDirectory, "SKILL.md"),
      `---
name: review
description: Review one issue
arguments: [issue, branch]
---
Issue=$issue
Branch=$1
First=$ARGUMENTS[0]
All=$ARGUMENTS
Dir=\${CLAUDE_SKILL_DIR}
Effort=\${CLAUDE_EFFORT}
Session=\${CLAUDE_SESSION_ID}
`,
    );
    await writeFile(path.join(plainSkillDirectory, "SKILL.md"), "Follow the checklist.\n");
    const canonicalSkillPath = await realpath(path.join(skillDirectory, "SKILL.md"));
    const canonicalSkillDirectory = path.dirname(canonicalSkillPath);
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
      effort: "high",
      sessionId: "session-123",
      skills: [
        {
          id: "review-skill",
          name: "review",
          filePath: skillDirectory,
          description: "Review one issue",
        },
        { id: "plain", filePath: plainSkillDirectory },
      ],
    });
    const skill = tools.find((tool) => tool.name === "Skill") as LocalMcpTool;
    expect(skill.description).toContain("review-skill");

    await expect(skill.call({ skill: "review", args: '"hello world" dev' })).resolves.toMatchObject(
      {
        content: expect.stringContaining("Issue=hello world"),
        structuredContent: expect.objectContaining({
          skill: "review-skill",
          args: '"hello world" dev',
          sourcePath: canonicalSkillPath,
        }),
      },
    );
    const expanded = (await skill.call({
      skill: "review-skill",
      args: '"hello world" dev',
    })) as LocalToolResult;
    expect(expanded.content).not.toContain("description: Review one issue");
    expect(expanded.content).toContain("Branch=dev");
    expect(expanded.content).toContain("First=hello world");
    expect(expanded.content).toContain('All="hello world" dev');
    expect(expanded.content).toContain(`Dir=${canonicalSkillDirectory}`);
    expect(expanded.content).toContain("Effort=high");
    expect(expanded.content).toContain("Session=session-123");

    await expect(skill.call({ skill: "plain", args: "one two" })).resolves.toMatchObject({
      content: "Follow the checklist.\n\nARGUMENTS: one two\n",
    });
    await expect(skill.call({ skill: "missing" })).rejects.toThrow(/not available/i);
  });

  it("V386-V389 bridges questions and enforces real plan approval state", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "existing.txt"), "before\n");
    const interactionRequests: ClaudeInteractionRequest[] = [];
    const approvals = [false, true];
    const closeInteractions = vi.fn();
    const interact = async (
      request: ClaudeInteractionRequest,
    ): Promise<ClaudeInteractionResponse> => {
      interactionRequests.push(request);
      if (request.kind === "questions") {
        return {
          kind: "questions",
          answers: {
            "Which runtime?": "Node",
            "Which features?": "Tests, Custom telemetry",
          },
        };
      }
      const approved = approvals.shift() ?? true;
      return {
        kind: "plan_approval",
        approved,
        ...(approved ? {} : { feedback: "Add rollback steps" }),
      };
    };
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-opus-4-6",
      interact,
      closeInteractions,
    });
    const ask = tools.find((tool) => tool.name === "AskUserQuestion") as LocalMcpTool;
    const enter = tools.find((tool) => tool.name === "EnterPlanMode") as LocalMcpTool;
    const exit = tools.find((tool) => tool.name === "ExitPlanMode") as LocalMcpTool;
    const read = tools.find((tool) => tool.name === "Read") as LocalMcpTool;
    const write = tools.find((tool) => tool.name === "Write") as LocalMcpTool;
    const edit = tools.find((tool) => tool.name === "Edit") as LocalMcpTool;
    const bash = tools.find((tool) => tool.name === "Bash") as LocalMcpTool;
    const notebook = tools.find((tool) => tool.name === "NotebookEdit") as LocalMcpTool;

    expect(tools.map((tool) => tool.name)).toEqual(
      expect.arrayContaining(["AskUserQuestion", "EnterPlanMode", "ExitPlanMode"]),
    );
    await expect(
      ask.call({
        questions: [
          {
            question: "Which runtime?",
            header: "Runtime",
            options: [
              { label: "Node", description: "Use Node.js" },
              { label: "Bun", description: "Use Bun" },
            ],
            multiSelect: false,
          },
          {
            question: "Which features?",
            header: "Features",
            options: [
              { label: "Tests", description: "Add tests", preview: "pnpm test" },
              { label: "Telemetry", description: "Add telemetry" },
            ],
            multiSelect: true,
          },
        ],
        harmless_extra: true,
      }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("Which runtime?: Node"),
      structuredContent: {
        questions: expect.any(Array),
        answers: {
          "Which runtime?": "Node",
          "Which features?": "Tests, Custom telemetry",
        },
      },
    });
    await expect(
      ask.call({
        questions: [
          {
            question: "Invalid?",
            header: "This header is too long",
            options: [
              { label: "A", description: "A" },
              { label: "B", description: "B" },
            ],
            multiSelect: false,
          },
        ],
      }),
    ).rejects.toThrow(/at most 12 characters/i);

    const entered = (await enter.call({ ignored: true })) as LocalToolResult;
    const message = (entered.structuredContent as { message: string }).message;
    const planPath = /to (.+) using Write/.exec(message)?.[1];
    expect(planPath).toBeTruthy();
    if (!planPath) throw new Error("EnterPlanMode did not return a plan path");
    expect(path.isAbsolute(planPath)).toBe(true);
    expect(planPath.startsWith(root)).toBe(false);

    await expect(bash.call({ command: "pwd" })).rejects.toThrow(/unavailable in plan mode/i);
    await expect(
      write.call({ file_path: path.join(root, "blocked.txt"), content: "blocked\n" }),
    ).rejects.toThrow(/unavailable in plan mode/i);
    await expect(
      edit.call({
        file_path: path.join(root, "existing.txt"),
        old_string: "before",
        new_string: "after",
      }),
    ).rejects.toThrow(/unavailable in plan mode/i);
    await expect(
      notebook.call({ notebook_path: path.join(root, "missing.ipynb"), new_source: "x" }),
    ).rejects.toThrow(/unavailable in plan mode/i);

    await expect(
      write.call({ file_path: planPath, content: "# Plan\n\n1. Implement.\n" }),
    ).resolves.toMatchObject({ structuredContent: { filePath: planPath, type: "update" } });
    await expect(read.call({ file_path: planPath })).resolves.toMatchObject({
      content: expect.stringContaining("# Plan"),
      structuredContent: { type: "text", file: { filePath: planPath } },
    });
    await expect(
      exit.call({ allowedPrompts: [{ tool: "Bash", prompt: "run tests" }] }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("Add rollback steps"),
      isError: true,
      structuredContent: {
        plan: expect.stringContaining("Implement"),
        isAgent: false,
        filePath: planPath,
      },
    });
    await expect(
      write.call({ file_path: path.join(root, "still-blocked.txt"), content: "blocked\n" }),
    ).rejects.toThrow(/unavailable in plan mode/i);
    await write.call({
      file_path: planPath,
      content: "# Plan\n\n1. Implement.\n2. Roll back safely.\n",
    });
    await expect(exit.call({ ignored: true })).resolves.toMatchObject({
      content: expect.stringContaining("approved"),
      structuredContent: {
        plan: expect.stringContaining("Roll back safely"),
        isAgent: false,
        filePath: planPath,
        planWasEdited: false,
      },
    });
    await expect(
      write.call({ file_path: path.join(root, "approved.txt"), content: "approved\n" }),
    ).resolves.toMatchObject({ structuredContent: { type: "create" } });
    expect(interactionRequests.map((request) => request.kind)).toEqual([
      "questions",
      "plan_approval",
      "plan_approval",
    ]);

    await bash.dispose?.();
    expect(closeInteractions).toHaveBeenCalledOnce();
    await expect(readFile(planPath, "utf8")).rejects.toThrow();
  });

  it("V425/V427 exposes exact Monitor and Cron bridges only for session-backed Claude runs", async () => {
    const root = await temporaryDirectory();
    const bridge: ClaudeSessionToolBridge = {
      monitor: vi.fn(async (request) => ({
        taskId: "42",
        timeoutMs: request.timeoutMs,
        ...(request.persistent ? { persistent: true } : {}),
      })),
      createCron: vi.fn(async (request) => ({
        id: "cron_1",
        humanSchedule: "Every minute",
        recurring: request.recurring,
      })),
      deleteCron: vi.fn(async (id) => ({ id })),
      listCrons: vi.fn(async () => ({
        jobs: [
          {
            id: "cron_1",
            cron: "* * * * *",
            humanSchedule: "Every minute",
            prompt: "check status",
            recurring: true,
          },
        ],
      })),
    };
    const shell = new WorkspaceShell(root);
    const close = vi.spyOn(shell, "close");
    const tools = workspaceAgentTools(new WorkspaceTools(root), shell, {
      model: "claude-opus-4-6",
      sessionTools: bridge,
      borrowShell: true,
    });
    const monitor = tools.find((tool) => tool.name === "Monitor") as LocalMcpTool;
    const create = tools.find((tool) => tool.name === "CronCreate") as LocalMcpTool;
    const remove = tools.find((tool) => tool.name === "CronDelete") as LocalMcpTool;
    const list = tools.find((tool) => tool.name === "CronList") as LocalMcpTool;

    expect(monitor.inputSchema).toMatchObject({
      additionalProperties: false,
      required: ["command", "description"],
      properties: {
        timeout_ms: { minimum: 1_000, default: 300_000 },
        persistent: { default: false },
      },
    });
    await expect(
      monitor.call({ command: "pnpm dev", description: "dev server", persistent: true }),
    ).resolves.toMatchObject({
      content: "Monitor started with task ID: 42",
      structuredContent: { taskId: "42", timeoutMs: 300_000, persistent: true },
    });
    expect(bridge.monitor).toHaveBeenCalledWith({
      command: "pnpm dev",
      description: "dev server",
      timeoutMs: 300_000,
      persistent: true,
    });
    await expect(create.call({ cron: "* * * * *", prompt: "check status" })).resolves.toMatchObject(
      {
        structuredContent: { id: "cron_1", humanSchedule: "Every minute", recurring: true },
      },
    );
    expect(bridge.createCron).toHaveBeenCalledWith({
      cron: "* * * * *",
      prompt: "check status",
      recurring: true,
      durable: false,
    });
    await expect(remove.call({ id: "cron_1" })).resolves.toMatchObject({
      structuredContent: { id: "cron_1" },
    });
    await expect(list.call({})).resolves.toMatchObject({
      structuredContent: { jobs: [expect.objectContaining({ id: "cron_1" })] },
    });

    await tools.find((tool) => tool.name === "Bash")?.dispose?.();
    expect(close).not.toHaveBeenCalled();
    await shell.close();
    expect(close).toHaveBeenCalledOnce();

    const codexNames = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "gpt-5-codex",
      sessionTools: bridge,
    }).map((tool) => tool.name);
    expect(codexNames).not.toEqual(expect.arrayContaining(["Monitor", "CronCreate"]));
  });

  it("V376 replaces Claude todos and validates review findings", async () => {
    const root = await temporaryDirectory();
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-opus-4-6",
    });
    const todoWrite = tools.find((tool) => tool.name === "TodoWrite") as LocalMcpTool;
    const reportFindings = tools.find((tool) => tool.name === "ReportFindings") as LocalMcpTool;
    const firstTodos = [
      { content: "Inspect code", status: "in_progress", activeForm: "Inspecting code" },
      { content: "Run tests", status: "pending", activeForm: "Running tests" },
    ];

    await expect(todoWrite.call({ todos: firstTodos })).resolves.toMatchObject({
      structuredContent: { oldTodos: [], newTodos: firstTodos },
    });
    await expect(
      todoWrite.call({
        todos: [{ content: "Run tests", status: "completed", activeForm: "Running tests" }],
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        oldTodos: firstTodos,
        newTodos: [{ content: "Run tests", status: "completed", activeForm: "Running tests" }],
      },
    });
    await expect(
      reportFindings.call({
        level: "high",
        findings: [
          {
            file: "src/runtime.ts",
            line: 12,
            summary: "Cancellation is ignored",
            failure_scenario: "Abort during wait -> child continues",
            category: "correctness",
            verdict: "CONFIRMED",
          },
        ],
      }),
    ).resolves.toMatchObject({
      content: "Reported 1 finding.",
      structuredContent: {
        count: 1,
        level: "high",
        findings: [expect.objectContaining({ file: "src/runtime.ts", line: 12 })],
      },
    });
    await expect(
      reportFindings.call({
        findings: [
          {
            file: "../outside.ts",
            summary: "Bad path",
            failure_scenario: "Path escapes",
          },
        ],
      }),
    ).rejects.toThrow(/traversal/i);
    await expect(
      reportFindings.call({
        findings: Array.from({ length: 33 }, (_, index) => ({
          file: `src/${index}.ts`,
          summary: "Finding",
          failure_scenario: "Scenario",
        })),
      }),
    ).rejects.toThrow(/at most 32/i);
  });

  it("V377 edits notebook cells through guarded Project writes", async () => {
    const root = await temporaryDirectory();
    const notebookPath = path.join(root, "analysis.ipynb");
    await writeFile(
      notebookPath,
      `${JSON.stringify(
        {
          cells: [
            {
              cell_type: "markdown",
              id: "intro",
              metadata: {},
              source: ["# Before\n"],
            },
            {
              cell_type: "code",
              execution_count: null,
              id: "code-1",
              metadata: {},
              outputs: [],
              source: ["print('before')\n"],
            },
          ],
          metadata: { language_info: { name: "python" } },
          nbformat: 4,
          nbformat_minor: 5,
        },
        null,
        2,
      )}\n`,
    );
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-sonnet-4-6",
    });
    const notebookEdit = tools.find((tool) => tool.name === "NotebookEdit") as LocalMcpTool;

    await expect(
      notebookEdit.call({
        notebook_path: notebookPath,
        cell_id: "intro",
        new_source: "# After\n",
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        new_source: "# After\n",
        old_source: "# Before\n",
        cell_id: "intro",
        cell_type: "markdown",
        language: "python",
        edit_mode: "replace",
        notebook_path: notebookPath,
        original_file: expect.stringContaining("# Before"),
        updated_file: expect.stringContaining("# After"),
      },
    });
    const inserted = (await notebookEdit.call({
      notebook_path: notebookPath,
      cell_id: "intro",
      new_source: "print('inserted')\n",
      cell_type: "code",
      edit_mode: "insert",
    })) as LocalToolResult;
    const insertedId = (inserted.structuredContent as { cell_id: string }).cell_id;
    expect(insertedId).toMatch(/^[a-f0-9]{8}$/);
    await expect(
      notebookEdit.call({
        notebook_path: notebookPath,
        cell_id: "code-1",
        new_source: "",
        edit_mode: "delete",
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        old_source: "print('before')\n",
        cell_id: "code-1",
        cell_type: "code",
        edit_mode: "delete",
      },
    });
    const notebook = JSON.parse(await readFile(notebookPath, "utf8")) as {
      cells: Array<{ id: string; source: string | string[] }>;
    };
    expect(notebook.cells.map((cell) => cell.id)).toEqual(["intro", insertedId]);
    expect(notebook.cells[0]?.source).toEqual(["# After\n"]);
    await expect(
      notebookEdit.call({
        notebook_path: "../outside.ipynb",
        cell_id: "intro",
        new_source: "x",
      }),
    ).rejects.toThrow(/traversal/i);
  });

  it.runIf(process.platform === "darwin")(
    "V364-V366-V383 returns native-shaped background and session outputs",
    async () => {
      const root = await temporaryDirectory();
      const claude = workspaceAgentTools(new WorkspaceTools(root), undefined, {
        model: "claude-sonnet-4-6",
      });
      const bash = claude.find((tool) => tool.name === "Bash") as LocalMcpTool;
      const taskOutput = claude.find((tool) => tool.name === "TaskOutput") as LocalMcpTool;
      try {
        const started = (await bash.call({
          command: "sleep 0.1; printf background-done",
          run_in_background: true,
        })) as LocalToolResult;
        const taskId = (started.structuredContent as { backgroundTaskId: string }).backgroundTaskId;
        expect(started).toMatchObject({
          content: expect.stringContaining(`ID: ${taskId}`),
          structuredContent: {
            stdout: expect.any(String),
            stderr: expect.any(String),
            interrupted: false,
            backgroundTaskId: taskId,
          },
        });

        const completed = (await taskOutput.call({
          task_id: taskId,
          block: true,
          timeout: 2_000,
        })) as LocalToolResult;
        expect(completed).toMatchObject({
          content: expect.stringContaining("<status>completed</status>"),
          structuredContent: expect.objectContaining({
            stdout: "background-done",
            status: "completed",
            exitCode: 0,
          }),
        });

        const moved = (await bash.call({
          command: "printf foreground-start; sleep 0.15; printf background-done",
          timeout: 10,
        })) as LocalToolResult;
        const movedTaskId = (moved.structuredContent as { backgroundTaskId: string })
          .backgroundTaskId;
        expect(moved).toMatchObject({
          content: expect.stringContaining("moved to the background"),
          structuredContent: expect.objectContaining({
            backgroundTaskId: movedTaskId,
          }),
        });
        await expect(
          taskOutput.call({ task_id: movedTaskId, block: true, timeout: 2_000 }),
        ).resolves.toMatchObject({
          structuredContent: expect.objectContaining({
            status: "completed",
            stdout: "foreground-startbackground-done",
          }),
        });

        await expect(bash.call({ command: "sleep 0.1", timeout: 10 })).resolves.toMatchObject({
          content: expect.stringContaining("timed out"),
          structuredContent: expect.not.objectContaining({ backgroundTaskId: expect.anything() }),
        });
      } finally {
        await bash.dispose?.();
      }

      const codex = workspaceAgentTools(new WorkspaceTools(root), undefined, {
        model: "gpt-5.4",
        apiProtocol: "openai_responses",
      });
      const execCommand = codex.find((tool) => tool.name === "exec_command") as LocalMcpTool;
      const writeStdin = codex.find((tool) => tool.name === "write_stdin") as LocalMcpTool;
      try {
        const started = (await execCommand.call({
          cmd: 'read -r value; printf "received:%s" "$value"',
          yield_time_ms: 250,
        })) as LocalToolResult;
        const sessionId = (started.structuredContent as { session_id: number }).session_id;
        expect(started.content).toContain(`Process running with session ID ${sessionId}`);

        const completed = (await writeStdin.call({
          session_id: sessionId,
          chars: "hello\n",
          yield_time_ms: 2_000,
        })) as LocalToolResult;
        expect(completed.content).toContain("Process exited with code 0");
        expect(completed.structuredContent).toMatchObject({
          exit_code: 0,
          output: "received:hello",
          chunk_id: expect.any(String),
          wall_time_seconds: expect.any(Number),
        });

        const terminal = (await execCommand.call({
          cmd: 'test -t 0 && printf "terminal:%s" "$TERM"',
          tty: true,
          yield_time_ms: 2_000,
        })) as LocalToolResult;
        expect(terminal.content).toContain("Process exited with code 0");
        expect(terminal.structuredContent).toMatchObject({
          exit_code: 0,
          output: "terminal:xterm-256color",
          chunk_id: expect.any(String),
          wall_time_seconds: expect.any(Number),
        });
      } finally {
        await execCommand.dispose?.();
      }
    },
    15_000,
  );

  it("V358-V359 applies Codex freeform patches through guarded mutations", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, "update.txt"), "before\nkeep\n");
    await writeFile(path.join(root, "delete.txt"), "delete me\n");
    const workspace = new WorkspaceTools(root);
    const applyPatch = workspaceAgentTools(workspace, undefined, {
      model: "gpt-5.4",
      apiProtocol: "openai_responses",
    }).find((tool) => tool.name === "apply_patch") as LocalTextTool | undefined;

    await expect(
      applyPatch?.call(`*** Begin Patch
*** Update File: update.txt
@@
-before
+after
 keep
*** Delete File: delete.txt
*** Add File: added.txt
+new file
*** End Patch
`),
    ).resolves.toMatchObject({
      content: expect.stringContaining("Success."),
      structuredContent: { operations: expect.any(Array) },
    });
    await expect(readFile(path.join(root, "update.txt"), "utf8")).resolves.toBe("after\nkeep\n");
    await expect(readFile(path.join(root, "added.txt"), "utf8")).resolves.toBe("new file\n");
    await expect(readFile(path.join(root, "delete.txt"), "utf8")).rejects.toMatchObject({
      code: "ENOENT",
    });
    await expect(
      applyPatch?.call("*** Begin Patch\n*** Delete File: ../outside.txt\n*** End Patch\n"),
    ).rejects.toThrow(/traversal/i);
  });

  it("V357 exposes bounded Claude Glob and Grep adapters", async () => {
    const root = await temporaryDirectory();
    await mkdir(path.join(root, "src"));
    await writeFile(path.join(root, "src", "one.ts"), "const trainedTool = true;\n");
    await writeFile(path.join(root, "src", "two.md"), "trainedTool\n");
    const tools = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-haiku-4-5",
    });
    const glob = tools.find((tool) => tool.name === "Glob") as LocalMcpTool | undefined;
    const grep = tools.find((tool) => tool.name === "Grep") as LocalMcpTool | undefined;

    await expect(glob?.call({ pattern: "**/*.ts" })).resolves.toMatchObject({
      content: expect.stringContaining("src/one.ts"),
      structuredContent: expect.objectContaining({
        filenames: [path.join(root, "src", "one.ts")],
        numFiles: 1,
        truncated: false,
      }),
    });
    await expect(
      grep?.call({ pattern: "trainedTool", path: root, output_mode: "files_with_matches" }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("src/one.ts"),
      structuredContent: expect.objectContaining({
        mode: "files_with_matches",
        numFiles: 2,
      }),
    });
  });

  it("V382 applies Claude Glob ignore, mtime, and 100-result semantics", async () => {
    const root = await temporaryDirectory();
    await writeFile(path.join(root, ".gitignore"), "ignored.ts\n");
    await Promise.all([
      ...Array.from({ length: 101 }, (_, index) =>
        writeFile(path.join(root, `file-${String(index).padStart(3, "0")}.ts`), `${index}\n`),
      ),
      writeFile(path.join(root, "ignored.ts"), "ignored\n"),
      writeFile(path.join(root, ".hidden.ts"), "hidden\n"),
    ]);
    const newest = path.join(root, "ignored.ts");
    const oldest = path.join(root, "file-000.ts");
    await utimes(newest, new Date(Date.now() + 60_000), new Date(Date.now() + 60_000));
    await utimes(oldest, new Date(0), new Date(0));
    const glob = workspaceAgentTools(new WorkspaceTools(root), undefined, {
      model: "claude-haiku-4-5",
    }).find((tool) => tool.name === "Glob") as LocalMcpTool;

    const result = (await glob.call({ pattern: "**/*.ts" })) as LocalToolResult;
    const output = result.structuredContent as {
      filenames: string[];
      numFiles: number;
      totalMatches: number;
      truncated: boolean;
      countIsComplete: boolean;
    };
    expect(output).toMatchObject({
      numFiles: 100,
      totalMatches: 103,
      truncated: true,
      countIsComplete: true,
    });
    expect(output.filenames[0]).toBe(newest);
    expect(output.filenames).toContain(path.join(root, ".hidden.ts"));
    expect(output.filenames).not.toContain(oldest);
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
