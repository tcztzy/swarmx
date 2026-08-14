import { resolveHarnessPermissionLayers } from "@swarmx/core";
import type {
  LocalMcpTool,
  LocalTextTool,
  LocalToolCallContext,
} from "@swarmx/core/local-tool-contracts";
import { describe, expect, it, vi } from "vitest";
import { applyWorkspaceToolPolicy, workspaceToolAccess } from "./workspace-tool-permissions.js";

describe("Workspace tool permissions", () => {
  it("preserves tool and array identity when no policy is configured", () => {
    const tools = [functionTool("Read"), textTool("apply_patch")];

    const result = applyWorkspaceToolPolicy(tools, {});

    expect(result).toBe(tools);
    expect(result[0]).toBe(tools[0]);
    expect(result[1]).toBe(tools[1]);
  });

  it("classifies the stable read/write sets and treats unknown tools as execution", () => {
    expect(
      [
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
        "TodoList",
        "TodoWrite",
      ].every((name) => workspaceToolAccess(name) === "read"),
    ).toBe(true);
    expect(
      ["Edit", "NotebookEdit", "Write", "apply_patch"].every(
        (name) => workspaceToolAccess(name) === "write",
      ),
    ).toBe(true);
    expect(workspaceToolAccess("exec_command")).toBe("execute");
    expect(workspaceToolAccess("future_tool")).toBe("execute");
  });

  it("allows reads, rejects explicit denials, and fails closed without a bridge", async () => {
    const read = functionTool("Read");
    const denied = functionTool("exec_command");
    const write = functionTool("Write");
    const context = { invocationId: "invocation-1" } satisfies LocalToolCallContext;
    const guarded = applyWorkspaceToolPolicy([read, denied, write], {
      permissionPolicy: {
        mode: "default",
        allowedTools: [],
        deniedTools: ["exec_command"],
      },
    }) as LocalMcpTool[];

    await expect(guarded[0]?.call({ path: "README.md" }, context)).resolves.toEqual("Read:done");
    expect(read.call).toHaveBeenCalledWith({ path: "README.md" }, context);
    await expect(guarded[1]?.call({ cmd: "pwd" })).rejects.toThrow(/explicit_deny/i);
    await expect(guarded[2]?.call({ file_path: "a.txt" })).rejects.toThrow(
      /requires approval.*no interaction bridge/i,
    );
    expect(denied.call).not.toHaveBeenCalled();
    expect(write.call).not.toHaveBeenCalled();

    expect(() =>
      applyWorkspaceToolPolicy([read], {
        permissionPolicy: { mode: "not-a-mode" } as never,
      }),
    ).toThrow();
  });

  it("accepts only allow-once human responses and redacts executable content", async () => {
    const execute = functionTool("exec_command");
    const interact = vi
      .fn()
      .mockResolvedValueOnce({ kind: "questions", answers: {} })
      .mockResolvedValueOnce({ kind: "tool_approval", optionId: "reject_once" })
      .mockResolvedValueOnce({ kind: "tool_approval", optionId: "allow_always" })
      .mockResolvedValueOnce({ kind: "tool_approval", optionId: "allow_once" });
    const [guarded] = applyWorkspaceToolPolicy([execute], {
      permissionPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      interact,
    }) as LocalMcpTool[];
    const input = {
      cmd: "printenv PRIVATE_TOKEN",
      path: "x".repeat(260),
      ignored: "private body",
    };

    await expect(guarded?.call(input)).rejects.toThrow(/rejected by the user/i);
    await expect(guarded?.call(input)).rejects.toThrow(/rejected by the user/i);
    await expect(guarded?.call(input)).rejects.toThrow(/rejected by the user/i);
    await expect(guarded?.call(input)).resolves.toEqual("exec_command:done");
    expect(execute.call).toHaveBeenCalledTimes(1);
    const requests = interact.mock.calls.map(([request]) => request);
    expect(requests[3]).toMatchObject({
      kind: "tool_approval",
      title: "Allow exec_command?",
      toolKind: "execute",
      source: "direct",
      summary: expect.stringContaining("command: Project-sandboxed shell command"),
      options: [
        { optionId: "reject_once", name: "Reject", kind: "reject_once" },
        { optionId: "allow_once", name: "Allow once", kind: "allow_once" },
      ],
    });
    expect(requests[3]?.summary).toContain(`${"x".repeat(239)}…`);
    expect(JSON.stringify(requests)).not.toContain("PRIVATE_TOKEN");
    expect(JSON.stringify(requests)).not.toContain("private body");
  });

  it("uses automatic review first and falls back to the human bridge", async () => {
    const execute = functionTool("future_tool");
    const reviewPermission = vi
      .fn()
      .mockResolvedValueOnce(true)
      .mockResolvedValueOnce(false)
      .mockRejectedValueOnce(new Error("review unavailable"));
    const interact = vi.fn().mockResolvedValue({
      kind: "tool_approval",
      optionId: "allow_once",
    });
    const [guarded] = applyWorkspaceToolPolicy([execute], {
      permissionPolicy: { mode: "auto", allowedTools: [], deniedTools: [] },
      reviewPermission,
      interact,
    }) as LocalMcpTool[];
    const input = { command: "echo private", description: "Run the configured build." };

    await expect(guarded?.call(input)).resolves.toEqual("future_tool:done");
    await expect(guarded?.call(input)).resolves.toEqual("future_tool:done");
    await expect(guarded?.call(input)).resolves.toEqual("future_tool:done");
    expect(reviewPermission).toHaveBeenCalledTimes(3);
    expect(reviewPermission).toHaveBeenCalledWith(
      expect.objectContaining({
        source: "direct",
        toolName: "future_tool",
        toolKind: "execute",
        toolInput: input,
      }),
    );
    expect(interact).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(interact.mock.calls)).not.toContain("echo private");

    const [withoutReviewer] = applyWorkspaceToolPolicy([execute], {
      permissionPolicy: { mode: "auto", allowedTools: [], deniedTools: [] },
      interact,
    }) as LocalMcpTool[];
    await expect(withoutReviewer?.call({})).resolves.toEqual("future_tool:done");

    const defaultReviewer = vi.fn().mockResolvedValue(true);
    const rejectingHuman = vi.fn().mockResolvedValue({
      kind: "tool_approval",
      optionId: "reject_once",
    });
    const [defaultGuarded] = applyWorkspaceToolPolicy([execute], {
      permissionPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      reviewPermission: defaultReviewer,
      interact: rejectingHuman,
    }) as LocalMcpTool[];
    await expect(defaultGuarded?.call({})).rejects.toThrow(/rejected by the user/i);
    expect(defaultReviewer).not.toHaveBeenCalled();
  });

  it("preserves text-tool metadata/context and layered policy sources", async () => {
    const patch = textTool("apply_patch");
    const interact = vi.fn().mockResolvedValue({
      kind: "tool_approval",
      optionId: "allow_once",
    });
    const policy = resolveHarnessPermissionLayers([
      {
        id: "session-default",
        source: "session",
        mode: "default",
        allowedTools: [],
        deniedTools: [],
      },
    ]);
    const [guarded] = applyWorkspaceToolPolicy([patch], { permissionPolicy: policy, interact }) as
      | LocalTextTool[]
      | [];
    const context = { invocationId: "patch-1" } satisfies LocalToolCallContext;

    await expect(guarded?.call("*** Begin Patch\n", context)).resolves.toEqual("apply_patch:done");
    expect(patch.call).toHaveBeenCalledWith("*** Begin Patch\n", context);
    expect(guarded).toMatchObject({
      name: "apply_patch",
      kind: "text",
      description: "apply_patch description",
      format: patch.format,
      dispose: patch.dispose,
    });
    expect(interact).toHaveBeenCalledWith(
      expect.objectContaining({
        policySourceIds: ["session-default"],
        summary: "apply_patch requested a bounded Project patch.",
      }),
    );
  });

  it("uses a content-free fallback summary when no safe field is present", async () => {
    const tool = functionTool("unclassified");
    const interact = vi.fn().mockResolvedValue({
      kind: "tool_approval",
      optionId: "allow_once",
    });
    const [guarded] = applyWorkspaceToolPolicy([tool], {
      permissionPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      interact,
    }) as LocalMcpTool[];

    await expect(guarded?.call({ payload: "private" })).resolves.toEqual("unclassified:done");
    expect(interact).toHaveBeenCalledWith(
      expect.objectContaining({
        summary: "unclassified requested a execute operation in the active Project.",
      }),
    );
    expect(JSON.stringify(interact.mock.calls)).not.toContain("private");
  });
});

function functionTool(name: string): LocalMcpTool & { call: ReturnType<typeof vi.fn> } {
  const dispose = vi.fn();
  return {
    name,
    description: `${name} description`,
    inputSchema: { type: "object" },
    dispose,
    call: vi.fn(async () => `${name}:done`),
  };
}

function textTool(name: string): LocalTextTool & { call: ReturnType<typeof vi.fn> } {
  const dispose = vi.fn();
  return {
    kind: "text",
    name,
    description: `${name} description`,
    format: { type: "grammar", syntax: "lark", definition: "start: /.+/" },
    dispose,
    call: vi.fn(async () => `${name}:done`),
  };
}
