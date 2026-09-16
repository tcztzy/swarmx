import { type ToolCallMessagePart, useAuiState } from "@assistant-ui/react";
import { createContext, type ReactNode, useContext } from "react";
import { z } from "zod";
import type { ToolActivity } from "../message-activity.js";
import {
  ToolGroupContent,
  ToolGroupRoot,
  ToolGroupTrigger,
} from "./components/assistant-ui/elements/tool-group.js";
import { t, useTranslation } from "./i18n.js";

export const ToolActivityContext = createContext<Record<string, ToolActivity>>({});
const GroupedToolsContext = createContext(false);
const CommandSchema = z.object({ command: z.string() });
const ShellResultSchema = z
  .object({
    type: z.literal("commandExecution"),
    aggregatedOutput: z.string().nullable(),
    exitCode: z.number().int().nullable(),
  })
  .transform((value) => ({
    formatted_output: value.aggregatedOutput ?? "",
    exit_code: value.exitCode,
  }))
  .or(
    z.object({
      formatted_output: z.string(),
      exit_code: z.number().int().nullish(),
    }),
  )
  .or(
    z.object({ stdout: z.string(), stderr: z.string() }).transform((value) => ({
      formatted_output: [value.stdout, value.stderr].filter(Boolean).join("\n"),
      exit_code: null,
    })),
  )
  .or(
    z
      .object({ result: z.object({ output: z.string(), exit_code: z.number().int().nullish() }) })
      .transform(({ result }) => ({
        formatted_output: result.output,
        exit_code: result.exit_code,
      })),
  );

export function readShell({ args, result }: Pick<ToolCallMessagePart, "args" | "result">) {
  const command = CommandSchema.safeParse(args);
  const output = ShellResultSchema.safeParse(result);
  if (!output.success && (result !== undefined || !command.success)) return undefined;
  return {
    command: command.success ? command.data.command : undefined,
    output: output.success ? output.data.formatted_output : undefined,
    exitCode: output.success ? output.data.exit_code : undefined,
  };
}

export function ToolGroup({
  calls,
  running = false,
  children,
}: {
  calls: readonly ToolCallMessagePart[];
  running?: boolean;
  children: ReactNode;
}) {
  useTranslation();
  const activity = useContext(ToolActivityContext);
  const grouped = useContext(GroupedToolsContext);
  if (grouped || calls.length === 0) return children;
  const actions = new Set(
    calls.map((call) => {
      switch (activity[call.toolCallId]?.kind) {
        case "read":
          return t("读取文件");
        case "search":
          return t("搜索内容");
        case "edit":
          return t("修改文件");
        default:
          return readShell(call) ? t("运行命令") : t("调用工具");
      }
    }),
  );
  const label = [...actions]
    .map((action, index) => (index === 0 ? action : action.toLowerCase()))
    .join(", ");
  return (
    <ToolGroupRoot variant="ghost" className="tool-group">
      <ToolGroupTrigger active={running} count={calls.length}>
        {label}
      </ToolGroupTrigger>
      <GroupedToolsContext.Provider value={true}>
        <ToolGroupContent>{children}</ToolGroupContent>
      </GroupedToolsContext.Provider>
    </ToolGroupRoot>
  );
}

export function MessageToolGroup({
  indices,
  running,
  children,
}: {
  indices: readonly number[];
  running: boolean;
  children: ReactNode;
}) {
  const parts = useAuiState((state) => state.message.parts);
  const calls = indices.flatMap((index) => {
    const part = parts[index];
    return part?.type === "tool-call" ? [part] : [];
  });
  return (
    <ToolGroup calls={calls} running={running}>
      {children}
    </ToolGroup>
  );
}
