export interface LocalMcpTool {
  name: string;
  description?: string;
  inputSchema: Record<string, unknown>;
  isEnabled?: () => boolean;
  /** @deprecated Reserved for protocol adapters. */
  serverName?: string;
  /** @deprecated Reserved for protocol adapters. */
  remoteName?: string;
  /** @deprecated Reserved for host-only protocol adapters. */
  hostOnly?: boolean;
  kind?: "function";
  dispose?: () => Promise<void> | void;
  call(arguments_: Record<string, unknown>, context?: LocalToolCallContext): Promise<unknown>;
}

export interface LocalTextTool {
  kind: "text";
  name: string;
  description?: string;
  isEnabled?: () => boolean;
  dispose?: () => Promise<void> | void;
  format?: {
    type: "grammar";
    syntax: "lark" | "regex";
    definition: string;
  };
  call(input: string, context?: LocalToolCallContext): Promise<unknown>;
}

export type LocalTool = LocalMcpTool | LocalTextTool;

export interface LocalToolProgress {
  content: string;
  structuredContent?: unknown;
}

export interface LocalToolCallContext {
  invocationId?: string;
  onProgress?: (progress: LocalToolProgress) => void;
}

export interface LocalToolResult {
  content: string;
  structuredContent?: unknown;
  isError?: boolean;
}

const LOCAL_TOOL_RESULT = Symbol("swarmx.local-tool-result");

/** Separates model-facing tool text from client-facing structured output. */
export function localToolResult(
  content: string,
  structuredContent?: unknown,
  options: { isError?: boolean } = {},
): LocalToolResult {
  return {
    [LOCAL_TOOL_RESULT]: true,
    content,
    ...(structuredContent === undefined ? {} : { structuredContent }),
    ...(options.isError === undefined ? {} : { isError: options.isError }),
  } as LocalToolResult;
}

export function isLocalToolResult(result: unknown): result is LocalToolResult {
  return (
    result !== null &&
    typeof result === "object" &&
    !Array.isArray(result) &&
    (result as Record<PropertyKey, unknown>)[LOCAL_TOOL_RESULT] === true &&
    typeof (result as { content?: unknown }).content === "string"
  );
}
