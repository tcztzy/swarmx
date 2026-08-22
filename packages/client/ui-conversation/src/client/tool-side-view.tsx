import type { ConversationSnapshot, ToolCallBlock } from "@deepseek-ai/dsh-client-runtime/client";
import type { PropsRenderSlots, PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SideViewContentOwnerProps, SideViewEntry } from "./side-view.js";
import css from "./side-view.module.css";

interface ToolEntryPayload {
  readonly callId: string;
}

interface ToolDetailsActionInjected {
  readonly openTool: (block: ToolCallBlock) => void;
}

interface ToolDetailsActionProps extends ToolDetailsActionInjected {
  readonly matched: number;
  readonly useSession: PropsRuntime<"conversation.chat.turnTail">["useSession"];
}

type ToolSideViewContentProps = PropsRuntime<"side-view.content"> &
  PropsRenderSlots<"side-view.tool.actions"> &
  SideViewContentOwnerProps;

function rootFromNode(node: unknown): ToolCallBlock | undefined {
  if (typeof node !== "object" || node === null) return undefined;
  if (!("kind" in node) || node.kind !== "tool-call" || !("data" in node)) return undefined;
  const data = node.data;
  if (typeof data !== "object" || data === null || !("root" in data)) return undefined;
  const root = data.root;
  if (typeof root !== "object" || root === null || !("callId" in root)) return undefined;
  return typeof root.callId === "string" ? (root as ToolCallBlock) : undefined;
}

function findNested(block: ToolCallBlock, callId: string): ToolCallBlock | undefined {
  if (block.callId === callId) return block;
  for (const child of block.subCalls) {
    const found = findNested(child, callId);
    if (found !== undefined) return found;
  }
  return undefined;
}

/** Resolve one call from public Chat nodes/running calls without private ui-conversation state. */
export function findToolCall(
  snapshot: ConversationSnapshot,
  callId: string,
): ToolCallBlock | undefined {
  for (const node of snapshot.chat.nodes.values()) {
    const root = rootFromNode(node);
    if (root === undefined) continue;
    const found = findNested(root, callId);
    if (found !== undefined) return found;
  }
  for (const root of snapshot.runningCalls) {
    const found = findNested(root, callId);
    if (found !== undefined) return found;
  }
  return undefined;
}

/** Last Tool root placed in one public Chat turn. */
export function latestToolCallInTurn(
  snapshot: ConversationSnapshot,
  turn: number,
): ToolCallBlock | undefined {
  const keys = snapshot.chat.locations.getTurn(turn);
  for (let index = keys.length - 1; index >= 0; index -= 1) {
    const key = keys[index];
    if (key === undefined) continue;
    const root = rootFromNode(snapshot.chat.nodes.get(key));
    if (root !== undefined) return root;
  }
  return undefined;
}

function toolPayload(payload: SideViewEntry["payload"]): ToolEntryPayload | null {
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) return null;
  const callId = (payload as { readonly [key: string]: unknown }).callId;
  return typeof callId === "string" ? { callId } : null;
}

function toolName(block: ToolCallBlock): string {
  return "kind" in block ? (block.call?.name ?? block.callId) : block.name;
}

function toolArgs(block: ToolCallBlock): string | null {
  return "kind" in block ? (block.call?.argsRaw ?? null) : block.argsRaw;
}

function rawToolResult(block: ToolCallBlock): string {
  if (!("kind" in block)) return "Tool is still running.";
  const parts = block.content.map((item) =>
    item.type === "text" ? item.text : JSON.stringify(item, null, 2),
  );
  if (parts.length === 0 && block.error !== undefined) {
    parts.push(`${block.error.name}: ${block.error.code}`);
  }
  return parts.join("\n");
}

export function toolSideViewEntry(block: ToolCallBlock): SideViewEntry {
  return {
    id: `tool:${block.callId}`,
    kind: "tool",
    title: toolName(block),
    mode: "inspect",
    payload: { callId: block.callId },
  };
}

/** Keyed Tool content projected only from the public conversation snapshot. */
export function ToolSideViewContent({ entry, useSession, renderSlot }: ToolSideViewContentProps) {
  const payload = toolPayload(entry.payload);
  const block = useSession((snapshot) =>
    payload === null ? undefined : findToolCall(snapshot, payload.callId),
  );
  if (payload === null) return <p>Tool locator is invalid.</p>;
  if (block === undefined) return <p>Tool call is outside the loaded Session window.</p>;
  const args = toolArgs(block);
  return (
    <article className={css.toolContent}>
      <h2>{toolName(block)}</h2>
      {renderSlot("side-view.tool.actions", { block })}
      {args !== null && (
        <section>
          <h3>Input</h3>
          <pre>{args}</pre>
        </section>
      )}
      <section>
        <h3>Output</h3>
        <pre>{rawToolResult(block)}</pre>
      </section>
    </article>
  );
}

/** One contextual entry below a completed turn's latest Tool call. */
export function ToolDetailsAction({ matched, useSession, openTool }: ToolDetailsActionProps) {
  const block = useSession((snapshot) => latestToolCallInTurn(snapshot, matched));
  if (block === undefined) return null;
  return (
    <button type="button" className={css.toolAction} onClick={() => openTool(block)}>
      Inspect {toolName(block)}
    </button>
  );
}
