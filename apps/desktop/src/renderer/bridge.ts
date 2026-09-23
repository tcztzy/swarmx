import { z } from "zod";
import type { AgUiEventMessage } from "../bridge-contract.js";
import type { WorkCommandSchema, WorkReadRequestSchema } from "../work.js";

export interface SwarmxBridge {
  bootstrap(): Promise<unknown>;
  work: {
    read(input: z.input<typeof WorkReadRequestSchema>): Promise<unknown>;
    command(input: z.input<typeof WorkCommandSchema>): Promise<unknown>;
  };
  tool(input: { requestId: string; name: string; args: unknown }): Promise<unknown>;
  cancelTool(input: { requestId: string }): Promise<unknown>;
  settings: {
    read(): Promise<unknown>;
    update(policy: unknown): Promise<unknown>;
  };
  language: { write(input: { language: "zh" | "en" }): Promise<unknown> };
  environment: {
    read(): Promise<unknown>;
    act(input: { action: "setup" | "inspect" | "cancel" }): Promise<unknown>;
  };
  sessions: {
    list(input: { agent: string }): Promise<unknown>;
    create(input: { agent: string }): Promise<unknown>;
    history(input: { agent: string; sessionId: string }): Promise<unknown>;
  };
  models: { read(input: { agent: string; session?: string }): Promise<unknown> };
  logs: {
    evidence(input: { sources: string[] }): Promise<unknown>;
    read(input: {
      after?: number;
      limit?: number;
      session?: string;
      run?: string;
      descendants?: "true" | "false";
    }): Promise<unknown>;
  };
  runs: {
    control(input: {
      runId: string;
      command: { action: "steer"; text: string } | { action: "cancel" };
    }): Promise<unknown>;
  };
  science: {
    workspace(): Promise<unknown>;
    researchObject(input: { projectId: string }): Promise<unknown>;
    notebookExecutions(input: { projectId: string; includeArtifactId?: string }): Promise<unknown>;
    artifactPreview(input: { id: string }): Promise<unknown>;
    artifactContent(input: { id: string }): Promise<unknown>;
    import(input: {
      requestId: string;
      projectId: string;
      name: string;
      dataBase64: string;
    }): Promise<unknown>;
  };
  agui: {
    start(input: { agent: string; input: unknown }): Promise<unknown>;
    cancel(input: { agent: string; threadId: string }): Promise<unknown>;
    subscribe(listener: (message: AgUiEventMessage) => void): () => void;
  };
}

declare global {
  interface Window {
    readonly swarmx?: SwarmxBridge;
  }
}

export function bridge(): SwarmxBridge {
  const value = window.swarmx;
  if (!value) throw new Error("SwarmX requires the desktop application.");
  return value;
}

const Envelope = z.object({ action: z.string(), data: z.unknown() });

/** Calls one product tool and unwraps its `{action, data}` envelope. */
export async function tool<T>(
  name: string,
  args: Record<string, unknown>,
  schema: z.ZodType<T>,
  signal?: AbortSignal,
): Promise<T> {
  const requestId = crypto.randomUUID();
  const cancel = () => void bridge().cancelTool({ requestId });
  signal?.addEventListener("abort", cancel, { once: true });
  try {
    const result = await bridge().tool({ requestId, name, args });
    return schema.parse(Envelope.parse(result).data);
  } finally {
    signal?.removeEventListener("abort", cancel);
  }
}

/** Calls one science tool, refreshes dependent views and unwraps its `{data}` envelope. */
export async function scienceTool<T>(
  name: string,
  action: string,
  request: unknown,
  schema: z.ZodType<T>,
  signal?: AbortSignal,
): Promise<T> {
  const result = await tool(name, { action, request }, z.object({ data: schema }), signal);
  window.dispatchEvent(new Event("swarmx:science-changed"));
  return result.data;
}

export function download(name: string, content: string | Uint8Array, mime = "application/json") {
  const payload: BlobPart = typeof content === "string" ? content : new Uint8Array(content);
  const url = URL.createObjectURL(new Blob([payload], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
