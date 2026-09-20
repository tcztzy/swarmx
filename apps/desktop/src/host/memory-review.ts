import { loadAgent } from "../agent.js";
import type { AgentOptions, EventAttributes } from "../agents/types.js";
import { policyPermissions } from "../permissions.js";
import { DEFAULT_POLICY } from "../settings.js";

export async function reviewMemory(
  options: AgentOptions,
  prompt: string,
  signal: AbortSignal,
  harness: "codex" | "claude",
  reportIdentity?: (attributes: EventAttributes) => void,
) {
  const cancelled = AbortSignal.any([signal, AbortSignal.timeout(120_000)]);
  cancelled.throwIfAborted();
  const policy = options.executionPolicy?.() ?? DEFAULT_POLICY;
  const models = policyPermissions(policy).harnesses[harness];
  if (models === undefined || models?.length === 0)
    throw new Error(`Memory review harness "${harness}" is not permitted.`);
  const reviewOptions: AgentOptions = {
    ...options,
    reviewOnly: true,
    executionPolicy: () => ({ ...policy, tools: [], delegation: false }),
  };
  const agent = await loadAgent(harness, reviewOptions);
  let sessionId: string | undefined;
  let interruption: Promise<PromiseSettledResult<unknown>[]> | undefined;
  const stop = () => {
    if (sessionId) interruption = Promise.allSettled([agent.interrupt(sessionId)]);
  };
  cancelled.addEventListener("abort", stop, { once: true });
  try {
    cancelled.throwIfAborted();
    sessionId = await agent.create();
    cancelled.throwIfAborted();
    const output: string[] = [];
    reportIdentity?.({ "gen_ai.request.model": models?.[0] ?? null });
    const result = await agent.start(
      sessionId,
      prompt,
      {
        text: (_id, text, role = "assistant") => {
          if (role === "assistant") output.push(text);
        },
        tool: () => {
          throw new Error("Memory review attempted a tool call.");
        },
        raw(_event, attributes) {
          if (attributes) reportIdentity?.(attributes);
        },
        interact: async () => {
          throw new Error("Memory review cannot request permissions.");
        },
      },
      models === null ? undefined : { model: models[0] },
    );
    const [stopped] = (await interruption) ?? [];
    if (stopped?.status === "rejected") throw stopped.reason;
    cancelled.throwIfAborted();
    if (result.stopReason !== "end_turn")
      throw new Error(`Memory review did not succeed: ${result.stopReason}`);
    return output.join("");
  } finally {
    cancelled.removeEventListener("abort", stop);
    await agent.dispose();
  }
}
