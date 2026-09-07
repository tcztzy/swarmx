import { createAcpHarness } from "../agents/acp-harness.js";
import type { AgentOptions } from "../agents/types.js";
import { projectPermissions } from "../permissions.js";
import { DEFAULT_POLICY } from "../settings.js";
import { acpAgent } from "./acp.js";
import { acpClient } from "./acp-client.js";

export async function reviewMemory(
  options: AgentOptions,
  prompt: string,
  signal: AbortSignal,
  harness: "codex" | "claude",
) {
  const cancelled = AbortSignal.any([signal, AbortSignal.timeout(120_000)]);
  cancelled.throwIfAborted();
  const policy = options.executionPolicy?.() ?? DEFAULT_POLICY;
  const models = projectPermissions(policy).harnesses[harness];
  if (models === undefined || models?.length === 0)
    throw new Error(`Memory review harness "${harness}" is not permitted.`);
  const reviewOptions: AgentOptions = {
    ...options,
    reviewOnly: true,
    executionPolicy: () => ({ ...policy, tools: [], delegation: false }),
  };
  const native = await createAcpHarness(harness, reviewOptions);
  const permissions = projectPermissions(reviewOptions.executionPolicy?.() ?? DEFAULT_POLICY);
  let nativeInterruption: Promise<void> | undefined;
  const agent = acpClient(
    native.name,
    native.capabilities,
    options.cwd,
    (client) =>
      client.connect(
        acpAgent(
          {
            ...native,
            permissions: async () => permissions,
            models: async (session) => {
              const catalog = await native.models(session);
              const { mode: _mode, ...current } = catalog.current;
              return {
                current,
                models:
                  models === null
                    ? catalog.models
                    : catalog.models.filter((model) => models.includes(model.id)),
              };
            },
            start: (session, text, observer, selection) =>
              native.start(
                session,
                text,
                {
                  ...observer,
                  tool() {
                    throw new Error("Memory review attempted a tool call.");
                  },
                  interact: async () => {
                    throw new Error("Memory review cannot request permissions.");
                  },
                },
                selection,
              ),
            interrupt: (session) => {
              nativeInterruption = native.interrupt(session);
              return nativeInterruption;
            },
          },
          options.cwd,
        ),
      ),
    () => permissions,
  );
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
        raw() {},
        interact: async () => {
          throw new Error("Memory review cannot request permissions.");
        },
      },
      models === null ? undefined : { model: models[0] },
    );
    const [stopped] = (await interruption) ?? [];
    if (stopped?.status === "rejected") throw stopped.reason;
    await nativeInterruption;
    cancelled.throwIfAborted();
    if (result.stopReason !== "end_turn")
      throw new Error(`Memory review did not succeed: ${result.stopReason}`);
    return output.join("");
  } finally {
    cancelled.removeEventListener("abort", stop);
    await agent.dispose();
    await native.dispose();
  }
}
