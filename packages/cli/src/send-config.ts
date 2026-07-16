import { getHarness, harnessModelRuntimeEnv, harnessModelRuntimeModel } from "@swarmx/core";
import type { AgentConfig, SwarmConfig } from "@swarmx/core";

export interface SendSwarmConfigOptions {
  harnessId: string;
  model?: string;
  effort?: string;
  env?: NodeJS.ProcessEnv;
}

export function createSendSwarmConfig(options: SendSwarmConfigOptions): SwarmConfig {
  const harness = getHarness(options.harnessId);
  if (!harness) throw new Error(`Unknown harness: ${options.harnessId}`);
  const requestedModel = options.model?.trim();
  if (harness.modelControl === "unsupported") {
    throw new Error(
      `Harness "${options.harnessId}" does not support request-scoped model selection.`,
    );
  }
  const runtimeModel = requestedModel
    ? harnessModelRuntimeModel(options.harnessId, {
        modelId: requestedModel,
        runtimeModel: requestedModel,
      })
    : undefined;
  const runtimeEnv = requestedModel
    ? harnessModelRuntimeEnv(options.harnessId, {
        modelId: requestedModel,
        runtimeModel,
        effort: options.effort,
        env: options.env ?? process.env,
      })
    : undefined;
  const agent: AgentConfig = {
    name: "agent",
    instructions: "You are a helpful assistant.",
    backend: harness.backend,
    ...(runtimeModel ? { model: runtimeModel } : {}),
    ...(options.effort ? { parameters: { reasoning: { effort: options.effort } } } : {}),
    ...(runtimeEnv && Object.keys(runtimeEnv).length > 0
      ? { process: { clearEnv: false, env: runtimeEnv } }
      : {}),
  };
  return {
    name: "default",
    root: "agent",
    nodes: { agent: { kind: "agent", agent } },
    edges: [],
  };
}
