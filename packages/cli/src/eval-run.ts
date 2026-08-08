import { readFileSync } from "node:fs";
import {
  type EvalRunResult,
  EvalRunResultSchema,
  loadSkillFragmentContent,
  parseSkillInstructionDelivery,
  type SkillInstructionDelivery,
  Swarm,
  type SwarmConfig,
  type SwarmRuntimeOptions,
} from "@swarmx/core";

export interface EvalRunOptions {
  config?: string;
  inputJson?: string;
  inputFile?: string;
  pretty?: boolean;
  skillDelivery?: string;
  skillContentPath?: string;
  skillDeliveryAgent?: string;
  resolveSkill?: string[];
  evolutionRoot?: string;
}

export function buildEvalArguments(
  message: string | undefined,
  options: Pick<EvalRunOptions, "inputJson" | "inputFile">,
): Record<string, unknown> {
  if (options.inputJson && options.inputFile) {
    throw new Error("Use either --input-json or --input-file, not both");
  }

  if (options.inputJson) {
    return parseEvalArguments(options.inputJson, "--input-json");
  }

  if (options.inputFile) {
    return parseEvalArguments(readFileSync(options.inputFile, "utf-8"), options.inputFile);
  }

  if (message !== undefined) {
    return {
      messages: [{ role: "user", content: message }],
    };
  }

  throw new Error("Provide a message, --input-json, or --input-file");
}

export async function runEval(
  message: string | undefined,
  options: EvalRunOptions,
): Promise<EvalRunResult> {
  const config = loadSwarmConfig(options.config);
  const swarmOptions = await evalSwarmOptions(options, config);
  const swarm = new Swarm(config, swarmOptions);
  return swarm.executeForEval(buildEvalArguments(message, options));
}

export async function evalSwarmOptions(
  options: Pick<
    EvalRunOptions,
    "skillDelivery" | "skillContentPath" | "skillDeliveryAgent" | "resolveSkill" | "evolutionRoot"
  >,
  config?: SwarmConfig,
): Promise<SwarmRuntimeOptions> {
  const explicit = await explicitSkillDelivery(options, config);
  if (options.resolveSkill && options.resolveSkill.length > 0) {
    if (explicit) {
      throw new Error("Use either --skill-delivery or --resolve-skill, not both");
    }
    const { nativeAgentTargetId, resolveActiveSkillDeliveriesForAgent } = await import(
      "./evolution-command.js"
    );
    const bindings = options.resolveSkill.map(parseSkillBinding);
    const perAgent: Record<string, SkillInstructionDelivery[]> = {};
    for (const node of Object.values(config?.nodes ?? {})) {
      if (node.kind !== "agent" || !node.agent) continue;
      const backend = node.agent.backend?.type ?? "swarmx";
      if (backend !== "swarmx" && backend !== "echo") continue;
      const deliveries = await resolveActiveSkillDeliveriesForAgent({
        bindings,
        agentName: node.agent.name,
        targetAgentId: nativeAgentTargetId(node.agent),
        evolutionRoot: options.evolutionRoot,
      });
      const agentDeliveries = deliveries[node.agent.name];
      if (agentDeliveries?.length) perAgent[node.agent.name] = agentDeliveries;
    }
    return Object.keys(perAgent).length > 0
      ? { agent: { skillInstructionsByAgent: perAgent } }
      : {};
  }
  return explicit ?? {};
}

export function parseSkillBinding(value: string): { skillId: string; variantId: string } {
  const separator = value.indexOf(":");
  if (separator <= 0 || separator === value.length - 1) {
    throw new Error('Skill binding must be "<skillId>:<variantId>".');
  }
  return { skillId: value.slice(0, separator), variantId: value.slice(separator + 1) };
}

async function explicitSkillDelivery(
  options: Pick<EvalRunOptions, "skillDelivery" | "skillContentPath" | "skillDeliveryAgent">,
  config?: SwarmConfig,
): Promise<SwarmRuntimeOptions | undefined> {
  if (!options.skillDelivery) {
    if (options.skillContentPath) {
      throw new Error("--skill-content-path requires --skill-delivery");
    }
    if (options.skillDeliveryAgent) {
      throw new Error("--skill-delivery-agent requires --skill-delivery");
    }
    return undefined;
  }
  if (!options.skillContentPath) {
    throw new Error("--skill-delivery requires --skill-content-path");
  }
  const agentNodes = Object.values(config?.nodes ?? []).filter(
    (node) => node.kind === "agent" && Boolean(node.agent),
  ) as Array<{
    kind: "agent";
    agent: { name: string; model?: string; backend?: { type?: string } };
  }>;
  if (options.skillDeliveryAgent) {
    const target = agentNodes.find((node) => node.agent.name === options.skillDeliveryAgent);
    if (!target) {
      throw new Error(
        `--skill-delivery-agent names no agent "${options.skillDeliveryAgent}" in the config.`,
      );
    }
  } else if (agentNodes.length > 1) {
    throw new Error(
      "Skill delivery with multiple Agent nodes requires --skill-delivery-agent <name>; a candidate must not be injected into every agent.",
    );
  }
  const deliveryInput = JSON.parse(options.skillDelivery) as {
    skillId?: string;
    variantId?: string;
    revisionId?: string;
    contentDigest?: string;
    mode?: string;
  };
  if (deliveryInput.mode !== "prompt_fragment") {
    throw new Error(
      `Skill delivery mode "${deliveryInput.mode ?? "<missing>"}" cannot be injected.`,
    );
  }
  const content = await loadSkillFragmentContent(options.skillContentPath, {
    expectedDigest: deliveryInput.contentDigest ?? "",
  });
  const delivery = parseSkillInstructionDelivery({
    skillId: deliveryInput.skillId ?? "",
    variantId: deliveryInput.variantId ?? "",
    revisionId: deliveryInput.revisionId ?? "",
    contentDigest: deliveryInput.contentDigest ?? "",
    mode: "prompt_fragment",
    content,
  });
  if (options.skillDeliveryAgent) {
    return {
      agent: {
        skillInstructionsByAgent: { [options.skillDeliveryAgent]: [delivery] },
      },
    };
  }
  return { agent: { skillInstructions: [delivery] } };
}

export function formatEvalResult(result: EvalRunResult, pretty = false): string {
  return `${JSON.stringify(EvalRunResultSchema.parse(result), null, pretty ? 2 : 0)}\n`;
}

export function errorEvalResult(error: unknown): EvalRunResult {
  return EvalRunResultSchema.parse({
    output: "",
    messages: [],
    trace: [],
    error: errorMessage(error),
    metrics: {
      steps: 0,
      messages: 0,
      toolCalls: 0,
      toolResults: 0,
      contextTokens: 0,
    },
  });
}

function loadSwarmConfig(configPath?: string): SwarmConfig {
  if (!configPath) {
    return {
      name: "default",
      root: "agent",
      nodes: {
        agent: {
          kind: "agent",
          agent: {
            name: "agent",
            instructions: "You are a helpful assistant.",
          },
        },
      },
      edges: [],
    };
  }

  return JSON.parse(readFileSync(configPath, "utf-8")) as SwarmConfig;
}

function parseEvalArguments(source: string, label: string): Record<string, unknown> {
  const parsed = JSON.parse(source);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed as Record<string, unknown>;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
