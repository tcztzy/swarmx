import { readFileSync } from "node:fs";
import { type EvalRunResult, EvalRunResultSchema, Swarm, type SwarmConfig } from "@swarmx/core";

export interface EvalRunOptions {
  config?: string;
  inputJson?: string;
  inputFile?: string;
  pretty?: boolean;
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
  const swarm = new Swarm(loadSwarmConfig(options.config));
  return swarm.executeForEval(buildEvalArguments(message, options));
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
