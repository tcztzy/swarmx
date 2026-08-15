import {
  closeSync,
  fchmodSync,
  fstatSync,
  fsyncSync,
  lstatSync,
  openSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import {
  type AblationProfile,
  AblationProfileSchema,
  type ContextEvaluationExecutor,
  type ContextEvaluationReport,
  ContextEvaluationReportSchema,
  type ContextEvaluationResult,
  type ContextEvaluationSuite,
  ContextEvaluationSuiteSchema,
  classifyContextEvaluationError,
  createBuiltinAgentServiceRegistry,
  createSessionContextEngine,
  type EvalRunResult,
  EvalRunResultSchema,
  formatContextEvaluationJsonl,
  type GlobalMemorySnapshot,
  GlobalMemorySnapshotSchema,
  loadSkillFragmentContent,
  parseSkillInstructionDelivery,
  runContextEvaluation,
  type SkillInstructionDelivery,
  Swarm,
  type SwarmConfig,
  type SwarmRuntimeOptions,
} from "@swarmx/core";

export interface EvalRunOptions {
  config?: string;
  contextSuite?: string;
  contextJsonl?: string;
  inputJson?: string;
  inputFile?: string;
  pretty?: boolean;
  skillDelivery?: string;
  skillContentPath?: string;
  skillDeliveryAgent?: string;
  resolveSkill?: string[];
  evolutionRoot?: string;
  ablationProfile?: string;
  memorySnapshot?: string;
}

export interface ContextEvalSuiteDependencies {
  executor?: ContextEvaluationExecutor;
  now?: () => Date;
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

export function isContextEvalSuiteRequest(options: EvalRunOptions): boolean {
  return Boolean(options.contextSuite || options.contextJsonl);
}

export function loadContextEvaluationSuite(path: string): ContextEvaluationSuite {
  return ContextEvaluationSuiteSchema.parse(JSON.parse(readFileSync(path, "utf8")));
}

export function loadAblationProfile(path: string): AblationProfile {
  return AblationProfileSchema.parse(JSON.parse(readFileSync(path, "utf8")));
}

export function loadGlobalMemorySnapshot(path: string): GlobalMemorySnapshot {
  return GlobalMemorySnapshotSchema.parse(JSON.parse(readFileSync(path, "utf8")));
}

export async function runContextEvalSuite(
  message: string | undefined,
  options: EvalRunOptions,
  dependencies: ContextEvalSuiteDependencies = {},
): Promise<ContextEvaluationResult> {
  validateContextEvalSuiteOptions(message, options);
  const suite = loadContextEvaluationSuite(options.contextSuite as string);
  const reservation = options.contextJsonl ? reserveContextJsonl(options.contextJsonl) : undefined;
  try {
    const result = await runContextEvaluation({
      suite,
      ...(dependencies.executor ? { executor: dependencies.executor } : {}),
      ...(dependencies.now ? { now: dependencies.now } : {}),
    });
    if (reservation) commitContextJsonl(reservation, result);
    return result;
  } catch (error) {
    if (reservation) abandonContextJsonl(reservation);
    throw error;
  }
}

export function formatContextEvaluationReport(
  report: ContextEvaluationReport,
  pretty = false,
): string {
  return `${JSON.stringify(ContextEvaluationReportSchema.parse(report), null, pretty ? 2 : 0)}\n`;
}

export function formatContextEvaluationError(error: unknown, pretty = false): string {
  return `${JSON.stringify(
    {
      schemaVersion: 2,
      recordType: "context_evaluation_error",
      failure: classifyContextEvaluationError(error),
    },
    null,
    pretty ? 2 : 0,
  )}\n`;
}

export async function evalSwarmOptions(
  options: Pick<
    EvalRunOptions,
    | "skillDelivery"
    | "skillContentPath"
    | "skillDeliveryAgent"
    | "resolveSkill"
    | "evolutionRoot"
    | "ablationProfile"
    | "memorySnapshot"
  >,
  config?: SwarmConfig,
): Promise<SwarmRuntimeOptions> {
  if (options.memorySnapshot && !options.ablationProfile) {
    throw new Error("--memory-snapshot requires --ablation-profile");
  }
  const explicit = await explicitSkillDelivery(options, config);
  const ablation = options.ablationProfile ? ablationRuntimeOptions(options, config) : {};
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
    return {
      agent: {
        ...ablation,
        ...(Object.keys(perAgent).length > 0 ? { skillInstructionsByAgent: perAgent } : {}),
      },
    };
  }
  return {
    agent: {
      ...ablation,
      ...(explicit?.agent ?? {}),
    },
  };
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

interface ContextJsonlReservation {
  path: string;
  fileDescriptor: number;
  device: bigint;
  inode: bigint;
  closed: boolean;
}

function validateContextEvalSuiteOptions(
  message: string | undefined,
  options: EvalRunOptions,
): void {
  if (!options.contextSuite) {
    throw new Error("--context-jsonl requires --context-suite <path>.");
  }
  const conflicts = [
    ...(message === undefined ? [] : ["positional message"]),
    ...(options.config ? ["--config"] : []),
    ...(options.inputJson ? ["--input-json"] : []),
    ...(options.inputFile ? ["--input-file"] : []),
    ...(options.skillDelivery ? ["--skill-delivery"] : []),
    ...(options.skillContentPath ? ["--skill-content-path"] : []),
    ...(options.skillDeliveryAgent ? ["--skill-delivery-agent"] : []),
    ...(options.resolveSkill?.length ? ["--resolve-skill"] : []),
    ...(options.evolutionRoot ? ["--evolution-root"] : []),
    ...(options.ablationProfile ? ["--ablation-profile"] : []),
    ...(options.memorySnapshot ? ["--memory-snapshot"] : []),
  ];
  if (conflicts.length > 0) {
    throw new Error(`--context-suite cannot be combined with ${conflicts.join(", ")}.`);
  }
}

function ablationRuntimeOptions(
  options: Pick<EvalRunOptions, "ablationProfile" | "memorySnapshot">,
  config?: SwarmConfig,
): NonNullable<SwarmRuntimeOptions["agent"]> {
  if (!options.ablationProfile) return {};
  return {
    serviceRegistry: createBuiltinAgentServiceRegistry(),
    ablationProfile: loadAblationProfile(options.ablationProfile),
    contextEngine: createSessionContextEngine({
      sessionId: `ablation_eval_${config?.name ?? "default"}`,
    }),
    ...(options.memorySnapshot
      ? { globalMemory: loadGlobalMemorySnapshot(options.memorySnapshot) }
      : {}),
  };
}

function reserveContextJsonl(path: string): ContextJsonlReservation {
  let fileDescriptor: number;
  try {
    fileDescriptor = openSync(path, "wx", 0o600);
  } catch (error) {
    if (errorCode(error) === "EEXIST") {
      throw new Error(`Context evaluation JSONL already exists; refusing to overwrite: ${path}`);
    }
    throw error;
  }
  let stats: ReturnType<typeof fstatSync>;
  try {
    stats = fstatSync(fileDescriptor, { bigint: true });
  } catch (error) {
    closeSync(fileDescriptor);
    throw error;
  }
  const reservation: ContextJsonlReservation = {
    path,
    fileDescriptor,
    device: stats.dev as bigint,
    inode: stats.ino as bigint,
    closed: false,
  };
  try {
    fchmodSync(fileDescriptor, 0o600);
    return reservation;
  } catch (error) {
    abandonContextJsonl(reservation);
    throw error;
  }
}

function commitContextJsonl(
  reservation: ContextJsonlReservation,
  result: ContextEvaluationResult,
): void {
  writeFileSync(reservation.fileDescriptor, formatContextEvaluationJsonl(result.records), "utf8");
  fsyncSync(reservation.fileDescriptor);
  closeSync(reservation.fileDescriptor);
  reservation.closed = true;
}

function abandonContextJsonl(reservation: ContextJsonlReservation): void {
  if (!reservation.closed) {
    try {
      closeSync(reservation.fileDescriptor);
    } catch {
      // Preserve the original evaluation error.
    }
    reservation.closed = true;
  }
  try {
    const stats = lstatSync(reservation.path, { bigint: true });
    if (stats.dev === reservation.device && stats.ino === reservation.inode) {
      unlinkSync(reservation.path);
    }
  } catch {
    // Preserve the original evaluation error or an already-cleaned reservation.
  }
}

function errorCode(error: unknown): string | undefined {
  return error && typeof error === "object" && "code" in error
    ? String((error as { code?: unknown }).code)
    : undefined;
}
