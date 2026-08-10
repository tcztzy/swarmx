import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { mkdtemp, readFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AppAttachedTaskControlService,
  AuditStore,
  assertEvalSafeSwarmConfig,
  canonicalSkillOptimizerConfig,
  createSkillEvolutionCapabilityGateway,
  SkillDeliveryError,
  type SkillEvaluationGate,
  type SkillEvaluationManifest,
  type SkillEvolutionModelHandler,
  SkillEvolutionService,
  SkillEvolutionStore,
  SkillExternalEvaluationEvidenceSchema,
  type SkillInstructionDelivery,
  type SkillOptimizationRequest,
  SkillPromotionReceiptSchema,
  Swarm,
  type SwarmConfig,
  TaskRuntimeStore,
  type TaskWorkerLaunchSpec,
} from "@swarmx/core";

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const DEFAULT_WORKER_PATH = path.join(REPO_ROOT, "src", "swarmx", "worker.py");
const DEFAULT_PROJECT_PATH = path.join(REPO_ROOT, "pyproject.toml");
const DEFAULT_LOCK_PATH = path.join(REPO_ROOT, "uv.lock");
const DEFAULT_TASK_ROOT = path.join(os.homedir(), ".swarmx", "task-runtime");
const DEFAULT_EVOLUTION_ROOT = path.join(os.homedir(), ".swarmx", "skill-evolution");
const RSI_SOURCES_ROOT = path.join(REPO_ROOT, "src", "swarmx", "rsi");

export interface EvolutionCliContext {
  audit: AuditStore;
  service: SkillEvolutionService;
  controlService?: AppAttachedTaskControlService;
  ownerId: string;
}

export interface EvolutionRequestFile {
  schemaVersion: 1;
  skillId: string;
  variantId: string;
  parentRevisionId: string;
  baselineContentPath: string;
  trainDatasetPath: string;
  devDatasetPath: string;
  targetAgentId: string;
  targetModelFingerprint: string;
  optimizer: {
    optimizerId: string;
    optimizerVersion: string;
    environmentDigest: string;
    seed: number;
  };
  budget: {
    maxWallTimeMs?: number;
    maxModelCalls?: number;
    maxTokens?: number;
    maxArtifactBytes?: number;
  };
  proposer?: "none" | "gateway" | "deterministic";
  requestedBy?: string;
}

export function createEvolutionCliContext(
  options: {
    taskRoot?: string;
    evolutionRoot?: string;
    audit?: AuditStore;
    ownerId?: string;
    modelHandler?: SkillEvolutionModelHandler;
  } = {},
): EvolutionCliContext {
  const audit = options.audit ?? new AuditStore();
  let controlService: AppAttachedTaskControlService | undefined;
  if (options.taskRoot) {
    const taskStore = new TaskRuntimeStore({ rootDir: options.taskRoot });
    const gateway = createSkillEvolutionCapabilityGateway({
      taskStore,
      modelHandler: options.modelHandler,
    });
    controlService = new AppAttachedTaskControlService({
      store: taskStore,
      capabilityGateway: gateway,
      ownerId: options.ownerId ?? `cli:${process.pid}`,
    });
    controlService.recoverOnStartup();
  }
  const service = new SkillEvolutionService({
    ledger: new SkillEvolutionStore({ rootDir: options.evolutionRoot ?? DEFAULT_EVOLUTION_ROOT }),
    controlService,
    audit,
  });
  return { audit, service, controlService, ownerId: options.ownerId ?? `cli:${process.pid}` };
}

export function launchDigestForWorker(workerPath: string, pythonLabel: string): string {
  const source = readFileSync(workerPath);
  const workerSha256 = createHash("sha256").update(source).digest("hex");
  const canonical = JSON.stringify({
    schemaVersion: 1,
    workerSha256,
    pythonLabel,
  });
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

export async function computeSkillEvolutionLaunchDigest(options: {
  workerPath?: string;
  pythonPath?: string;
  projectPath?: string;
  lockPath?: string;
  rsiSources?: string[];
  dependencyVersions?: { dspy: string; mcp: string };
}): Promise<string> {
  const workerPath = path.resolve(options.workerPath ?? DEFAULT_WORKER_PATH);
  const pythonPath = options.pythonPath ?? defaultPythonPath();
  const [workerSha256, projectSha256, lockSha256, ...sources] = await Promise.all([
    hashFile(workerPath),
    hashFile(options.projectPath ?? DEFAULT_PROJECT_PATH),
    hashFile(options.lockPath ?? DEFAULT_LOCK_PATH),
    ...(options.rsiSources ?? defaultRsiSources()).map(async (source) => ({
      source,
      sha256: await hashFile(source),
    })),
  ]);
  const pythonVersion = await detectPythonVersion(pythonPath);
  const canonical = JSON.stringify({
    schemaVersion: 5,
    workerSha256,
    projectSha256,
    lockSha256,
    rsiSources: sources.map((entry) => entry.sha256).sort(),
    dependencyVersions: options.dependencyVersions ?? null,
    pythonVersion,
  });
  return `sha256:${createHash("sha256").update(canonical).digest("hex")}`;
}

/**
 * Resolves the interpreter's installed DSPy and MCP versions and verifies they
 * equal the versions pinned by the standard `swarmx` distribution, and that
 * the interpreter's environment satisfies the strict locked sync
 * (`uv sync --locked --check --no-default-groups`), so a
 * launch can never silently run against unverified optimizer dependencies.
 */
export async function resolveLockedEvolutionVersions(options: {
  pythonPath: string;
  projectPath?: string;
}): Promise<{ dspy: string; mcp: string }> {
  const projectPath = options.projectPath ?? DEFAULT_PROJECT_PATH;
  const pyproject = readFileSync(projectPath, "utf8");
  const expected = {
    dspy: /dspy\s*==\s*([0-9]+\.[0-9]+\.[0-9]+)/.exec(pyproject)?.[1],
    mcp: /mcp\s*==\s*([0-9]+\.[0-9]+\.[0-9]+)/.exec(pyproject)?.[1],
  };
  if (!expected.dspy || !expected.mcp) {
    throw new Error("The locked swarmx project must pin dspy==X.Y.Z and mcp==X.Y.Z.");
  }
  const envRoot = environmentRootForInterpreter(options.pythonPath);
  const check = await runCommand(
    "uv",
    [
      "sync",
      "--project",
      path.dirname(projectPath),
      "--locked",
      "--check",
      "--no-default-groups",
      "--python",
      options.pythonPath,
      "--managed-python",
      "--no-python-downloads",
      "--offline",
      "--no-cache",
    ],
    { env: { ...process.env, UV_PROJECT_ENVIRONMENT: envRoot } },
  );
  if (check.code !== 0) {
    throw new Error(
      `The interpreter "${options.pythonPath}" is not the strictly locked SwarmX environment (uv sync --locked --check --no-default-groups failed). ${check.stderr.slice(0, 300)}`,
    );
  }
  const actual = await new Promise<{ dspy: string; mcp: string }>((resolve, reject) => {
    execFile(
      options.pythonPath,
      [
        "-c",
        "import importlib.metadata, json; print(json.dumps({'dspy': importlib.metadata.version('dspy'), 'mcp': importlib.metadata.version('mcp')}))",
      ],
      { timeout: 30_000, env: { PATH: process.env.PATH ?? "" } },
      (error, stdout, stderr) => {
        if (error) {
          reject(
            new Error(
              `The interpreter "${options.pythonPath}" cannot import the locked DSPy/MCP dependencies; sync the SwarmX project (${(stderr || "").slice(0, 200)}).`,
            ),
          );
          return;
        }
        try {
          const parsed = JSON.parse(stdout) as { dspy?: unknown; mcp?: unknown };
          if (typeof parsed.dspy !== "string" || typeof parsed.mcp !== "string") {
            throw new Error("invalid dependency version response");
          }
          resolve({ dspy: parsed.dspy, mcp: parsed.mcp });
        } catch {
          reject(new Error(`The interpreter "${options.pythonPath}" reported invalid versions.`));
        }
      },
    );
  });
  if (actual.dspy !== expected.dspy || actual.mcp !== expected.mcp) {
    throw new Error(
      `The interpreter "${options.pythonPath}" has dspy ${actual.dspy} and mcp ${actual.mcp}; swarmx requires dspy ${expected.dspy} and mcp ${expected.mcp}.`,
    );
  }
  return actual;
}

/**
 * The environment root is the parent of the interpreter's bin directory
 * (`<envRoot>/bin/python`), matching uv-managed environments.
 */
function environmentRootForInterpreter(pythonPath: string): string {
  const resolved = path.resolve(pythonPath);
  const match = /^(.*)[/\\]bin[/\\][^/\\]+$/.exec(resolved);
  if (!match?.[1]) {
    throw new Error(
      `The interpreter "${pythonPath}" is not inside a uv-managed environment (expected <envRoot>/bin/python).`,
    );
  }
  return match[1];
}

function runCommand(
  program: string,
  args: string[],
  options: { env: NodeJS.ProcessEnv },
): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile(program, args, { timeout: 60_000, env: options.env }, (error, stdout, stderr) => {
      if (error && (error as NodeJS.ErrnoException).code === "ENOENT") {
        reject(new Error(`"${program}" is required to verify the locked evolution environment.`));
        return;
      }
      resolve({ code: error ? ((error as { code?: number }).code ?? 1) : 0, stdout, stderr });
    });
  });
}

function defaultRsiSources(): string[] {
  try {
    return readdirSync(RSI_SOURCES_ROOT)
      .filter((entry) => entry.endsWith(".py"))
      .sort()
      .map((entry) => path.join(RSI_SOURCES_ROOT, entry));
  } catch {
    return [];
  }
}

async function hashFile(target: string): Promise<string> {
  const content = await readFile(target);
  return createHash("sha256").update(content).digest("hex");
}

async function detectPythonVersion(pythonPath: string): Promise<string> {
  const result = await new Promise<string>((resolve, reject) => {
    execFile(pythonPath, ["-V"], { timeout: 15_000 }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`Cannot run python interpreter "${pythonPath}": ${error.message}`));
        return;
      }
      const version = `${stdout}\n${stderr}`.match(/\bPython\s+(\d+\.\d+\.\d+)/)?.[1];
      if (!version) {
        reject(new Error(`Python interpreter "${pythonPath}" did not report a version.`));
        return;
      }
      resolve(version);
    });
  });
  return result;
}

export async function runEvolutionEvolve(
  requestFile: string,
  options: {
    taskRoot?: string;
    evolutionRoot?: string;
    workerPath?: string;
    python?: string;
    cwd?: string;
    requestedBy?: string;
    modelCommand?: string;
    signal?: AbortSignal;
  } = {},
): Promise<{ workItemId: string; candidateId: string }> {
  const raw = JSON.parse(readFileSync(requestFile, "utf-8")) as EvolutionRequestFile;
  if (raw.schemaVersion !== 1) throw new Error("Evolution request schemaVersion must be 1.");
  const proposer = raw.proposer ?? "none";
  const modelHandler = options.modelCommand
    ? createModelCommandHandler(options.modelCommand)
    : undefined;
  if (proposer === "gateway" && !modelHandler) {
    throw new Error(
      'proposer "gateway" requires --model-command <cmd> so model calls can cross the capability gateway; credentials never enter the worker.',
    );
  }
  if (proposer !== "gateway" && options.modelCommand) {
    throw new Error('--model-command is only valid with proposer "gateway".');
  }
  const workerPath = path.resolve(options.workerPath ?? DEFAULT_WORKER_PATH);
  const isGepa = raw.optimizer.optimizerId === "dspy.gepa.v1";
  const pythonPath =
    options.python ?? (isGepa ? await discoverEvolutionPythonPath() : defaultPythonPath());
  const dependencyVersions = isGepa
    ? await resolveLockedEvolutionVersions({ pythonPath })
    : undefined;
  const digest = await computeSkillEvolutionLaunchDigest({
    workerPath,
    pythonPath,
    dependencyVersions,
  });
  if (raw.optimizer.environmentDigest !== digest) {
    throw new Error(
      `Optimizer environmentDigest ${raw.optimizer.environmentDigest} does not match the launch digest ${digest} (worker, lockfile, RSI sources, locked DSPy/MCP versions, and Python version).`,
    );
  }
  const baseline = readFileSync(path.resolve(raw.baselineContentPath));
  const trainRecords = parseDatasetRecords(
    readFileSync(path.resolve(raw.trainDatasetPath), "utf8"),
  );
  const devRecords = parseDatasetRecords(readFileSync(path.resolve(raw.devDatasetPath), "utf8"));
  const context = createEvolutionCliContext({
    taskRoot: options.taskRoot ?? DEFAULT_TASK_ROOT,
    evolutionRoot: options.evolutionRoot,
    modelHandler,
  });
  const taskStore = context.controlService?.store;
  if (!taskStore) {
    throw new Error("The evolution CLI could not attach the durable task runtime.");
  }
  const baselineRef = taskStore.putBytes(baseline).ref;
  const trainRef = taskStore.putJson(trainRecords).ref;
  const devRef = taskStore.putJson(devRecords).ref;
  const request: SkillOptimizationRequest = {
    schemaVersion: 1,
    skillId: raw.skillId,
    variantId: raw.variantId,
    parentRevisionId: raw.parentRevisionId,
    parentRevisionDigest: baselineRef,
    baselineContentRef: baselineRef,
    baselineContentDigest: baselineRef,
    targetAgentId: raw.targetAgentId,
    targetModelFingerprint: raw.targetModelFingerprint,
    trainDataset: {
      role: "train",
      contentRef: trainRef,
      contentDigest: trainRef,
      caseCount: trainRecords.length,
      format: "swarmx.eval.jsonl",
    },
    devDataset: {
      role: "dev",
      contentRef: devRef,
      contentDigest: devRef,
      caseCount: devRecords.length,
      format: "swarmx.eval.jsonl",
    },
    optimizer: {
      optimizerId: raw.optimizer.optimizerId,
      optimizerVersion: raw.optimizer.optimizerVersion,
      environmentDigest: raw.optimizer.environmentDigest,
      configDigest: canonicalSkillOptimizerConfig({
        optimizerId: raw.optimizer.optimizerId,
        seed: raw.optimizer.seed,
        proposer,
        budget: raw.budget,
      }),
      seed: raw.optimizer.seed,
    },
    budget: raw.budget,
    proposer,
    requestedBy: options.requestedBy ?? raw.requestedBy ?? "cli",
  };
  const cwd = options.cwd ?? (await temporaryDirectory());
  const launch = evolutionLaunchSpec({ workerPath, python: pythonPath, digest, cwd });
  const { workItem, grant } = context.service.createOptimizationWorkItem({
    request,
    launch,
    requestedBy: request.requestedBy,
  });
  await context.controlService?.runWorkItem(workItem.id, {
    launch,
    grants: [grant],
    signal: options.signal,
  });
  const candidate = context.service.ingestCandidate({ workItemId: workItem.id });
  return { workItemId: workItem.id, candidateId: candidate.candidateId };
}

export async function runEvolutionEvaluate(
  candidateId: string,
  options: {
    holdoutPath?: string;
    evidencePath?: string;
    configPath?: string;
    seed?: number;
    taskRoot?: string;
    evolutionRoot?: string;
    actor?: string;
  },
): Promise<SkillEvaluationManifest> {
  const context = createEvolutionCliContext({
    taskRoot: options.taskRoot ?? DEFAULT_TASK_ROOT,
    evolutionRoot: options.evolutionRoot,
  });
  if (options.evidencePath) {
    if (!options.holdoutPath) {
      throw new Error(
        "evaluate --evidence requires --holdout <jsonl> so the claimed holdout digest can be verified.",
      );
    }
    if (!options.configPath) {
      throw new Error(
        "evaluate --evidence requires -c/--config <path> so the host can verify the evaluator's runtime fingerprint.",
      );
    }
    const holdoutContent = readFileSync(path.resolve(options.holdoutPath), "utf8");
    const configContent = readFileSync(path.resolve(options.configPath));
    const hostConfigFingerprint = `swarmx.inspect.config:${createHash("sha256")
      .update(configContent)
      .digest("hex")}`;
    const evidence = SkillExternalEvaluationEvidenceSchema.parse(
      JSON.parse(readFileSync(path.resolve(options.evidencePath), "utf8")),
    );
    return context.service.recordExternalEvaluation({
      candidateId,
      evaluatorId: evidence.evaluatorId,
      scorerFingerprint: evidence.scorerFingerprint,
      runtimeFingerprint: evidence.runtimeFingerprint,
      seed: evidence.seed,
      holdoutContentDigest: evidence.holdoutContentDigest,
      holdoutCaseCount: evidence.holdoutCaseCount,
      baselineRevisionId: evidence.baselineRevisionId,
      candidateRevisionId: evidence.candidateRevisionId,
      targetAgentId: evidence.targetAgentId,
      targetModelFingerprint: evidence.targetModelFingerprint,
      samples: evidence.samples,
      gate: defaultEvaluationGate(),
      holdoutContent,
      hostConfigFingerprint,
    });
  }
  if (!options.holdoutPath) {
    throw new Error("evaluate requires --holdout <jsonl> or --evidence <json>");
  }
  const holdout = readFileSync(path.resolve(options.holdoutPath), "utf8");
  const config = options.configPath
    ? assertEvalSafeSwarmConfig(JSON.parse(readFileSync(path.resolve(options.configPath), "utf-8")))
    : assertEvalSafeSwarmConfig(defaultEvalSwarmConfig());
  const manifest = await context.service.evaluateCandidate({
    candidateId,
    holdoutContent: holdout,
    createSwarm: (delivery) => new Swarm(config, { agent: { skillInstructions: [delivery] } }),
    evaluatorId: `cli:${options.actor ?? process.getuid?.() ?? "user"}`,
    scorerFingerprint: "swarmx.cli.deterministic.v1",
    runtimeFingerprint: "swarmx.runtime.direct.v1",
    seed: options.seed ?? 0,
    gate: defaultEvaluationGate(),
  });
  return manifest;
}

export function runEvolutionStatus(options: { evolutionRoot?: string } = {}): string {
  const ledger = new SkillEvolutionStore({
    rootDir: options.evolutionRoot ?? DEFAULT_EVOLUTION_ROOT,
  });
  const state = ledger.state();
  const lines: string[] = [];
  for (const pointer of Object.values(state.activePointers)) {
    lines.push(
      `active ${pointer.skillId} -> ${pointer.revisionId} (${pointer.contentDigest.slice(0, 20)}...)`,
    );
  }
  for (const candidate of Object.values(state.candidates)) {
    const manifest = candidate.manifest;
    const evaluation = Object.values(state.evaluations)
      .filter((entry) => entry.candidateId === manifest.candidateId)
      .at(-1);
    lines.push(
      `${manifest.candidateId} ${candidate.status} skill=${manifest.skillId} revision=${manifest.revisionId} parent=${manifest.parentRevisionId} eval=${evaluation?.verdict ?? "none"}`,
    );
  }
  return lines.length > 0 ? `${lines.join("\n")}\n` : "No skill evolution records.\n";
}

export function runEvolutionPromote(
  candidateId: string,
  options: { actor: string; reason: string; evolutionRoot?: string; yes?: boolean },
): string {
  if (!options.yes) {
    throw new Error("Promotion requires --yes to confirm the human approval.");
  }
  const context = createEvolutionCliContext({ evolutionRoot: options.evolutionRoot });
  const receipt = context.service.promote({
    candidateId,
    actor: options.actor,
    reason: options.reason,
    gate: "human",
  });
  return `${JSON.stringify(SkillPromotionReceiptSchema.parse(receipt), null, 2)}\n`;
}

export function runEvolutionDecide(
  candidateId: string,
  decision: "reject" | "quarantine",
  options: { actor: string; reason: string; evolutionRoot?: string },
): string {
  const context = createEvolutionCliContext({ evolutionRoot: options.evolutionRoot });
  context.service.decideCandidate({
    candidateId,
    decision,
    actor: options.actor,
    reason: options.reason,
  });
  return `Candidate ${candidateId} marked ${decision}.\n`;
}

export function runEvolutionRollback(
  skillId: string,
  options: {
    revision: string;
    actor: string;
    reason: string;
    evolutionRoot?: string;
    yes?: boolean;
  },
): string {
  if (!options.yes) {
    throw new Error("Rollback requires --yes to confirm the human decision.");
  }
  const context = createEvolutionCliContext({ evolutionRoot: options.evolutionRoot });
  const receipt = context.service.rollback({
    skillId,
    targetRevisionId: options.revision,
    actor: options.actor,
    reason: options.reason,
  });
  return `${JSON.stringify(SkillPromotionReceiptSchema.parse(receipt), null, 2)}\n`;
}

/**
 * Resolves the evolved active revision for one Agent node's Skill bindings
 * from the evolution ledger. This is the production entry used by `swarmx
 * send` and `swarmx eval-run` so new executions receive the promoted
 * revision while already-constructed Swarms stay frozen. The binding's
 * variant id and (when supplied) target agent must match the promoted
 * candidate manifest, otherwise delivery is refused.
 */
export async function resolveActiveSkillDeliveriesForAgent(options: {
  bindings: Array<{ skillId: string; variantId: string }>;
  agentName: string;
  evolutionRoot?: string;
  taskRoot?: string;
  targetAgentId: string;
  targetModelFingerprint?: string;
}): Promise<Record<string, SkillInstructionDelivery[]>> {
  const ledger = new SkillEvolutionStore({
    rootDir: options.evolutionRoot ?? DEFAULT_EVOLUTION_ROOT,
  });
  const taskStore = new TaskRuntimeStore({ rootDir: options.taskRoot ?? DEFAULT_TASK_ROOT });
  const state = ledger.state();
  const deliveries: SkillInstructionDelivery[] = [];
  for (const binding of options.bindings) {
    const pointer = state.activePointers[binding.skillId];
    if (!pointer) continue;
    // Prefer the promoted candidate manifest; fall back to the retained
    // revision metadata (baselines and rolled-back revisions) so target
    // validation still applies after a rollback.
    const receipt = state.promotionReceipts.find(
      (candidate) => candidate.receiptId === pointer.receiptId,
    );
    const candidateManifest = receipt?.candidateId
      ? state.candidates[receipt.candidateId]?.manifest
      : undefined;
    const retained = state.retainedRevisions[binding.skillId]?.[pointer.revisionId];
    const metadata = candidateManifest ?? retained;
    if (metadata && metadata.variantId !== binding.variantId) {
      throw new SkillDeliveryError(
        "variant_mismatch",
        `Active revision "${pointer.revisionId}" for Skill "${binding.skillId}" targets variant "${metadata.variantId}", not "${binding.variantId}".`,
      );
    }
    if (metadata && options.targetAgentId !== metadata.targetAgentId) {
      throw new SkillDeliveryError(
        "target_mismatch",
        `Active revision "${pointer.revisionId}" targets agent "${metadata.targetAgentId}", not "${options.targetAgentId}".`,
      );
    }
    if (
      metadata &&
      options.targetModelFingerprint &&
      options.targetModelFingerprint !== metadata.targetModelFingerprint
    ) {
      throw new SkillDeliveryError(
        "model_mismatch",
        `Active revision "${pointer.revisionId}" targets model fingerprint "${metadata.targetModelFingerprint}", not "${options.targetModelFingerprint}".`,
      );
    }
    const content = taskStore.readBytes(pointer.contentRef).toString("utf8");
    if (`sha256:${createHash("sha256").update(content).digest("hex")}` !== pointer.contentDigest) {
      throw new SkillDeliveryError(
        "digest_mismatch",
        `Active Skill delivery digest mismatch for "${binding.skillId}".`,
      );
    }
    deliveries.push({
      skillId: binding.skillId,
      variantId: binding.variantId,
      revisionId: pointer.revisionId,
      contentDigest: pointer.contentDigest,
      mode: "prompt_fragment",
      content,
    });
  }
  return deliveries.length > 0 ? { [options.agentName]: deliveries } : {};
}

/** Native direct Agent identity used for delivery binding: `<harness>:<model>`. */
export function nativeAgentTargetId(agent: { model?: string; name?: string }): string {
  return `swarmx:${agent.model?.trim() || "env"}`;
}

/**
 * Discovers the strictly locked SwarmX interpreter (passing
 * `uv sync --locked --check --no-default-groups`), or throws with explicit
 * setup instructions.
 */
export async function discoverEvolutionPythonPath(): Promise<string> {
  const candidates = [".venv/bin/python"];
  const failures: string[] = [];
  for (const candidate of candidates) {
    const pythonPath = path.resolve(REPO_ROOT, candidate);
    try {
      await resolveLockedEvolutionVersions({ pythonPath });
      return pythonPath;
    } catch (error) {
      failures.push(`${candidate}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  throw new Error(
    "No strictly locked SwarmX environment was found. Create one with:\n" +
      "  uv sync --locked --no-default-groups\n" +
      "  uv sync --locked --check --no-default-groups\n" +
      "then pass --python .venv/bin/python.\n" +
      `Attempts: ${failures.join(" | ")}`,
  );
}

export function createModelCommandHandler(command: string): SkillEvolutionModelHandler {
  return async (request) => {
    const payload = JSON.stringify(request);
    const { stdout, stderr } = await runModelCommand(command, payload);
    let parsed: unknown;
    try {
      parsed = JSON.parse(stdout);
    } catch (error) {
      throw new Error(
        `--model-command returned invalid JSON: ${error instanceof Error ? error.message : String(error)}${stderr ? ` (stderr: ${stderr.slice(0, 200)})` : ""}`,
      );
    }
    const record = parsed as {
      content?: unknown;
      usage?: { totalTokens?: unknown };
      latencyMs?: unknown;
      costUsd?: unknown;
    };
    if (
      typeof record.content !== "string" ||
      typeof record.usage?.totalTokens !== "number" ||
      !Number.isInteger(record.usage.totalTokens) ||
      record.usage.totalTokens < 0
    ) {
      throw new Error(
        "--model-command must respond with {content: string, usage: {totalTokens: number}}.",
      );
    }
    return {
      content: record.content,
      usage: {
        inputTokens: record.usage.totalTokens,
        outputTokens: 0,
        reasoningTokens: 0,
        cachedInputTokens: 0,
        totalTokens: record.usage.totalTokens,
        estimated: true,
      },
      latencyMs:
        typeof record.latencyMs === "number" && record.latencyMs >= 0 ? record.latencyMs : 0,
      costUsd: typeof record.costUsd === "number" && record.costUsd >= 0 ? record.costUsd : 0,
    };
  };
}

function runModelCommand(
  command: string,
  payload: string,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, {
      shell: true,
      env: { PATH: process.env.PATH ?? "" },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const MAX_OUTPUT = 512 * 1024;
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`--model-command timed out after 120s.`));
    }, 120_000);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout = `${stdout}${chunk.toString("utf8")}`.slice(-MAX_OUTPUT);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = `${stderr}${chunk.toString("utf8")}`.slice(-4_096);
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(new Error(`--model-command failed to start: ${error.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(
          new Error(
            `--model-command exited with code ${code ?? "none"}.${stderr ? ` stderr: ${stderr.slice(0, 300)}` : ""}`,
          ),
        );
        return;
      }
      resolve({ stdout, stderr });
    });
    child.stdin.write(payload);
    child.stdin.end();
  });
}

export function evolutionLaunchSpec(input: {
  workerPath: string;
  python: string;
  digest: string;
  cwd: string;
}): TaskWorkerLaunchSpec {
  return {
    backendId: "python",
    program: input.python,
    args: ["-I", "-B", "-u", input.workerPath, "--environment-digest", input.digest],
    cwd: input.cwd,
    env: {
      PATH: process.env.PATH ?? "",
      PYTHONDONTWRITEBYTECODE: "1",
      PYTHONUNBUFFERED: "1",
      PYTHONUTF8: "1",
    },
    environmentDigest: input.digest,
    artifactRoot: input.cwd,
  };
}

export function parseDatasetRecords(content: string): Array<Record<string, unknown>> {
  const records: Array<Record<string, unknown>> = [];
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const parsed = JSON.parse(line) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("Evolution dataset lines must be JSON objects.");
    }
    records.push(parsed as Record<string, unknown>);
  }
  return records;
}

export async function temporaryDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), "swarmx-evolution-"));
}

export function defaultEvalSwarmConfig(): SwarmConfig {
  return {
    name: "skill-eval",
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

export function defaultEvaluationGate(): SkillEvaluationGate {
  return {
    minSampleCount: 4,
    minQualityImprovement: 0.1,
    minImprovedRatio: 0.5,
  };
}

export function defaultPythonPath(): string {
  const venvPython = path.join(REPO_ROOT, ".venv", "bin", "python");
  try {
    readFileSync(venvPython);
    return venvPython;
  } catch {
    return "python3";
  }
}

export function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : "Error";
}
