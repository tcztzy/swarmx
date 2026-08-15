import { z } from "zod";
import {
  type HarnessEnvironmentHarness,
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupRequest,
  type HarnessEnvironmentSetupResult,
  type HarnessEnvironmentStatus,
} from "./harness-environment.js";

export const DoctorIssueSeveritySchema = z.enum(["error", "warning"]);
export const DoctorIssueClassificationSchema = z.enum([
  "warning",
  "blocking",
  "repairable",
  "decision",
]);
export const DoctorRepairRiskSchema = z.enum(["safe", "install", "admin"]);
export const DoctorOverallStatusSchema = z.enum(["ok", "warning", "blocking"]);

export const DoctorReadinessSchema = z
  .object({
    provider: z.enum(["ready", "missing", "invalid_reference", "not_required"]).optional(),
    project: z.enum(["ready", "missing", "not_writable"]).optional(),
    network: z.enum(["online", "offline", "not_required"]).optional(),
  })
  .strict();

export const DoctorInspectOptionsSchema = z
  .object({
    harnessId: z.string().trim().min(1).max(160).optional(),
    readiness: DoctorReadinessSchema.optional(),
  })
  .strict();

export const DoctorFixOptionsSchema = DoctorInspectOptionsSchema.extend({
  confirmed: z.boolean(),
}).strict();

export const DoctorIssueSchema = z
  .object({
    id: z.string().min(1).max(256),
    severity: DoctorIssueSeveritySchema,
    classification: DoctorIssueClassificationSchema,
    scope: z.enum([
      "doctor",
      "protection",
      "requirement",
      "harness",
      "provider",
      "project",
      "network",
    ]),
    targetId: z.string().min(1).max(256).optional(),
    symptom: z.string().min(1).max(1_024),
    cause: z.string().min(1).max(1_024),
    impact: z.string().min(1).max(1_024),
    nextAction: z.string().min(1).max(1_024),
    message: z.string().min(1).max(1_024),
    repairActionId: z.string().min(1).max(256).optional(),
  })
  .strict();

const HarnessEnvironmentSetupRequestSchema = z
  .object({
    harnessId: z.string().min(1).optional(),
    harnessToolId: z.string().min(1).optional(),
    requirementIds: z.array(z.string().min(1)).optional(),
    containerRuntimeId: z.literal("apple_container").optional(),
    includeContainerRuntime: z.boolean().optional(),
  })
  .strict();

export const DoctorRepairActionSchema = z
  .object({
    id: z.string().min(1).max(256),
    label: z.string().min(1).max(512),
    risk: DoctorRepairRiskSchema,
    request: HarnessEnvironmentSetupRequestSchema,
    changes: z.array(z.string().min(1).max(512)).min(1).max(32),
    idempotent: z.literal(true),
  })
  .strict();

export type DoctorIssueSeverity = z.infer<typeof DoctorIssueSeveritySchema>;
export type DoctorIssueClassification = z.infer<typeof DoctorIssueClassificationSchema>;
export type DoctorRepairRisk = z.infer<typeof DoctorRepairRiskSchema>;
export type DoctorOverallStatus = z.infer<typeof DoctorOverallStatusSchema>;
export type DoctorReadiness = z.infer<typeof DoctorReadinessSchema>;
export interface DoctorIssue {
  id: string;
  severity: DoctorIssueSeverity;
  classification?: DoctorIssueClassification;
  scope: "doctor" | "protection" | "requirement" | "harness" | "provider" | "project" | "network";
  targetId?: string;
  symptom?: string;
  cause?: string;
  impact?: string;
  nextAction?: string;
  message: string;
  repairActionId?: string;
}
export interface DoctorRepairAction {
  id: string;
  label: string;
  risk: DoctorRepairRisk;
  request: HarnessEnvironmentSetupRequest;
  changes?: string[];
  idempotent?: true;
}
export type DoctorInspectOptions = z.infer<typeof DoctorInspectOptionsSchema>;
export type DoctorFixOptions = z.infer<typeof DoctorFixOptionsSchema>;

export interface DoctorReport {
  checkedAt: string;
  healthy: boolean;
  status?: DoctorOverallStatus;
  harnessId?: string;
  summary: {
    readyHarnesses: number;
    totalHarnesses: number;
    issueCount: number;
    warningCount?: number;
    blockingCount?: number;
    decisionCount?: number;
    fixableCount: number;
  };
  issues: DoctorIssue[];
  repairActions: DoctorRepairAction[];
  firstRun?: {
    availableHarnessIds: string[];
    recommendedHarnessId?: string;
    provider: DoctorReadiness["provider"] | "unknown";
    project: DoctorReadiness["project"] | "unknown";
    network: DoctorReadiness["network"] | "unknown";
    nextStep: string;
  };
  environment: HarnessEnvironmentStatus;
}

export interface DoctorRepairPlan {
  actions: DoctorRepairAction[];
  changes?: string[];
  requiresConfirmation: boolean;
  requiresAdmin: boolean;
  idempotent?: true;
}

export interface DoctorFixResult {
  executed: boolean;
  before: DoctorReport;
  plan: DoctorRepairPlan;
  setupResults: HarnessEnvironmentSetupResult[];
  after: DoctorReport;
}

export interface HarnessEnvironmentDoctorHost {
  status(): Promise<HarnessEnvironmentStatus>;
  setup(request?: HarnessEnvironmentSetupRequest): Promise<HarnessEnvironmentSetupResult>;
}

export class HarnessDoctor {
  constructor(
    private readonly environment: HarnessEnvironmentDoctorHost = new HarnessEnvironmentService(),
  ) {}

  async inspect(optionsInput: DoctorInspectOptions = {}): Promise<DoctorReport> {
    const options = DoctorInspectOptionsSchema.parse(optionsInput);
    const environment = await this.environment.status();
    const selectedHarnesses = options.harnessId
      ? environment.harnesses.filter((harness) => harness.harnessId === options.harnessId)
      : environment.harnesses;
    const issues: DoctorIssue[] = [];
    const repairActions: DoctorRepairAction[] = [];

    if (options.harnessId && selectedHarnesses.length === 0) {
      issues.push(
        issue({
          id: `doctor:unknown-harness:${options.harnessId}`,
          classification: "decision",
          scope: "doctor",
          targetId: options.harnessId,
          symptom: `SwarmX does not recognize Harness "${options.harnessId}".`,
          cause: "The selected Harness is not present in the current runtime inventory.",
          impact: "SwarmX cannot start a task with this Harness.",
          nextAction:
            "Choose one of the available Harnesses or install its owning Extension first.",
        }),
      );
    }

    const nodeRuntime = environment.requirements.find((requirement) => requirement.id === "node");
    if (!options.harnessId && nodeRuntime && nodeRuntime.status !== "ready") {
      issues.push(
        issue({
          id: "doctor:requirement:node",
          classification: "blocking",
          scope: "requirement",
          targetId: "node",
          symptom: "Node.js is not ready.",
          cause: nodeRuntime.note ?? "The required Node.js runtime was not found on PATH.",
          impact: "SwarmX CLI and Node-based Harnesses cannot run reliably.",
          nextAction:
            "Install an active Node.js LTS release with your preferred version manager, then run Doctor again.",
        }),
      );
    }

    if (options.harnessId) {
      for (const harness of selectedHarnesses) {
        if (harness.status === "ready") continue;
        const action = repairActionForHarness(harness, environment);
        if (action) repairActions.push(action);
        issues.push(
          issue({
            id: `doctor:harness:${harness.harnessId}`,
            classification: action ? "repairable" : "blocking",
            scope: harness.protectionRequired ? "protection" : "harness",
            targetId: harness.harnessId,
            symptom: `${harness.harnessLabel} is not ready.`,
            cause:
              harness.note ??
              missingHarnessCause(harness, environment) ??
              "One or more required runtime components are unavailable.",
            impact: `Tasks selected for ${harness.harnessLabel} cannot start.`,
            nextAction: action
              ? "Review the proposed changes, then confirm the repair explicitly."
              : "Install or configure the missing component, then run Doctor again.",
            ...(action ? { repairActionId: action.id } : {}),
          }),
        );
      }
    }

    addReadinessIssues(options.readiness, issues);
    const availableHarnessIds = environment.harnesses
      .filter((harness) => harness.status === "ready")
      .map((harness) => harness.harnessId)
      .sort();
    const recommendedHarnessId = recommendedHarness(availableHarnessIds);
    const warningCount = issues.filter((entry) => entry.classification === "warning").length;
    const decisionCount = issues.filter((entry) => entry.classification === "decision").length;
    const blockingCount = issues.filter(
      (entry) =>
        entry.classification === "blocking" ||
        entry.classification === "repairable" ||
        entry.classification === "decision",
    ).length;
    const status: DoctorOverallStatus =
      blockingCount > 0 ? "blocking" : warningCount > 0 ? "warning" : "ok";
    const readyHarnesses = selectedHarnesses.filter((harness) => harness.status === "ready").length;
    const filteredEnvironment = options.harnessId
      ? {
          ...environment,
          ready: blockingCount === 0,
          setupAvailable: repairActions.length > 0,
          containerRuntimes: selectedHarnesses.some((harness) => harness.protectionRequired)
            ? environment.containerRuntimes
            : [],
          protection: selectedHarnesses.some((harness) => harness.protectionRequired)
            ? environment.protection
            : { mode: "native" as const, ready: true, requiredHarnessIds: [] },
          sandbox: selectedHarnesses.some((harness) => harness.protectionRequired)
            ? environment.sandbox
            : {
                strategy: "native_allowed" as const,
                mode: "native" as const,
                ready: true,
                profileIds: [],
              },
          requirements: environment.requirements.filter((requirement) =>
            selectedHarnesses.some((harness) => harness.requirements.includes(requirement.id)),
          ),
          harnesses: selectedHarnesses,
        }
      : environment;
    const nextStep =
      issues.find((entry) => entry.severity === "error")?.nextAction ??
      (options.harnessId
        ? "Configure Provider authentication and choose or create a writable Project."
        : recommendedHarnessId
          ? `Start with ${recommendedHarnessId}, then configure Provider authentication and a Project.`
          : "Install one supported Harness, then run Doctor again.");

    return {
      checkedAt: environment.checkedAt,
      healthy: status !== "blocking",
      status,
      ...(options.harnessId ? { harnessId: options.harnessId } : {}),
      summary: {
        readyHarnesses,
        totalHarnesses: selectedHarnesses.length,
        issueCount: issues.length,
        warningCount,
        blockingCount,
        decisionCount,
        fixableCount: repairActions.length,
      },
      issues,
      repairActions,
      firstRun: {
        availableHarnessIds,
        ...(recommendedHarnessId ? { recommendedHarnessId } : {}),
        provider: options.readiness?.provider ?? "unknown",
        project: options.readiness?.project ?? "unknown",
        network: options.readiness?.network ?? "unknown",
        nextStep,
      },
      environment: filteredEnvironment,
    };
  }

  plan(report: DoctorReport): DoctorRepairPlan {
    return {
      actions: report.repairActions,
      changes: report.repairActions.flatMap((action) => action.changes ?? []),
      requiresConfirmation: report.repairActions.length > 0,
      requiresAdmin: report.repairActions.some((action) => action.risk === "admin"),
      idempotent: true,
    };
  }

  async fix(optionsInput: DoctorFixOptions): Promise<DoctorFixResult> {
    const options = DoctorFixOptionsSchema.parse(optionsInput);
    const { confirmed, ...inspectOptions } = options;
    const before = await this.inspect(inspectOptions);
    const plan = this.plan(before);
    if (!confirmed || plan.actions.length === 0) {
      return { executed: false, before, plan, setupResults: [], after: before };
    }

    const setupResults: HarnessEnvironmentSetupResult[] = [];
    for (const action of plan.actions) {
      setupResults.push(await this.environment.setup(action.request));
    }
    const after = await this.inspect(inspectOptions);
    return { executed: true, before, plan, setupResults, after };
  }
}

function issue(input: {
  id: string;
  classification: DoctorIssueClassification;
  scope: DoctorIssue["scope"];
  targetId?: string;
  symptom: string;
  cause: string;
  impact: string;
  nextAction: string;
  repairActionId?: string;
}): DoctorIssue {
  return DoctorIssueSchema.parse({
    ...input,
    severity: input.classification === "warning" ? "warning" : "error",
    message: input.symptom,
  });
}

function repairActionForHarness(
  harness: HarnessEnvironmentHarness,
  environment: HarnessEnvironmentStatus,
): DoctorRepairAction | undefined {
  const requirements = environment.requirements.filter(
    (requirement) =>
      harness.requirements.includes(requirement.id) &&
      requirement.status !== "ready" &&
      requirement.installable,
  );
  const container = harness.protectionRequired
    ? environment.containerRuntimes.find(
        (runtime) =>
          runtime.id === (harness.containerRuntimeId ?? environment.protection.selectedRuntimeId) &&
          runtime.status !== "ready",
      )
    : undefined;
  if (requirements.length === 0 && !container?.installable) return undefined;
  const changes = [
    ...requirements.map((requirement) => `Install or repair ${requirement.label}.`),
    ...(container ? [`Install or start ${container.label} for protected execution.`] : []),
  ];
  return DoctorRepairActionSchema.parse({
    id: `harness:${harness.harnessId}`,
    label: `Set up ${harness.harnessLabel}`,
    risk: container ? "admin" : "install",
    request: {
      harnessId: harness.harnessId,
      requirementIds: requirements.map((requirement) => requirement.id),
      ...(container ? { containerRuntimeId: container.id, includeContainerRuntime: true } : {}),
    },
    changes,
    idempotent: true,
  });
}

function missingHarnessCause(
  harness: HarnessEnvironmentHarness,
  environment: HarnessEnvironmentStatus,
): string | undefined {
  const missing = environment.requirements.find(
    (requirement) =>
      harness.requirements.includes(requirement.id) && requirement.status !== "ready",
  );
  if (missing) return missing.note ?? `${missing.label} is ${missing.status}.`;
  if (harness.protectionRequired && !environment.protection.ready) {
    return environment.protection.note ?? "The protected runtime is not ready.";
  }
  return undefined;
}

function addReadinessIssues(readiness: DoctorReadiness | undefined, issues: DoctorIssue[]): void {
  if (!readiness) return;
  if (readiness.provider === "missing") {
    issues.push(
      issue({
        id: "doctor:provider:missing",
        classification: "decision",
        scope: "provider",
        targetId: "provider-auth",
        symptom: "No Provider authentication is configured.",
        cause: "The selected model route has no usable authentication reference.",
        impact: "The first model-backed task cannot start.",
        nextAction: "Choose a Provider and configure its authentication explicitly.",
      }),
    );
  } else if (readiness.provider === "invalid_reference") {
    issues.push(
      issue({
        id: "doctor:provider:invalid-reference",
        classification: "blocking",
        scope: "provider",
        targetId: "provider-auth",
        symptom: "The Provider authentication reference is unavailable.",
        cause: "Settings point to a missing or unreadable local authentication entry.",
        impact: "Provider requests will fail before a model call is made.",
        nextAction:
          "Reconfigure the Provider in the current schema; Doctor will not rewrite credentials.",
      }),
    );
  }
  if (readiness.project === "missing") {
    issues.push(
      issue({
        id: "doctor:project:missing",
        classification: "decision",
        scope: "project",
        targetId: "project",
        symptom: "No Project is selected.",
        cause: "The first-run flow has not chosen or created a Project.",
        impact: "SwarmX has no bounded workspace for the first task.",
        nextAction: "Choose an existing Project or explicitly create one.",
      }),
    );
  } else if (readiness.project === "not_writable") {
    issues.push(
      issue({
        id: "doctor:project:not-writable",
        classification: "blocking",
        scope: "project",
        targetId: "project",
        symptom: "The selected Project is not writable.",
        cause: "The current process cannot write to the Project root.",
        impact: "Tasks that create or modify Project files will fail.",
        nextAction: "Choose a writable Project or correct filesystem ownership outside Doctor.",
      }),
    );
  }
  if (readiness.network === "offline") {
    issues.push(
      issue({
        id: "doctor:network:offline",
        classification: "warning",
        scope: "network",
        targetId: "network",
        symptom: "The network appears to be offline.",
        cause: "No network route was observed for services that may need one.",
        impact: "Remote Providers and downloads may fail; local discovery remains available.",
        nextAction: "Continue with local capabilities or reconnect before using a remote Provider.",
      }),
    );
  }
}

function recommendedHarness(available: string[]): string | undefined {
  const priority = [
    "swarmx",
    "codex",
    "claude_code",
    "pi",
    "kimi",
    "opencode",
    "hermes",
    "openclaw",
  ];
  return priority.find((id) => available.includes(id)) ?? available[0];
}
