import {
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupRequest,
  type HarnessEnvironmentSetupResult,
  type HarnessEnvironmentStatus,
} from "./harness-environment.js";

export type DoctorIssueSeverity = "error" | "warning";
export type DoctorRepairRisk = "safe" | "install" | "admin";

export interface DoctorIssue {
  id: string;
  severity: DoctorIssueSeverity;
  scope: "doctor" | "protection" | "requirement" | "harness";
  targetId?: string;
  message: string;
  repairActionId?: string;
}

export interface DoctorRepairAction {
  id: string;
  label: string;
  risk: DoctorRepairRisk;
  request: HarnessEnvironmentSetupRequest;
}

export interface DoctorReport {
  checkedAt: string;
  healthy: boolean;
  harnessId?: string;
  summary: {
    readyHarnesses: number;
    totalHarnesses: number;
    issueCount: number;
    fixableCount: number;
  };
  issues: DoctorIssue[];
  repairActions: DoctorRepairAction[];
  environment: HarnessEnvironmentStatus;
}

export interface DoctorRepairPlan {
  actions: DoctorRepairAction[];
  requiresConfirmation: boolean;
  requiresAdmin: boolean;
}

export interface DoctorFixResult {
  executed: boolean;
  before: DoctorReport;
  plan: DoctorRepairPlan;
  setupResults: HarnessEnvironmentSetupResult[];
  after: DoctorReport;
}

export interface DoctorInspectOptions {
  harnessId?: string;
}

export interface DoctorFixOptions extends DoctorInspectOptions {
  confirmed: boolean;
}

export interface HarnessEnvironmentDoctorHost {
  status(): Promise<HarnessEnvironmentStatus>;
  setup(request?: HarnessEnvironmentSetupRequest): Promise<HarnessEnvironmentSetupResult>;
}

export class HarnessDoctor {
  constructor(
    private readonly environment: HarnessEnvironmentDoctorHost = new HarnessEnvironmentService(),
  ) {}

  async inspect(options: DoctorInspectOptions = {}): Promise<DoctorReport> {
    const environment = await this.environment.status();
    const selectedHarnesses = options.harnessId
      ? environment.harnesses.filter((harness) => harness.harnessId === options.harnessId)
      : environment.harnesses;
    const issues: DoctorIssue[] = [];

    if (options.harnessId && selectedHarnesses.length === 0) {
      issues.push({
        id: `doctor:unknown-harness:${options.harnessId}`,
        severity: "error",
        scope: "doctor",
        targetId: options.harnessId,
        message: `Unknown harness: ${options.harnessId}`,
      });
    }

    const nodeRuntime = environment.requirements.find((requirement) => requirement.id === "node");
    if (!options.harnessId && nodeRuntime && nodeRuntime.status !== "ready") {
      issues.push({
        id: "doctor:requirement:node",
        severity: "error",
        scope: "requirement",
        targetId: "node",
        message:
          nodeRuntime.note ??
          "Node.js is not ready. Install an active LTS release and refresh Runtime.",
      });
    }

    const repairActions: DoctorRepairAction[] = [];
    const readyHarnesses = selectedHarnesses.filter((harness) => harness.status === "ready").length;
    const filteredEnvironment = options.harnessId
      ? {
          ...environment,
          ready: issues.length === 0,
          setupAvailable: false,
          containerRuntimes: [],
          protection: { mode: "native" as const, ready: true, requiredHarnessIds: [] },
          requirements: [],
          harnesses: selectedHarnesses,
        }
      : environment;
    return {
      checkedAt: environment.checkedAt,
      healthy: issues.length === 0,
      harnessId: options.harnessId,
      summary: {
        readyHarnesses,
        totalHarnesses: selectedHarnesses.length,
        issueCount: issues.length,
        fixableCount: repairActions.length,
      },
      issues,
      repairActions,
      environment: filteredEnvironment,
    };
  }

  plan(report: DoctorReport): DoctorRepairPlan {
    return {
      actions: report.repairActions,
      requiresConfirmation: report.repairActions.length > 0,
      requiresAdmin: report.repairActions.some((action) => action.risk === "admin"),
    };
  }

  async fix(options: DoctorFixOptions): Promise<DoctorFixResult> {
    const before = await this.inspect(options);
    const plan = this.plan(before);
    if (!options.confirmed || plan.actions.length === 0) {
      return { executed: false, before, plan, setupResults: [], after: before };
    }

    const setupResults: HarnessEnvironmentSetupResult[] = [];
    for (const action of plan.actions) {
      setupResults.push(await this.environment.setup(action.request));
    }
    const after = await this.inspect(options);
    return { executed: true, before, plan, setupResults, after };
  }
}
