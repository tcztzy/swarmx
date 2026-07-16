import { createInterface } from "node:readline/promises";
import {
  type DoctorFixOptions,
  type DoctorFixResult,
  type DoctorInspectOptions,
  type DoctorRepairPlan,
  type DoctorReport,
  HarnessDoctor,
} from "@swarmx/runtime";

export interface DoctorCommandOptions {
  fix?: boolean;
  harness?: string;
  json?: boolean;
  yes?: boolean;
}

export interface DoctorCommandIo {
  write(value: string): void;
  confirm(prompt: string): Promise<boolean>;
}

export interface DoctorCommandRunner {
  inspect(options?: DoctorInspectOptions): Promise<DoctorReport>;
  plan(report: DoctorReport): DoctorRepairPlan;
  fix(options: DoctorFixOptions): Promise<DoctorFixResult>;
}

export async function runDoctorCommand(
  options: DoctorCommandOptions,
  io: DoctorCommandIo = processDoctorIo(),
  doctor: DoctorCommandRunner = new HarnessDoctor(),
): Promise<number> {
  if (options.yes && !options.fix) throw new Error("--yes requires --fix");
  if (options.json && options.fix && !options.yes) {
    throw new Error("JSON fix mode requires --yes so stdout remains valid JSON");
  }

  const inspectOptions = options.harness ? { harnessId: options.harness } : {};
  const report = await doctor.inspect(inspectOptions);
  if (!options.fix) {
    io.write(options.json ? formatJson(report) : formatDoctorReport(report));
    return report.healthy ? 0 : 1;
  }

  const plan = doctor.plan(report);
  if (plan.actions.length === 0) {
    io.write(options.json ? formatJson({ report, plan }) : formatDoctorReport(report, plan));
    return report.healthy ? 0 : 1;
  }

  if (!options.json) io.write(formatDoctorReport(report, plan));
  const confirmed = options.yes || (await io.confirm("Apply this repair plan? [y/N] "));
  if (!confirmed) {
    if (!options.json) io.write("No changes applied.\n");
    return 1;
  }

  const result = await doctor.fix({ ...inspectOptions, confirmed: true });
  io.write(options.json ? formatJson(result) : formatDoctorFixResult(result));
  return result.after.healthy ? 0 : 1;
}

export function formatDoctorReport(report: DoctorReport, plan?: DoctorRepairPlan): string {
  const lines = [
    "SwarmX Doctor",
    `Status: ${report.healthy ? "healthy" : "attention needed"}`,
    `Harnesses: ${report.summary.readyHarnesses}/${report.summary.totalHarnesses} ready`,
  ];
  if (report.harnessId) lines.push(`Filter: ${report.harnessId}`);

  if (report.issues.length === 0) {
    lines.push("", "No issues found.");
  } else {
    lines.push("", "Issues:");
    for (const issue of report.issues) {
      lines.push(`  ${issue.severity.toUpperCase()} [${issue.id}] ${issue.message}`);
    }
  }

  const actions = plan?.actions ?? report.repairActions;
  if (actions.length > 0) {
    lines.push("", "Repair plan:");
    actions.forEach((action, index) => {
      lines.push(`  ${index + 1}. [${action.risk}] ${action.label}`);
    });
    if (!plan) lines.push("", "Run `swarmx doctor --fix` to review and apply repairs.");
  }
  return `${lines.join("\n")}\n`;
}

export function formatDoctorFixResult(result: DoctorFixResult): string {
  const logs = result.setupResults.flatMap((setup) => setup.log).filter(Boolean);
  const lines = [
    "",
    result.executed ? "Repair complete." : "No repairs executed.",
    `Status: ${result.after.healthy ? "healthy" : "attention still needed"}`,
  ];
  if (logs.length > 0) lines.push("", "Repair log:", ...logs.map((line) => `  ${line}`));
  if (result.after.issues.length > 0) {
    lines.push("", "Remaining issues:");
    for (const issue of result.after.issues) lines.push(`  ${issue.message}`);
  }
  return `${lines.join("\n")}\n`;
}

function formatJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function processDoctorIo(): DoctorCommandIo {
  return {
    write: (value) => process.stdout.write(value),
    confirm: async (prompt) => {
      if (!process.stdin.isTTY || !process.stdout.isTTY) return false;
      const readline = createInterface({ input: process.stdin, output: process.stdout });
      try {
        const answer = await readline.question(prompt);
        return /^(y|yes)$/i.test(answer.trim());
      } finally {
        readline.close();
      }
    },
  };
}
