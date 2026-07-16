import type { DoctorFixResult, DoctorRepairPlan, DoctorReport } from "@swarmx/runtime";
import { describe, expect, it, vi } from "vitest";
import {
  type DoctorCommandIo,
  type DoctorCommandRunner,
  formatDoctorReport,
  runDoctorCommand,
} from "../src/doctor.js";

describe("swarmx doctor", () => {
  it("V212 prints a read-only human report with a stable failure exit", async () => {
    const report = unhealthyReport();
    const runner = fakeRunner(report);
    const io = fakeIo();

    const exitCode = await runDoctorCommand({}, io, runner);

    expect(exitCode).toBe(1);
    expect(io.output()).toContain("SwarmX Doctor");
    expect(io.output()).toContain("Run `swarmx doctor --fix`");
    expect(runner.inspect).toHaveBeenCalledWith({});
    expect(runner.fix).not.toHaveBeenCalled();
  });

  it("V212 supports harness-filtered JSON output", async () => {
    const report = { ...unhealthyReport(), harnessId: "hermes" };
    const runner = fakeRunner(report);
    const io = fakeIo();

    await runDoctorCommand({ harness: "hermes", json: true }, io, runner);

    expect(runner.inspect).toHaveBeenCalledWith({ harnessId: "hermes" });
    expect(JSON.parse(io.output())).toMatchObject({ harnessId: "hermes", healthy: false });
  });

  it("V210 previews repairs and refuses mutation when confirmation is declined", async () => {
    const runner = fakeRunner(unhealthyReport());
    const io = fakeIo(false);

    const exitCode = await runDoctorCommand({ fix: true }, io, runner);

    expect(exitCode).toBe(1);
    expect(io.output()).toContain("Repair plan:");
    expect(io.output()).toContain("No changes applied.");
    expect(runner.fix).not.toHaveBeenCalled();
  });

  it("V210 applies and reports a confirmed repair", async () => {
    const before = unhealthyReport();
    const after = { ...before, healthy: true, issues: [], repairActions: [] };
    const runner = fakeRunner(before, {
      executed: true,
      before,
      plan: runnerPlan(before),
      setupResults: [],
      after,
    });
    const io = fakeIo();

    const exitCode = await runDoctorCommand({ fix: true, yes: true }, io, runner);

    expect(exitCode).toBe(0);
    expect(runner.fix).toHaveBeenCalledWith({ confirmed: true });
    expect(io.output()).toContain("Repair complete.");
    expect(io.output()).toContain("Status: healthy");
  });

  it("keeps JSON fix output machine-readable", async () => {
    const before = unhealthyReport();
    const result: DoctorFixResult = {
      executed: true,
      before,
      plan: runnerPlan(before),
      setupResults: [],
      after: { ...before, healthy: true, issues: [], repairActions: [] },
    };
    const io = fakeIo();

    await runDoctorCommand({ fix: true, yes: true, json: true }, io, fakeRunner(before, result));

    expect(JSON.parse(io.output())).toMatchObject({ executed: true, after: { healthy: true } });
  });

  it("requires an explicit non-interactive confirmation for JSON fix mode", async () => {
    await expect(
      runDoctorCommand({ fix: true, json: true }, fakeIo(), fakeRunner(unhealthyReport())),
    ).rejects.toThrow("requires --yes");
  });

  it("formats a healthy report without a repair hint", () => {
    const report = { ...unhealthyReport(), healthy: true, issues: [], repairActions: [] };
    expect(formatDoctorReport(report)).toContain("No issues found.");
    expect(formatDoctorReport(report)).not.toContain("--fix");
  });
});

function fakeRunner(report: DoctorReport, fixResult?: DoctorFixResult) {
  const plan = runnerPlan(report);
  return {
    inspect: vi.fn(async () => report),
    plan: vi.fn(() => plan),
    fix: vi.fn(
      async () =>
        fixResult ?? { executed: false, before: report, plan, setupResults: [], after: report },
    ),
  } satisfies DoctorCommandRunner;
}

function runnerPlan(report: DoctorReport): DoctorRepairPlan {
  return {
    actions: report.repairActions,
    requiresConfirmation: report.repairActions.length > 0,
    requiresAdmin: report.repairActions.some((action) => action.risk === "admin"),
  };
}

function fakeIo(confirmed = true) {
  const chunks: string[] = [];
  return {
    write: (value: string) => chunks.push(value),
    confirm: vi.fn(async () => confirmed),
    output: () => chunks.join(""),
  } satisfies DoctorCommandIo & { output(): string };
}

function unhealthyReport(): DoctorReport {
  const action = {
    id: "harness:hermes",
    label: "Set up Hermes",
    risk: "install" as const,
    request: { harnessId: "hermes" },
  };
  return {
    checkedAt: "2026-07-11T00:00:00.000Z",
    healthy: false,
    summary: { readyHarnesses: 0, totalHarnesses: 1, issueCount: 1, fixableCount: 1 },
    issues: [
      {
        id: "requirement:hermes",
        severity: "error",
        scope: "requirement",
        targetId: "hermes",
        message: "Hermes is missing.",
        repairActionId: action.id,
      },
    ],
    repairActions: [action],
    environment: {
      checkedAt: "2026-07-11T00:00:00.000Z",
      path: "/usr/bin",
      ready: false,
      setupAvailable: true,
      containerRuntimes: [],
      protection: { mode: "native", ready: true, requiredHarnessIds: [] },
      requirements: [],
      harnesses: [],
    },
  };
}
