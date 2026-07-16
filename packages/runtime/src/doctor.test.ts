import { describe, expect, it, vi } from "vitest";
import { HarnessDoctor, type HarnessEnvironmentDoctorHost } from "./doctor.js";
import {
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupResult,
  type HarnessEnvironmentStatus,
} from "./harness-environment.js";

describe("HarnessDoctor", () => {
  it("V208 V217 inspects optional harness availability without creating global issues", async () => {
    const host = fakeHost(unhealthyStatus());
    const report = await new HarnessDoctor(host).inspect();

    expect(report.healthy).toBe(true);
    expect(report.summary).toMatchObject({ issueCount: 0, fixableCount: 0 });
    expect(report.repairActions).toEqual([]);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("V209 V218 keeps a selected missing harness out of the repair plan", async () => {
    const doctor = new HarnessDoctor(fakeHost(unhealthyStatus()));
    const plan = doctor.plan(await doctor.inspect({ harnessId: "hermes" }));

    expect(plan.requiresConfirmation).toBe(false);
    expect(plan.requiresAdmin).toBe(false);
    expect(plan.actions).toEqual([]);
  });

  it("V210 refuses repair without explicit confirmation", async () => {
    const host = fakeHost(unhealthyStatus());
    const result = await new HarnessDoctor(host).fix({ harnessId: "hermes", confirmed: false });

    expect(result.executed).toBe(false);
    expect(result.after).toBe(result.before);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("V210 does not repair a harness even after confirmation", async () => {
    const host = fakeHost(unhealthyStatus());

    const result = await new HarnessDoctor(host).fix({ harnessId: "hermes", confirmed: true });

    expect(result.executed).toBe(false);
    expect(host.setup).not.toHaveBeenCalled();
    expect(result.after.healthy).toBe(true);
  });

  it("V218 does not diagnose a selected harness's protected runtime", async () => {
    const report = await new HarnessDoctor(fakeHost(unhealthyStatus())).inspect({
      harnessId: "claude_code",
    });

    expect(report.healthy).toBe(true);
    expect(report.issues).toEqual([]);
    expect(report.repairActions).toEqual([]);
  });

  it("does not diagnose a selected missing OpenClaw CLI", async () => {
    const status = healthyStatus();
    status.requirements.push({
      id: "openclaw",
      label: "OpenClaw CLI",
      command: "openclaw",
      status: "missing",
      installable: true,
      requiredBy: ["openclaw"],
    });
    status.harnesses.push({
      harnessId: "openclaw",
      harnessLabel: "OpenClaw",
      status: "needs_setup",
      requirements: ["openclaw"],
      executionMode: "native",
      protectionRequired: false,
    });

    const report = await new HarnessDoctor(fakeHost(status)).inspect({ harnessId: "openclaw" });

    expect(report.healthy).toBe(true);
    expect(report.issues).toEqual([]);
    expect(report.repairActions).toEqual([]);
  });

  it("V219 never repairs optional harnesses from an unfiltered fix", async () => {
    const host = fakeHost(unhealthyStatus());
    const result = await new HarnessDoctor(host).fix({ confirmed: true });

    expect(result.executed).toBe(false);
    expect(result.after.healthy).toBe(true);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("V207 reports an unknown harness without inventing a repair", async () => {
    const report = await new HarnessDoctor(fakeHost(healthyStatus())).inspect({
      harnessId: "missing",
    });

    expect(report.healthy).toBe(false);
    expect(report.issues[0]?.id).toBe("doctor:unknown-harness:missing");
    expect(report.repairActions).toEqual([]);
  });

  it("V312 diagnoses the Node.js baseline without treating optional Harnesses as required", async () => {
    const status = healthyStatus();
    status.requirements.push({
      id: "node",
      label: "Node.js runtime",
      command: "node",
      status: "missing",
      installable: false,
      requiredBy: [],
      note: "Install an active Node.js LTS release.",
    });

    const report = await new HarnessDoctor(fakeHost(status)).inspect();

    expect(report.healthy).toBe(false);
    expect(report.issues).toContainEqual(
      expect.objectContaining({ id: "doctor:requirement:node", targetId: "node" }),
    );
    expect(report.repairActions).toEqual([]);
  });

  it("V207 keeps filtered environment readiness consistent with the report", async () => {
    const report = await new HarnessDoctor(fakeHost(protectionOnlyUnhealthyStatus())).inspect({
      harnessId: "hermes",
    });

    expect(report.healthy).toBe(true);
    expect(report.environment.ready).toBe(true);
    expect(report.environment.setupAvailable).toBe(false);
    expect(report.environment.harnesses.map((harness) => harness.harnessId)).toEqual(["hermes"]);
  });

  it("V211 prefers the existing local Hermes checkout during detection", async () => {
    const commands: string[] = [];
    const service = new HarnessEnvironmentService({
      env: { PATH: "/usr/bin" },
      platform: "linux",
      homeDir: "/Users/test",
      findExecutable: vi.fn(async (command, envPath) => {
        if (command === "hermes") {
          expect(envPath.split(":")[0]).toBe("/Users/test/.hermes/hermes-agent");
          return "/Users/test/.hermes/hermes-agent/hermes";
        }
        return `/usr/bin/${command}`;
      }),
      runCommand: vi.fn(async (program) => {
        commands.push(program);
        return { exitCode: 0, stdout: "1.0.0\n", stderr: "" };
      }),
    });

    const status = await service.status();
    expect(status.requirements.find((item) => item.id === "hermes")?.path).toBe(
      "/Users/test/.hermes/hermes-agent/hermes",
    );
    expect(commands).not.toContain("git");
    expect(commands).not.toContain("curl");
  });
});

function fakeHost(status: HarnessEnvironmentStatus) {
  return {
    status: vi.fn(async () => status),
    setup: vi.fn(async () => setupResult(status)),
  } satisfies HarnessEnvironmentDoctorHost;
}

function setupResult(status: HarnessEnvironmentStatus): HarnessEnvironmentSetupResult {
  return {
    success: status.ready,
    status,
    installedRequirementIds: [],
    skippedRequirementIds: [],
    failedRequirementIds: [],
    installedContainerRuntimeIds: [],
    skippedContainerRuntimeIds: [],
    failedContainerRuntimeIds: [],
    log: [],
  };
}

function healthyStatus(): HarnessEnvironmentStatus {
  return {
    checkedAt: "2026-07-11T00:00:00.000Z",
    path: "/usr/bin",
    ready: true,
    setupAvailable: false,
    containerRuntimes: [
      {
        id: "apple_container",
        label: "Apple Container",
        command: "container",
        status: "ready",
        supported: true,
        installable: true,
        serviceReady: true,
        preferred: true,
      },
    ],
    protection: {
      mode: "protected",
      ready: true,
      requiredHarnessIds: ["claude_code"],
      selectedRuntimeId: "apple_container",
    },
    requirements: [
      {
        id: "hermes",
        label: "Hermes Agent CLI",
        command: "hermes",
        status: "ready",
        installable: true,
        requiredBy: ["hermes"],
      },
    ],
    harnesses: [
      {
        harnessId: "claude_code",
        harnessLabel: "Claude Code",
        status: "ready",
        requirements: ["bun"],
        executionMode: "protected",
        protectionRequired: true,
      },
      {
        harnessId: "hermes",
        harnessLabel: "Hermes",
        status: "ready",
        requirements: ["hermes"],
        executionMode: "native",
        protectionRequired: false,
      },
    ],
  };
}

function unhealthyStatus(): HarnessEnvironmentStatus {
  const status = healthyStatus();
  return {
    ...status,
    ready: false,
    setupAvailable: true,
    containerRuntimes: status.containerRuntimes.map((runtime) => ({
      ...runtime,
      status: "missing",
      serviceReady: false,
      note: "Apple Container is missing.",
    })),
    protection: {
      ...status.protection,
      ready: false,
      note: "Apple Container is missing.",
    },
    requirements: status.requirements.map((requirement) => ({
      ...requirement,
      status: "missing",
      note: "Hermes is missing.",
    })),
    harnesses: status.harnesses.map((harness) => ({
      ...harness,
      status: "needs_setup",
    })),
  };
}

function protectionOnlyUnhealthyStatus(): HarnessEnvironmentStatus {
  const status = healthyStatus();
  return {
    ...status,
    ready: false,
    setupAvailable: true,
    containerRuntimes: status.containerRuntimes.map((runtime) => ({
      ...runtime,
      status: "missing",
      serviceReady: false,
      note: "Apple Container is missing.",
    })),
    protection: {
      ...status.protection,
      ready: false,
      note: "Apple Container is missing.",
    },
    harnesses: status.harnesses.map((harness) =>
      harness.executionMode === "protected" ? { ...harness, status: "needs_setup" } : harness,
    ),
  };
}
