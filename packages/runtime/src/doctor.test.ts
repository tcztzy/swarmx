import { describe, expect, it, vi } from "vitest";
import { HarnessDoctor, type HarnessEnvironmentDoctorHost } from "./doctor.js";
import {
  HarnessEnvironmentService,
  type HarnessEnvironmentSetupResult,
  type HarnessEnvironmentStatus,
} from "./harness-environment.js";

describe("HarnessDoctor", () => {
  it("inspects optional harness availability without creating global issues", async () => {
    const host = fakeHost(unhealthyStatus());
    const report = await new HarnessDoctor(host).inspect();

    expect(report.healthy).toBe(true);
    expect(report.summary).toMatchObject({ issueCount: 0, fixableCount: 0 });
    expect(report.repairActions).toEqual([]);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("plans an idempotent repair for a selected installable Harness", async () => {
    const doctor = new HarnessDoctor(fakeHost(unhealthyStatus()));
    const plan = doctor.plan(await doctor.inspect({ harnessId: "hermes" }));

    expect(plan.requiresConfirmation).toBe(true);
    expect(plan.requiresAdmin).toBe(false);
    expect(plan.idempotent).toBe(true);
    expect(plan.actions).toEqual([
      expect.objectContaining({ id: "harness:hermes", risk: "install", idempotent: true }),
    ]);
    expect(plan.changes).toContain("Install or repair Hermes Agent CLI.");
  });

  it("refuses repair without explicit confirmation", async () => {
    const host = fakeHost(unhealthyStatus());
    const result = await new HarnessDoctor(host).fix({ harnessId: "hermes", confirmed: false });

    expect(result.executed).toBe(false);
    expect(result.after).toBe(result.before);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("repairs a selected Harness only after confirmation", async () => {
    const host = fakeHost(unhealthyStatus());

    const result = await new HarnessDoctor(host).fix({ harnessId: "hermes", confirmed: true });

    expect(result.executed).toBe(true);
    expect(host.setup).toHaveBeenCalledWith({
      harnessId: "hermes",
      requirementIds: ["hermes"],
    });
    expect(result.after.healthy).toBe(false);
  });

  it("diagnoses and previews a selected Harness's protected runtime", async () => {
    const report = await new HarnessDoctor(fakeHost(unhealthyStatus())).inspect({
      harnessId: "claude_code",
    });

    expect(report.healthy).toBe(false);
    expect(report.issues).toEqual([
      expect.objectContaining({ classification: "repairable", scope: "protection" }),
    ]);
    expect(report.repairActions).toEqual([
      expect.objectContaining({
        risk: "admin",
        changes: expect.arrayContaining([expect.stringMatching(/Apple Container/)]),
      }),
    ]);
  });

  it("diagnoses a selected missing OpenClaw CLI without running it", async () => {
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

    expect(report.healthy).toBe(false);
    expect(report.issues).toEqual([
      expect.objectContaining({ classification: "repairable", targetId: "openclaw" }),
    ]);
    expect(report.repairActions).toEqual([
      expect.objectContaining({ request: { harnessId: "openclaw", requirementIds: ["openclaw"] } }),
    ]);
  });

  it("never repairs optional harnesses from an unfiltered fix", async () => {
    const host = fakeHost(unhealthyStatus());
    const result = await new HarnessDoctor(host).fix({ confirmed: true });

    expect(result.executed).toBe(false);
    expect(result.after.healthy).toBe(true);
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("reports an unknown harness without inventing a repair", async () => {
    const report = await new HarnessDoctor(fakeHost(healthyStatus())).inspect({
      harnessId: "missing",
    });

    expect(report.healthy).toBe(false);
    expect(report.issues[0]?.id).toBe("doctor:unknown-harness:missing");
    expect(report.repairActions).toEqual([]);
  });

  it("diagnoses the Node.js baseline without treating optional Harnesses as required", async () => {
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

  it("keeps filtered environment readiness consistent with the report", async () => {
    const report = await new HarnessDoctor(fakeHost(protectionOnlyUnhealthyStatus())).inspect({
      harnessId: "hermes",
    });

    expect(report.healthy).toBe(true);
    expect(report.environment.ready).toBe(true);
    expect(report.environment.setupAvailable).toBe(false);
    expect(report.environment.harnesses.map((harness) => harness.harnessId)).toEqual(["hermes"]);
  });

  it("explains fresh Provider, Project, and offline readiness without mutating", async () => {
    const host = fakeHost(healthyStatus());
    const report = await new HarnessDoctor(host).inspect({
      readiness: { provider: "missing", project: "missing", network: "offline" },
    });

    expect(report.status).toBe("blocking");
    expect(report.issues).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ scope: "provider", classification: "decision" }),
        expect.objectContaining({ scope: "project", classification: "decision" }),
        expect.objectContaining({ scope: "network", classification: "warning" }),
      ]),
    );
    expect(report.issues[0]).toMatchObject({
      symptom: expect.any(String),
      cause: expect.any(String),
      impact: expect.any(String),
      nextAction: expect.any(String),
    });
    expect(host.setup).not.toHaveBeenCalled();
  });

  it("blocks invalid auth references and unwritable Projects", async () => {
    const report = await new HarnessDoctor(fakeHost(healthyStatus())).inspect({
      readiness: {
        provider: "invalid_reference",
        project: "not_writable",
        network: "not_required",
      },
    });

    expect(report.issues.map((entry) => entry.id)).toEqual(
      expect.arrayContaining(["doctor:provider:invalid-reference", "doctor:project:not-writable"]),
    );
    expect(report.repairActions).toEqual([]);
  });

  it("chooses one deterministic default when multiple Harnesses are ready", async () => {
    const report = await new HarnessDoctor(fakeHost(healthyStatus())).inspect();

    expect(report.firstRun).toMatchObject({
      availableHarnessIds: ["claude_code", "hermes"],
      recommendedHarnessId: "claude_code",
    });
  });

  it("is stable after an idempotent repair succeeds", async () => {
    let status = unhealthyStatus();
    const setup = vi.fn(async () => {
      status = healthyStatus();
      return setupResult(status);
    });
    const host = {
      status: vi.fn(async () => status),
      setup,
    } satisfies HarnessEnvironmentDoctorHost;
    const doctor = new HarnessDoctor(host);

    const first = await doctor.fix({ harnessId: "hermes", confirmed: true });
    const second = await doctor.fix({ harnessId: "hermes", confirmed: true });

    expect(first.executed).toBe(true);
    expect(first.after.healthy).toBe(true);
    expect(second.executed).toBe(false);
    expect(second.after.repairActions).toEqual([]);
    expect(setup).toHaveBeenCalledTimes(1);
  });

  it("prefers the existing local Hermes checkout during detection", async () => {
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
    sandbox: {
      strategy: "protected_required",
      mode: "protected",
      ready: true,
      profileIds: ["claude_code"],
      runtimeId: "apple_container",
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
