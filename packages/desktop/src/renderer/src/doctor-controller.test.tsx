/** @vitest-environment jsdom */

import type {
  DoctorFixResult,
  DoctorReport,
  HarnessEnvironmentSetupResult,
  HarnessEnvironmentStatus,
} from "@swarmx/runtime";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { type PropsWithChildren, StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type DoctorControllerApi,
  parseDoctorSlashCommand,
  useDoctorController,
} from "./doctor-controller.js";

const HARNESSES = [
  { id: "codex", label: "Codex" },
  { id: "kimi", label: "Kimi Code" },
] as const;

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("parseDoctorSlashCommand", () => {
  it.each(["", "hello", "/Doctor", "/setupper"])("ignores %j", (input) => {
    expect(parseDoctorSlashCommand(input)).toBeNull();
  });

  it.each([
    [" /doctor ", { kind: "doctor", fix: false }],
    ["/doctor --fix --fix", { kind: "doctor", fix: true }],
    ["/doctor codex", { kind: "doctor", fix: false, harnessId: "codex" }],
    ["/doctor --harness codex", { kind: "doctor", fix: false, harnessId: "codex" }],
    ["/doctor --harness=codex", { kind: "doctor", fix: false, harnessId: "codex" }],
    ["/setup", { kind: "setup", fix: false }],
    ["/setup kimi", { kind: "setup", fix: false, harnessId: "kimi" }],
    ["/setup --harness -x", { kind: "error", message: "--harness requires a harness id." }],
    ["/setup --harness=-x", { kind: "setup", fix: false, harnessId: "-x" }],
  ])("parses %s", (input, expected) => {
    expect(parseDoctorSlashCommand(input)).toEqual(expected);
  });

  it.each([
    ["/setup --fix", "Use /setup without --fix, then confirm repairs."],
    ["/doctor --harness", "--harness requires a harness id."],
    ["/doctor --harness=", "--harness requires a harness id."],
    ["/doctor codex --harness kimi", "Specify only one harness id."],
    ["/doctor codex kimi", "Specify only one harness id."],
    ["/doctor --wat", "Unknown doctor option: --wat"],
    ["/setup --wat", "Unknown setup option: --wat"],
  ])("rejects %s", (input, message) => {
    expect(parseDoctorSlashCommand(input)).toEqual({ kind: "error", message });
  });
});

describe("useDoctorController", () => {
  it("opens globally, checks versions once, and keeps scoped reports out of the global cache", async () => {
    const global = report("global", undefined, true);
    const scoped = report("scoped", "codex");
    const api = createApi({
      inspectDoctor: vi.fn().mockResolvedValueOnce(global).mockResolvedValueOnce(scoped),
    });
    const fixture = renderController(api);

    await act(() => fixture.result.current.open({ requestFix: true }));
    expect(fixture.result.current).toMatchObject({
      panelOpen: true,
      panelMode: "doctor",
      harnessId: null,
      report: global,
      loading: false,
      fixPending: true,
      error: null,
    });
    expect(api.inspectDoctor).toHaveBeenNthCalledWith(1, {});
    expect(api.getHarnessVersion).toHaveBeenCalledTimes(2);
    expect(fixture.publishEnvironment).toHaveBeenCalledWith(global.environment);

    act(() => fixture.result.current.close());
    await act(() => fixture.result.current.open({ mode: "setup", harnessId: "codex" }));
    expect(fixture.result.current).toMatchObject({
      panelOpen: true,
      panelMode: "setup",
      harnessId: "codex",
      report: scoped,
      fixPending: false,
    });
    expect(api.inspectDoctor).toHaveBeenNthCalledWith(2, { harnessId: "codex" });
    expect(api.getHarnessVersion).toHaveBeenCalledTimes(2);
    expect(fixture.publishEnvironment).toHaveBeenCalledOnce();
  });

  it("lets only the latest inspection own report, error, loading, and cache publication", async () => {
    const codex = deferred<DoctorReport>();
    const kimi = deferred<DoctorReport>();
    const api = createApi({
      inspectDoctor: vi.fn().mockReturnValueOnce(codex.promise).mockReturnValueOnce(kimi.promise),
    });
    const fixture = renderController(api, { harnesses: [] });

    let codexRequest!: Promise<void>;
    let kimiRequest!: Promise<void>;
    act(() => {
      codexRequest = fixture.result.current.open({ harnessId: "codex" });
      kimiRequest = fixture.result.current.open({ harnessId: "kimi" });
    });
    await act(() => codex.reject(new Error("old failure")));
    expect(fixture.result.current).toMatchObject({
      harnessId: "kimi",
      report: null,
      loading: true,
      error: null,
    });

    const latest = report("kimi", "kimi");
    await act(() => kimi.resolve(latest));
    await Promise.all([codexRequest, kimiRequest]);
    expect(fixture.result.current).toMatchObject({ report: latest, loading: false, error: null });
    expect(fixture.publishEnvironment).not.toHaveBeenCalled();
  });

  it("invalidates a pending inspection when closed or unmounted", async () => {
    const pending = deferred<DoctorReport>();
    const api = createApi({ inspectDoctor: vi.fn(() => pending.promise) });
    const fixture = renderController(api, { harnesses: [] });

    let request!: Promise<void>;
    act(() => {
      request = fixture.result.current.open();
      fixture.result.current.close();
    });
    await act(() => pending.resolve(report("late")));
    await request;
    expect(fixture.result.current.panelOpen).toBe(false);
    expect(fixture.result.current.report).toBeNull();
    expect(fixture.publishEnvironment).not.toHaveBeenCalled();

    const unmounted = deferred<DoctorReport>();
    api.inspectDoctor.mockReturnValueOnce(unmounted.promise);
    act(() => {
      void fixture.result.current.open();
    });
    const closeAfterUnmount = fixture.result.current.close;
    fixture.unmount();
    act(() => closeAfterUnmount());
    await act(() => unmounted.reject(new Error("after unmount")));
    expect(fixture.publishEnvironment).not.toHaveBeenCalled();
  });

  it("records only a current inspection failure", async () => {
    const api = createApi({
      inspectDoctor: vi.fn().mockRejectedValue(new Error("inspect failed")),
    });
    const fixture = renderController(api, { harnesses: [] });
    await act(() => fixture.result.current.open());
    expect(fixture.result.current).toMatchObject({ loading: false, error: "inspect failed" });
  });

  it("deduplicates Runtime inspection across StrictMode replay and refreshes on a new activation", async () => {
    const first = deferred<DoctorReport>();
    const api = createApi({
      inspectDoctor: vi
        .fn()
        .mockReturnValueOnce(first.promise)
        .mockResolvedValueOnce(report("next")),
    });
    const publishEnvironment = vi.fn(async () => undefined);
    const wrapper = ({ children }: PropsWithChildren) => <StrictMode>{children}</StrictMode>;
    const fixture = renderHook(
      ({ runtimeVisible }) =>
        useDoctorController({
          api,
          harnesses: [],
          runtimeVisible,
          publishEnvironment,
          refreshAfterRepair: vi.fn(async () => undefined),
        }),
      { initialProps: { runtimeVisible: true }, wrapper },
    );

    await waitFor(() => expect(api.inspectDoctor).toHaveBeenCalledOnce());
    expect(publishEnvironment).not.toHaveBeenCalled();
    await act(() => first.resolve(report("strict")));
    await waitFor(() => expect(publishEnvironment).toHaveBeenCalledOnce());
    fixture.rerender({ runtimeVisible: false });
    fixture.rerender({ runtimeVisible: true });
    await waitFor(() => expect(api.inspectDoctor).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(publishEnvironment).toHaveBeenCalledTimes(2));
  });

  it("keeps the latest version result per Harness and retains the previous version on failure", async () => {
    const oldCheck = deferred<{ harnessId: string; version?: string }>();
    const refresh = deferred<{ harnessId: string; version?: string }>();
    const api = createApi({
      getHarnessVersion: vi
        .fn()
        .mockReturnValueOnce(oldCheck.promise)
        .mockReturnValueOnce(refresh.promise)
        .mockRejectedValueOnce(new Error("current failure")),
    });
    const fixture = renderController(api, { harnesses: [] });

    let oldRequest!: Promise<void>;
    let refreshRequest!: Promise<void>;
    act(() => {
      oldRequest = fixture.result.current.refreshHarnessVersion("codex", false);
      refreshRequest = fixture.result.current.refreshHarnessVersion("codex");
    });
    await act(() => refresh.resolve({ harnessId: "codex", version: "2.0.0" }));
    await act(() => oldCheck.resolve({ harnessId: "codex", version: "1.0.0" }));
    await Promise.all([oldRequest, refreshRequest]);
    expect(fixture.result.current.harnessVersions.codex).toEqual({
      status: "loaded",
      version: "2.0.0",
    });

    await act(() => fixture.result.current.refreshHarnessVersion("codex"));
    expect(fixture.result.current.harnessVersions.codex).toEqual({
      status: "loaded",
      version: "2.0.0",
    });
    expect(fixture.result.current.error).toBe("current failure");
  });

  it("ignores a stale version failure", async () => {
    const stale = deferred<{ harnessId: string; version?: string }>();
    const api = createApi({
      getHarnessVersion: vi
        .fn()
        .mockReturnValueOnce(stale.promise)
        .mockResolvedValueOnce({ harnessId: "codex", version: "current" }),
    });
    const fixture = renderController(api, { harnesses: [] });
    act(() => {
      void fixture.result.current.refreshHarnessVersion("codex", false);
    });
    await act(() => fixture.result.current.refreshHarnessVersion("codex"));
    await act(() => stale.reject(new Error("stale version failure")));
    expect(fixture.result.current.harnessVersions.codex?.version).toBe("current");
    expect(fixture.result.current.error).toBeNull();
  });

  it("does not start an obsolete Runtime inspection after refreshed versions settle", async () => {
    const codex = deferred<{ harnessId: string; version?: string }>();
    const kimi = deferred<{ harnessId: string; version?: string }>();
    const api = createApi({
      getHarnessVersion: vi
        .fn()
        .mockReturnValueOnce(codex.promise)
        .mockReturnValueOnce(kimi.promise),
      inspectDoctor: vi.fn(async ({ harnessId } = {}) => report(harnessId ?? "global", harnessId)),
    });
    const fixture = renderController(api);
    act(() => {
      void fixture.result.current.refreshRuntime(true);
    });
    await act(() => fixture.result.current.open({ harnessId: "codex" }));
    await act(() => codex.resolve({ harnessId: "codex", version: "2" }));
    await act(() => kimi.resolve({ harnessId: "kimi", version: "2" }));
    expect(api.inspectDoctor).toHaveBeenCalledOnce();
    expect(api.inspectDoctor).toHaveBeenCalledWith({ harnessId: "codex" });
  });

  it("runs a confirmed global repair once and refreshes dependent projections", async () => {
    const before = report("before", undefined, true);
    const after = report("after");
    const pending = deferred<DoctorFixResult>();
    const api = createApi({
      inspectDoctor: vi.fn(async () => before),
      fixDoctor: vi.fn(() => pending.promise),
    });
    const fixture = renderController(api, { harnesses: [] });
    await act(() => fixture.result.current.open({ requestFix: true }));

    let first!: Promise<void>;
    let second!: Promise<void>;
    act(() => {
      first = fixture.result.current.confirmFix();
      second = fixture.result.current.confirmFix();
    });
    expect(first).toBe(second);
    expect(api.fixDoctor).toHaveBeenCalledOnce();
    expect(api.fixDoctor).toHaveBeenCalledWith({ confirmed: true });
    expect(fixture.result.current.fixRunning).toBe(true);

    await act(() => pending.resolve(fixResult(before, after)));
    await first;
    expect(fixture.result.current).toMatchObject({
      report: after,
      fixResult: fixResult(before, after),
      fixPending: false,
      fixRunning: false,
    });
    expect(fixture.publishEnvironment).toHaveBeenLastCalledWith(after.environment);
    expect(fixture.refreshAfterRepair).toHaveBeenCalledOnce();

    act(() => {
      fixture.result.current.requestFix();
      fixture.result.current.cancelFix();
    });
    expect(fixture.result.current.fixPending).toBe(false);
  });

  it("does not project a completed scoped repair into a newer view", async () => {
    const before = report("before", "codex", true);
    const after = report("after", "codex");
    const repair = deferred<DoctorFixResult>();
    const api = createApi({
      inspectDoctor: vi
        .fn()
        .mockResolvedValueOnce(before)
        .mockResolvedValueOnce(report("kimi", "kimi")),
      fixDoctor: vi.fn(() => repair.promise),
    });
    const fixture = renderController(api, { harnesses: [] });
    await act(() => fixture.result.current.open({ harnessId: "codex", requestFix: true }));

    let operation!: Promise<void>;
    act(() => {
      operation = fixture.result.current.confirmFix();
    });
    await act(() => fixture.result.current.open({ harnessId: "kimi" }));
    await act(() => repair.resolve(fixResult(before, after)));
    await operation;
    expect(fixture.result.current.report?.harnessId).toBe("kimi");
    expect(fixture.result.current.fixResult).toBeNull();
    expect(fixture.publishEnvironment).not.toHaveBeenCalled();
    expect(fixture.refreshAfterRepair).toHaveBeenCalledOnce();
  });

  it("does not publish or update state when a repair settles after unmount", async () => {
    const before = report("before", undefined, true);
    const repair = deferred<DoctorFixResult>();
    const api = createApi({
      inspectDoctor: vi.fn(async () => before),
      fixDoctor: vi.fn(() => repair.promise),
    });
    const fixture = renderController(api, { harnesses: [] });
    await act(() => fixture.result.current.open());
    let operation!: Promise<void>;
    act(() => {
      operation = fixture.result.current.confirmFix();
    });
    fixture.unmount();
    await act(() => repair.resolve(fixResult(before, report("after"))));
    await operation;
    expect(fixture.publishEnvironment).toHaveBeenCalledOnce();
    expect(fixture.refreshAfterRepair).not.toHaveBeenCalled();
  });

  it("serializes Harness installs, publishes their status, and refreshes the installed version", async () => {
    const setup = deferred<HarnessEnvironmentSetupResult>();
    const installed = environment("installed");
    const api = createApi({
      setupHarnessEnvironment: vi.fn(() => setup.promise),
      inspectDoctor: vi.fn(async () => report("installed")),
      getHarnessVersion: vi.fn(async ({ harnessId }) => ({ harnessId, version: "3.0.0" })),
    });
    const fixture = renderController(api, { harnesses: HARNESSES });

    let first!: Promise<void>;
    let second!: Promise<void>;
    act(() => {
      first = fixture.result.current.installHarness("codex");
      second = fixture.result.current.installHarness("kimi");
    });
    expect(first).toBe(second);
    expect(api.setupHarnessEnvironment).toHaveBeenCalledOnce();
    expect(api.setupHarnessEnvironment).toHaveBeenCalledWith({ harnessToolId: "codex" });
    await act(() => setup.resolve(setupResult(installed)));
    await first;

    expect(fixture.publishEnvironment).toHaveBeenCalledWith(installed);
    expect(api.inspectDoctor).toHaveBeenCalledWith({});
    expect(api.getHarnessVersion).toHaveBeenCalledWith({ harnessId: "codex", refresh: true });
    expect(fixture.result.current).toMatchObject({
      installingHarnessId: null,
      report: report("installed"),
    });
  });

  it("surfaces setup failures with a bounded label and unlocks a retry", async () => {
    const api = createApi({
      setupHarnessEnvironment: vi
        .fn()
        .mockResolvedValueOnce(setupResult(environment("failed"), false))
        .mockRejectedValueOnce(new Error("retry failed")),
    });
    const fixture = renderController(api, { harnesses: HARNESSES });

    await act(() => fixture.result.current.installHarness("codex"));
    expect(fixture.result.current.error).toBe("Could not install Codex.");
    await act(() => fixture.result.current.installHarness("codex"));
    expect(api.setupHarnessEnvironment).toHaveBeenCalledTimes(2);
    expect(fixture.result.current.error).toBe("retry failed");
  });

  it("does not continue an install that settles after unmount", async () => {
    const setup = deferred<HarnessEnvironmentSetupResult>();
    const api = createApi({ setupHarnessEnvironment: vi.fn(() => setup.promise) });
    const fixture = renderController(api);
    let operation!: Promise<void>;
    act(() => {
      operation = fixture.result.current.installHarness("codex");
    });
    fixture.unmount();
    await act(() => setup.resolve(setupResult(environment("late"))));
    await operation;
    expect(fixture.publishEnvironment).not.toHaveBeenCalled();
    expect(api.inspectDoctor).not.toHaveBeenCalled();
    expect(api.getHarnessVersion).not.toHaveBeenCalled();
  });

  it("keeps a scoped install follow-up out of a newer view", async () => {
    const followup = deferred<DoctorReport>();
    const api = createApi({
      inspectDoctor: vi
        .fn()
        .mockResolvedValueOnce(report("codex", "codex"))
        .mockReturnValueOnce(followup.promise)
        .mockResolvedValueOnce(report("kimi", "kimi")),
    });
    const fixture = renderController(api, { harnesses: [] });
    await act(() => fixture.result.current.open({ harnessId: "codex" }));
    let installation!: Promise<void>;
    act(() => {
      installation = fixture.result.current.installHarness("codex");
    });
    await waitFor(() => expect(api.inspectDoctor).toHaveBeenCalledTimes(2));
    await act(() => fixture.result.current.open({ harnessId: "kimi" }));
    await act(() => followup.resolve(report("old followup", "codex")));
    await installation;
    expect(fixture.result.current.report?.harnessId).toBe("kimi");
  });

  it("skips repair without actions and records the current repair failure", async () => {
    const api = createApi({
      inspectDoctor: vi
        .fn()
        .mockResolvedValueOnce(report("empty"))
        .mockResolvedValueOnce(report("repairable", "codex", true)),
      fixDoctor: vi.fn().mockRejectedValueOnce(new Error("repair failed")),
    });
    const fixture = renderController(api, { harnesses: [] });

    await act(() => fixture.result.current.open());
    await act(() => fixture.result.current.confirmFix());
    expect(api.fixDoctor).not.toHaveBeenCalled();

    await act(() => fixture.result.current.open({ harnessId: "codex" }));
    await act(() => fixture.result.current.confirmFix());
    expect(api.fixDoctor).toHaveBeenCalledWith({ harnessId: "codex", confirmed: true });
    expect(fixture.result.current.error).toBe("repair failed");
    expect(fixture.result.current.fixRunning).toBe(false);
  });
});

interface MockDoctorApi extends DoctorControllerApi {
  getHarnessVersion: ReturnType<typeof vi.fn>;
  inspectDoctor: ReturnType<typeof vi.fn>;
  fixDoctor: ReturnType<typeof vi.fn>;
  setupHarnessEnvironment: ReturnType<typeof vi.fn>;
}

function createApi(overrides: Partial<MockDoctorApi> = {}): MockDoctorApi {
  const defaultReport = report("default");
  return {
    getHarnessVersion: vi.fn(async ({ harnessId }) => ({ harnessId, version: "1.0.0" })),
    inspectDoctor: vi.fn(async () => defaultReport),
    fixDoctor: vi.fn(async () => fixResult(defaultReport, defaultReport)),
    setupHarnessEnvironment: vi.fn(async () => setupResult(defaultReport.environment)),
    ...overrides,
  } as MockDoctorApi;
}

function renderController(
  api: MockDoctorApi,
  options: { harnesses?: readonly { id: string; label: string }[] } = {},
) {
  const publishEnvironment = vi.fn(async () => undefined);
  const refreshAfterRepair = vi.fn(async () => undefined);
  const hook = renderHook(() =>
    useDoctorController({
      api,
      harnesses: options.harnesses ?? HARNESSES,
      runtimeVisible: false,
      publishEnvironment,
      refreshAfterRepair,
    }),
  );
  return { ...hook, publishEnvironment, refreshAfterRepair };
}

function environment(tag: string): HarnessEnvironmentStatus {
  return {
    checkedAt: `2026-08-13T00:00:00.000Z#${tag}`,
    path: `/runtime/${tag}`,
    ready: true,
    setupAvailable: false,
    containerRuntimes: [],
    protection: { mode: "native", ready: true, requiredHarnessIds: [] },
    requirements: [],
    harnesses: [],
  };
}

function report(tag: string, harnessId?: string, repairable = false): DoctorReport {
  const status = environment(tag);
  return {
    checkedAt: status.checkedAt,
    healthy: !repairable,
    ...(harnessId ? { harnessId } : {}),
    summary: {
      readyHarnesses: 0,
      totalHarnesses: 0,
      issueCount: repairable ? 1 : 0,
      fixableCount: repairable ? 1 : 0,
    },
    issues: repairable
      ? [
          {
            id: `issue:${tag}`,
            severity: "error",
            scope: "doctor",
            message: `Issue ${tag}`,
            repairActionId: `repair:${tag}`,
          },
        ]
      : [],
    repairActions: repairable
      ? [
          {
            id: `repair:${tag}`,
            label: `Repair ${tag}`,
            risk: "install",
            request: harnessId ? { harnessId } : {},
          },
        ]
      : [],
    environment: status,
  };
}

function fixResult(before: DoctorReport, after: DoctorReport): DoctorFixResult {
  return {
    executed: true,
    before,
    plan: { actions: before.repairActions, requiresConfirmation: true, requiresAdmin: false },
    setupResults: [],
    after,
  };
}

function setupResult(
  status: HarnessEnvironmentStatus,
  success = true,
): HarnessEnvironmentSetupResult {
  return {
    success,
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}
