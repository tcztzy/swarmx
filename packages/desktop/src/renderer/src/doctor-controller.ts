import type { DoctorFixResult, DoctorReport, HarnessEnvironmentStatus } from "@swarmx/runtime";
import { useCallback, useEffect, useRef, useState } from "react";
import type { SwarmxAPI } from "../../shared/desktop-api.js";
import { errorMessage } from "./text-utils.js";

export type DoctorControllerApi = Pick<
  SwarmxAPI,
  "getHarnessVersion" | "inspectDoctor" | "fixDoctor" | "setupHarnessEnvironment"
>;

export interface DoctorHarnessVersionState {
  status: "loading" | "loaded";
  version?: string;
}

export type DoctorPanelMode = "doctor" | "setup";

export type DoctorSlashCommand =
  | { kind: "doctor"; fix: boolean; harnessId?: string }
  | { kind: "setup"; fix: false; harnessId?: string }
  | { kind: "error"; message: string };

export interface DoctorControllerOptions {
  api: DoctorControllerApi;
  harnesses: readonly { id: string; label: string }[];
  runtimeVisible: boolean;
  publishEnvironment: (environment: HarnessEnvironmentStatus) => Promise<void> | void;
  refreshAfterRepair: () => Promise<void>;
}

type OpenDoctorOptions = { mode?: DoctorPanelMode; harnessId?: string; requestFix?: boolean };

const INITIAL_STATE = {
  panelOpen: false,
  panelMode: "doctor" as DoctorPanelMode,
  harnessId: null as string | null,
  report: null as DoctorReport | null,
  loading: false,
  harnessVersions: {} as Record<string, DoctorHarnessVersionState>,
  fixPending: false,
  fixRunning: false,
  fixResult: null as DoctorFixResult | null,
  installingHarnessId: null as string | null,
  error: null as string | null,
};
type DoctorControllerState = typeof INITIAL_STATE;

export function parseDoctorSlashCommand(value: string): DoctorSlashCommand | null {
  const tokens = value.trim().split(/\s+/);
  const command = tokens.shift();
  if (command !== "/doctor" && command !== "/setup") return null;

  const kind = command === "/doctor" ? "doctor" : "setup";
  let fix = false;
  let harnessId: string | undefined;
  while (tokens.length > 0) {
    const token = tokens.shift() as string;
    if (token === "--fix") {
      if (kind === "setup") {
        return { kind: "error", message: "Use /setup without --fix, then confirm repairs." };
      }
      fix = true;
      continue;
    }
    if (token === "--harness" || token.startsWith("--harness=")) {
      const separate = token === "--harness";
      const next = separate ? tokens.shift() : token.slice("--harness=".length);
      if (!next || (separate && next.startsWith("-"))) {
        return { kind: "error", message: "--harness requires a harness id." };
      }
      if (harnessId) return { kind: "error", message: "Specify only one harness id." };
      harnessId = next;
      continue;
    }
    if (token.startsWith("-")) {
      return { kind: "error", message: `Unknown ${kind} option: ${token}` };
    }
    if (harnessId) return { kind: "error", message: "Specify only one harness id." };
    harnessId = token;
  }

  return kind === "doctor"
    ? { kind, fix, ...(harnessId ? { harnessId } : {}) }
    : { kind, fix: false, ...(harnessId ? { harnessId } : {}) };
}

export function useDoctorController({
  api,
  harnesses,
  runtimeVisible,
  publishEnvironment,
  refreshAfterRepair,
}: DoctorControllerOptions) {
  const [state, setState] = useState(INITIAL_STATE);
  const stateRef = useRef(state);
  const mounted = useRef(true);
  const viewGeneration = useRef(0);
  const versionGenerations = useRef(new Map<string, number>());
  const versionsStarted = useRef(false);
  const runtimeStarted = useRef(false);
  const fixOperation = useRef<Promise<void> | null>(null);
  const installOperation = useRef<Promise<void> | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const patchState = useCallback((patch: Partial<DoctorControllerState>) => {
    if (!mounted.current) return;
    stateRef.current = { ...stateRef.current, ...patch };
    setState(stateRef.current);
  }, []);

  const ownsView = useCallback(
    (generation: number) => mounted.current && viewGeneration.current === generation,
    [],
  );

  const patchHarnessVersion = useCallback(
    (harnessId: string, version: DoctorHarnessVersionState, error?: string) =>
      patchState({
        harnessVersions: { ...stateRef.current.harnessVersions, [harnessId]: version },
        ...(error ? { error } : {}),
      }),
    [patchState],
  );

  const refreshVersion = useCallback(
    async (harnessId: string, refresh = true) => {
      const generations = versionGenerations.current;
      const generation = (generations.get(harnessId) ?? 0) + 1;
      const previousVersion = stateRef.current.harnessVersions[harnessId]?.version;
      generations.set(harnessId, generation);
      patchHarnessVersion(harnessId, { status: "loading", version: previousVersion });
      try {
        const result = await api.getHarnessVersion({
          harnessId,
          ...(refresh ? { refresh: true } : {}),
        });
        if (!mounted.current || generations.get(harnessId) !== generation) return;
        patchHarnessVersion(harnessId, { status: "loaded", version: result.version });
      } catch (error) {
        if (!mounted.current || generations.get(harnessId) !== generation) return;
        patchHarnessVersion(
          harnessId,
          { status: "loaded", version: previousVersion },
          errorMessage(error),
        );
      }
    },
    [api, patchHarnessVersion],
  );

  const ensureVersions = useCallback(() => {
    if (versionsStarted.current) return;
    versionsStarted.current = true;
    for (const harness of harnesses) void refreshVersion(harness.id, false);
  }, [harnesses, refreshVersion]);

  const inspect = useCallback(
    async (
      generation: number,
      harnessId: string | null,
      requestFix: boolean,
      omitGlobalArgument: boolean,
    ) => {
      try {
        const report = harnessId
          ? await api.inspectDoctor({ harnessId })
          : omitGlobalArgument
            ? await api.inspectDoctor()
            : await api.inspectDoctor({});
        if (!ownsView(generation)) return;
        patchState({ report, fixPending: requestFix && report.repairActions.length > 0 });
        if (!harnessId) await publishEnvironment(report.environment);
      } catch (error) {
        if (ownsView(generation)) patchState({ error: errorMessage(error) });
      } finally {
        if (ownsView(generation)) patchState({ loading: false });
      }
    },
    [api, ownsView, patchState, publishEnvironment],
  );

  const open = useCallback(
    (options: OpenDoctorOptions = {}) => {
      const { mode = "doctor", harnessId, requestFix = false } = options;
      const generation = viewGeneration.current + 1;
      viewGeneration.current = generation;
      patchState({
        panelOpen: true,
        panelMode: mode,
        harnessId: harnessId ?? null,
        loading: true,
        fixPending: false,
        fixResult: null,
        error: null,
      });
      ensureVersions();
      return inspect(generation, harnessId ?? null, requestFix, false);
    },
    [ensureVersions, inspect, patchState],
  );

  const close = useCallback(() => {
    viewGeneration.current += 1;
    patchState({ panelOpen: false });
  }, [patchState]);

  const refreshRuntime = useCallback(
    async (refreshVersions = false) => {
      const generation = viewGeneration.current + 1;
      viewGeneration.current = generation;
      patchState({ harnessId: null, loading: true, fixPending: false, error: null });
      if (refreshVersions) {
        versionsStarted.current = true;
        await Promise.all(harnesses.map((harness) => refreshVersion(harness.id)));
      } else {
        ensureVersions();
      }
      if (!ownsView(generation)) return;
      await inspect(generation, null, false, true);
    },
    [ensureVersions, harnesses, inspect, ownsView, patchState, refreshVersion],
  );

  useEffect(() => {
    if (!runtimeVisible) runtimeStarted.current = false;
    else if (!runtimeStarted.current) {
      runtimeStarted.current = true;
      void refreshRuntime();
    }
  }, [refreshRuntime, runtimeVisible]);

  const requestFix = useCallback(() => patchState({ fixPending: true }), [patchState]);
  const cancelFix = useCallback(() => patchState({ fixPending: false }), [patchState]);

  const confirmFix = useCallback(
    () =>
      serializeOperation(fixOperation, async () => {
        const current = stateRef.current;
        if (!current.report?.repairActions.length) return;
        const generation = viewGeneration.current;
        const harnessId = current.harnessId;
        patchState({ fixRunning: true, error: null });
        try {
          const result = await api.fixDoctor({
            ...(harnessId ? { harnessId } : {}),
            confirmed: true,
          });
          if (mounted.current && !harnessId) await publishEnvironment(result.after.environment);
          if (mounted.current) await refreshAfterRepair();
          if (ownsView(generation))
            patchState({ report: result.after, fixResult: result, fixPending: false });
        } catch (error) {
          if (ownsView(generation)) patchState({ error: errorMessage(error) });
        } finally {
          if (mounted.current) patchState({ fixRunning: false });
        }
      }),
    [api, ownsView, patchState, publishEnvironment, refreshAfterRepair],
  );

  const installHarness = useCallback(
    (harnessToolId: string) =>
      serializeOperation(installOperation, async () => {
        const generation = viewGeneration.current;
        const reportHarnessId = stateRef.current.harnessId;
        patchState({ installingHarnessId: harnessToolId, error: null });
        try {
          const result = await api.setupHarnessEnvironment({ harnessToolId });
          if (!mounted.current) return;
          await publishEnvironment(result.status);
          if (ownsView(generation)) {
            const report = reportHarnessId
              ? await api.inspectDoctor({ harnessId: reportHarnessId })
              : await api.inspectDoctor({});
            if (ownsView(generation)) patchState({ report });
          }
          await refreshVersion(harnessToolId);
          if (!result.success && ownsView(generation)) {
            const label = harnesses.find((harness) => harness.id === harnessToolId)?.label;
            patchState({ error: result.error ?? `Could not install ${label ?? harnessToolId}.` });
          }
        } catch (error) {
          if (ownsView(generation)) patchState({ error: errorMessage(error) });
        } finally {
          if (mounted.current) patchState({ installingHarnessId: null });
        }
      }),
    [api, harnesses, ownsView, patchState, publishEnvironment, refreshVersion],
  );

  return {
    ...state,
    open,
    close,
    refreshRuntime,
    refreshHarnessVersion: refreshVersion,
    requestFix,
    cancelFix,
    confirmFix,
    installHarness,
  };
}

function serializeOperation(ref: { current: Promise<void> | null }, start: () => Promise<void>) {
  if (ref.current) return ref.current;
  const operation = start();
  ref.current = operation;
  const release = () => (ref.current = null);
  void operation.then(release, release);
  return operation;
}
