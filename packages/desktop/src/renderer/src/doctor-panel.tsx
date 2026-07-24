import type {
  ContainerRuntimeStatus,
  DoctorFixResult,
  DoctorReport,
  HarnessRequirementStatus,
} from "@swarmx/runtime";
import {
  ChevronRight,
  CircleCheck,
  Loader2,
  PanelRight,
  RefreshCw,
  Wrench,
  XCircle,
} from "lucide-react";
import { HarnessBrandIcon, type HarnessOption, harnessOption } from "./harness-presentation.js";
import { Badge, Button, cx } from "./ui-primitives.js";

export interface DoctorHarnessVersionState {
  status: "loading" | "loaded";
  version?: string;
}

export type DoctorPanelMode = "doctor" | "setup";

export function DoctorPanel({
  mode,
  report,
  loading,
  harnessOptions,
  harnessVersions,
  error,
  fixPending,
  fixRunning,
  fixResult,
  onRefresh,
  onRequestFix,
  onCancelFix,
  onConfirmFix,
  installingHarnessId,
  onInstallHarness,
  onRefreshHarnessVersion,
  onClose,
}: {
  mode: DoctorPanelMode;
  report: DoctorReport | null;
  loading: boolean;
  harnessOptions: HarnessOption[];
  harnessVersions: Record<string, DoctorHarnessVersionState>;
  error: string | null;
  fixPending: boolean;
  fixRunning: boolean;
  fixResult: DoctorFixResult | null;
  onRefresh: () => void;
  onRequestFix: () => void;
  onCancelFix: () => void;
  onConfirmFix: () => void;
  installingHarnessId: string | null;
  onInstallHarness: (harnessId: string) => void;
  onRefreshHarnessVersion: (harnessId: string) => void;
  onClose: () => void;
}) {
  const issues = (report?.issues ?? []).filter((issue) => issue.scope === "doctor");
  const visibleRepairActionIds = new Set(issues.flatMap((issue) => issue.repairActionId ?? []));
  const repairActions = (report?.repairActions ?? []).filter((action) =>
    visibleRepairActionIds.has(action.id),
  );
  const reportedHarnesses = new Map(
    (report?.environment.harnesses ?? []).map((harness) => [harness.harnessId, harness]),
  );
  const harnesses = harnessOptions.map(
    (harness) =>
      reportedHarnesses.get(harness.id) ?? {
        harnessId: harness.id,
        harnessLabel: harness.label,
        version: undefined,
      },
  );
  const requirements = report?.environment.requirements ?? [];
  const containerRuntimes = report?.environment.containerRuntimes ?? [];
  const setupLogs = fixResult?.setupResults.flatMap((result) => result.log) ?? [];
  const title = mode === "setup" ? "Setup" : "Doctor";
  const panelHealthy = Boolean(report && issues.length === 0);
  const summaryTitle = loading
    ? "Checking environment"
    : panelHealthy
      ? "Environment ready"
      : report
        ? issues.length + (issues.length === 1 ? " issue found" : " issues found")
        : "Status unavailable";
  const summaryCopy = panelHealthy
    ? "Harnesses are optional; install one only when you plan to use it."
    : mode === "setup"
      ? "Review the missing pieces, then confirm before SwarmX changes anything."
      : "Review diagnostics and the repair plan before applying fixes.";

  return (
    <aside
      className="runtime-right-panel doctor-panel"
      aria-label={mode === "setup" ? "Setup panel" : "Doctor panel"}
    >
      <div className="runtime-panel__header">
        <div>
          <span>Environment</span>
          <h2>
            {title}
            {report?.harnessId ? ` · ${report.harnessId}` : ""}
          </h2>
        </div>
        <div className="doctor-panel__header-actions">
          <Button
            variant="ghost"
            size="icon"
            onClick={onRefresh}
            disabled={loading || fixRunning}
            title="Refresh diagnostics"
            aria-label="Refresh diagnostics"
          >
            <RefreshCw aria-hidden="true" />
          </Button>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label={`Close ${title}`}>
            <PanelRight aria-hidden="true" />
          </Button>
        </div>
      </div>

      <section className={cx("doctor-summary", panelHealthy && "is-healthy")} aria-live="polite">
        <span className="doctor-summary__icon">
          {loading ? (
            <Loader2 aria-hidden="true" />
          ) : panelHealthy ? (
            <CircleCheck aria-hidden="true" />
          ) : (
            <Wrench aria-hidden="true" />
          )}
        </span>
        <div>
          <h3>{summaryTitle}</h3>
          <p>{summaryCopy}</p>
        </div>
      </section>

      {error && (
        <div className="doctor-notice doctor-notice--error" role="alert">
          <XCircle aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {fixResult?.executed && (
        <output
          className={cx(
            "doctor-notice",
            fixResult.after.healthy ? "doctor-notice--success" : "doctor-notice--error",
          )}
        >
          {fixResult.after.healthy ? (
            <CircleCheck aria-hidden="true" />
          ) : (
            <XCircle aria-hidden="true" />
          )}
          <span>
            {fixResult.after.healthy
              ? "Repairs completed. The environment is ready."
              : "Repairs completed, but some issues still need attention."}
          </span>
        </output>
      )}

      {!loading && report && repairActions.length > 0 && (
        <section className="doctor-section" aria-labelledby="doctor-repair-title">
          <div className="doctor-section__heading">
            <h3 id="doctor-repair-title">Repair plan</h3>
            <span>{repairActions.length}</span>
          </div>
          {fixPending ? (
            <div className="doctor-confirmation">
              <strong>
                Confirm {repairActions.length} {repairActions.length === 1 ? "repair" : "repairs"}
              </strong>
              <p>No changes are made until you confirm this plan.</p>
              <ul className="doctor-list">
                {repairActions.map((action) => (
                  <li key={action.id} className="doctor-action">
                    <span>{action.label}</span>
                    <Badge tone={action.risk === "admin" ? "danger" : "neutral"}>
                      {action.risk}
                    </Badge>
                  </li>
                ))}
              </ul>
              <div className="doctor-confirmation__actions">
                <Button variant="ghost" size="sm" onClick={onCancelFix} disabled={fixRunning}>
                  Cancel
                </Button>
                <Button size="sm" onClick={onConfirmFix} disabled={fixRunning}>
                  {fixRunning ? (
                    <Loader2 data-icon="inline-start" aria-hidden="true" />
                  ) : (
                    <Wrench data-icon="inline-start" aria-hidden="true" />
                  )}
                  Confirm {repairActions.length}
                </Button>
              </div>
            </div>
          ) : (
            <Button size="sm" onClick={onRequestFix}>
              <Wrench data-icon="inline-start" aria-hidden="true" />
              {mode === "setup" ? "Set up missing" : "Fix issues"}
            </Button>
          )}
        </section>
      )}

      {!loading && report && issues.length > 0 && (
        <section className="doctor-section" aria-labelledby="doctor-issues-title">
          <div className="doctor-section__heading">
            <h3 id="doctor-issues-title">Diagnostics</h3>
            <span>{issues.length}</span>
          </div>
          <ul className="doctor-list">
            {issues.map((issue) => (
              <li key={issue.id} className="doctor-issue">
                <XCircle aria-hidden="true" />
                <div>
                  <strong>{issue.targetId ?? issue.scope}</strong>
                  <span>{issue.message}</span>
                </div>
                <Badge tone={issue.severity === "error" ? "danger" : "neutral"}>
                  {issue.severity}
                </Badge>
              </li>
            ))}
          </ul>
          {issues.length > 0 && repairActions.length === 0 && (
            <p className="doctor-section__hint">These issues require manual review.</p>
          )}
        </section>
      )}

      <section className="doctor-section" aria-labelledby="doctor-harnesses-title">
        <div className="doctor-section__heading">
          <h3 id="doctor-harnesses-title">Harnesses</h3>
          <span>
            {
              harnesses.filter((harness) => {
                const state = harnessVersions[harness.harnessId];
                return state?.status === "loaded" && Boolean(state.version ?? harness.version);
              }).length
            }
            /{harnesses.length}
          </span>
        </div>
        <ul className="doctor-list">
          {harnesses.map((harness) => {
            const versionState = harnessVersions[harness.harnessId];
            const version = versionState?.version ?? harness.version;
            const versionLoading = !versionState || versionState.status === "loading";
            return (
              <li key={harness.harnessId} className="doctor-harness">
                <span className="doctor-harness__icon">
                  <HarnessBrandIcon
                    harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                  />
                </span>
                <div>
                  <strong>{harness.harnessLabel}</strong>
                </div>
                {versionLoading ? (
                  <output
                    className="badge doctor-harness__version is-loading"
                    aria-label={`Checking ${harness.harnessLabel} version`}
                  >
                    <Loader2 data-icon aria-hidden="true" />
                  </output>
                ) : version ? (
                  <button
                    type="button"
                    className="badge badge--active doctor-harness__version"
                    aria-label={`Check ${harness.harnessLabel} version again`}
                    title="Check version again"
                    onClick={() => onRefreshHarnessVersion(harness.harnessId)}
                  >
                    {version}
                  </button>
                ) : (
                  <Button
                    variant="secondary"
                    size="sm"
                    aria-label={`Install ${harness.harnessLabel}`}
                    disabled={Boolean(installingHarnessId)}
                    onClick={() => onInstallHarness(harness.harnessId)}
                  >
                    {installingHarnessId === harness.harnessId && (
                      <Loader2 data-icon="inline-start" aria-hidden="true" />
                    )}
                    Install
                  </Button>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {report && (
        <details className="doctor-advanced">
          <summary>
            <span>
              <strong>Advanced details</strong>
              <small>Runtime tools, PATH, and repair logs</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </summary>
          <div className="doctor-advanced__body">
            {requirements.length > 0 && (
              <section>
                <h4>Runtime tools</h4>
                <ul className="doctor-list">
                  {requirements.map((requirement) => (
                    <li key={requirement.id} className="doctor-diagnostic">
                      <div>
                        <strong>{requirement.label}</strong>
                        <span>
                          {[
                            requirement.command,
                            requirement.version,
                            requirement.path,
                            requirement.note,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </div>
                      <Badge tone={requirement.status === "ready" ? "active" : "danger"}>
                        {requirementStatusLabel(requirement.status)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            {containerRuntimes.length > 0 && (
              <section>
                <h4>Container runtime</h4>
                <ul className="doctor-list">
                  {containerRuntimes.map((runtime) => (
                    <li key={runtime.id} className="doctor-diagnostic">
                      <div>
                        <strong>{runtime.label}</strong>
                        <span>
                          {[runtime.command, runtime.version, runtime.path, runtime.note]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </div>
                      <Badge tone={runtime.status === "ready" ? "active" : "danger"}>
                        {containerRuntimeStatusLabel(runtime.status)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <section>
              <h4>Environment PATH</h4>
              <pre className="doctor-code">{report.environment.path}</pre>
            </section>
            {setupLogs.length > 0 && (
              <section>
                <h4>Repair log</h4>
                <pre className="doctor-code">{setupLogs.join("\n\n")}</pre>
              </section>
            )}
          </div>
        </details>
      )}
    </aside>
  );
}

export function requirementStatusLabel(status: HarnessRequirementStatus): string {
  switch (status) {
    case "ready":
      return "ready";
    case "missing":
      return "missing";
    case "unsupported":
      return "unsupported";
    case "failed":
      return "failed";
  }
}

function containerRuntimeStatusLabel(status: ContainerRuntimeStatus): string {
  switch (status) {
    case "ready":
      return "ready";
    case "missing":
      return "missing";
    case "service_stopped":
      return "service stopped";
    case "unsupported":
      return "unsupported";
    case "failed":
      return "failed";
  }
}
