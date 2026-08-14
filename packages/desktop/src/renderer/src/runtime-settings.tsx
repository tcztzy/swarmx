import type {
  ContainerRuntimeStatus,
  DoctorFixResult,
  DoctorReport,
  HarnessEnvironmentStatus,
  HarnessRequirementStatus,
} from "@swarmx/runtime";
import { CircleCheck, Download, Loader2, RefreshCw, Wrench, XCircle } from "lucide-react";
import { useState } from "react";
import type { DesktopTaskRuntimeListResult } from "../../shared/desktop-api.js";
import type { DoctorHarnessVersionState } from "./doctor-controller.js";
import { requirementStatusLabel } from "./doctor-panel.js";
import { HarnessBrandIcon, harnessOption } from "./harness-presentation.js";
import { errorMessage, formatTimestamp } from "./text-utils.js";
import { Badge, Button, badgeVariants, cx, doctorNoticeVariants } from "./ui-primitives.js";

const RUNTIME_STATUS_CLASS = {
  ready: "is-ready",
  missing: "is-missing",
  unsupported: "is-unsupported",
  failed: "is-failed",
  service_stopped: "is-service_stopped",
} satisfies Record<ContainerRuntimeStatus | HarnessRequirementStatus, string>;

export function RuntimeSettings({
  environment,
  loading,
  error,
  doctorReport,
  doctorLoading,
  doctorError,
  harnessVersions,
  fixPending,
  fixRunning,
  fixResult,
  installingHarnessId,
  taskRuntime,
  taskRuntimeLoading,
  taskRuntimeError,
  onRefresh,
  onSetupContainer,
  onInstallHarness,
  onRefreshHarnessVersion,
  onRequestFix,
  onCancelFix,
  onConfirmFix,
  onRefreshTasks,
  onCancelTask,
  onDecideApproval,
}: {
  environment?: HarnessEnvironmentStatus;
  loading: boolean;
  error: unknown;
  doctorReport: DoctorReport | null;
  doctorLoading: boolean;
  doctorError: string | null;
  harnessVersions: Record<string, DoctorHarnessVersionState>;
  fixPending: boolean;
  fixRunning: boolean;
  fixResult: DoctorFixResult | null;
  installingHarnessId: string | null;
  taskRuntime?: DesktopTaskRuntimeListResult;
  taskRuntimeLoading: boolean;
  taskRuntimeError: unknown;
  onRefresh: () => Promise<void>;
  onSetupContainer: (containerRuntimeId: string) => Promise<void>;
  onInstallHarness: (harnessId: string) => Promise<void>;
  onRefreshHarnessVersion: (harnessId: string) => void;
  onRequestFix: () => void;
  onCancelFix: () => void;
  onConfirmFix: () => void;
  onRefreshTasks: () => Promise<void>;
  onCancelTask: (workItemId: string) => Promise<void>;
  onDecideApproval: (
    approvalId: string,
    status: "approved" | "rejected" | "waived",
  ) => Promise<void>;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const nodeRuntime = environment?.requirements.find((requirement) => requirement.id === "node");
  const harnesses = environment?.harnesses ?? [];
  const doctorIssues = doctorReport?.issues ?? [];
  const repairActions = doctorReport?.repairActions ?? [];
  const doctorHealthy = Boolean(doctorReport?.healthy && doctorIssues.length === 0);
  const repairLogs = fixResult?.setupResults.flatMap((result) => result.log) ?? [];
  const activeWorkItemIds = new Set(taskRuntime?.activeWorkItemIds ?? []);
  const pendingApprovals = new Map(
    (taskRuntime?.approvals ?? [])
      .filter((approval) => approval.status === "requested")
      .map((approval) => [approval.approvalId, approval]),
  );
  const run = async (id: string, action: () => Promise<void>) => {
    setBusyId(id);
    setActionError(null);
    try {
      await action();
    } catch (error) {
      setActionError(errorMessage(error));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <section
      className="settings-workspace [width:100%] [height:100%] [min-width:0] [min-height:0] [overflow:hidden] [display:grid] [grid-template-rows:minmax(0,_1fr)] [color:var(--foreground)] [background:var(--background)]"
      aria-label="Runtime settings"
    >
      <div className="settings-workspace__body max-680:[display:block] [min-width:0] [min-height:0] [overflow:hidden] [display:block] [&.custom-agent-layout]:[height:100%] [&.custom-agent-layout]:[display:grid] [&.custom-agent-layout]:[grid-template-columns:260px_minmax(0,_1fr)]">
        <div className="settings-workspace__content [min-width:0] [min-height:0] [overflow-y:auto] [height:100%] [padding:48px_clamp(32px,_6vw,_84px)_64px] [background:var(--background)] max-680:[padding:24px_18px_40px]">
          <section className="runtime-settings [width:min(100%,_1120px)] [margin:0_auto]">
            <div
              className={String.raw`settings-content-heading [&_>_span]:[min-width:0] [&_h2]:[margin:0] [&_p]:[margin:0] [&_small]:[margin:0] [&_small]:[display:block] [&_small]:[margin-bottom:5px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[font-weight:560] [&_small]:[letter-spacing:0.035em] [&_small]:[text-transform:uppercase] [&_h2]:[font-size:28px] [&_h2]:[font-weight:620] [&_h2]:[letter-spacing:-0.025em] [&_h2]:[line-height:1.25] [&_p]:[max-width:590px] [&_p]:[margin-top:7px] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:12.5px] [&_p]:[line-height:1.5] [&_>_div]:[flex:0_0_auto] [&_>_div]:[display:flex] [&_>_div]:[align-items:center] [&_>_div]:[gap:7px] [&_>_div]:[flex-wrap:wrap] max-680:[align-items:flex-start] max-680:[flex-direction:column] max-680:[gap:14px] max-680:[&_>_div]:[width:100%] max-680:[&_>_div_button]:[flex:1_1_0] [&.permission-settings\_\_heading]:[margin-bottom:20px] [min-width:0] [margin-bottom:24px] [display:flex] [align-items:flex-end] [justify-content:space-between] [gap:20px]`}
            >
              <span>
                <small>Local environment</small>
                <h2>Runtime</h2>
                <p>
                  Node.js is the shared baseline. Harness tools and environment diagnostics are
                  managed here independently from each Custom Agent recipe.
                </p>
              </span>
              <div>
                <button
                  type="button"
                  className="settings-primary-action [color:var(--primary-foreground)] [background:var(--primary)] [border-color:var(--primary)] [min-height:32px] [padding:0_10px] [display:inline-flex] [align-items:center] [justify-content:center] [gap:6px] [border:1px_solid_var(--border)] [border-radius:7px] [font-size:11.5px] [font-weight:580] [cursor:pointer] [&_svg]:[width:13px] [&_svg]:[height:13px]"
                  disabled={busyId !== null}
                  onClick={() => void run("refresh", onRefresh)}
                >
                  <RefreshCw
                    className={
                      busyId === "refresh"
                        ? "is-spinning [animation:spin_0.9s_linear_infinite]"
                        : undefined
                    }
                    aria-hidden="true"
                  />
                  Refresh
                </button>
              </div>
            </div>

            {Boolean(actionError || doctorError || error) && (
              <div
                className="settings-provider-error [margin:-10px_0_16px] [padding:9px_11px] [color:var(--danger)] [background:var(--danger-muted)] [border:1px_solid_color-mix(in_srgb,_var(--danger)_24%,_transparent)] [border-radius:7px] [font-size:11px] [line-height:1.4]"
                role="alert"
              >
                {actionError ?? doctorError ?? errorMessage(error)}
              </div>
            )}
            {loading && !environment ? (
              <div className="runtime-settings__empty [padding:28px] [color:var(--muted)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)]">
                Detecting local runtimes…
              </div>
            ) : (
              <>
                <div className="runtime-settings__summary [margin-bottom:28px] [display:grid] [grid-template-columns:repeat(4,_minmax(0,_1fr))] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow-inset)] [&_>_span]:[min-width:0] [&_>_span]:[padding:16px] [&_>_span]:[display:grid] [&_>_span]:[gap:4px] [&_>_span]:[border-right:1px_solid_var(--border-subtle)] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:18px] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] max-860:[grid-template-columns:repeat(2,_minmax(0,_1fr))] max-680:[grid-template-columns:1fr] max-680:[&_>_span]:[border-right:0] max-680:[&_>_span]:[border-bottom:1px_solid_var(--border-subtle)]">
                  <span>
                    <strong>
                      {harnesses.filter((harness) => Boolean(harness.version)).length}
                    </strong>
                    <small>Harness tools detected</small>
                  </span>
                  <span>
                    <strong>{harnesses.filter((harness) => !harness.version).length}</strong>
                    <small>Harness tools missing</small>
                  </span>
                  <span>
                    <strong>{nodeRuntime?.version ?? "—"}</strong>
                    <small>Node.js</small>
                  </span>
                  <span>
                    <strong>
                      {environment?.checkedAt ? formatTimestamp(environment.checkedAt) : "—"}
                    </strong>
                    <small>last checked</small>
                  </span>
                </div>

                <section
                  className="runtime-settings__doctor [margin:0_0_28px] [padding:16px] [display:grid] [gap:12px] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow-inset)]"
                  aria-labelledby="runtime-doctor-title"
                >
                  <div className="runtime-settings__doctor-heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:12px] [&_>_span]:[min-width:0] [&_>_span]:[display:grid] [&_>_span]:[gap:2px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:9.5px] [&_small]:[font-weight:650] [&_small]:[letter-spacing:0.06em] [&_small]:[text-transform:uppercase] [&_h3]:[margin:0] [&_h3]:[font-size:13px]">
                    <span>
                      <small>Built-in diagnostics</small>
                      <h3 id="runtime-doctor-title">Environment Doctor</h3>
                    </span>
                    <Badge tone={doctorHealthy ? "success" : "neutral"}>
                      {doctorLoading
                        ? "Checking"
                        : doctorHealthy
                          ? "Healthy"
                          : `${doctorIssues.length} ${doctorIssues.length === 1 ? "issue" : "issues"}`}
                    </Badge>
                  </div>
                  <div
                    className={cx(
                      String.raw`doctor-summary [min-width:0] [padding:12px] [display:grid] [grid-template-columns:34px_minmax(0,_1fr)] [align-items:start] [gap:10px] [color:var(--foreground)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-md)] [&.is-healthy_.doctor-summary\_\_icon]:[color:var(--success)] [&.is-healthy_.doctor-summary\_\_icon]:[background:rgba(52,_211,_153,_0.12)] [&_h3]:[margin:0] [&_p]:[margin:0] [&_h3]:[font-size:12.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.35] [&_p]:[margin-top:3px] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:11px] [&_p]:[line-height:1.45]`,
                      doctorHealthy && "is-healthy",
                    )}
                    aria-live="polite"
                  >
                    <span className="doctor-summary__icon [width:34px] [height:34px] [display:grid] [place-items:center] [color:var(--accent)] [background:var(--accent-muted)] [border-radius:10px] [&_svg]:[width:17px] [&_svg]:[height:17px] [&_.lucide-loader-circle]:[animation:spin_900ms_linear_infinite] [&_.lucide-loader]:[animation:spin_900ms_linear_infinite]">
                      {doctorLoading ? (
                        <Loader2 aria-hidden="true" />
                      ) : doctorHealthy ? (
                        <CircleCheck aria-hidden="true" />
                      ) : (
                        <Wrench aria-hidden="true" />
                      )}
                    </span>
                    <div>
                      <h3>
                        {doctorLoading
                          ? "Checking local environment"
                          : doctorHealthy
                            ? "Environment ready"
                            : doctorReport
                              ? "Review the diagnostics below"
                              : "Doctor status unavailable"}
                      </h3>
                      <p>
                        Harnesses remain optional. Doctor checks the shared baseline and applies no
                        repair until you confirm its plan.
                      </p>
                    </div>
                  </div>

                  {fixResult?.executed && (
                    <output
                      className={doctorNoticeVariants({
                        tone: fixResult.after.healthy ? "success" : "error",
                      })}
                    >
                      {fixResult.after.healthy ? (
                        <CircleCheck aria-hidden="true" />
                      ) : (
                        <XCircle aria-hidden="true" />
                      )}
                      <span>
                        {fixResult.after.healthy
                          ? "Repairs completed. The environment is ready."
                          : "Repairs completed, but some diagnostics still need attention."}
                      </span>
                    </output>
                  )}

                  {!doctorLoading && repairActions.length > 0 && (
                    <section
                      className="doctor-section [min-width:0] [display:grid] [gap:9px] [&_h3]:[margin:0] [&_+_.doctor-section]:[padding-top:12px] [&_+_.doctor-section]:[border-top:1px_solid_var(--border-subtle)]"
                      aria-labelledby="runtime-repair-title"
                    >
                      <div className="doctor-section__heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [&_h3]:[color:var(--foreground)] [&_h3]:[font-size:11.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.3] [&_>_span]:[color:var(--muted-foreground)] [&_>_span]:[font-family:var(--font-mono)] [&_>_span]:[font-size:10px]">
                        <h3 id="runtime-repair-title">Repair plan</h3>
                        <span>{repairActions.length}</span>
                      </div>
                      {fixPending ? (
                        <div className="doctor-confirmation [padding:10px] [display:grid] [gap:8px] [background:var(--accent-muted)] [border:1px_solid_color-mix(in_srgb,_var(--accent)_26%,_transparent)] [border-radius:8px] [&_p]:[margin:0] [&_>_strong]:[font-size:11.5px] [&_>_strong]:[font-weight:680] [&_>_p]:[color:var(--muted-foreground)] [&_>_p]:[font-size:10.5px] [&_>_p]:[line-height:1.4]">
                          <strong>Confirm environment changes</strong>
                          <p>No installer or system change runs until this plan is confirmed.</p>
                          <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
                            {repairActions.map((action) => (
                              <li
                                key={action.id}
                                className="doctor-action [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [display:flex] [align-items:flex-start] [gap:8px] [&_>_span]:[min-width:0] [&_>_span]:[flex:1_1_auto] [&_>_span]:[color:var(--foreground)] [&_>_span]:[font-size:11px] [&_>_span]:[line-height:1.4] [&_>_span]:[overflow-wrap:anywhere]"
                              >
                                <span>{action.label}</span>
                                <Badge tone={action.risk === "admin" ? "danger" : "neutral"}>
                                  {action.risk}
                                </Badge>
                              </li>
                            ))}
                          </ul>
                          <div className="doctor-confirmation__actions [justify-content:flex-end] [display:flex] [align-items:center] [gap:4px]">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={onCancelFix}
                              disabled={fixRunning}
                            >
                              Cancel
                            </Button>
                            <Button size="sm" onClick={onConfirmFix} disabled={fixRunning}>
                              {fixRunning ? (
                                <Loader2 data-icon="inline-start" aria-hidden="true" />
                              ) : (
                                <Wrench data-icon="inline-start" aria-hidden="true" />
                              )}
                              Confirm repairs
                            </Button>
                          </div>
                        </div>
                      ) : (
                        <Button size="sm" onClick={onRequestFix}>
                          <Wrench data-icon="inline-start" aria-hidden="true" />
                          Review repair plan
                        </Button>
                      )}
                    </section>
                  )}

                  {!doctorLoading && doctorIssues.length > 0 && (
                    <section
                      className="doctor-section [min-width:0] [display:grid] [gap:9px] [&_h3]:[margin:0] [&_+_.doctor-section]:[padding-top:12px] [&_+_.doctor-section]:[border-top:1px_solid_var(--border-subtle)]"
                      aria-labelledby="runtime-diagnostics-title"
                    >
                      <div className="doctor-section__heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [&_h3]:[color:var(--foreground)] [&_h3]:[font-size:11.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.3] [&_>_span]:[color:var(--muted-foreground)] [&_>_span]:[font-family:var(--font-mono)] [&_>_span]:[font-size:10px]">
                        <h3 id="runtime-diagnostics-title">Diagnostics</h3>
                        <span>{doctorIssues.length}</span>
                      </div>
                      <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
                        {doctorIssues.map((issue) => (
                          <li
                            key={issue.id}
                            className="doctor-issue [display:grid] [grid-template-columns:15px_minmax(0,_1fr)_auto] [align-items:start] [gap:7px] [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[min-width:0] [&_strong]:[overflow-wrap:anywhere] [&_span]:[min-width:0] [&_span]:[overflow-wrap:anywhere] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:650] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4] [&_>_svg]:[width:15px] [&_>_svg]:[height:15px] [&_>_svg]:[color:var(--danger)]"
                          >
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
                    </section>
                  )}
                </section>

                <section
                  className="runtime-settings__section [margin-top:24px] [&_>_div_p]:[color:var(--muted-foreground)] [&_>_div_p]:[font-size:10.5px] [&_>_div_h3]:[margin:0] [&_>_div_p]:[margin:0] [&_>_div_h3]:[font-size:13px] [&_>_div_p]:[margin-top:4px]"
                  aria-labelledby="runtime-node-title"
                >
                  <div>
                    <h3 id="runtime-node-title">Node.js</h3>
                    <p>
                      Shared JavaScript runtime for npm/npx-based adapters and package management.
                    </p>
                  </div>
                  <ul className="runtime-settings__list runtime-settings__list--node [margin:12px_0_0] [padding:0] [overflow:hidden] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [list-style:none] [box-shadow:var(--shadow-inset)] [&_li]:[min-width:0] [&_li]:[min-height:62px] [&_li]:[padding:11px_14px] [&_li]:[display:grid] [&_li]:[align-items:center] [&_li]:[gap:10px] [&_li]:[border-bottom:1px_solid_var(--border-subtle)] max-860:[&_li]:[grid-template-columns:28px_minmax(130px,_1fr)_auto] max-680:[&_li]:[grid-template-columns:28px_minmax(0,_1fr)_auto] [&_li]:[grid-template-columns:28px_minmax(180px,_1fr)_auto]">
                    {nodeRuntime && (
                      <li>
                        <span
                          className={cx(
                            "runtime-status-icon [width:26px] [height:26px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:rgba(255,_255,_255,_0.04)] [border-radius:8px] [&_svg]:[width:15px]",
                            RUNTIME_STATUS_CLASS[nodeRuntime.status],
                          )}
                        >
                          {nodeRuntime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity [min-width:0] [display:grid] [gap:3px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[overflow:hidden] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap]">
                          <strong>{nodeRuntime.label}</strong>
                          <small>{nodeRuntime.path ?? nodeRuntime.command}</small>
                        </span>
                        {nodeRuntime.version ? (
                          <button
                            type="button"
                            className={badgeVariants({
                              tone: "success",
                              className:
                                "doctor-harness__version [min-width:58px] [justify-content:center]",
                            })}
                            aria-label="Check Node.js version again"
                            title="Check version again"
                            disabled={busyId !== null}
                            onClick={() => void run("refresh-node", onRefresh)}
                          >
                            {nodeRuntime.version}
                          </button>
                        ) : (
                          <Badge tone="danger">{requirementStatusLabel(nodeRuntime.status)}</Badge>
                        )}
                      </li>
                    )}
                  </ul>
                </section>

                <section
                  className="runtime-settings__section [margin-top:24px] [&_>_div_p]:[color:var(--muted-foreground)] [&_>_div_p]:[font-size:10.5px] [&_>_div_h3]:[margin:0] [&_>_div_p]:[margin:0] [&_>_div_h3]:[font-size:13px] [&_>_div_p]:[margin-top:4px]"
                  aria-labelledby="runtime-harnesses-title"
                >
                  <div>
                    <h3 id="runtime-harnesses-title">Harness tools</h3>
                    <p>
                      Tool versions are detected independently. Click a version to check it again.
                    </p>
                  </div>
                  <ul className="runtime-harness-list [margin:12px_0_0] [padding:0] [overflow:hidden] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [list-style:none] [box-shadow:var(--shadow-inset)] [&_li]:[min-width:0] [&_li]:[min-height:62px] [&_li]:[padding:11px_14px] [&_li]:[display:grid] [&_li]:[align-items:center] [&_li]:[gap:10px] [&_li]:[border-bottom:1px_solid_var(--border-subtle)] [&_li]:[grid-template-columns:28px_minmax(0,_1fr)_auto]">
                    {harnesses.map((harness) => {
                      const versionState = harnessVersions[harness.harnessId];
                      const version = versionState?.version ?? harness.version;
                      const versionLoading = versionState?.status === "loading";
                      return (
                        <li key={harness.harnessId}>
                          <span className="runtime-harness-list__icon [width:26px] [height:26px] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--input)] [border-radius:8px] [&_.harness-brand-icon]:[width:16px] [&_.harness-brand-icon]:[height:16px] [&_svg]:[width:16px] [&_svg]:[height:16px]">
                            <HarnessBrandIcon
                              harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                            />
                          </span>
                          <span className="runtime-settings__identity [min-width:0] [display:grid] [gap:3px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[overflow:hidden] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap]">
                            <strong>{harness.harnessLabel}</strong>
                            <small>{harness.path ?? harness.command}</small>
                          </span>
                          {versionLoading ? (
                            <output
                              className={badgeVariants({
                                tone: "loading",
                                className:
                                  "doctor-harness__version [min-width:58px] [justify-content:center]",
                              })}
                              aria-label={`Checking ${harness.harnessLabel} version`}
                            >
                              <Loader2 data-icon aria-hidden="true" />
                            </output>
                          ) : version ? (
                            <button
                              type="button"
                              className={badgeVariants({
                                tone: "success",
                                className:
                                  "doctor-harness__version [min-width:58px] [justify-content:center]",
                              })}
                              aria-label={`Check ${harness.harnessLabel} version again`}
                              title="Check version again"
                              onClick={() => onRefreshHarnessVersion(harness.harnessId)}
                            >
                              {version}
                            </button>
                          ) : harness.installable ? (
                            <Button
                              size="sm"
                              disabled={Boolean(installingHarnessId)}
                              aria-label={`Install ${harness.harnessLabel}`}
                              onClick={() => void onInstallHarness(harness.harnessId)}
                            >
                              {installingHarnessId === harness.harnessId ? (
                                <Loader2 data-icon="inline-start" aria-hidden="true" />
                              ) : (
                                <Download data-icon="inline-start" aria-hidden="true" />
                              )}
                              Install
                            </Button>
                          ) : (
                            <Badge tone="neutral">Not detected</Badge>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </section>

                <section
                  className="runtime-settings__section [margin-top:24px] [&_>_div_p]:[color:var(--muted-foreground)] [&_>_div_p]:[font-size:10.5px] [&_>_div_h3]:[margin:0] [&_>_div_p]:[margin:0] [&_>_div_h3]:[font-size:13px] [&_>_div_p]:[margin-top:4px]"
                  aria-labelledby="runtime-container-title"
                >
                  <div>
                    <h3 id="runtime-container-title">Container runtime</h3>
                    <p>
                      Apple Container is preferred for protected local harness execution on
                      supported macOS hosts.
                    </p>
                  </div>
                  <ul className="runtime-settings__list [margin:12px_0_0] [padding:0] [overflow:hidden] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [list-style:none] [box-shadow:var(--shadow-inset)] [&_li]:[min-width:0] [&_li]:[min-height:62px] [&_li]:[padding:11px_14px] [&_li]:[display:grid] [&_li]:[align-items:center] [&_li]:[gap:10px] [&_li]:[border-bottom:1px_solid_var(--border-subtle)] [&_li]:[grid-template-columns:28px_minmax(150px,_1.2fr)_minmax(90px,_0.65fr)_minmax(130px,_1fr)_auto] max-860:[&_li]:[grid-template-columns:28px_minmax(130px,_1fr)_auto] max-680:[&_li]:[grid-template-columns:28px_minmax(0,_1fr)_auto]">
                    {(environment?.containerRuntimes ?? []).map((runtime) => (
                      <li key={runtime.id}>
                        <span
                          className={cx(
                            "runtime-status-icon [width:26px] [height:26px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:rgba(255,_255,_255,_0.04)] [border-radius:8px] [&_svg]:[width:15px]",
                            RUNTIME_STATUS_CLASS[runtime.status],
                          )}
                        >
                          {runtime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity [min-width:0] [display:grid] [gap:3px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px] [&_small]:[overflow:hidden] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap]">
                          <strong>{runtime.label}</strong>
                          <small>{runtime.path ?? runtime.command}</small>
                        </span>
                        <span className="runtime-settings__version [color:var(--muted)] [font-family:var(--font-mono)] [font-size:10.5px] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap] max-680:[grid-column:2_/_-1]">
                          {runtime.version ?? runtime.status.replaceAll("_", " ")}
                        </span>
                        <span className="runtime-settings__consumers [color:var(--muted-foreground)] [font-size:10.5px] max-860:[grid-column:2_/_-1]">
                          {runtime.preferred
                            ? "Preferred"
                            : runtime.supported
                              ? "Supported"
                              : "Unavailable on this host"}
                        </span>
                        {runtime.status !== "ready" && runtime.installable && (
                          <Button
                            size="sm"
                            disabled={busyId !== null}
                            onClick={() =>
                              void run(`container:${runtime.id}`, () =>
                                onSetupContainer(runtime.id),
                              )
                            }
                          >
                            <Download aria-hidden="true" />
                            Set up
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                </section>

                <section
                  className="runtime-settings__section [margin-top:24px]"
                  aria-labelledby="runtime-work-items-title"
                >
                  <div className="[display:flex] [align-items:flex-end] [justify-content:space-between] [gap:12px] [&_h3]:[margin:0] [&_h3]:[font-size:13px] [&_p]:[margin:4px_0_0] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:10.5px]">
                    <span>
                      <h3 id="runtime-work-items-title">Detached WorkItems</h3>
                      <p>
                        Active credential-free workers continue under the local supervisor after
                        Desktop closes. Sessions only observe this durable state.
                      </p>
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={taskRuntimeLoading || busyId !== null}
                      onClick={() => void run("tasks:refresh", onRefreshTasks)}
                    >
                      <RefreshCw aria-hidden="true" />
                      Refresh
                    </Button>
                  </div>
                  {taskRuntimeError ? (
                    <div
                      className="[margin-top:10px] [color:var(--danger)] [font-size:10.5px]"
                      role="alert"
                    >
                      {errorMessage(taskRuntimeError)}
                    </div>
                  ) : (
                    <ul className="[margin:12px_0_0] [padding:0] [overflow:hidden] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [list-style:none] [&_li]:[padding:11px_14px] [&_li]:[display:flex] [&_li]:[align-items:center] [&_li]:[justify-content:space-between] [&_li]:[gap:12px] [&_li]:[border-bottom:1px_solid_var(--border-subtle)]">
                      {taskRuntimeLoading && !taskRuntime ? (
                        <li>Loading durable work…</li>
                      ) : (taskRuntime?.workItems.length ?? 0) === 0 ? (
                        <li>No durable WorkItems yet.</li>
                      ) : (
                        taskRuntime?.workItems.map((workItem) => {
                          const approval = workItem.approvalIds
                            .map((approvalId) => pendingApprovals.get(approvalId))
                            .find(Boolean);
                          return (
                            <li key={workItem.id}>
                              <span className="[min-width:0] [display:grid] [gap:3px] [&_strong]:[font-family:var(--font-mono)] [&_strong]:[font-size:11px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10px]">
                                <strong>{workItem.id}</strong>
                                <small>
                                  {workItem.executor.backend} · {workItem.executor.operation} ·{" "}
                                  {workItem.status}
                                  {approval?.reason ? ` · ${approval.reason}` : ""}
                                </small>
                              </span>
                              <span className="[display:flex] [align-items:center] [gap:5px]">
                                <Badge
                                  tone={activeWorkItemIds.has(workItem.id) ? "success" : "neutral"}
                                >
                                  {activeWorkItemIds.has(workItem.id)
                                    ? "Supervisor active"
                                    : workItem.status}
                                </Badge>
                                {approval ? (
                                  <>
                                    <Button
                                      size="sm"
                                      disabled={busyId !== null}
                                      onClick={() =>
                                        void run(`approval:${approval.approvalId}:approve`, () =>
                                          onDecideApproval(approval.approvalId, "approved"),
                                        )
                                      }
                                    >
                                      Approve
                                    </Button>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      disabled={busyId !== null}
                                      onClick={() =>
                                        void run(`approval:${approval.approvalId}:reject`, () =>
                                          onDecideApproval(approval.approvalId, "rejected"),
                                        )
                                      }
                                    >
                                      Reject
                                    </Button>
                                  </>
                                ) : activeWorkItemIds.has(workItem.id) ? (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    disabled={busyId !== null}
                                    onClick={() =>
                                      void run(`task:${workItem.id}:cancel`, () =>
                                        onCancelTask(workItem.id),
                                      )
                                    }
                                  >
                                    Cancel
                                  </Button>
                                ) : null}
                              </span>
                            </li>
                          );
                        })
                      )}
                    </ul>
                  )}
                </section>

                <div className="runtime-settings__path [margin-top:24px] [padding:12px_14px] [display:grid] [gap:5px] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:10px] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_code]:[overflow:hidden] [&_code]:[text-overflow:ellipsis] [&_code]:[white-space:nowrap] [&_code]:[color:var(--muted)] [&_code]:[font-size:10.5px] [&_pre]:[max-height:160px] [&_pre]:[margin:8px_0_0] [&_pre]:[overflow:auto] [&_pre]:[padding-top:10px] [&_pre]:[color:var(--muted-foreground)] [&_pre]:[border-top:1px_solid_var(--border-subtle)] [&_pre]:[font-family:var(--font-mono)] [&_pre]:[font-size:9.5px] [&_pre]:[white-space:pre-wrap]">
                  <span>Detected PATH</span>
                  <code>{environment?.path ?? "Unavailable"}</code>
                  {repairLogs.length > 0 && <pre>{repairLogs.join("\n\n")}</pre>}
                </div>
              </>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}
