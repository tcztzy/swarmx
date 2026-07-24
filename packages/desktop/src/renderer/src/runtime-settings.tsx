import type { DoctorFixResult, DoctorReport, HarnessEnvironmentStatus } from "@swarmx/runtime";
import { CircleCheck, Download, Loader2, RefreshCw, Wrench, XCircle } from "lucide-react";
import { useState } from "react";
import { type DoctorHarnessVersionState, requirementStatusLabel } from "./doctor-panel.js";
import { HarnessBrandIcon, harnessOption } from "./harness-presentation.js";
import { errorMessage, formatTimestamp } from "./text-utils.js";
import { Badge, Button, cx } from "./ui-primitives.js";

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
  onRefresh,
  onSetupContainer,
  onInstallHarness,
  onRefreshHarnessVersion,
  onRequestFix,
  onCancelFix,
  onConfirmFix,
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
  onRefresh: () => Promise<void>;
  onSetupContainer: (containerRuntimeId: string) => Promise<void>;
  onInstallHarness: (harnessId: string) => Promise<void>;
  onRefreshHarnessVersion: (harnessId: string) => void;
  onRequestFix: () => void;
  onCancelFix: () => void;
  onConfirmFix: () => void;
}) {
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const nodeRuntime = environment?.requirements.find((requirement) => requirement.id === "node");
  const harnesses = environment?.harnesses ?? [];
  const doctorIssues = doctorReport?.issues ?? [];
  const repairActions = doctorReport?.repairActions ?? [];
  const doctorHealthy = Boolean(doctorReport?.healthy && doctorIssues.length === 0);
  const repairLogs = fixResult?.setupResults.flatMap((result) => result.log) ?? [];
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
    <section className="settings-workspace" aria-label="Runtime settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content">
          <section className="runtime-settings">
            <div className="settings-content-heading">
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
                  className="settings-primary-action"
                  disabled={busyId !== null}
                  onClick={() => void run("refresh", onRefresh)}
                >
                  <RefreshCw
                    className={busyId === "refresh" ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  Refresh
                </button>
              </div>
            </div>

            {Boolean(actionError || doctorError || error) && (
              <div className="settings-provider-error" role="alert">
                {actionError ?? doctorError ?? errorMessage(error)}
              </div>
            )}
            {loading && !environment ? (
              <div className="runtime-settings__empty">Detecting local runtimes…</div>
            ) : (
              <>
                <div className="runtime-settings__summary">
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
                  className="runtime-settings__doctor"
                  aria-labelledby="runtime-doctor-title"
                >
                  <div className="runtime-settings__doctor-heading">
                    <span>
                      <small>Built-in diagnostics</small>
                      <h3 id="runtime-doctor-title">Environment Doctor</h3>
                    </span>
                    <Badge tone={doctorHealthy ? "active" : "neutral"}>
                      {doctorLoading
                        ? "Checking"
                        : doctorHealthy
                          ? "Healthy"
                          : `${doctorIssues.length} ${doctorIssues.length === 1 ? "issue" : "issues"}`}
                    </Badge>
                  </div>
                  <div
                    className={cx("doctor-summary", doctorHealthy && "is-healthy")}
                    aria-live="polite"
                  >
                    <span className="doctor-summary__icon">
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
                          : "Repairs completed, but some diagnostics still need attention."}
                      </span>
                    </output>
                  )}

                  {!doctorLoading && repairActions.length > 0 && (
                    <section className="doctor-section" aria-labelledby="runtime-repair-title">
                      <div className="doctor-section__heading">
                        <h3 id="runtime-repair-title">Repair plan</h3>
                        <span>{repairActions.length}</span>
                      </div>
                      {fixPending ? (
                        <div className="doctor-confirmation">
                          <strong>Confirm environment changes</strong>
                          <p>No installer or system change runs until this plan is confirmed.</p>
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
                    <section className="doctor-section" aria-labelledby="runtime-diagnostics-title">
                      <div className="doctor-section__heading">
                        <h3 id="runtime-diagnostics-title">Diagnostics</h3>
                        <span>{doctorIssues.length}</span>
                      </div>
                      <ul className="doctor-list">
                        {doctorIssues.map((issue) => (
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
                    </section>
                  )}
                </section>

                <section className="runtime-settings__section" aria-labelledby="runtime-node-title">
                  <div>
                    <h3 id="runtime-node-title">Node.js</h3>
                    <p>
                      Shared JavaScript runtime for npm/npx-based adapters and package management.
                    </p>
                  </div>
                  <ul className="runtime-settings__list runtime-settings__list--node">
                    {nodeRuntime && (
                      <li>
                        <span className={`runtime-status-icon is-${nodeRuntime.status}`}>
                          {nodeRuntime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity">
                          <strong>{nodeRuntime.label}</strong>
                          <small>{nodeRuntime.path ?? nodeRuntime.command}</small>
                        </span>
                        {nodeRuntime.version ? (
                          <button
                            type="button"
                            className="badge badge--active doctor-harness__version"
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
                  className="runtime-settings__section"
                  aria-labelledby="runtime-harnesses-title"
                >
                  <div>
                    <h3 id="runtime-harnesses-title">Harness tools</h3>
                    <p>
                      Tool versions are detected independently. Click a version to check it again.
                    </p>
                  </div>
                  <ul className="runtime-harness-list">
                    {harnesses.map((harness) => {
                      const versionState = harnessVersions[harness.harnessId];
                      const version = versionState?.version ?? harness.version;
                      const versionLoading = versionState?.status === "loading";
                      return (
                        <li key={harness.harnessId}>
                          <span className="runtime-harness-list__icon">
                            <HarnessBrandIcon
                              harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                            />
                          </span>
                          <span className="runtime-settings__identity">
                            <strong>{harness.harnessLabel}</strong>
                            <small>{harness.path ?? harness.command}</small>
                          </span>
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
                  className="runtime-settings__section"
                  aria-labelledby="runtime-container-title"
                >
                  <div>
                    <h3 id="runtime-container-title">Container runtime</h3>
                    <p>
                      Apple Container is preferred for protected local harness execution on
                      supported macOS hosts.
                    </p>
                  </div>
                  <ul className="runtime-settings__list">
                    {(environment?.containerRuntimes ?? []).map((runtime) => (
                      <li key={runtime.id}>
                        <span className={`runtime-status-icon is-${runtime.status}`}>
                          {runtime.status === "ready" ? (
                            <CircleCheck aria-hidden="true" />
                          ) : (
                            <XCircle aria-hidden="true" />
                          )}
                        </span>
                        <span className="runtime-settings__identity">
                          <strong>{runtime.label}</strong>
                          <small>{runtime.path ?? runtime.command}</small>
                        </span>
                        <span className="runtime-settings__version">
                          {runtime.version ?? runtime.status.replaceAll("_", " ")}
                        </span>
                        <span className="runtime-settings__consumers">
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

                <div className="runtime-settings__path">
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
