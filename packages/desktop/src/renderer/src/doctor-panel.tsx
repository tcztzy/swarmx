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
import type { DoctorHarnessVersionState, DoctorPanelMode } from "./doctor-controller.js";
import { HarnessBrandIcon, type HarnessOption, harnessOption } from "./harness-presentation.js";
import {
  Badge,
  Button,
  badgeVariants,
  cx,
  doctorNoticeVariants,
  rightPanelVariants,
} from "./ui-primitives.js";

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
  const sandbox = report?.environment.sandbox;
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
      className={rightPanelVariants({ kind: "doctor" })}
      aria-label={mode === "setup" ? "Setup panel" : "Doctor panel"}
    >
      <div className="runtime-panel__header [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:12px] [&_>_div]:[min-width:0] [&_span]:[display:block] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[font-weight:650] [&_span]:[letter-spacing:0.04em] [&_span]:[text-transform:uppercase] [&_h2]:[margin:2px_0_0] [&_h2]:[overflow:hidden] [&_h2]:[color:var(--foreground)] [&_h2]:[font-size:13px] [&_h2]:[font-weight:650] [&_h2]:[line-height:1.25] [&_h2]:[text-overflow:ellipsis] [&_h2]:[white-space:nowrap]">
        <div>
          <span>Environment</span>
          <h2>
            {title}
            {report?.harnessId ? ` · ${report.harnessId}` : ""}
          </h2>
        </div>
        <div className="doctor-panel__header-actions [display:flex] [align-items:center] [gap:4px]">
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

      <section
        className={cx(
          String.raw`doctor-summary [min-width:0] [padding:12px] [display:grid] [grid-template-columns:34px_minmax(0,_1fr)] [align-items:start] [gap:10px] [color:var(--foreground)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-md)] [&.is-healthy_.doctor-summary\_\_icon]:[color:var(--success)] [&.is-healthy_.doctor-summary\_\_icon]:[background:rgba(52,_211,_153,_0.12)] [&_h3]:[margin:0] [&_p]:[margin:0] [&_h3]:[font-size:12.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.35] [&_p]:[margin-top:3px] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:11px] [&_p]:[line-height:1.45]`,
          panelHealthy && "is-healthy",
        )}
        aria-live="polite"
      >
        <span className="doctor-summary__icon [width:34px] [height:34px] [display:grid] [place-items:center] [color:var(--accent)] [background:var(--accent-muted)] [border-radius:10px] [&_svg]:[width:17px] [&_svg]:[height:17px] [&_.lucide-loader-circle]:[animation:spin_900ms_linear_infinite] [&_.lucide-loader]:[animation:spin_900ms_linear_infinite]">
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
        <div className={doctorNoticeVariants({ tone: "error" })} role="alert">
          <XCircle aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

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
              : "Repairs completed, but some issues still need attention."}
          </span>
        </output>
      )}

      {!loading && report && repairActions.length > 0 && (
        <section
          className="doctor-section [min-width:0] [display:grid] [gap:9px] [&_h3]:[margin:0] [&_+_.doctor-section]:[padding-top:12px] [&_+_.doctor-section]:[border-top:1px_solid_var(--border-subtle)]"
          aria-labelledby="doctor-repair-title"
        >
          <div className="doctor-section__heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [&_h3]:[color:var(--foreground)] [&_h3]:[font-size:11.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.3] [&_>_span]:[color:var(--muted-foreground)] [&_>_span]:[font-family:var(--font-mono)] [&_>_span]:[font-size:10px]">
            <h3 id="doctor-repair-title">Repair plan</h3>
            <span>{repairActions.length}</span>
          </div>
          {fixPending ? (
            <div className="doctor-confirmation [padding:10px] [display:grid] [gap:8px] [background:var(--accent-muted)] [border:1px_solid_color-mix(in_srgb,_var(--accent)_26%,_transparent)] [border-radius:8px] [&_p]:[margin:0] [&_>_strong]:[font-size:11.5px] [&_>_strong]:[font-weight:680] [&_>_p]:[color:var(--muted-foreground)] [&_>_p]:[font-size:10.5px] [&_>_p]:[line-height:1.4]">
              <strong>
                Confirm {repairActions.length} {repairActions.length === 1 ? "repair" : "repairs"}
              </strong>
              <p>No changes are made until you confirm this plan.</p>
              <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
                {repairActions.map((action) => (
                  <li
                    key={action.id}
                    className="doctor-action [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [display:flex] [align-items:flex-start] [gap:8px] [&_>_span]:[min-width:0] [&_>_span]:[flex:1_1_auto] [&_>_span]:[color:var(--foreground)] [&_>_span]:[font-size:11px] [&_>_span]:[line-height:1.4] [&_>_span]:[overflow-wrap:anywhere]"
                  >
                    <span>
                      {action.label}
                      {(action.changes ?? []).map((change) => (
                        <span key={change}> · {change}</span>
                      ))}
                    </span>
                    <Badge tone={action.risk === "admin" ? "danger" : "neutral"}>
                      {action.risk}
                    </Badge>
                  </li>
                ))}
              </ul>
              <div className="doctor-confirmation__actions [justify-content:flex-end] [display:flex] [align-items:center] [gap:4px]">
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
        <section
          className="doctor-section [min-width:0] [display:grid] [gap:9px] [&_h3]:[margin:0] [&_+_.doctor-section]:[padding-top:12px] [&_+_.doctor-section]:[border-top:1px_solid_var(--border-subtle)]"
          aria-labelledby="doctor-issues-title"
        >
          <div className="doctor-section__heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [&_h3]:[color:var(--foreground)] [&_h3]:[font-size:11.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.3] [&_>_span]:[color:var(--muted-foreground)] [&_>_span]:[font-family:var(--font-mono)] [&_>_span]:[font-size:10px]">
            <h3 id="doctor-issues-title">Diagnostics</h3>
            <span>{issues.length}</span>
          </div>
          <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
            {issues.map((issue) => (
              <li
                key={issue.id}
                className="doctor-issue [display:grid] [grid-template-columns:15px_minmax(0,_1fr)_auto] [align-items:start] [gap:7px] [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[min-width:0] [&_strong]:[overflow-wrap:anywhere] [&_span]:[min-width:0] [&_span]:[overflow-wrap:anywhere] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:650] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4] [&_>_svg]:[width:15px] [&_>_svg]:[height:15px] [&_>_svg]:[color:var(--danger)]"
              >
                <XCircle aria-hidden="true" />
                <div>
                  <strong>{issue.targetId ?? issue.scope}</strong>
                  <span>{issue.symptom ?? issue.message}</span>
                  {issue.cause && <span>Cause: {issue.cause}</span>}
                  {issue.impact && <span>Impact: {issue.impact}</span>}
                  {issue.nextAction && <span>Next: {issue.nextAction}</span>}
                </div>
                <Badge tone={issue.severity === "error" ? "danger" : "neutral"}>
                  {issue.classification ?? issue.severity}
                </Badge>
              </li>
            ))}
          </ul>
          {issues.length > 0 && repairActions.length === 0 && (
            <p className="doctor-section__hint [margin:0] [color:var(--muted-foreground)] [font-size:10.5px] [line-height:1.45]">
              These issues require manual review.
            </p>
          )}
        </section>
      )}

      <section
        className="doctor-section [min-width:0] [display:grid] [gap:9px] [&_h3]:[margin:0] [&_+_.doctor-section]:[padding-top:12px] [&_+_.doctor-section]:[border-top:1px_solid_var(--border-subtle)]"
        aria-labelledby="doctor-harnesses-title"
      >
        <div className="doctor-section__heading [min-width:0] [display:flex] [align-items:center] [justify-content:space-between] [gap:8px] [&_h3]:[color:var(--foreground)] [&_h3]:[font-size:11.5px] [&_h3]:[font-weight:680] [&_h3]:[line-height:1.3] [&_>_span]:[color:var(--muted-foreground)] [&_>_span]:[font-family:var(--font-mono)] [&_>_span]:[font-size:10px]">
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
        <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
          {harnesses.map((harness) => {
            const versionState = harnessVersions[harness.harnessId];
            const version = versionState?.version ?? harness.version;
            const versionLoading = !versionState || versionState.status === "loading";
            return (
              <li
                key={harness.harnessId}
                className="doctor-harness [align-items:center] [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [display:flex] [gap:8px] [&_>_.badge]:[flex:0_0_auto] [&_>_.button]:[flex:0_0_auto] [&_>_.badge]:[font-family:var(--font-mono)] [&_>_div]:[min-width:0] [&_>_div]:[flex:1_1_auto] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[min-width:0] [&_strong]:[overflow-wrap:anywhere] [&_span]:[min-width:0] [&_span]:[overflow-wrap:anywhere] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:650] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4]"
              >
                <span className="doctor-harness__icon [width:16px] [height:16px] [margin-top:1px] [display:grid] [place-items:center] [flex:0_0_auto] [color:var(--muted-foreground)] [&_svg]:[width:16px] [&_svg]:[height:16px] [&_.harness-brand-icon]:[width:16px] [&_.harness-brand-icon]:[height:16px]">
                  <HarnessBrandIcon
                    harness={harnessOption(harness.harnessId, harness.harnessLabel)}
                  />
                </span>
                <div>
                  <strong>{harness.harnessLabel}</strong>
                </div>
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
        <details className="doctor-advanced [min-width:0] [overflow:hidden] [border-top:1px_solid_var(--border-subtle)] [&_h4]:[margin:0] [&_h4]:[color:var(--foreground)] [&_h4]:[font-size:11.5px] [&_h4]:[font-weight:680] [&_h4]:[line-height:1.3] [&_>_summary]:[min-height:48px] [&_>_summary]:[display:flex] [&_>_summary]:[align-items:center] [&_>_summary]:[justify-content:space-between] [&_>_summary]:[gap:8px] [&_>_summary]:[cursor:pointer] [&_>_summary]:[list-style:none] [&_>_summary_>_span]:[min-width:0] [&_>_summary_>_span]:[display:grid] [&_>_summary_>_span]:[gap:1px] [&_>_summary_strong]:[color:var(--foreground)] [&_>_summary_strong]:[font-size:11.5px] [&_>_summary_strong]:[font-weight:650] [&_>_summary_small]:[color:var(--muted-foreground)] [&_>_summary_small]:[font-size:10px] [&_>_summary_>_svg]:[width:15px] [&_>_summary_>_svg]:[height:15px] [&_>_summary_>_svg]:[flex:0_0_auto] [&_>_summary_>_svg]:[color:var(--muted-foreground)] [&_>_summary_>_svg]:[transition:transform_var(--duration-fast)_var(--ease-out)] [&[open]_>_summary_>_svg]:[transform:rotate(90deg)]">
          <summary>
            <span>
              <strong>Advanced details</strong>
              <small>Runtime tools, PATH, and repair logs</small>
            </span>
            <ChevronRight aria-hidden="true" />
          </summary>
          <div className="doctor-advanced__body [display:grid] [gap:12px] [&_>_section]:[min-width:0] [&_>_section]:[display:grid] [&_>_section]:[gap:6px]">
            {sandbox && (
              <section>
                <h4>OS sandbox</h4>
                <div className="doctor-diagnostic [display:flex] [align-items:flex-start] [gap:8px] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [&_>_div]:[min-width:0] [&_>_div]:[flex:1_1_auto] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[font-size:11px] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4]">
                  <div>
                    <strong>{sandbox.strategy}</strong>
                    <span>
                      {[sandbox.mode, sandbox.runtimeId, sandbox.note].filter(Boolean).join(" · ")}
                    </span>
                  </div>
                  <Badge tone={sandbox.ready ? "success" : "danger"}>
                    {sandbox.ready ? "ready" : "blocked"}
                  </Badge>
                </div>
              </section>
            )}
            {requirements.length > 0 && (
              <section>
                <h4>Runtime tools</h4>
                <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
                  {requirements.map((requirement) => (
                    <li
                      key={requirement.id}
                      className="doctor-diagnostic [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [display:flex] [align-items:flex-start] [gap:8px] [&_>_div]:[min-width:0] [&_>_div]:[flex:1_1_auto] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[min-width:0] [&_strong]:[overflow-wrap:anywhere] [&_span]:[min-width:0] [&_span]:[overflow-wrap:anywhere] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:650] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4] [&_span]:[font-family:var(--font-mono)]"
                    >
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
                      <Badge tone={requirement.status === "ready" ? "success" : "danger"}>
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
                <ul className="doctor-list [min-width:0] [margin:0] [padding:0] [display:grid] [list-style:none]">
                  {containerRuntimes.map((runtime) => (
                    <li
                      key={runtime.id}
                      className="doctor-diagnostic [min-width:0] [padding:8px_0] [border-top:1px_solid_var(--border-subtle)] [display:flex] [align-items:flex-start] [gap:8px] [&_>_div]:[min-width:0] [&_>_div]:[flex:1_1_auto] [&_>_div]:[display:grid] [&_>_div]:[gap:2px] [&_strong]:[min-width:0] [&_strong]:[overflow-wrap:anywhere] [&_span]:[min-width:0] [&_span]:[overflow-wrap:anywhere] [&_strong]:[color:var(--foreground)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:650] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.4] [&_span]:[font-family:var(--font-mono)]"
                    >
                      <div>
                        <strong>{runtime.label}</strong>
                        <span>
                          {[runtime.command, runtime.version, runtime.path, runtime.note]
                            .filter(Boolean)
                            .join(" · ")}
                        </span>
                      </div>
                      <Badge tone={runtime.status === "ready" ? "success" : "danger"}>
                        {containerRuntimeStatusLabel(runtime.status)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            <section>
              <h4>Environment PATH</h4>
              <pre className="doctor-code [min-width:0] [max-height:180px] [margin:0] [overflow:auto] [padding:8px] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-family:var(--font-mono)] [font-size:9.5px] [line-height:1.45] [white-space:pre-wrap] [overflow-wrap:anywhere]">
                {report.environment.path}
              </pre>
            </section>
            {setupLogs.length > 0 && (
              <section>
                <h4>Repair log</h4>
                <pre className="doctor-code [min-width:0] [max-height:180px] [margin:0] [overflow:auto] [padding:8px] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-family:var(--font-mono)] [font-size:9.5px] [line-height:1.45] [white-space:pre-wrap] [overflow-wrap:anywhere]">
                  {setupLogs.join("\n\n")}
                </pre>
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
