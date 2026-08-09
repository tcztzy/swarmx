import { Sparkles, User, Wrench } from "lucide-react";
import { useMemo, useState } from "react";

export interface ActivityProfileSummary {
  generatedAt: string;
  trackingSince?: string;
  lifetime: {
    totalTokens: number;
    inputTokens: number;
    outputTokens: number;
    reasoningTokens: number;
    cachedInputTokens: number;
    estimatedTokens: number;
    peakDayTokens: number;
    longestTaskMs: number;
    currentStreakDays: number;
    longestStreakDays: number;
    totalTasks: number;
    completedTasks: number;
    toolCalls: number;
    skillCalls: number;
    skillsExplored: number;
  };
  daily: Array<{
    date: string;
    tokens: number;
    estimatedTokens: number;
    tasks: number;
    tools: number;
    skills: number;
  }>;
  topTools: ActivityRank[];
  topSkills: ActivityRank[];
  reasoningEfforts: ActivityRank[];
  models: ActivityRank[];
}

interface ActivityRank {
  name: string;
  count: number;
}

type ActivityMode = "daily" | "weekly" | "cumulative";
type RankingMode = "skills" | "tools";

interface HeatmapDay {
  date: string;
  tokens: number;
  intensity: number;
}

export function ProfileWorkspace({
  summary,
  loading,
  error,
}: {
  summary?: ActivityProfileSummary;
  loading: boolean;
  error?: string;
}) {
  const [activityMode, setActivityMode] = useState<ActivityMode>("daily");
  const [rankingMode, setRankingMode] = useState<RankingMode>("skills");
  const heatmapDays = useMemo(
    () => buildHeatmapDays(summary, activityMode),
    [activityMode, summary],
  );
  const monthLabels = useMemo(() => buildMonthLabels(heatmapDays), [heatmapDays]);

  if (loading && !summary) {
    return (
      <section
        className="profile-workspace [width:100%] [height:100%] [min-width:0] [min-height:0] [overflow-y:auto] [color:var(--foreground)] [background:var(--background)]"
        aria-label="Settings"
      >
        <output className="profile-workspace__state [height:100%] [display:grid] [place-content:center] [gap:4px] [color:var(--muted-foreground)] [font-size:13px] [text-align:center] [&.is-error_strong]:[color:var(--foreground)] [&.is-error_strong]:[font-size:15px]">
          Loading local activity…
        </output>
      </section>
    );
  }

  if (error || !summary) {
    return (
      <section
        className="profile-workspace [width:100%] [height:100%] [min-width:0] [min-height:0] [overflow-y:auto] [color:var(--foreground)] [background:var(--background)]"
        aria-label="Settings"
      >
        <div className="profile-workspace__state is-error [height:100%] [display:grid] [place-content:center] [gap:4px] [color:var(--muted-foreground)] [font-size:13px] [text-align:center] [&.is-error_strong]:[color:var(--foreground)] [&.is-error_strong]:[font-size:15px]">
          <strong>Activity is unavailable</strong>
          <span>{error ?? "The local activity store could not be read."}</span>
        </div>
      </section>
    );
  }

  const lifetime = summary.lifetime;
  const ranked = rankingMode === "skills" ? summary.topSkills : summary.topTools;
  const measuredTokens = Math.max(0, lifetime.totalTokens - lifetime.estimatedTokens);
  const measuredShare = lifetime.totalTokens
    ? Math.round((measuredTokens / lifetime.totalTokens) * 100)
    : 0;
  const topReasoning = summary.reasoningEfforts[0];
  const topModel = summary.models[0];

  return (
    <section
      className="profile-workspace [width:100%] [height:100%] [min-width:0] [min-height:0] [overflow-y:auto] [color:var(--foreground)] [background:var(--background)]"
      aria-label="Settings"
    >
      <div className="profile-workspace__content [width:min(100%,_1080px)] [min-width:0] [margin:0_auto] [padding:82px_clamp(30px,_5vw,_72px)_56px] max-680:[padding:36px_18px_40px]">
        <header className="profile-hero [display:flex] [flex-direction:column] [align-items:center] [text-align:center] [&_h2]:[margin:18px_0_5px] [&_h2]:[font-size:23px] [&_h2]:[font-weight:620] [&_h2]:[letter-spacing:-0.025em] [&_p]:[margin:0] [&_p]:[display:flex] [&_p]:[align-items:center] [&_p]:[gap:7px] [&_p]:[color:var(--muted-foreground)] [&_p]:[font-size:12.5px] [&_p_em]:[padding:2px_7px] [&_p_em]:[background:var(--input)] [&_p_em]:[border:1px_solid_var(--border-subtle)] [&_p_em]:[border-radius:999px] [&_p_em]:[font-size:10px] [&_p_em]:[font-style:normal] [&_small]:[margin-top:9px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:10.5px]">
          <span
            className="profile-hero__avatar [width:78px] [height:78px] [display:grid] [place-items:center] [color:var(--primary-foreground)] [background:var(--primary)] [border-radius:999px] [&_svg]:[width:33px] [&_svg]:[height:33px] [&_svg]:[stroke-width:1.7]"
            aria-hidden="true"
          >
            <User />
          </span>
          <h2>Anonymous user</h2>
          <p>
            <span>@swarmx</span>
            <span aria-hidden="true">·</span>
            <em>Local</em>
          </p>
          <small>Usage stays on this device and never includes prompt or response text.</small>
        </header>

        <dl
          className="profile-metrics [margin:42px_0_0] [padding:11px_2px] [display:grid] [grid-template-columns:repeat(5,_minmax(0,_1fr))] [border:1px_solid_var(--border-subtle)] [border-radius:14px] [&_>_div]:[min-width:0] [&_>_div]:[padding:3px_12px] [&_>_div]:[display:flex] [&_>_div]:[flex-direction:column] [&_>_div]:[align-items:center] [&_>_div]:[border-right:1px_solid_var(--border-subtle)] [&_>_div]:[text-align:center] [&_dt]:[color:var(--muted-foreground)] [&_dt]:[font-size:11.5px] [&_dt]:[white-space:nowrap] [&_dd]:[order:-1] [&_dd]:[margin:0_0_1px] [&_dd]:[overflow:hidden] [&_dd]:[font-size:13.5px] [&_dd]:[font-variant-numeric:tabular-nums] [&_dd]:[font-weight:590] [&_dd]:[text-overflow:ellipsis] [&_dd]:[white-space:nowrap] max-680:[grid-template-columns:repeat(2,_minmax(0,_1fr))] max-680:[&_>_div]:[padding:9px] max-680:[&_>_div]:[border-right:0] max-680:[&_>_div]:[border-bottom:1px_solid_var(--border-subtle)]"
          aria-label="Lifetime activity summary"
        >
          <ProfileMetric
            label="Lifetime tokens"
            value={formatCompactNumber(lifetime.totalTokens)}
          />
          <ProfileMetric label="Peak day" value={formatCompactNumber(lifetime.peakDayTokens)} />
          <ProfileMetric label="Longest task" value={formatDuration(lifetime.longestTaskMs)} />
          <ProfileMetric label="Current streak" value={`${lifetime.currentStreakDays} days`} />
          <ProfileMetric label="Longest streak" value={`${lifetime.longestStreakDays} days`} />
        </dl>

        <section
          className="profile-activity [margin-top:38px]"
          aria-labelledby="profile-token-activity"
        >
          <div className="profile-section-heading [min-height:28px] [display:flex] [align-items:center] [justify-content:space-between] [gap:16px] [&_h3]:[margin:0] [&_h3]:[font-size:13px] [&_h3]:[font-weight:620]">
            <h3 id="profile-token-activity">Token activity</h3>
            <fieldset
              className="profile-tabs [margin:0] [padding:0] [display:flex] [align-items:center] [gap:2px] [border:0] [&_button]:[min-height:26px] [&_button]:[padding:3px_8px] [&_button]:[color:var(--muted-foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[font-size:11.5px] [&_button]:[cursor:pointer] [&_button.is-active]:[color:var(--foreground)] [&_button.is-active]:[font-weight:600]"
              aria-label="Token activity aggregation"
            >
              {(["daily", "weekly", "cumulative"] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={activityMode === mode ? "is-active" : undefined}
                  aria-pressed={activityMode === mode}
                  onClick={() => setActivityMode(mode)}
                >
                  {capitalize(mode)}
                </button>
              ))}
            </fieldset>
          </div>
          <div className="profile-heatmap-scroll [width:100%] [margin-top:10px] [padding-bottom:4px] [overflow-x:auto]">
            <div className="profile-heatmap-frame [width:100%] [min-width:680px]">
              <div
                className="profile-heatmap [display:grid] [grid-auto-flow:column] [grid-auto-columns:minmax(8px,_1fr)] [grid-template-rows:repeat(7,_12px)] [gap:4px]"
                role="img"
                aria-label={`${capitalize(activityMode)} token activity for the last 53 weeks`}
              >
                {heatmapDays.map((day) => (
                  <span
                    key={day.date}
                    className="profile-heatmap__day [display:block] [width:100%] [height:12px] [background:color-mix(in_srgb,_var(--muted-foreground)_9%,_transparent)] [border-radius:3px]"
                    data-level={day.intensity}
                    title={`${formatCalendarDate(day.date)}: ${formatNumber(day.tokens)} tokens`}
                  />
                ))}
              </div>
              <div
                className="profile-heatmap__months [margin-top:8px] [display:grid] [grid-template-columns:repeat(53,_minmax(8px,_1fr))] [gap:4px] [color:var(--muted-foreground)] [font-size:9.5px] [&_span]:[white-space:nowrap]"
                aria-hidden="true"
              >
                {monthLabels.map((month) => (
                  <span key={`${month.label}-${month.column}`} style={{ gridColumn: month.column }}>
                    {month.label}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="profile-token-breakdown [margin-top:20px] [display:flex] [flex-wrap:wrap] [gap:7px_18px] [color:var(--muted-foreground)] [font-size:10.5px] [font-variant-numeric:tabular-nums]">
            <span>Input {formatCompactNumber(lifetime.inputTokens)}</span>
            <span>Output {formatCompactNumber(lifetime.outputTokens)}</span>
            <span>Reasoning {formatCompactNumber(lifetime.reasoningTokens)}</span>
            <span>Cached {formatCompactNumber(lifetime.cachedInputTokens)}</span>
          </div>
        </section>

        <div className="profile-details-grid [margin-top:40px] [display:grid] [grid-template-columns:minmax(0,_1fr)_minmax(0,_1fr)] [gap:54px] max-680:[grid-template-columns:1fr] max-680:[gap:34px]">
          <section
            className="profile-insights [&_h3]:[margin:0] [&_h3]:[font-size:13px] [&_h3]:[font-weight:620] [&_dl]:[margin:13px_0_0] [&_dl]:[display:grid] [&_dl]:[gap:10px] [&_dl_>_div]:[min-width:0] [&_dl_>_div]:[display:flex] [&_dl_>_div]:[align-items:baseline] [&_dl_>_div]:[justify-content:space-between] [&_dl_>_div]:[gap:18px] [&_dt]:[color:var(--muted-foreground)] [&_dt]:[font-size:11.5px] [&_dd]:[max-width:60%] [&_dd]:[margin:0] [&_dd]:[overflow:hidden] [&_dd]:[font-size:11.5px] [&_dd]:[font-variant-numeric:tabular-nums] [&_dd]:[font-weight:570] [&_dd]:[text-overflow:ellipsis] [&_dd]:[white-space:nowrap]"
            aria-labelledby="profile-insights-title"
          >
            <h3 id="profile-insights-title">Activity insights</h3>
            <dl>
              <Insight label="Measured tokens" value={`${measuredShare}%`} />
              <Insight
                label="Most used reasoning"
                value={
                  topReasoning ? `${topReasoning.name} · ${topReasoning.count}` : "Not recorded"
                }
              />
              <Insight
                label="Most used model"
                value={topModel ? `${topModel.name} · ${topModel.count}` : "Not recorded"}
              />
              <Insight label="Skills explored" value={formatNumber(lifetime.skillsExplored)} />
              <Insight label="Total skill loads" value={formatNumber(lifetime.skillCalls)} />
              <Insight label="Total tool calls" value={formatNumber(lifetime.toolCalls)} />
              <Insight label="Total tasks" value={formatNumber(lifetime.totalTasks)} />
            </dl>
          </section>

          <section
            className="profile-ranking [&_ol]:[margin:8px_0_0] [&_ol]:[padding:0] [&_ol]:[display:grid] [&_ol]:[gap:4px] [&_ol]:[list-style:none] [&_li]:[min-width:0] [&_li]:[min-height:30px] [&_li]:[padding:3px_4px] [&_li]:[display:grid] [&_li]:[grid-template-columns:26px_minmax(0,_1fr)_auto] [&_li]:[align-items:center] [&_li]:[gap:7px] [&_li]:[border-radius:7px] [&_li_strong]:[overflow:hidden] [&_li_strong]:[font-size:11.5px] [&_li_strong]:[font-weight:570] [&_li_strong]:[text-overflow:ellipsis] [&_li_strong]:[white-space:nowrap]"
            aria-labelledby="profile-ranking-title"
          >
            <div className="profile-section-heading [min-height:28px] [display:flex] [align-items:center] [justify-content:space-between] [gap:16px] [&_h3]:[margin:0] [&_h3]:[font-size:13px] [&_h3]:[font-weight:620]">
              <h3 id="profile-ranking-title">Most used</h3>
              <fieldset
                className="profile-tabs [margin:0] [padding:0] [display:flex] [align-items:center] [gap:2px] [border:0] [&_button]:[min-height:26px] [&_button]:[padding:3px_8px] [&_button]:[color:var(--muted-foreground)] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[font-size:11.5px] [&_button]:[cursor:pointer] [&_button.is-active]:[color:var(--foreground)] [&_button.is-active]:[font-weight:600]"
                aria-label="Most used capability type"
              >
                {(["skills", "tools"] as const).map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={rankingMode === mode ? "is-active" : undefined}
                    aria-pressed={rankingMode === mode}
                    onClick={() => setRankingMode(mode)}
                  >
                    {capitalize(mode)}
                  </button>
                ))}
              </fieldset>
            </div>
            {ranked.length > 0 ? (
              <ol>
                {ranked.slice(0, 6).map((item) => (
                  <li key={item.name}>
                    <span
                      className="profile-ranking__icon [width:24px] [height:24px] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:999px] [&_svg]:[width:12px] [&_svg]:[height:12px]"
                      aria-hidden="true"
                    >
                      {rankingMode === "skills" ? <Sparkles /> : <Wrench />}
                    </span>
                    <strong title={item.name}>{item.name}</strong>
                    <span>{formatNumber(item.count)} runs</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="profile-ranking__empty [margin:14px_0_0] [color:var(--muted-foreground)] [font-size:11.5px]">
                {rankingMode === "skills" ? "No skills loaded yet." : "No tools called yet."}
              </p>
            )}
          </section>
        </div>

        <footer className="profile-tracking-note [margin-top:42px] [padding-top:14px] [color:var(--muted-foreground)] [border-top:1px_solid_var(--border-subtle)] [font-size:9.5px] [text-align:center]">
          {summary.trackingSince
            ? `Tracking since ${new Date(summary.trackingSince).toLocaleDateString()}`
            : "Tracking starts with your next task"}
          {lifetime.estimatedTokens > 0 &&
            ` · ${formatCompactNumber(lifetime.estimatedTokens)} tokens estimated for runtimes without usage data`}
        </footer>
      </div>
    </section>
  );
}

function ProfileMetric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function Insight({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export function buildHeatmapDays(
  summary: ActivityProfileSummary | undefined,
  mode: ActivityMode,
): HeatmapDay[] {
  const now = summary ? new Date(summary.generatedAt) : new Date();
  const end = startOfDay(now);
  end.setDate(end.getDate() + (6 - end.getDay()));
  const start = new Date(end);
  start.setDate(start.getDate() - 370);
  const source = new Map((summary?.daily ?? []).map((day) => [day.date, day.tokens]));
  const days = Array.from({ length: 371 }, (_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    const key = localDateKey(date);
    return { date: key, rawTokens: source.get(key) ?? 0, displayTokens: 0 };
  });

  if (mode === "daily") {
    for (const day of days) day.displayTokens = day.rawTokens;
  } else if (mode === "weekly") {
    for (let index = 0; index < days.length; index += 7) {
      const week = days.slice(index, index + 7);
      const total = week.reduce((sum, day) => sum + day.rawTokens, 0);
      for (const day of week) day.displayTokens = total;
    }
  } else {
    let total = 0;
    for (const day of days) {
      total += day.rawTokens;
      day.displayTokens = total;
    }
  }

  const maximum = Math.max(0, ...days.map((day) => day.displayTokens));
  return days.map((day) => ({
    date: day.date,
    tokens: day.displayTokens,
    intensity: heatmapIntensity(day.displayTokens, maximum),
  }));
}

function buildMonthLabels(days: readonly HeatmapDay[]): Array<{ label: string; column: number }> {
  const labels: Array<{ label: string; column: number }> = [];
  let previousMonth = -1;
  for (let index = 0; index < days.length; index += 7) {
    const date = localDateFromKey(days[index]?.date ?? "1970-01-01");
    const month = date.getMonth();
    if (month === previousMonth) continue;
    previousMonth = month;
    labels.push({
      label: date.toLocaleDateString(undefined, { month: "short" }),
      column: index / 7 + 1,
    });
  }
  return labels;
}

function heatmapIntensity(value: number, maximum: number): number {
  if (value <= 0 || maximum <= 0) return 0;
  return Math.min(4, Math.max(1, Math.ceil(Math.sqrt(value / maximum) * 4)));
}

function formatCompactNumber(value: number): string {
  if (value < 1_000) return formatNumber(value);
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(
    value,
  );
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function formatDuration(durationMs: number): string {
  if (durationMs <= 0) return "0m";
  const totalMinutes = Math.max(1, Math.round(durationMs / 60_000));
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

function formatCalendarDate(key: string): string {
  return localDateFromKey(key).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function capitalize(value: string): string {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function startOfDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function localDateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function localDateFromKey(key: string): Date {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year ?? 1970, (month ?? 1) - 1, day ?? 1);
}
