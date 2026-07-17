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
      <section className="profile-workspace" aria-label="Settings">
        <output className="profile-workspace__state">Loading local activity…</output>
      </section>
    );
  }

  if (error || !summary) {
    return (
      <section className="profile-workspace" aria-label="Settings">
        <div className="profile-workspace__state is-error">
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
    <section className="profile-workspace" aria-label="Settings">
      <div className="profile-workspace__content">
        <header className="profile-hero">
          <span className="profile-hero__avatar" aria-hidden="true">
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

        <dl className="profile-metrics" aria-label="Lifetime activity summary">
          <ProfileMetric
            label="Lifetime tokens"
            value={formatCompactNumber(lifetime.totalTokens)}
          />
          <ProfileMetric label="Peak day" value={formatCompactNumber(lifetime.peakDayTokens)} />
          <ProfileMetric label="Longest task" value={formatDuration(lifetime.longestTaskMs)} />
          <ProfileMetric label="Current streak" value={`${lifetime.currentStreakDays} days`} />
          <ProfileMetric label="Longest streak" value={`${lifetime.longestStreakDays} days`} />
        </dl>

        <section className="profile-activity" aria-labelledby="profile-token-activity">
          <div className="profile-section-heading">
            <h3 id="profile-token-activity">Token activity</h3>
            <fieldset className="profile-tabs" aria-label="Token activity aggregation">
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
          <div className="profile-heatmap-scroll">
            <div className="profile-heatmap-frame">
              <div
                className="profile-heatmap"
                role="img"
                aria-label={`${capitalize(activityMode)} token activity for the last 53 weeks`}
              >
                {heatmapDays.map((day) => (
                  <span
                    key={day.date}
                    className="profile-heatmap__day"
                    data-level={day.intensity}
                    title={`${formatCalendarDate(day.date)}: ${formatNumber(day.tokens)} tokens`}
                  />
                ))}
              </div>
              <div className="profile-heatmap__months" aria-hidden="true">
                {monthLabels.map((month) => (
                  <span key={`${month.label}-${month.column}`} style={{ gridColumn: month.column }}>
                    {month.label}
                  </span>
                ))}
              </div>
            </div>
          </div>
          <div className="profile-token-breakdown">
            <span>Input {formatCompactNumber(lifetime.inputTokens)}</span>
            <span>Output {formatCompactNumber(lifetime.outputTokens)}</span>
            <span>Reasoning {formatCompactNumber(lifetime.reasoningTokens)}</span>
            <span>Cached {formatCompactNumber(lifetime.cachedInputTokens)}</span>
          </div>
        </section>

        <div className="profile-details-grid">
          <section className="profile-insights" aria-labelledby="profile-insights-title">
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

          <section className="profile-ranking" aria-labelledby="profile-ranking-title">
            <div className="profile-section-heading">
              <h3 id="profile-ranking-title">Most used</h3>
              <fieldset className="profile-tabs" aria-label="Most used capability type">
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
                    <span className="profile-ranking__icon" aria-hidden="true">
                      {rankingMode === "skills" ? <Sparkles /> : <Wrench />}
                    </span>
                    <strong title={item.name}>{item.name}</strong>
                    <span>{formatNumber(item.count)} runs</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="profile-ranking__empty">
                {rankingMode === "skills" ? "No skills loaded yet." : "No tools called yet."}
              </p>
            )}
          </section>
        </div>

        <footer className="profile-tracking-note">
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
