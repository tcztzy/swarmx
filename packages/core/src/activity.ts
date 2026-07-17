import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import { homedir } from "node:os";
import * as path from "node:path";
import { z } from "zod";
import { ModelTokenUsageSchema } from "./types.js";
import type { MessageChunk, ModelTokenUsage } from "./types.js";

const DEFAULT_ACTIVITY_FILE = path.join(homedir(), ".swarmx", "activity.jsonl");

export const ActivityEventTypeSchema = z.enum([
  "task_started",
  "task_finished",
  "token_usage",
  "tool_called",
  "skill_used",
]);

export const ActivityEventSchema = z
  .object({
    eventId: z.string().min(1),
    timestamp: z.string().datetime(),
    type: ActivityEventTypeSchema,
    taskId: z.string().min(1),
    sessionId: z.string().min(1).optional(),
    harnessId: z.string().min(1).optional(),
    modelId: z.string().min(1).optional(),
    reasoningEffort: z.string().min(1).optional(),
    status: z.enum(["completed", "failed", "canceled"]).optional(),
    durationMs: z.number().int().nonnegative().optional(),
    name: z.string().min(1).optional(),
    tokens: ModelTokenUsageSchema.optional(),
  })
  .superRefine((event, ctx) => {
    if (event.type === "task_finished" && !event.status) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["status"], message: "Status required" });
    }
    if (event.type === "token_usage" && !event.tokens) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["tokens"], message: "Tokens required" });
    }
    if ((event.type === "tool_called" || event.type === "skill_used") && !event.name) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, path: ["name"], message: "Name required" });
    }
  });

export const ActivityDaySchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  tokens: z.number().int().nonnegative(),
  estimatedTokens: z.number().int().nonnegative(),
  tasks: z.number().int().nonnegative(),
  tools: z.number().int().nonnegative(),
  skills: z.number().int().nonnegative(),
});

export const ActivityRankSchema = z.object({
  name: z.string().min(1),
  count: z.number().int().nonnegative(),
});

export const ActivityProfileSummarySchema = z.object({
  generatedAt: z.string().datetime(),
  trackingSince: z.string().datetime().optional(),
  lifetime: z.object({
    totalTokens: z.number().int().nonnegative(),
    inputTokens: z.number().int().nonnegative(),
    outputTokens: z.number().int().nonnegative(),
    reasoningTokens: z.number().int().nonnegative(),
    cachedInputTokens: z.number().int().nonnegative(),
    estimatedTokens: z.number().int().nonnegative(),
    peakDayTokens: z.number().int().nonnegative(),
    longestTaskMs: z.number().int().nonnegative(),
    currentStreakDays: z.number().int().nonnegative(),
    longestStreakDays: z.number().int().nonnegative(),
    totalTasks: z.number().int().nonnegative(),
    completedTasks: z.number().int().nonnegative(),
    toolCalls: z.number().int().nonnegative(),
    skillCalls: z.number().int().nonnegative(),
    skillsExplored: z.number().int().nonnegative(),
  }),
  daily: z.array(ActivityDaySchema),
  topTools: z.array(ActivityRankSchema),
  topSkills: z.array(ActivityRankSchema),
  reasoningEfforts: z.array(ActivityRankSchema),
  models: z.array(ActivityRankSchema),
});

export type ActivityEventType = z.infer<typeof ActivityEventTypeSchema>;
export type ActivityEvent = z.infer<typeof ActivityEventSchema>;
export type ActivityDay = z.infer<typeof ActivityDaySchema>;
export type ActivityRank = z.infer<typeof ActivityRankSchema>;
export type ActivityProfileSummary = z.infer<typeof ActivityProfileSummarySchema>;

export type ActivityEventInput = Omit<ActivityEvent, "eventId" | "timestamp"> & {
  eventId?: string;
  timestamp?: string;
};

export interface ActivityStoreOptions {
  filePath?: string;
  now?: () => Date;
}

export class ActivityStore {
  readonly filePath: string;
  private readonly now: () => Date;

  constructor(options: ActivityStoreOptions = {}) {
    this.filePath = options.filePath ?? DEFAULT_ACTIVITY_FILE;
    this.now = options.now ?? (() => new Date());
  }

  append(input: ActivityEventInput): ActivityEvent {
    const event = ActivityEventSchema.parse({
      ...input,
      eventId: input.eventId ?? `act_${randomUUID()}`,
      timestamp: input.timestamp ?? this.now().toISOString(),
    });
    fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    fs.appendFileSync(this.filePath, `${JSON.stringify(event)}\n`, {
      encoding: "utf8",
      mode: 0o600,
    });
    return event;
  }

  events(): ActivityEvent[] {
    if (!fs.existsSync(this.filePath)) return [];
    try {
      return fs
        .readFileSync(this.filePath, "utf8")
        .split(/\r?\n/u)
        .flatMap((line) => {
          const trimmed = line.trim();
          if (!trimmed) return [];
          try {
            const parsed = ActivityEventSchema.safeParse(JSON.parse(trimmed));
            return parsed.success ? [parsed.data] : [];
          } catch {
            return [];
          }
        });
    } catch {
      return [];
    }
  }

  summary(): ActivityProfileSummary {
    return summarizeActivityEvents(this.events(), this.now());
  }
}

export function summarizeActivityEvents(
  eventsInput: readonly ActivityEvent[],
  now = new Date(),
): ActivityProfileSummary {
  const events = eventsInput
    .flatMap((event) => {
      const parsed = ActivityEventSchema.safeParse(event);
      return parsed.success ? [parsed.data] : [];
    })
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp));
  const days = new Map<string, ActivityDay>();
  const tools = new Map<string, number>();
  const skills = new Map<string, number>();
  const reasoningEfforts = new Map<string, number>();
  const models = new Map<string, number>();
  const lifetime = {
    totalTokens: 0,
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    cachedInputTokens: 0,
    estimatedTokens: 0,
    peakDayTokens: 0,
    longestTaskMs: 0,
    currentStreakDays: 0,
    longestStreakDays: 0,
    totalTasks: 0,
    completedTasks: 0,
    toolCalls: 0,
    skillCalls: 0,
    skillsExplored: 0,
  };

  for (const event of events) {
    const date = localDateKey(new Date(event.timestamp));
    const day = days.get(date) ?? {
      date,
      tokens: 0,
      estimatedTokens: 0,
      tasks: 0,
      tools: 0,
      skills: 0,
    };
    if (event.type === "task_started") {
      increment(reasoningEfforts, event.reasoningEffort);
      increment(models, event.modelId);
    } else if (event.type === "task_finished") {
      lifetime.totalTasks += 1;
      lifetime.completedTasks += event.status === "completed" ? 1 : 0;
      lifetime.longestTaskMs = Math.max(lifetime.longestTaskMs, event.durationMs ?? 0);
      day.tasks += 1;
    } else if (event.type === "token_usage" && event.tokens) {
      lifetime.totalTokens += event.tokens.totalTokens;
      lifetime.inputTokens += event.tokens.inputTokens;
      lifetime.outputTokens += event.tokens.outputTokens;
      lifetime.reasoningTokens += event.tokens.reasoningTokens;
      lifetime.cachedInputTokens += event.tokens.cachedInputTokens;
      lifetime.estimatedTokens += event.tokens.estimated ? event.tokens.totalTokens : 0;
      day.tokens += event.tokens.totalTokens;
      day.estimatedTokens += event.tokens.estimated ? event.tokens.totalTokens : 0;
    } else if (event.type === "tool_called" && event.name) {
      lifetime.toolCalls += 1;
      day.tools += 1;
      increment(tools, event.name);
    } else if (event.type === "skill_used" && event.name) {
      lifetime.skillCalls += 1;
      day.skills += 1;
      increment(skills, event.name);
    }
    days.set(date, day);
  }

  const daily = [...days.values()].sort((left, right) => left.date.localeCompare(right.date));
  lifetime.peakDayTokens = daily.reduce((peak, day) => Math.max(peak, day.tokens), 0);
  lifetime.skillsExplored = skills.size;
  const streaks = activityStreaks(daily, now);
  lifetime.currentStreakDays = streaks.current;
  lifetime.longestStreakDays = streaks.longest;

  return ActivityProfileSummarySchema.parse({
    generatedAt: now.toISOString(),
    trackingSince: events[0]?.timestamp,
    lifetime,
    daily,
    topTools: ranked(tools),
    topSkills: ranked(skills),
    reasoningEfforts: ranked(reasoningEfforts),
    models: ranked(models),
  });
}

export function mergeModelTokenUsage(usages: readonly ModelTokenUsage[]): ModelTokenUsage {
  const parsed = usages.map((usage) => ModelTokenUsageSchema.parse(usage));
  return ModelTokenUsageSchema.parse({
    inputTokens: parsed.reduce((sum, usage) => sum + usage.inputTokens, 0),
    outputTokens: parsed.reduce((sum, usage) => sum + usage.outputTokens, 0),
    reasoningTokens: parsed.reduce((sum, usage) => sum + usage.reasoningTokens, 0),
    cachedInputTokens: parsed.reduce((sum, usage) => sum + usage.cachedInputTokens, 0),
    totalTokens: parsed.reduce((sum, usage) => sum + usage.totalTokens, 0),
    estimated: parsed.length === 0 || parsed.some((usage) => usage.estimated),
    model: uniqueValue(parsed.map((usage) => usage.model)),
    provider: uniqueValue(parsed.map((usage) => usage.provider)),
  });
}

export function estimateModelTokenUsage(
  input: string,
  messages: readonly MessageChunk[],
  metadata: { model?: string; provider?: string } = {},
): ModelTokenUsage {
  const inputTokens = estimateTextTokens(input);
  const reasoningTokens = messages
    .filter((message) => message.kind === "thinking")
    .reduce((sum, message) => sum + estimateTextTokens(message.content), 0);
  const outputTokens = messages
    .filter((message) => message.kind !== "thinking" && message.role !== "user")
    .reduce((sum, message) => sum + estimateTextTokens(message.content), 0);
  return ModelTokenUsageSchema.parse({
    inputTokens,
    outputTokens,
    reasoningTokens,
    cachedInputTokens: 0,
    totalTokens: inputTokens + outputTokens + reasoningTokens,
    estimated: true,
    model: metadata.model,
    provider: metadata.provider,
  });
}

export function estimateTextTokens(text: string): number {
  const compact = text.trim();
  if (!compact) return 0;
  const cjkCount = (
    compact.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) ?? []
  ).length;
  const otherCount = [...compact].length - cjkCount;
  return cjkCount + Math.ceil(otherCount / 4);
}

function activityStreaks(
  days: readonly ActivityDay[],
  now: Date,
): { current: number; longest: number } {
  const active = new Set(
    days
      .filter((day) => day.tokens > 0 || day.tasks > 0 || day.tools > 0 || day.skills > 0)
      .map((day) => day.date),
  );
  const sorted = [...active].sort();
  let longest = 0;
  let running = 0;
  let previous: Date | undefined;
  for (const key of sorted) {
    const date = localDateFromKey(key);
    running = previous && differenceInCalendarDays(previous, date) === 1 ? running + 1 : 1;
    longest = Math.max(longest, running);
    previous = date;
  }

  let cursor = startOfLocalDay(now);
  if (!active.has(localDateKey(cursor))) cursor = addLocalDays(cursor, -1);
  let current = 0;
  while (active.has(localDateKey(cursor))) {
    current += 1;
    cursor = addLocalDays(cursor, -1);
  }
  return { current, longest };
}

function ranked(values: ReadonlyMap<string, number>): ActivityRank[] {
  return [...values.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((left, right) => right.count - left.count || left.name.localeCompare(right.name));
}

function increment(values: Map<string, number>, key: string | undefined): void {
  if (key) values.set(key, (values.get(key) ?? 0) + 1);
}

function uniqueValue(values: Array<string | undefined>): string | undefined {
  const unique = [...new Set(values.filter((value): value is string => Boolean(value)))];
  return unique.length === 1 ? unique[0] : undefined;
}

function localDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function localDateFromKey(key: string): Date {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year ?? 1970, (month ?? 1) - 1, day ?? 1);
}

function startOfLocalDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function addLocalDays(date: Date, days: number): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + days);
}

function differenceInCalendarDays(left: Date, right: Date): number {
  const leftUtc = Date.UTC(left.getFullYear(), left.getMonth(), left.getDate());
  const rightUtc = Date.UTC(right.getFullYear(), right.getMonth(), right.getDate());
  return Math.round((rightUtc - leftUtc) / 86_400_000);
}
