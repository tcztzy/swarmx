import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  ActivityStore,
  estimateModelTokenUsage,
  mergeModelTokenUsage,
  summarizeActivityEvents,
} from "../src/activity.js";
import type { ActivityEvent } from "../src/activity.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

describe("local activity profile", () => {
  it("aggregates tokens, tasks, streaks, tools, skills, models, and reasoning", () => {
    const events: ActivityEvent[] = [
      event("task_started", "2026-07-13T09:00:00.000Z", {
        modelId: "gpt-5",
        reasoningEffort: "high",
      }),
      event("token_usage", "2026-07-13T09:01:00.000Z", {
        tokens: usage(100, 40, 20, 10, false),
      }),
      event("task_finished", "2026-07-13T09:02:00.000Z", {
        status: "completed",
        durationMs: 120_000,
      }),
      event("tool_called", "2026-07-13T09:01:30.000Z", { name: "read_file" }),
      event("skill_used", "2026-07-13T09:00:05.000Z", { name: "paper-reviewer" }),
      event("task_started", "2026-07-14T09:00:00.000Z", {
        modelId: "gpt-5",
        reasoningEffort: "high",
      }),
      event("token_usage", "2026-07-14T09:01:00.000Z", {
        tokens: usage(20, 10, 0, 0, true),
      }),
      event("task_finished", "2026-07-14T09:02:00.000Z", {
        status: "failed",
        durationMs: 180_000,
      }),
      event("tool_called", "2026-07-14T09:01:30.000Z", { name: "read_file" }),
      event("skill_used", "2026-07-14T09:00:05.000Z", { name: "code-reviewer" }),
      event("task_finished", "2026-07-16T09:02:00.000Z", {
        status: "completed",
        durationMs: 60_000,
      }),
    ];

    const profile = summarizeActivityEvents(events, new Date("2026-07-16T12:00:00.000Z"));

    expect(profile.lifetime).toMatchObject({
      totalTokens: 190,
      inputTokens: 120,
      outputTokens: 50,
      reasoningTokens: 20,
      cachedInputTokens: 10,
      estimatedTokens: 30,
      peakDayTokens: 160,
      longestTaskMs: 180_000,
      currentStreakDays: 1,
      longestStreakDays: 2,
      totalTasks: 3,
      completedTasks: 2,
      toolCalls: 2,
      skillCalls: 2,
      skillsExplored: 2,
    });
    expect(profile.topTools[0]).toEqual({ name: "read_file", count: 2 });
    expect(profile.topSkills).toEqual([
      { name: "code-reviewer", count: 1 },
      { name: "paper-reviewer", count: 1 },
    ]);
    expect(profile.models[0]).toEqual({ name: "gpt-5", count: 2 });
    expect(profile.reasoningEfforts[0]).toEqual({ name: "high", count: 2 });
  });

  it("persists privacy-safe JSONL and skips malformed records", () => {
    const directory = mkdtempSync(path.join(tmpdir(), "swarmx-activity-"));
    temporaryDirectories.push(directory);
    const filePath = path.join(directory, "activity.jsonl");
    const store = new ActivityStore({
      filePath,
      now: () => new Date("2026-07-16T10:00:00.000Z"),
    });

    store.append({
      type: "tool_called",
      taskId: "task-1",
      sessionId: "session-1",
      name: "workspace_read_file",
    });
    writeFileSync(filePath, `${readFileSync(filePath, "utf8")}not-json\n`, "utf8");

    expect(store.events()).toHaveLength(1);
    expect(store.summary().topTools).toEqual([{ name: "workspace_read_file", count: 1 }]);
    expect(readFileSync(filePath, "utf8")).not.toContain("prompt");
  });

  it("estimates missing runtime usage and merges measured provider steps", () => {
    const estimated = estimateModelTokenUsage("你好 test", [
      { role: "assistant", kind: "thinking", content: "考虑" },
      { role: "assistant", kind: "message", content: "done" },
    ]);
    expect(estimated).toMatchObject({ estimated: true, inputTokens: 4, reasoningTokens: 2 });

    expect(
      mergeModelTokenUsage([usage(10, 5, 2, 3, false), usage(20, 8, 4, 6, false)]),
    ).toMatchObject({
      inputTokens: 30,
      outputTokens: 13,
      reasoningTokens: 6,
      cachedInputTokens: 9,
      totalTokens: 49,
      estimated: false,
    });
  });
});

function event(
  type: ActivityEvent["type"],
  timestamp: string,
  extra: Partial<ActivityEvent>,
): ActivityEvent {
  return {
    eventId: `event-${timestamp}-${type}`,
    taskId: `task-${timestamp}`,
    timestamp,
    type,
    ...extra,
  } as ActivityEvent;
}

function usage(
  inputTokens: number,
  outputTokens: number,
  reasoningTokens: number,
  cachedInputTokens: number,
  estimated: boolean,
) {
  return {
    inputTokens,
    outputTokens,
    reasoningTokens,
    cachedInputTokens,
    totalTokens: inputTokens + outputTokens + reasoningTokens,
    estimated,
  };
}
