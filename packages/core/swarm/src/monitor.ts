import type { MonitorFinding, SwarmAttempt, SwarmTeamState } from "./contracts.js";

export interface SwarmMonitorOptions {
  readonly now: number;
  readonly stallMs: number;
  readonly maxPendingMessagesPerMember: number;
  readonly runningMemberIds?: ReadonlySet<string>;
}

export type MonitorFindingDraft = Omit<MonitorFinding, "id">;

function draft(
  options: SwarmMonitorOptions,
  input: Omit<MonitorFindingDraft, "dedupeKey" | "recordedAt">,
  suffix: string,
): MonitorFindingDraft {
  return {
    ...input,
    dedupeKey: `${input.code}:${input.subject.kind}:${input.subject.id}:${suffix}`,
    recordedAt: options.now,
  };
}

function knownLimitFindings(
  attempt: SwarmAttempt,
  options: SwarmMonitorOptions,
): MonitorFindingDraft[] {
  const actor = attempt.actors.find((candidate) => candidate.endedAt === undefined);
  const budget = actor?.budget ?? attempt.budget;
  const usage = actor?.usage ?? attempt.usage;
  if (
    (attempt.status !== "active" && attempt.status !== "verifying") ||
    usage.availability !== "known"
  ) {
    return [];
  }
  const subject = { kind: "attempt" as const, id: attempt.id };
  const suffix = `${actor?.phase ?? "implementation"}:${actor?.memberName ?? attempt.memberName}`;
  const findings: MonitorFindingDraft[] = [];
  if (budget?.maxTurns !== undefined && usage.turns > budget.maxTurns) {
    findings.push(
      draft(
        options,
        {
          action: "needs_attention",
          code: "attempt_turns_exhausted",
          severity: "block",
          subject,
          summary: "Attempt exceeded its observed turn ceiling.",
        },
        suffix,
      ),
    );
  }
  if (
    budget?.maxInputTokens !== undefined &&
    usage.inputTokens + usage.cacheReadTokens + usage.cacheWriteTokens > budget.maxInputTokens
  ) {
    findings.push(
      draft(
        options,
        {
          action: "needs_attention",
          code: "attempt_input_tokens_exhausted",
          severity: "block",
          subject,
          summary: "Attempt exceeded its observed input-token ceiling.",
        },
        suffix,
      ),
    );
  }
  if (budget?.maxOutputTokens !== undefined && usage.outputTokens > budget.maxOutputTokens) {
    findings.push(
      draft(
        options,
        {
          action: "needs_attention",
          code: "attempt_output_tokens_exhausted",
          severity: "block",
          subject,
          summary: "Attempt exceeded its observed output-token ceiling.",
        },
        suffix,
      ),
    );
  }
  return findings;
}

/** Pure, event-invoked monitor evaluation. It performs no I/O and never calls a model. */
export function evaluateSwarmMonitor(
  team: SwarmTeamState,
  options: SwarmMonitorOptions,
): MonitorFindingDraft[] {
  const findings: MonitorFindingDraft[] = [];
  for (const attempt of team.attempts) {
    const subject = { kind: "attempt" as const, id: attempt.id };
    const actor = attempt.actors.find((candidate) => candidate.endedAt === undefined);
    const budget = actor?.budget ?? attempt.budget;
    const usage = actor?.usage ?? attempt.usage;
    const actorMember = actor
      ? team.members.find((member) => member.name === actor.memberName)
      : undefined;
    const startedAt = actor?.startedAt ?? attempt.startedAt;
    const actorSuffix = `${actor?.phase ?? "implementation"}:${actor?.memberName ?? attempt.memberName}`;
    const wall = Math.max(0, options.now - startedAt);
    if (
      (attempt.status === "active" || attempt.status === "verifying") &&
      budget?.maxWallMs !== undefined
    ) {
      if (wall > budget.maxWallMs) {
        findings.push(
          draft(
            options,
            {
              action: "needs_attention",
              code: "attempt_wall_exhausted",
              severity: "block",
              subject,
              summary: "Attempt exceeded its hard wall-clock deadline.",
            },
            actorSuffix,
          ),
        );
      } else if (wall >= budget.maxWallMs * budget.warningFraction) {
        findings.push(
          draft(
            options,
            {
              action: "notify",
              code: "attempt_wall_warning",
              severity: "warning",
              subject,
              summary: "Attempt is approaching its hard wall-clock deadline.",
            },
            actorSuffix,
          ),
        );
      }
    }
    const consideredRunning =
      options.runningMemberIds === undefined ||
      options.runningMemberIds.has(actorMember?.id ?? attempt.ownerId);
    if (
      consideredRunning &&
      ["active", "submitted", "verifying"].includes(attempt.status) &&
      options.now - attempt.lastProgressAt > options.stallMs
    ) {
      const task = team.tasks.find((candidate) => candidate.id === attempt.taskId);
      findings.push(
        draft(
          options,
          {
            action: "lead_review",
            code:
              attempt.status === "active" && task?.kind === "write"
                ? "write_attempt_stalled"
                : "attempt_stalled",
            severity: "escalate",
            subject,
            summary: "Attempt has no observable progress beyond the configured interval.",
          },
          attempt.status,
        ),
      );
    }
    if (
      usage.availability === "unknown" &&
      (budget?.maxInputTokens !== undefined || budget?.maxOutputTokens !== undefined) &&
      attempt.status !== "active"
    ) {
      findings.push(
        draft(
          options,
          {
            action: "notify",
            code: "usage_unknown",
            severity: "warning",
            subject,
            summary: "Provider token usage is unavailable for this attempt.",
          },
          attempt.status,
        ),
      );
    }
    findings.push(...knownLimitFindings(attempt, options));
  }

  for (const member of team.members) {
    if (member.phase === "failed") {
      findings.push(
        draft(
          options,
          {
            action: "lead_review",
            code: "member_lifecycle_failure",
            severity: "escalate",
            subject: { kind: "member", id: member.name },
            summary: "A Team member failed during provisioning or lifecycle recovery.",
          },
          member.phase,
        ),
      );
    }
    const pending = team.messages.filter(
      (message) => message.targetId === member.id && message.deliveredAt === undefined,
    ).length;
    const near = Math.ceil(options.maxPendingMessagesPerMember * 0.9);
    if (pending >= options.maxPendingMessagesPerMember) {
      findings.push(
        draft(
          options,
          {
            action: "lead_review",
            code: "mailbox_limit_reached",
            severity: "block",
            subject: { kind: "member", id: member.name },
            summary: "A Team mailbox reached its configured pending-message limit.",
          },
          "full",
        ),
      );
    } else if (pending >= near) {
      findings.push(
        draft(
          options,
          {
            action: "notify",
            code: "mailbox_near_limit",
            severity: "warning",
            subject: { kind: "member", id: member.name },
            summary: "A Team mailbox is approaching its pending-message limit.",
          },
          "near",
        ),
      );
    }
  }

  for (const task of team.tasks) {
    if (task.status === "submitted" && task.acceptance && task.submission) {
      const labels = new Set(task.submission.artifactLocators.map((locator) => locator.label));
      for (const expected of task.acceptance.expectedArtifacts) {
        if (labels.has(expected)) continue;
        findings.push(
          draft(
            options,
            {
              action: "lead_review",
              code: "submission_missing_artifact",
              severity: "block",
              subject: { kind: "task", id: task.id },
              summary: "Submission is missing an expected artifact label.",
            },
            expected,
          ),
        );
      }
      if (
        task.acceptance.requiredChecks.length > 0 &&
        task.submission.evidenceDigests.length === 0
      ) {
        findings.push(
          draft(
            options,
            {
              action: "lead_review",
              code: "submission_missing_evidence",
              severity: "block",
              subject: { kind: "task", id: task.id },
              summary: "Submission has required checks but no evidence digest.",
            },
            task.submission.id,
          ),
        );
      }
    }
    const failures = team.attempts.filter(
      (attempt) => attempt.taskId === task.id && attempt.verification?.verdict === "fail",
    ).length;
    if (failures >= 2) {
      findings.push(
        draft(
          options,
          {
            action: "lead_review",
            code: "verification_repeated_failure",
            severity: "escalate",
            subject: { kind: "task", id: task.id },
            summary: "Task has repeated independent verification failures.",
          },
          String(failures),
        ),
      );
    }
  }

  const existing = new Set(team.findings.map((finding) => finding.dedupeKey));
  return findings.filter((finding) => !existing.has(finding.dedupeKey));
}
