import {
  type AuditStore,
  projectSessionTimeline,
  readSessionTimelineSource,
  type SessionTimeline,
  sessionTimelineAuditRecords,
} from "@swarmx/core";

export interface SessionTimelineCommandOptions {
  json?: boolean;
}

export interface SessionTimelineCommandResult {
  exitCode: number;
  output: string;
}

export function runSessionTimelineCommand(
  sessionId: string,
  options: SessionTimelineCommandOptions,
  audit: Pick<AuditStore, "queryReadOnly">,
): SessionTimelineCommandResult {
  const source = readSessionTimelineSource(sessionId);
  if (!source) {
    return { exitCode: 1, output: `Session ${sessionId} was not found.\n` };
  }
  const evidence = audit.queryReadOnly({ sessionId: source.sessionId, limit: 10_000 });
  const timeline = projectSessionTimeline(source, sessionTimelineAuditRecords(evidence));
  return {
    exitCode: 0,
    output: options.json
      ? `${JSON.stringify(timeline, null, 2)}\n`
      : formatSessionTimeline(timeline),
  };
}

export function formatSessionTimeline(timeline: SessionTimeline): string {
  const lines = [
    `Session ${timeline.sessionId}`,
    "Derived diagnostic timeline — Session JSONL remains authoritative.",
  ];
  for (const [index, turn] of timeline.turns.entries()) {
    lines.push("", `Turn ${index + 1}: ${turn.status}`, `  ${turn.statusReason}`);
    const eventIds = new Set(turn.eventIds);
    for (const event of timeline.events.filter((candidate) => eventIds.has(candidate.eventId))) {
      const markers = [event.late ? "late" : "", event.inferred ? "inferred" : ""]
        .filter(Boolean)
        .join(", ");
      lines.push(`  ${event.ordinal}. ${event.summary}${markers ? ` (${markers})` : ""}`);
    }
  }
  if (timeline.unsettled.length > 0) {
    lines.push("", "Still unsettled:");
    for (const item of timeline.unsettled) lines.push(`  - ${item.summary}`);
  }
  if (timeline.diagnostics.length > 0) {
    lines.push("", "Diagnostic notes:");
    for (const item of timeline.diagnostics) lines.push(`  - ${item.summary}`);
  }
  if (timeline.turns.length === 0) lines.push("", "No Turn evidence was found.");
  return `${lines.join("\n")}\n`;
}
