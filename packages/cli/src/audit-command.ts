import { chmodSync, writeFileSync } from "node:fs";
import {
  type AuditEvent,
  type AuditQuery,
  AuditQuerySchema,
  AuditStore,
  type AuditVerification,
} from "@swarmx/core";

export interface AuditCommandOptions {
  verify?: boolean;
  json?: boolean;
  output?: string;
  limit?: number | string;
  category?: string;
  action?: string;
  outcome?: string;
  actorId?: string;
  targetId?: string;
  sessionId?: string;
  taskId?: string;
  requestId?: string;
  from?: string;
  to?: string;
  reverse?: boolean;
}

export interface AuditCommandResult {
  output: string;
  exitCode: number;
  events?: AuditEvent[];
  verification: AuditVerification;
  exportedTo?: string;
}

export function runAuditCommand(
  options: AuditCommandOptions = {},
  store = new AuditStore(),
): AuditCommandResult {
  const query = auditQuery(options);
  const verification = store.verify();
  if (!verification.ok) {
    return {
      output: options.json
        ? `${JSON.stringify({ verification }, null, 2)}\n`
        : formatVerification(verification),
      exitCode: 1,
      verification,
    };
  }

  if (options.output) {
    store.append({
      category: "system",
      action: "audit.export",
      outcome: "attempted",
      actor: { kind: "user", id: "cli" },
      metadata: { filtered: hasFilters(query) },
    });
    try {
      const exported = store.exportJsonl(query);
      writeFileSync(options.output, exported, { encoding: "utf8", mode: 0o600 });
      chmodSync(options.output, 0o600);
      store.append({
        category: "system",
        action: "audit.export",
        outcome: "completed",
        actor: { kind: "user", id: "cli" },
        metadata: { eventCount: countJsonlRecords(exported), filtered: hasFilters(query) },
      });
      return {
        output: options.json
          ? `${JSON.stringify({ exported: true, verification }, null, 2)}\n`
          : "Exported verified audit JSONL.\n",
        exitCode: 0,
        verification,
        exportedTo: options.output,
      };
    } catch (error) {
      store.append({
        category: "system",
        action: "audit.export",
        outcome: "failed",
        actor: { kind: "user", id: "cli" },
        metadata: { errorType: errorName(error) },
      });
      throw error;
    }
  }

  if (options.verify) {
    return {
      output: options.json
        ? `${JSON.stringify({ verification }, null, 2)}\n`
        : formatVerification(verification),
      exitCode: 0,
      verification,
    };
  }

  const events = store.query(query);
  return {
    output: options.json ? `${JSON.stringify(events, null, 2)}\n` : formatEvents(events),
    exitCode: 0,
    events,
    verification,
  };
}

function auditQuery(options: AuditCommandOptions): AuditQuery {
  const limit =
    typeof options.limit === "string" ? Number.parseInt(options.limit, 10) : options.limit;
  return AuditQuerySchema.parse({
    ...(options.category ? { category: options.category } : {}),
    ...(options.action ? { action: options.action } : {}),
    ...(options.outcome ? { outcome: options.outcome } : {}),
    ...(options.actorId ? { actorId: options.actorId } : {}),
    ...(options.targetId ? { targetId: options.targetId } : {}),
    ...(options.sessionId ? { sessionId: options.sessionId } : {}),
    ...(options.taskId ? { taskId: options.taskId } : {}),
    ...(options.requestId ? { requestId: options.requestId } : {}),
    ...(options.from ? { from: options.from } : {}),
    ...(options.to ? { to: options.to } : {}),
    ...(limit !== undefined ? { limit } : {}),
    reverse: options.reverse ?? true,
  });
}

function formatVerification(verification: AuditVerification): string {
  if (!verification.ok) {
    return `Audit verification failed: ${verification.issue?.code ?? "unknown"}.\n`;
  }
  return `Audit chain verified: ${verification.eventCount} events, head ${verification.headSequence}.\n`;
}

function formatEvents(events: readonly AuditEvent[]): string {
  if (events.length === 0) return "No audit events matched.\n";
  return `${events
    .map((event) => {
      const correlations = [
        event.requestId ? `request=${event.requestId}` : "",
        event.sessionId ? `session=${event.sessionId}` : "",
        event.taskId ? `task=${event.taskId}` : "",
      ].filter(Boolean);
      return `${event.sequence}\t${event.timestamp}\t${event.category}.${event.action}\t${event.outcome}${correlations.length > 0 ? `\t${correlations.join(" ")}` : ""}`;
    })
    .join("\n")}\n`;
}

function hasFilters(query: AuditQuery): boolean {
  return Object.keys(query).some((key) => key !== "reverse" && key !== "limit");
}

function countJsonlRecords(value: string): number {
  return value ? value.trimEnd().split("\n").length : 0;
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : "Error";
}
