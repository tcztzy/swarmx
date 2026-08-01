import {
  migrateLegacySessions,
  type SessionMigrationOptions,
  type SessionMigrationResult,
} from "@swarmx/core";

export interface SessionMigrationCommandOptions extends SessionMigrationOptions {
  json?: boolean;
}

export interface SessionMigrationCommandResult {
  result: SessionMigrationResult;
  output: string;
  exitCode: number;
}

export function runSessionMigrationCommand(
  options: SessionMigrationCommandOptions = {},
): SessionMigrationCommandResult {
  const result = migrateLegacySessions(options);
  return {
    result,
    output: options.json ? `${JSON.stringify(result, null, 2)}\n` : formatSessionMigration(result),
    exitCode: result.failed > 0 ? 1 : 0,
  };
}

export function formatSessionMigration(result: SessionMigrationResult): string {
  const lines = [
    `${result.planned > 0 ? "Would migrate" : "Migrated"} ${result.migrated + result.planned} of ${result.discovered} legacy Session files.`,
    `Skipped: ${result.skipped}; failed: ${result.failed}.`,
  ];
  if (result.backupDir) lines.push(`Backup: ${result.backupDir}`);
  for (const entry of result.sessions) {
    const detail = entry.error ? ` — ${entry.error}` : "";
    lines.push(`${entry.status.toUpperCase()} ${entry.id}${detail}`);
  }
  return `${lines.join("\n")}\n`;
}
