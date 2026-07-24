import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { runSessionMigrationCommand } from "../src/session-migration.js";

describe("Session migration command", () => {
  it("V523 reports a dry run without moving legacy data", () => {
    const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-cli-migration-"));
    const legacyPath = path.join(sessionsDir, "legacy-cli.json");
    fs.writeFileSync(
      legacyPath,
      JSON.stringify({
        id: "legacy-cli",
        title: "CLI migration",
        agentName: "agent",
        harness: "swarmx",
        messages: [],
        createdAt: "2026-07-01T00:00:00.000Z",
        updatedAt: "2026-07-01T00:00:00.000Z",
      }),
      "utf8",
    );

    try {
      const command = runSessionMigrationCommand({ sessionsDir, dryRun: true });
      expect(command.exitCode).toBe(0);
      expect(command.output).toContain("Would migrate 1 of 1");
      expect(command.output).toContain("PLANNED legacy-cli");
      expect(fs.existsSync(legacyPath)).toBe(true);
      expect(fs.existsSync(path.join(sessionsDir, "legacy-cli.jsonl"))).toBe(false);
    } finally {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
    }
  });

  it("V523 returns a failing exit code without exposing Session message content", () => {
    const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-cli-migration-"));
    fs.writeFileSync(
      path.join(sessionsDir, "invalid.json"),
      '{"messages":[{"content":"private marker"}]}',
      "utf8",
    );

    try {
      const command = runSessionMigrationCommand({ sessionsDir });
      expect(command.exitCode).toBe(1);
      expect(command.output).toContain("FAILED invalid");
      expect(command.output).not.toContain("private marker");
      expect(command.output.length).toBeLessThan(1_000);
      expect(
        runSessionMigrationCommand({ sessionsDir, dryRun: true, json: true }).output,
      ).not.toContain("private marker");
    } finally {
      fs.rmSync(sessionsDir, { recursive: true, force: true });
    }
  });
});
