import type { SessionInfo } from "@agentclientprotocol/sdk";
import { describe, expect, it } from "vitest";
import { acpSessionToDiscovered, groupDiscoveredSessions } from "../src/session-discovery.js";
import type { DiscoveredSession } from "../src/session-discovery.js";

describe("Session discovery", () => {
  it("groups sessions by harness", () => {
    const groups = groupDiscoveredSessions(
      [
        discovered({ id: "claude-1", harnessId: "claude_code", harnessLabel: "Claude Code" }),
        discovered({ id: "codex-1", harnessId: "codex", harnessLabel: "Codex" }),
        discovered({
          id: "codex-2",
          harnessId: "codex",
          harnessLabel: "Codex",
          updatedAt: "2026-01-03T00:00:00Z",
        }),
      ],
      "harness",
    );

    expect(groups).toHaveLength(2);
    expect(groups.map((group) => group.label)).toEqual(["Codex", "Claude Code"]);
    expect(groups[0].sessions.map((session) => session.id)).toEqual(["codex-2", "codex-1"]);
  });

  it("groups sessions by project working directory", () => {
    const groups = groupDiscoveredSessions(
      [
        discovered({ id: "root", cwd: "/Users/test/swarmx", updatedAt: "2026-01-02T00:00:00Z" }),
        discovered({ id: "other", cwd: "/Users/test/other", updatedAt: "2026-01-03T00:00:00Z" }),
        discovered({ id: "missing-cwd", cwd: "" }),
      ],
      "project",
    );

    expect(groups.map((group) => group.label)).toEqual([
      "/Users/test/other",
      "/Users/test/swarmx",
      "No project",
    ]);
    expect(groups[0].sessions[0].id).toBe("other");
    expect(groups[2].sessions[0].id).toBe("missing-cwd");
  });

  it("normalizes ACP session info with harness metadata", () => {
    const session = acpSessionToDiscovered(
      {
        sessionId: "codex-session",
        title: "Fix tests",
        cwd: "/Users/test/swarmx",
        updatedAt: "2026-01-04T00:00:00Z",
      } as SessionInfo,
      "codex",
    );

    expect(session).toEqual({
      id: "codex-session",
      title: "Fix tests",
      cwd: "/Users/test/swarmx",
      updatedAt: "2026-01-04T00:00:00Z",
      harnessId: "codex",
      harnessLabel: "Codex",
      source: "acp",
    });
  });
});

function discovered(overrides: Partial<DiscoveredSession>): DiscoveredSession {
  return {
    id: "session",
    title: "Session",
    cwd: "/Users/test/swarmx",
    updatedAt: "2026-01-01T00:00:00Z",
    harnessId: "codex",
    harnessLabel: "Codex",
    source: "acp",
    ...overrides,
  };
}
