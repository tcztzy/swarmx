import type { Agent } from "@deepseek-ai/dsh-agent";
import { describe, expect, it, vi } from "vitest";
import { leadToolGuard, memberToolGuard } from "../src/capabilities.js";

describe("V166/V170 member capabilities", () => {
  it("denies delegation and PKB, and fences mutation by exact write ownership", () => {
    const member = { id: "session-member" } as Agent;
    const hasActiveWriteAttempt = vi.fn(() => false);
    const coordinator = { hasActiveWriteAttempt };

    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "subagent" }),
    ).toMatch(/cannot delegate/iu);
    expect(memberToolGuard(member, coordinator as never, { agent: member, name: "pkb" })).toMatch(
      /PKB/u,
    );
    expect(memberToolGuard(member, coordinator as never, { agent: member, name: "write" })).toMatch(
      /active write task/iu,
    );
    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "run_code" }),
    ).toMatch(/active write task/iu);
    hasActiveWriteAttempt.mockReturnValue(true);
    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "write" }),
    ).toBeUndefined();
    expect(
      memberToolGuard(member, coordinator as never, {
        agent: { id: member.id } as Agent,
        name: "read",
      }),
    ).toMatch(/exact/iu);
  });

  it("requires the lead's own W attempt without removing lead administration", () => {
    const lead = { id: "session-lead" } as Agent;
    const hasActiveWriteAttempt = vi.fn(() => false);
    const coordinator = { hasActiveWriteAttempt };

    expect(leadToolGuard(lead, coordinator as never, { agent: lead, name: "write" })).toMatch(
      /lead.*active write/iu,
    );
    expect(
      leadToolGuard(lead, coordinator as never, { agent: lead, name: "swarm" }),
    ).toBeUndefined();
    hasActiveWriteAttempt.mockReturnValue(true);
    expect(
      leadToolGuard(lead, coordinator as never, { agent: lead, name: "write" }),
    ).toBeUndefined();
  });

  it("V188 denies monitor/verifier mutation even if an attempt predicate is true", () => {
    const member = { id: "session-reviewer" } as Agent;
    const coordinator = { hasActiveWriteAttempt: vi.fn(() => true) };
    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "write" }, "monitor"),
    ).toMatch(/monitor.*read-only/iu);
    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "bash" }, "verifier"),
    ).toMatch(/verifier.*read-only/iu);
    expect(
      memberToolGuard(member, coordinator as never, { agent: member, name: "read" }, "verifier"),
    ).toBeUndefined();
  });
});
