import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SWARM_PROVISIONING_INTERRUPTED_ERROR, SwarmJournal } from "@swarmx/dsh-swarm";
import { describe, expect, it } from "vitest";
import { startSwarmRecoveryOwner } from "../src/runtime/swarm-recovery-owner.js";

describe("desktop Swarm recovery owner", () => {
  it("owns cold and final provisioning recovery outside configurable Harness plugins", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-recovery-owner-"));
    const journalRoot = join(root, "swarm");
    try {
      const seed = new SwarmJournal(journalRoot);
      const teamId = "dsh-lead";
      seed.append(teamId, {
        type: "team/created",
        data: {
          createdAt: 1,
          lead: {
            createdAt: 1,
            description: "Lead",
            id: teamId,
            name: "lead",
            phase: "active",
            role: "lead",
            modelPolicy: { source: "observed" },
          },
          name: "Recovery owner team",
          workspaceKey: `swarmx--${"a".repeat(64)}`,
        },
      });
      const firstMember = randomUUID();
      seed.append(teamId, {
        type: "member/updated",
        data: {
          createdAt: 2,
          description: "Cold interrupted member",
          id: firstMember,
          name: "cold",
          phase: "provisioning",
          role: "legacy",
          modelPolicy: { source: "legacy-default" },
        },
      });
      seed.close();

      const owner = startSwarmRecoveryOwner(journalRoot);
      const live = new SwarmJournal(journalRoot, { mode: "client" });
      expect(live.get(teamId)?.members.find((member) => member.id === firstMember)).toMatchObject({
        error: SWARM_PROVISIONING_INTERRUPTED_ERROR,
        phase: "failed",
      });
      const finalMember = randomUUID();
      live.append(teamId, {
        type: "member/updated",
        data: {
          createdAt: 3,
          description: "Final interrupted member",
          id: finalMember,
          name: "final",
          phase: "provisioning",
          role: "legacy",
          modelPolicy: { source: "legacy-default" },
        },
      });
      live.close();

      await owner.dispose();
      await owner.dispose();
      const inspected = new SwarmJournal(journalRoot, { mode: "client" });
      expect(
        inspected.get(teamId)?.members.find((member) => member.id === finalMember),
      ).toMatchObject({ error: SWARM_PROVISIONING_INTERRUPTED_ERROR, phase: "failed" });
      inspected.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
