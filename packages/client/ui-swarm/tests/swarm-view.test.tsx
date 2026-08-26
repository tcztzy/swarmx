import type { SwarmUiSnapshot } from "@swarmx/dsh-swarm/contracts";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SwarmActivity, swarmSideViewEntry } from "../src/client/swarm-view.js";

const snapshot: SwarmUiSnapshot = {
  kind: "active",
  memberName: "lead",
  members: [
    { description: "Team lead", name: "lead", role: "lead", status: "idle" },
    { description: "Implementation", name: "alpha", role: "member", status: "running" },
  ],
  name: "Research team",
  pendingMessages: 0,
  revision: 7,
  role: "lead",
  tasks: [
    {
      blockedBy: [],
      id: "task-1",
      kind: "write",
      ownerName: "alpha",
      ready: true,
      revision: 2,
      status: "in_progress",
      subject: "Implement scheduler",
    },
  ],
  updatedAt: 100,
};

describe("V171 Swarm Side View", () => {
  it("uses a serializable per-Session entry and renders only safe projection fields", () => {
    const entry = swarmSideViewEntry(snapshot);
    expect(entry).toMatchObject({ id: "swarm-activity", kind: "swarm-activity", mode: "inspect" });
    expect(entry.payload).toEqual(snapshot);

    const markup = renderToStaticMarkup(<SwarmActivity snapshot={snapshot} />);
    expect(markup).toContain("Research team");
    expect(markup).toContain("alpha");
    expect(markup).toContain("Implement scheduler");
    expect(markup).not.toContain("session-");
    expect(markup).not.toContain("private coordination detail");
  });
});
