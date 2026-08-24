import { describe, expect, it, vi } from "vitest";
import { SideViewController, type SideViewEntry } from "../src/client/side-view.js";

function entry(id: string, title: string = id): SideViewEntry {
  return {
    id,
    kind: "test",
    title,
    mode: "inspect",
    payload: { id },
  };
}

function workbenchEntry(id: string): SideViewEntry {
  return { ...entry(id), mode: "workbench" };
}

function controller() {
  const layout = {
    openDetails: vi.fn(),
    closeDetails: vi.fn(),
  };
  return { layout, sideView: new SideViewController(layout) };
}

describe("V47/V48 generic Side View state", () => {
  it("opens, activates, upserts, and deterministically closes tabs within one Session", () => {
    const { layout, sideView } = controller();

    sideView.open("session-a" as never, entry("one"));
    sideView.open("session-a" as never, entry("two"));
    sideView.open("session-a" as never, entry("one", "One updated"));

    expect(sideView.getSnapshot("session-a" as never)).toEqual({
      entries: [entry("one", "One updated"), entry("two")],
      activeId: "one",
    });
    expect(layout.openDetails).toHaveBeenCalledTimes(3);
    expect(layout.openDetails).toHaveBeenNthCalledWith(1, undefined);

    sideView.close("session-a" as never, "one");
    expect(sideView.getSnapshot("session-a" as never).activeId).toBe("two");
    sideView.close("session-a" as never, "two");
    expect(sideView.getSnapshot("session-a" as never)).toEqual({
      entries: [],
      activeId: null,
    });
    expect(layout.closeDetails).toHaveBeenCalledOnce();
  });

  it("requests the wide preference only for a workbench entry", () => {
    const { layout, sideView } = controller();

    sideView.open("session-a" as never, workbenchEntry("artifact"));
    sideView.activate("session-a" as never, "artifact");

    expect(layout.openDetails).toHaveBeenNthCalledWith(1, 880);
    expect(layout.openDetails).toHaveBeenNthCalledWith(2, 880);
  });

  it("isolates tabs and subscriptions by Session", () => {
    const { sideView } = controller();
    const notifyA = vi.fn();
    const notifyB = vi.fn();
    sideView.subscribe("session-a" as never, notifyA);
    sideView.subscribe("session-b" as never, notifyB);

    sideView.open("session-a" as never, entry("one"));

    expect(notifyA).toHaveBeenCalledOnce();
    expect(notifyB).not.toHaveBeenCalled();
    expect(sideView.getSnapshot("session-b" as never).entries).toEqual([]);
  });

  it("rejects non-JSON payloads before they enter shared state", () => {
    const { sideView } = controller();

    expect(() =>
      sideView.open("session-a" as never, {
        ...entry("bad"),
        payload: { callback: () => undefined } as never,
      }),
    ).toThrow(/JSON-serializable/u);
    expect(sideView.getSnapshot("session-a" as never).entries).toEqual([]);
  });

  it("dismisses geometry without deleting tabs and clears all ownership on disposal", () => {
    const { layout, sideView } = controller();
    const notify = vi.fn();
    sideView.open("session-a" as never, entry("one"));
    sideView.subscribe("session-a" as never, notify);

    sideView.dismiss("session-a" as never);
    expect(sideView.getSnapshot("session-a" as never).entries).toHaveLength(1);

    sideView.dispose();

    expect(layout.closeDetails).toHaveBeenCalledTimes(2);
    expect(notify).toHaveBeenCalledOnce();
    expect(sideView.getSnapshot("session-a" as never).entries).toEqual([]);
    expect(() => sideView.open("session-a" as never, entry("two"))).toThrow(/disposed/u);
  });
});
