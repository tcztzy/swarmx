import { describe, expect, it, vi } from "vitest";
import { ScienceNavigationController } from "../src/client/science-navigation.js";

const target = {
  kind: "artifact" as const,
  artifactId: "artifact-1",
  projectId: "project-1",
  surface: "artifacts" as const,
};

describe("V53 Science fullscreen deep links", () => {
  it("deduplicates targets per Session and reports whether Science is mounted", () => {
    const navigation = new ScienceNavigationController();
    const listener = vi.fn();
    navigation.subscribe("session-a" as never, listener);

    expect(navigation.open("session-a" as never, target)).toBe(false);
    expect(navigation.open("session-a" as never, target)).toBe(false);
    expect(listener).toHaveBeenCalledOnce();
    expect(navigation.getSnapshot("session-b" as never)).toBeNull();

    const unmount = navigation.mount("session-a" as never);
    expect(navigation.open("session-a" as never, { ...target, artifactId: "artifact-2" })).toBe(
      true,
    );
    unmount();
    expect(navigation.open("session-a" as never, target)).toBe(false);
  });

  it("clears retained targets and listeners on HMR disposal", () => {
    const navigation = new ScienceNavigationController();
    const listener = vi.fn();
    navigation.open("session-a" as never, target);
    navigation.subscribe("session-a" as never, listener);

    navigation.dispose();

    expect(navigation.getSnapshot("session-a" as never)).toBeNull();
    expect(listener).toHaveBeenCalledOnce();
    expect(() => navigation.open("session-a" as never, target)).toThrow(/disposed/u);
  });
});
