import { describe, expect, it } from "vitest";
import {
  canRerunTurn,
  resolveForkBoundary,
  type Timeline,
  type TimelineTurn,
} from "../src/fork-boundary.js";

/** Build a timeline from `[turn, endSeq | null]` pairs; null leaves the turn open. */
function timeline(...turns: [number, number | null][]): Timeline {
  const entries = turns.map(([turn, endSeq]): [number, TimelineTurn] => [
    turn,
    endSeq === null ? { turn, status: "open" } : { turn, status: "closed", end: { seq: endSeq } },
  ]);
  return { turnOrder: turns.map(([turn]) => turn), turns: new Map(entries) };
}

describe("resolveForkBoundary", () => {
  it("anchors at the preceding turn's end, not the target's own", () => {
    // Re-running turn 2 must not keep turn 2. The host searches forward from
    // atSeq, so anchoring at 20 would find turn 2's own end and retain it.
    expect(resolveForkBoundary(timeline([1, 10], [2, 20], [3, 30]), 2, true)).toEqual({
      kind: "fork",
      atSeq: 10,
    });
  });

  it("uses a fresh session for the first turn", () => {
    expect(resolveForkBoundary(timeline([1, 10], [2, 20]), 1, true)).toEqual({
      kind: "fresh",
    });
  });

  it("refuses the first loaded turn when older history is unfetched", () => {
    // Treating this as the first turn would discard turns the user can still
    // scroll back to, so a fresh session is not safe yet.
    expect(resolveForkBoundary(timeline([7, 70], [8, 80]), 7, false)).toBeUndefined();
  });

  it("skips back over a turn that never closed", () => {
    expect(resolveForkBoundary(timeline([1, 10], [2, null], [3, 30]), 3, true)).toEqual({
      kind: "fork",
      atSeq: 10,
    });
  });

  it("returns undefined for a turn outside the timeline", () => {
    expect(resolveForkBoundary(timeline([1, 10]), 99, true)).toBeUndefined();
  });

  it("reads turn order rather than assuming contiguous indices", () => {
    expect(resolveForkBoundary(timeline([4, 40], [9, 90]), 9, true)).toEqual({
      kind: "fork",
      atSeq: 40,
    });
  });
});

describe("canRerunTurn", () => {
  it("declines a turn that is still running", () => {
    expect(canRerunTurn(timeline([1, 10], [2, null]), 2, true)).toBe(false);
  });

  it("accepts a closed turn with a resolvable boundary", () => {
    expect(canRerunTurn(timeline([1, 10], [2, 20]), 2, true)).toBe(true);
  });

  it("declines when no boundary can be resolved", () => {
    expect(canRerunTurn(timeline([7, 70]), 7, false)).toBe(false);
  });
});
