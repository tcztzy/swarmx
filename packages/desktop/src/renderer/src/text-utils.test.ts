import { describe, expect, it } from "vitest";
import { formatFullMessageTimestamp, formatMessageTimestamp } from "./text-utils.js";

describe("message timestamps", () => {
  it("shows only the time for messages less than 24 hours old", () => {
    const now = new Date(2026, 6, 25, 16, 0);
    const createdAt = new Date(2026, 6, 24, 16, 1).toISOString();

    expect(formatMessageTimestamp(createdAt, now)).toBe("16:01");
  });

  it("shows the weekday and time from 24 hours up to 7 days", () => {
    const now = new Date(2026, 6, 25, 16, 0);
    const created = new Date(2026, 6, 23, 9, 5);
    const weekday = new Intl.DateTimeFormat([], { weekday: "short" }).format(created);

    expect(formatMessageTimestamp(created.toISOString(), now)).toBe(`${weekday} 09:05`);
  });

  it("shows the full local date and time at 7 days and beyond", () => {
    const now = new Date(2026, 6, 25, 16, 0);
    const createdAt = new Date(2026, 6, 18, 16, 0).toISOString();

    expect(formatMessageTimestamp(createdAt, now)).toBe("2026-07-18 16:00");
    expect(formatFullMessageTimestamp(createdAt)).toBe("2026-07-18 16:00");
  });

  it("returns null for invalid timestamps", () => {
    expect(formatMessageTimestamp("not-a-date")).toBeNull();
    expect(formatFullMessageTimestamp("not-a-date")).toBeNull();
  });
});
