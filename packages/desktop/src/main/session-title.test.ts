import { describe, expect, it } from "vitest";
import {
  MAX_SESSION_TITLE_LENGTH,
  generatedSessionTitle,
  isPlaceholderSessionTitle,
  normalizeManualSessionTitle,
  sessionTitleMessages,
} from "./session-title.js";

describe("session titles", () => {
  it("recognizes only untitled local task placeholders", () => {
    expect(isPlaceholderSessionTitle("New Session")).toBe(true);
    expect(isPlaceholderSessionTitle(" untitled ")).toBe(true);
    expect(isPlaceholderSessionTitle("Fix session title")).toBe(false);
  });

  it("asks the title model to preserve the user's language", () => {
    expect(sessionTitleMessages("  修复 Project 上下文  ")).toEqual([
      expect.objectContaining({
        role: "system",
        content: expect.stringContaining("user's language"),
      }),
      { role: "user", content: "修复 Project 上下文" },
    ]);
  });

  it("sanitizes model output and ignores non-message output", () => {
    expect(
      generatedSessionTitle([
        { role: "assistant", kind: "thinking", content: "Thinking" },
        { role: "assistant", kind: "message", content: '标题："修复 SwarmX Project 上下文"\n说明' },
      ]),
    ).toBe("修复 SwarmX Project 上下文");
    expect(
      generatedSessionTitle([{ role: "assistant", kind: "thinking", content: "Only work" }]),
    ).toBeNull();
  });

  it("normalizes manual titles and applies the shared length limit", () => {
    expect(normalizeManualSessionTitle("  Rename   this task ")).toBe("Rename this task");
    expect(normalizeManualSessionTitle("   ")).toBeNull();
    expect(normalizeManualSessionTitle("a".repeat(100))).toHaveLength(MAX_SESSION_TITLE_LENGTH);
  });
});
