import type { MessageChunk } from "@swarmx/core";

export const SESSION_TITLE_MODEL_ID = "gpt-5.4-mini";
export const MAX_SESSION_TITLE_LENGTH = 60;

const PLACEHOLDER_TITLES = new Set(["", "new session", "untitled"]);

export function isPlaceholderSessionTitle(title: string): boolean {
  return PLACEHOLDER_TITLES.has(title.trim().toLocaleLowerCase());
}

export function normalizeManualSessionTitle(title: string): string | null {
  return normalizeTitleText(title);
}

export function sessionTitleMessages(userRequest: string): Array<{
  role: "system" | "user";
  content: string;
}> {
  return [
    {
      role: "system",
      content:
        "Write a short recognizable task title from the user's request. Use the user's language, preserve important technical names, stay under 60 characters, and output only the title without quotes or punctuation commentary.",
    },
    { role: "user", content: userRequest.trim() },
  ];
}

export function generatedSessionTitle(messages: MessageChunk[]): string | null {
  const response = [...messages]
    .reverse()
    .find(
      (message) => message.kind === "message" && message.role !== "user" && message.role !== "tool",
    );
  if (!response) return null;

  const firstLine = response.content
    .trim()
    .split(/\r?\n/u)
    .find((line) => line.trim().length > 0);
  if (!firstLine) return null;

  return normalizeTitleText(
    firstLine
      .replace(/^#{1,6}\s*/u, "")
      .replace(/^(?:title|标题)\s*[:：]\s*/iu, "")
      .replace(/^[`'"“‘]+|[`'"”’]+$/gu, ""),
  );
}

function normalizeTitleText(value: string): string | null {
  const normalized = value.replace(/\s+/gu, " ").trim();
  if (!normalized) return null;
  return Array.from(normalized).slice(0, MAX_SESSION_TITLE_LENGTH).join("");
}
