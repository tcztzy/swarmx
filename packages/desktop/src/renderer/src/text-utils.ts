export function capitalize(value: string): string {
  return value.length > 0 ? `${value[0]?.toUpperCase()}${value.slice(1)}` : value;
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const SHORT_WEEKDAY_FORMAT = new Intl.DateTimeFormat([], { weekday: "short" });

export function formatMessageTimestamp(value: string, now = new Date()): string | null {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;

  const ageMs = Math.max(0, now.getTime() - date.getTime());
  const time = `${twoDigits(date.getHours())}:${twoDigits(date.getMinutes())}`;
  if (ageMs < 24 * 60 * 60 * 1_000) return time;
  if (ageMs < 7 * 24 * 60 * 60 * 1_000) {
    return `${SHORT_WEEKDAY_FORMAT.format(date)} ${time}`;
  }
  return fullTimestamp(date);
}

export function formatFullMessageTimestamp(value: string): string | null {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return fullTimestamp(date);
}

function fullTimestamp(date: Date): string {
  return `${date.getFullYear()}-${twoDigits(date.getMonth() + 1)}-${twoDigits(date.getDate())} ${twoDigits(date.getHours())}:${twoDigits(date.getMinutes())}`;
}

export function lines(value: string): string[] {
  return [
    ...new Set(
      value
        .split(/\r?\n|,/)
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  ];
}

export function projectName(cwd: string): string {
  const parts = cwd.split(/[\\/]/).filter(Boolean);
  return parts.at(-1) ?? cwd;
}

export function slugId(value: string, fallback: string): string {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug || fallback;
}

function twoDigits(value: number): string {
  return String(value).padStart(2, "0");
}
