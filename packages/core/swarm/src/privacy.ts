const ABSOLUTE_PATH_PATTERN = /(^|\s)(?:\/(?!\/)[^\s]+|[a-z]:[\\/][^\s]+|\\\\[^\s]+)/giu;
const FILE_URL_PATTERN = /\bfile:[^\s]+/giu;
const SECRET_VALUE_PATTERN = /\b(?:api[_ -]?key|password|secret|token)\b\s*[:=]\s*[^\s]+/giu;

export function redactSwarmText(value: string, maxLength: number): string {
  return value
    .replace(ABSOLUTE_PATH_PATTERN, "$1[redacted-path]")
    .replace(FILE_URL_PATTERN, "[redacted-path]")
    .replace(SECRET_VALUE_PATTERN, "[redacted-secret]")
    .slice(0, maxLength);
}
