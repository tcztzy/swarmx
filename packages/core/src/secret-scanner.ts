export const REDACTED_VALUE = "[redacted]";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secretrefs",
  "secret_ref",
  "secret_refs",
  "secretrefid",
  "secret_ref_id",
  "secretstatus",
  "secret_status",
  "secretscan",
  "credentialref",
  "credential_ref",
  "credentialrefs",
  "credential_refs",
  "credentialreferences",
  "credential_references",
]);

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|api[_-]?token|access[_-]?token|auth[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|ingest[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password|host[_-]?login)/i;

export interface InlineSecretIssue {
  key: string;
  path: Array<string | number>;
}

export interface InlineSecretScanOptions {
  allowRedacted?: boolean;
  skippedPathPrefixes?: Array<Array<string | number>>;
}

export function findInlineSecretFields(
  value: unknown,
  options: InlineSecretScanOptions = {},
  path: Array<string | number> = [],
): InlineSecretIssue[] {
  const skippedPathPrefixes = options.skippedPathPrefixes ?? [];
  const allowRedacted = options.allowRedacted ?? true;
  if (isSkippedPath(path, skippedPathPrefixes)) return [];
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findInlineSecretFields(item, options, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: InlineSecretIssue[] = [];
  for (const [key, child] of Object.entries(value)) {
    const childPath = [...path, key];
    if (isSkippedPath(childPath, skippedPathPrefixes)) continue;
    if (isForbiddenSecretKey(key) && (!allowRedacted || child !== REDACTED_VALUE)) {
      issues.push({ key, path: childPath });
    }
    issues.push(...findInlineSecretFields(child, options, childPath));
  }
  return issues;
}

export function isForbiddenSecretKey(key: string): boolean {
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
  return containsSecretMarker(key) && !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey);
}

export function containsSecretMarker(value: string): boolean {
  return FORBIDDEN_SECRET_KEY_PATTERN.test(value);
}

function isSkippedPath(
  path: Array<string | number>,
  skippedPathPrefixes: Array<Array<string | number>>,
): boolean {
  return skippedPathPrefixes.some((prefix) => prefix.every((part, index) => path[index] === part));
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
