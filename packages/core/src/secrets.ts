import { z } from "zod";

const REDACTED_VALUE = "[redacted]";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secretrefs",
  "secret_ref",
  "secret_refs",
  "secretrefid",
  "secret_ref_id",
  "secretstatus",
  "secret_status",
  "credentialref",
  "credential_ref",
  "credentialrefs",
  "credential_refs",
  "credentialreferences",
  "credential_references",
]);

const FORBIDDEN_SECRET_KEY_PATTERN =
  /(api[_-]?key|api[_-]?token|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|ingest[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password|host[_-]?login|auth[_-]?token)/i;

export const SecretPurposeSchema = z.enum([
  "provider_api_key",
  "deployment_bearer_token",
  "telemetry_token",
  "smtp_password",
  "remote_compute_password",
  "custom",
]);

export const SecretSourceSchema = z.enum([
  "env",
  "local_auth_file",
  "local_keychain",
  "server_keychain",
  "prompt",
  "request_only",
]);

export const SecretRefSchema = z.preprocess(
  normalizeSecretRefInput,
  z
    .object({
      source: SecretSourceSchema,
      key: z.string().min(1),
      purpose: SecretPurposeSchema.default("custom"),
      service: z.string().min(1).optional(),
      account: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const SecretVaultEntrySchema = z.preprocess(
  normalizeSecretVaultEntryInput,
  z
    .object({
      value: z.string(),
      purpose: SecretPurposeSchema.default("custom"),
      updatedAt: z.string().min(1).optional(),
      metadata: z.record(z.string(), z.unknown()).default({}),
    })
    .passthrough()
    .superRefine((entry, ctx) => addSecretIssuesSkipping(entry, ctx, [["value"]])),
);

export const SecretVaultDocumentSchema = z.preprocess(
  normalizeSecretVaultDocumentInput,
  z
    .object({
      schemaVersion: z.literal(1).default(1),
      service: z.string().min(1).optional(),
      fileMode: z.string().min(3).optional(),
      secrets: z.record(z.string().min(1), SecretVaultEntrySchema).default({}),
    })
    .passthrough()
    .superRefine((document, ctx) => addSecretIssuesSkipping(document, ctx, [["secrets"]])),
);

export const SecretWriteRequestSchema = z.preprocess(
  normalizeSecretWriteRequestInput,
  z
    .object({
      ref: SecretRefSchema,
      value: z.string(),
      actor: z.string().min(1).optional(),
      reason: z.string().optional(),
      requestedAt: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine((request, ctx) => addSecretIssuesSkipping(request, ctx, [["value"]])),
);

export const SecretStatusSchema = z.preprocess(
  normalizeSecretStatusInput,
  z
    .object({
      ref: SecretRefSchema,
      configured: z.boolean(),
      checkedAt: z.string().min(1).optional(),
      message: z.string().optional(),
    })
    .passthrough()
    .superRefine(addSecretStatusIssues),
);

export const SecretFileModeStatusSchema = z.object({
  path: z.string().min(1).optional(),
  mode: z.string().min(3).optional(),
  expectedMode: z.string().min(3).default("0600"),
  secure: z.boolean(),
  message: z.string(),
});

export type SecretPurpose = z.infer<typeof SecretPurposeSchema>;
export type SecretSource = z.infer<typeof SecretSourceSchema>;
export type SecretRef = z.infer<typeof SecretRefSchema>;
export type SecretVaultEntry = z.infer<typeof SecretVaultEntrySchema>;
export type SecretVaultDocument = z.infer<typeof SecretVaultDocumentSchema>;
export type SecretWriteRequest = z.infer<typeof SecretWriteRequestSchema>;
export type SecretStatus = z.infer<typeof SecretStatusSchema>;
export type SecretFileModeStatus = z.infer<typeof SecretFileModeStatusSchema>;

export function parseSecretRef(input: unknown): SecretRef {
  const ref = SecretRefSchema.parse(input);
  assertSecretRefPolicy(ref);
  return ref;
}

export function parseSecretVaultDocument(input: unknown): SecretVaultDocument {
  return SecretVaultDocumentSchema.parse(input);
}

export function parseSecretWriteRequest(input: unknown): SecretWriteRequest {
  const request = SecretWriteRequestSchema.parse(input);
  assertSecretRefPolicy(request.ref);
  return request;
}

export function parseSecretStatus(input: unknown): SecretStatus {
  return SecretStatusSchema.parse(input);
}

export function redactSecretVaultDocument(input: unknown): SecretVaultDocument {
  const document = SecretVaultDocumentSchema.parse(input);
  return SecretVaultDocumentSchema.parse({
    ...document,
    secrets: Object.fromEntries(
      Object.entries(document.secrets).map(([key, entry]) => [
        key,
        {
          ...entry,
          value: REDACTED_VALUE,
        },
      ]),
    ),
  });
}

export function redactSecretWriteRequest(input: unknown): SecretWriteRequest {
  const request = SecretWriteRequestSchema.parse(input);
  return SecretWriteRequestSchema.parse({
    ...request,
    value: REDACTED_VALUE,
  });
}

export function secretStatusFromVault(input: {
  vault: unknown;
  ref: unknown;
  checkedAt?: string;
}): SecretStatus {
  const vault = SecretVaultDocumentSchema.parse(input.vault);
  const ref = parseSecretRef(input.ref);
  const entry = vault.secrets[ref.key];
  return SecretStatusSchema.parse({
    ref,
    configured: !!entry && entry.value.length > 0,
    checkedAt: input.checkedAt,
  });
}

export function secretValueFromVault(input: { vault: unknown; ref: unknown }): string {
  const vault = SecretVaultDocumentSchema.parse(input.vault);
  const ref = parseSecretRef(input.ref);
  const entry = vault.secrets[ref.key];
  if (!entry?.value) {
    throw new Error(`Secret "${ref.key}" is not configured in the vault.`);
  }
  return entry.value;
}

export function evaluateSecretFileMode(input: {
  mode?: string | number;
  expectedMode?: string;
  path?: string;
}): SecretFileModeStatus {
  const expectedMode = normalizeMode(input.expectedMode ?? "0600");
  const mode = input.mode === undefined ? undefined : normalizeMode(input.mode);
  const secure = mode === expectedMode;
  return SecretFileModeStatusSchema.parse({
    path: input.path,
    mode,
    expectedMode,
    secure,
    message: secure
      ? `Secret file mode ${expectedMode} is secure.`
      : `Secret file mode must be ${expectedMode}.`,
  });
}

export function assertSecretRefPolicy(input: unknown): SecretRef {
  const ref = SecretRefSchema.parse(input);
  if (
    ref.purpose === "remote_compute_password" &&
    ref.source !== "prompt" &&
    ref.source !== "request_only"
  ) {
    throw new Error("Remote-compute passwords are request-only and must not be persisted.");
  }
  return ref;
}

function normalizeSecretRefInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    purpose: input.purpose ?? input.secret_purpose,
  };
}

function normalizeSecretVaultEntryInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    updatedAt: input.updatedAt ?? input.updated_at,
  };
}

function normalizeSecretVaultDocumentInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    schemaVersion: input.schemaVersion ?? input.schema_version,
    fileMode: input.fileMode ?? input.file_mode,
  };
}

function normalizeSecretWriteRequestInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    requestedAt: input.requestedAt ?? input.requested_at,
  };
}

function normalizeSecretStatusInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  return {
    ...input,
    checkedAt: input.checkedAt ?? input.checked_at,
  };
}

function normalizeMode(mode: string | number): string {
  const normalized = typeof mode === "number" ? mode.toString(8) : mode.trim();
  return normalized.padStart(4, "0").slice(-4);
}

function addSecretStatusIssues(status: unknown, ctx: z.RefinementCtx): void {
  if (
    isObjectRecord(status) &&
    ("value" in status || "secretValue" in status || "apiKey" in status)
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: [],
      message: "Secret status records must not include secret values.",
    });
  }
  addSecretIssues(status, ctx);
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  addSecretIssuesSkipping(value, ctx, []);
}

function addSecretIssuesSkipping(
  value: unknown,
  ctx: z.RefinementCtx,
  skippedPathPrefixes: Array<Array<string | number>>,
): void {
  for (const issue of findInlineSecrets(value, [], skippedPathPrefixes)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Secret records must not contain inline secret field "${issue.key}".`,
    });
  }
}

function findInlineSecrets(
  value: unknown,
  path: Array<string | number> = [],
  skippedPathPrefixes: Array<Array<string | number>> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (isSkippedPath(path, skippedPathPrefixes)) return [];
  if (Array.isArray(value)) {
    return value.flatMap((item, index) =>
      findInlineSecrets(item, [...path, index], skippedPathPrefixes),
    );
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    const childPath = [...path, key];
    if (isSkippedPath(childPath, skippedPathPrefixes)) continue;
    if (isForbiddenSecretKey(key)) {
      issues.push({ key, path: childPath });
    }
    issues.push(...findInlineSecrets(child, childPath, skippedPathPrefixes));
  }
  return issues;
}

function isSkippedPath(
  path: Array<string | number>,
  skippedPathPrefixes: Array<Array<string | number>>,
): boolean {
  return skippedPathPrefixes.some((prefix) => prefix.every((part, index) => path[index] === part));
}

function isForbiddenSecretKey(key: string): boolean {
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
  return (
    FORBIDDEN_SECRET_KEY_PATTERN.test(key) && !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
  );
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
