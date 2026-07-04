import { z } from "zod";

const REDACTED_VALUE = "[redacted]";
const OMITTED_VALUE = "[omitted]";
const DEFAULT_SCHEMA_VERSION = "swarmx.telemetry.v1";

const ALLOWED_SECRET_REFERENCE_KEYS = new Set([
  "secretref",
  "secret_ref",
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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|ingest[_-]?token)/i;
const RAW_CONTENT_KEY_PATTERN =
  /(prompt|response|message[_-]?text|conversation|wiki[_-]?body|source[_-]?code|terminal[_-]?output|stdout|stderr|stack[_-]?trace|run[_-]?log|process[_-]?log|worker[_-]?log|raw[_-]?payload)/i;

export const TelemetrySchemaVersionSchema = z.string().regex(/^[a-z0-9_.-]+\.telemetry\.v1$/);

export const TelemetryEventTypeSchema = z
  .string()
  .regex(/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/)
  .superRefine((eventType, ctx) => {
    if (eventType === "task_created" || eventType === "task_updated") {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `Telemetry event "${eventType}" is not allowed in v1.`,
      });
    }
  });

export const TelemetryReleaseSchema = z
  .object({
    appVersion: z.string().min(1).optional(),
    gitCommit: z.string().min(1).optional(),
    platform: z.string().min(1).optional(),
    arch: z.string().min(1).optional(),
  })
  .passthrough();

export const TelemetryEnvelopeSchema = z
  .object({
    schemaVersion: TelemetrySchemaVersionSchema.default(DEFAULT_SCHEMA_VERSION),
    eventId: z.string().regex(/^(evt|tel)_[A-Za-z0-9][A-Za-z0-9_-]*$/),
    timestamp: z.string().min(1),
    eventType: TelemetryEventTypeSchema,
    source: z.string().min(1),
    installationId: z.string().min(1),
    sessionId: z.string().min(1).optional(),
    release: TelemetryReleaseSchema.default({}),
    payload: z.record(z.string(), z.unknown()).default({}),
  })
  .passthrough()
  .superRefine((envelope, ctx) => {
    for (const issue of findUnsafeTelemetryFields(envelope.payload)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["payload", ...issue.path],
        message: `Telemetry payload contains unsafe field "${issue.key}". Sanitize it before building an envelope.`,
      });
    }
  });

export const TelemetryConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    url: z.string().min(1).optional(),
    token: z.string().min(1).optional(),
    source: z.string().min(1).default("swarmx"),
    installationId: z.string().min(1).optional(),
  })
  .passthrough();

export const TelemetryStatusSchema = z.object({
  enabled: z.boolean(),
  configured: z.boolean(),
  url: z.string().optional(),
  outboxCount: z.number().int().nonnegative().optional(),
  reason: z.string().optional(),
});

export const TelemetryIngestConfigSchema = z
  .object({
    enabled: z.boolean().default(false),
    ingestToken: z.string().min(1).optional(),
    acceptedSchemaVersions: z.array(TelemetrySchemaVersionSchema).default([DEFAULT_SCHEMA_VERSION]),
  })
  .passthrough();

export const TelemetryIngestAcceptedRecordSchema = z.object({
  ingestId: z.string().regex(/^ing_[A-Za-z0-9][A-Za-z0-9_-]*$/),
  receivedAt: z.string().min(1),
  envelope: TelemetryEnvelopeSchema,
  sourceIp: z.string().min(1).optional(),
  userAgent: z.string().min(1).optional(),
  storageRef: z.string().min(1).optional(),
});

export const TelemetryIngestDecisionSchema = z.object({
  status: z.enum(["accepted", "rejected"]),
  statusCode: z.number().int().positive(),
  reason: z.string().min(1).optional(),
  record: TelemetryIngestAcceptedRecordSchema.optional(),
  error: z.string().min(1).optional(),
});

export type TelemetrySchemaVersion = z.infer<typeof TelemetrySchemaVersionSchema>;
export type TelemetryRelease = z.infer<typeof TelemetryReleaseSchema>;
export type TelemetryEnvelope = z.infer<typeof TelemetryEnvelopeSchema>;
export type TelemetryConfig = z.infer<typeof TelemetryConfigSchema>;
export type TelemetryStatus = z.infer<typeof TelemetryStatusSchema>;
export type TelemetryIngestConfig = z.infer<typeof TelemetryIngestConfigSchema>;
export type TelemetryIngestAcceptedRecord = z.infer<typeof TelemetryIngestAcceptedRecordSchema>;
export type TelemetryIngestDecision = z.infer<typeof TelemetryIngestDecisionSchema>;

export interface BuildTelemetryEnvelopeInput {
  schemaVersion?: TelemetrySchemaVersion;
  eventId?: string;
  timestamp?: string;
  eventType: string;
  source: string;
  installationId: string;
  sessionId?: string;
  release?: TelemetryRelease;
  payload?: Record<string, unknown>;
  contentDiagnostics?: boolean;
}

export interface ResolveTelemetryConfigOptions {
  env?: Record<string, string | undefined>;
  settings?: {
    enabled?: boolean;
    url?: string;
    token?: string;
    source?: string;
    installationId?: string;
  };
  prefix?: "SWARMX" | "GEEPILOT";
}

export interface ResolveTelemetryIngestConfigOptions {
  env?: Record<string, string | undefined>;
  settings?: {
    enabled?: boolean;
    ingestToken?: string;
    acceptedSchemaVersions?: string[];
  };
  prefix?: "SWARMX" | "GEEPILOT";
}

export interface TelemetrySender {
  send(
    envelope: TelemetryEnvelope,
    options: {
      url: string;
      headers: Record<string, string>;
    },
  ): Promise<void>;
}

export interface TelemetryOutbox {
  append(envelope: TelemetryEnvelope, error: string): Promise<void>;
  count?(): Promise<number>;
}

export interface TelemetryClientOptions {
  config: TelemetryConfig;
  sender: TelemetrySender;
  outbox?: TelemetryOutbox;
  now?: () => string;
}

export interface TelemetryIngestRequest {
  headers?: Record<string, string | string[] | undefined>;
  body: unknown;
  receivedAt?: string;
  sourceIp?: string;
  userAgent?: string;
}

export interface TelemetryIngestStore {
  append(record: TelemetryIngestAcceptedRecord): Promise<undefined | { storageRef?: string }>;
}

export interface TelemetryIngestHandlerOptions {
  config: TelemetryIngestConfig;
  store?: TelemetryIngestStore;
  now?: () => string;
}

export interface TelemetrySendResult {
  status: "sent" | "skipped" | "outboxed" | "failed";
  envelope?: TelemetryEnvelope;
  reason?: string;
  error?: string;
}

export type TelemetryClientSendInput = Omit<
  BuildTelemetryEnvelopeInput,
  "source" | "installationId"
>;

export function sanitizeTelemetryPayload(
  input: unknown,
  options: { contentDiagnostics?: boolean } = {},
): unknown {
  if (Array.isArray(input)) return input.map((item) => sanitizeTelemetryPayload(item, options));
  if (!isObjectRecord(input)) return input;

  return Object.fromEntries(
    Object.entries(input)
      .map(([key, value]) => {
        if (isForbiddenSecretKey(key)) return [key, REDACTED_VALUE];
        if (!options.contentDiagnostics && isRawContentKey(key)) return [key, OMITTED_VALUE];
        return [key, sanitizeTelemetryPayload(value, options)];
      })
      .filter(([, value]) => value !== undefined),
  );
}

export function buildTelemetryEnvelope(input: BuildTelemetryEnvelopeInput): TelemetryEnvelope {
  const payload = sanitizeTelemetryPayload(input.payload ?? {}, {
    contentDiagnostics: input.contentDiagnostics,
  });
  const withoutId = {
    schemaVersion: input.schemaVersion ?? DEFAULT_SCHEMA_VERSION,
    timestamp: input.timestamp ?? new Date().toISOString(),
    eventType: input.eventType,
    source: input.source,
    installationId: input.installationId,
    sessionId: input.sessionId,
    release: input.release ?? {},
    payload,
  };
  return TelemetryEnvelopeSchema.parse({
    ...withoutId,
    eventId: input.eventId ?? `evt_${stableHash(stableJson(withoutId))}`,
  });
}

export function resolveTelemetryConfig(
  options: ResolveTelemetryConfigOptions = {},
): TelemetryConfig {
  const prefix = options.prefix ?? "SWARMX";
  const env = options.env ?? processLikeEnv();
  const settings = options.settings ?? {};
  const enabledRaw =
    settings.enabled ?? env[`${prefix}_TELEMETRY_ENABLED`] ?? env.GEEPILOT_TELEMETRY_ENABLED;
  const enabled = typeof enabledRaw === "boolean" ? enabledRaw : parseBoolean(enabledRaw);

  return TelemetryConfigSchema.parse({
    enabled,
    url: settings.url ?? env[`${prefix}_TELEMETRY_URL`] ?? env.GEEPILOT_TELEMETRY_URL,
    token: settings.token ?? env[`${prefix}_TELEMETRY_TOKEN`] ?? env.GEEPILOT_TELEMETRY_TOKEN,
    source: settings.source ?? env[`${prefix}_TELEMETRY_SOURCE`] ?? prefix.toLowerCase(),
    installationId:
      settings.installationId ?? env[`${prefix}_INSTALLATION_ID`] ?? env.GEEPILOT_INSTALLATION_ID,
  });
}

export function resolveTelemetryIngestConfig(
  options: ResolveTelemetryIngestConfigOptions = {},
): TelemetryIngestConfig {
  const prefix = options.prefix ?? "SWARMX";
  const env = options.env ?? processLikeEnv();
  const settings = options.settings ?? {};
  const enabledRaw =
    settings.enabled ??
    env[`${prefix}_TELEMETRY_INGEST_ENABLED`] ??
    env.GEEPILOT_TELEMETRY_INGEST_ENABLED;
  const acceptedSchemaVersions =
    settings.acceptedSchemaVersions ??
    splitCsv(env[`${prefix}_TELEMETRY_ACCEPTED_SCHEMA_VERSIONS`]) ??
    splitCsv(env.GEEPILOT_TELEMETRY_ACCEPTED_SCHEMA_VERSIONS);

  return TelemetryIngestConfigSchema.parse({
    enabled: typeof enabledRaw === "boolean" ? enabledRaw : parseBoolean(enabledRaw),
    ingestToken:
      settings.ingestToken ??
      env[`${prefix}_TELEMETRY_INGEST_TOKEN`] ??
      env.GEEPILOT_TELEMETRY_INGEST_TOKEN,
    acceptedSchemaVersions,
  });
}

export function telemetryStatus(
  configInput: unknown,
  options: { outboxCount?: number } = {},
): TelemetryStatus {
  const config = TelemetryConfigSchema.parse(configInput);
  const configured = !!config.enabled && !!config.url;
  return TelemetryStatusSchema.parse({
    enabled: config.enabled,
    configured,
    url: config.url,
    outboxCount: options.outboxCount,
    reason: configured
      ? undefined
      : config.enabled
        ? "telemetry url is not configured"
        : "telemetry is disabled",
  });
}

export function createTelemetryClient(options: TelemetryClientOptions) {
  const config = TelemetryConfigSchema.parse(options.config);

  return {
    async send(input: TelemetryClientSendInput): Promise<TelemetrySendResult> {
      if (!config.enabled || !config.url) {
        return {
          status: "skipped",
          reason: config.enabled ? "telemetry url is not configured" : "telemetry is disabled",
        };
      }
      if (!config.installationId) {
        return { status: "skipped", reason: "installation id is not configured" };
      }

      const envelope = buildTelemetryEnvelope({
        ...input,
        timestamp: input.timestamp ?? options.now?.() ?? new Date().toISOString(),
        source: config.source,
        installationId: config.installationId,
      });
      const headers = telemetryHeaders(config);

      try {
        await options.sender.send(envelope, { url: config.url, headers });
        return { status: "sent", envelope };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (options.outbox) {
          await options.outbox.append(envelope, message);
          return { status: "outboxed", envelope, error: message };
        }
        return { status: "failed", envelope, error: message };
      }
    },
    status: async (): Promise<TelemetryStatus> =>
      telemetryStatus(config, { outboxCount: await options.outbox?.count?.() }),
  };
}

export function telemetryHeaders(configInput: unknown): Record<string, string> {
  const config = TelemetryConfigSchema.parse(configInput);
  return {
    "Content-Type": "application/json",
    ...(config.token ? { Authorization: `Bearer ${config.token}` } : {}),
  };
}

export function evaluateTelemetryIngest(
  request: TelemetryIngestRequest,
  configInput: unknown,
): TelemetryIngestDecision {
  const config = TelemetryIngestConfigSchema.parse(configInput);
  if (!config.enabled) {
    return TelemetryIngestDecisionSchema.parse({
      status: "rejected",
      statusCode: 404,
      reason: "telemetry ingest is disabled",
    });
  }

  if (config.ingestToken && bearerToken(request.headers) !== config.ingestToken) {
    return TelemetryIngestDecisionSchema.parse({
      status: "rejected",
      statusCode: 401,
      reason: "telemetry ingest token is missing or invalid",
    });
  }

  const parsed = TelemetryEnvelopeSchema.safeParse(request.body);
  if (!parsed.success) {
    return TelemetryIngestDecisionSchema.parse({
      status: "rejected",
      statusCode: 400,
      reason: "telemetry envelope is invalid",
      error: parsed.error.issues.map((issue) => issue.message).join("; "),
    });
  }

  const envelope = parsed.data;
  if (!config.acceptedSchemaVersions.includes(envelope.schemaVersion)) {
    return TelemetryIngestDecisionSchema.parse({
      status: "rejected",
      statusCode: 400,
      reason: `unsupported telemetry schema version "${envelope.schemaVersion}"`,
    });
  }

  const receivedAt = request.receivedAt ?? new Date().toISOString();
  const record = TelemetryIngestAcceptedRecordSchema.parse({
    ingestId: `ing_${stableHash(
      stableJson({
        eventId: envelope.eventId,
        receivedAt,
        sourceIp: request.sourceIp,
      }),
    )}`,
    receivedAt,
    envelope,
    sourceIp: request.sourceIp,
    userAgent: request.userAgent ?? firstHeader(request.headers, "user-agent"),
  });

  return TelemetryIngestDecisionSchema.parse({
    status: "accepted",
    statusCode: 202,
    record,
  });
}

export function createTelemetryIngestHandler(options: TelemetryIngestHandlerOptions) {
  const config = TelemetryIngestConfigSchema.parse(options.config);

  return {
    async ingest(request: TelemetryIngestRequest): Promise<TelemetryIngestDecision> {
      const decision = evaluateTelemetryIngest(
        {
          ...request,
          receivedAt: request.receivedAt ?? options.now?.() ?? new Date().toISOString(),
        },
        config,
      );
      if (decision.status !== "accepted" || !decision.record || !options.store) {
        return decision;
      }

      try {
        const appendResult = await options.store.append(decision.record);
        const record = TelemetryIngestAcceptedRecordSchema.parse({
          ...decision.record,
          storageRef: appendResult?.storageRef ?? decision.record.storageRef,
        });
        return TelemetryIngestDecisionSchema.parse({
          status: "accepted",
          statusCode: 202,
          record,
        });
      } catch (error) {
        return TelemetryIngestDecisionSchema.parse({
          status: "rejected",
          statusCode: 500,
          reason: "telemetry append failed",
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
  };
}

export function parseTelemetryIngestAcceptedRecord(input: unknown): TelemetryIngestAcceptedRecord {
  return TelemetryIngestAcceptedRecordSchema.parse(input);
}

export function parseTelemetryIngestDecision(input: unknown): TelemetryIngestDecision {
  return TelemetryIngestDecisionSchema.parse(input);
}

function findUnsafeTelemetryFields(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findUnsafeTelemetryFields(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    if (isForbiddenSecretKey(key) && child !== REDACTED_VALUE) {
      issues.push({ key, path: [...path, key] });
    }
    if (isRawContentKey(key) && child !== OMITTED_VALUE) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findUnsafeTelemetryFields(child, [...path, key]));
  }
  return issues;
}

function isForbiddenSecretKey(key: string): boolean {
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
  return (
    FORBIDDEN_SECRET_KEY_PATTERN.test(key) && !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
  );
}

function isRawContentKey(key: string): boolean {
  return RAW_CONTENT_KEY_PATTERN.test(key);
}

function parseBoolean(value: string | undefined): boolean {
  if (!value) return false;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function splitCsv(value: string | undefined): string[] | undefined {
  if (!value) return undefined;
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function processLikeEnv(): Record<string, string | undefined> {
  if (typeof process !== "undefined" && process.env) return process.env;
  return {};
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  return `{${Object.entries(value)
    .filter(([, child]) => child !== undefined)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, child]) => `${JSON.stringify(key)}:${stableJson(child)}`)
    .join(",")}}`;
}

function stableHash(value: string): string {
  let hash = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  for (let index = 0; index < value.length; index++) {
    hash ^= BigInt(value.charCodeAt(index));
    hash = BigInt.asUintN(64, hash * prime);
  }
  return hash.toString(16).padStart(16, "0");
}

function firstHeader(
  headers: Record<string, string | string[] | undefined> | undefined,
  name: string,
): string | undefined {
  if (!headers) return undefined;
  const target = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() !== target) continue;
    if (Array.isArray(value)) return value[0];
    return value;
  }
  return undefined;
}

function bearerToken(
  headers: Record<string, string | string[] | undefined> | undefined,
): string | undefined {
  const authorization = firstHeader(headers, "authorization");
  const match = authorization?.match(/^Bearer\s+(.+)$/i);
  return match?.[1];
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
