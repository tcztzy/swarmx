import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import { z } from "zod";

const REDACTED_VALUE = "[redacted]";

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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|cluster[_-]?password|remote[_-]?compute[_-]?password|host[_-]?login)/i;

const OptionalStringListSchema = z
  .preprocess((value) => stringListFromUnknown(value), z.array(z.string().min(1)))
  .optional()
  .default([]);

const OptionalIntegerSchema = z.preprocess((value) => {
  if (value === undefined || value === null || value === "") return undefined;
  if (typeof value === "number") return value;
  if (typeof value === "string") return Number(value);
  return value;
}, z.number().int().positive().optional());

const OptionalBooleanSchema = z.preprocess((value) => {
  if (value === undefined || value === null || value === "") return undefined;
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true") return true;
    if (normalized === "false") return false;
  }
  return value;
}, z.boolean().optional());

export const AgentDefinitionSourceSchema = z
  .object({
    kind: z.enum(["local", "project", "user", "plugin", "host", "server", "imported"]),
    path: z.string().min(1).optional(),
    pluginId: z.string().min(1).optional(),
    label: z.string().min(1).optional(),
    readOnly: z.boolean().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AgentDefinitionGeepilotMetadataSchema = z
  .object({
    harness: z.string().min(1).optional(),
    provider: z.string().min(1).optional(),
    selector: z.string().min(1).optional(),
    enabled: OptionalBooleanSchema,
    source: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AgentDefinitionFrontmatterSchema = z
  .object({
    name: z.string().min(1).optional(),
    description: z.string().min(1).optional(),
    tools: OptionalStringListSchema,
    disallowedTools: OptionalStringListSchema,
    model: z.string().min(1).optional(),
    permissionMode: z.string().min(1).optional(),
    mcpServers: z.unknown().optional(),
    hooks: z.unknown().optional(),
    maxTurns: OptionalIntegerSchema,
    skills: OptionalStringListSchema,
    initialPrompt: z.string().min(1).optional(),
    memory: z.string().min(1).optional(),
    effort: z.string().min(1).optional(),
    background: OptionalBooleanSchema,
    isolation: z.string().min(1).optional(),
    color: z.string().min(1).optional(),
    geepilot: AgentDefinitionGeepilotMetadataSchema.optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const AgentDefinitionDocumentSchema = z.object({
  frontmatter: AgentDefinitionFrontmatterSchema,
  rawFrontmatter: z.record(z.string(), z.unknown()).default({}),
  body: z.string(),
  source: AgentDefinitionSourceSchema.optional(),
});

export const AgentProfileMetadataSchema = z
  .object({
    id: z.string().min(1),
    name: z.string().min(1),
    description: z.string().optional(),
    aliases: z.array(z.string().min(1)).default([]),
    harnessId: z.string().min(1).optional(),
    providerProfileId: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    instructions: z.string().optional(),
    tools: z.array(z.string().min(1)).default([]),
    disallowedTools: z.array(z.string().min(1)).default([]),
    skills: z.array(z.string().min(1)).default([]),
    mcpServers: z.array(z.string().min(1)).default([]),
    permissionMode: z.string().min(1).optional(),
    maxTurns: z.number().int().positive().optional(),
    memory: z.string().min(1).optional(),
    effort: z.string().min(1).optional(),
    background: z.boolean().optional(),
    isolation: z.string().min(1).optional(),
    color: z.string().min(1).optional(),
    enabled: z.boolean().default(true),
    readOnly: z.boolean().default(false),
    pluginIds: z.array(z.string().min(1)).default([]),
    source: AgentDefinitionSourceSchema.optional(),
    order: z.number().int().optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export type AgentDefinitionSource = z.infer<typeof AgentDefinitionSourceSchema>;
export type AgentDefinitionGeepilotMetadata = z.infer<typeof AgentDefinitionGeepilotMetadataSchema>;
export type AgentDefinitionFrontmatter = z.infer<typeof AgentDefinitionFrontmatterSchema>;
export type AgentDefinitionDocument = z.infer<typeof AgentDefinitionDocumentSchema>;
export type AgentProfileMetadata = z.infer<typeof AgentProfileMetadataSchema>;

export interface ParseAgentDefinitionOptions {
  source?: AgentDefinitionSource;
}

export interface CreateAgentProfileFromDefinitionOptions {
  id?: string;
  name?: string;
  aliases?: string[];
  harnessId?: string;
  providerProfileId?: string;
  model?: string;
  enabled?: boolean;
  readOnly?: boolean;
  pluginIds?: string[];
  order?: number;
}

export function parseAgentDefinitionMarkdown(
  markdown: string,
  options: ParseAgentDefinitionOptions = {},
): AgentDefinitionDocument {
  const { frontmatterText, body } = splitMarkdownFrontmatter(markdown);
  const rawFrontmatter = parseFrontmatterYaml(frontmatterText);
  return AgentDefinitionDocumentSchema.parse({
    frontmatter: rawFrontmatter,
    rawFrontmatter,
    body,
    source: options.source,
  });
}

export function serializeAgentDefinitionMarkdown(input: unknown): string {
  const definition = AgentDefinitionDocumentSchema.parse(input);
  const frontmatter = compactFrontmatter(definition.frontmatter);
  const yamlText = stringifyYaml(frontmatter).trimEnd();
  return `---\n${yamlText}\n---\n\n${definition.body.replace(/^\n/, "")}`;
}

export function projectAgentDefinitionForClaudeCode(input: unknown): string {
  const definition = AgentDefinitionDocumentSchema.parse(input);
  const { geepilot: _geepilot, ...frontmatter } = definition.frontmatter;
  return serializeAgentDefinitionMarkdown({
    ...definition,
    frontmatter,
    rawFrontmatter: frontmatter,
  });
}

export function createAgentProfileFromDefinition(
  definitionInput: unknown,
  options: CreateAgentProfileFromDefinitionOptions = {},
): AgentProfileMetadata {
  const definition = AgentDefinitionDocumentSchema.parse(definitionInput);
  const frontmatter = definition.frontmatter;
  const geepilot = frontmatter.geepilot;
  const name = options.name ?? frontmatter.name;
  if (!name) {
    throw new Error("Agent definition must provide a name before it can become a profile.");
  }

  const source = definition.source;
  const pluginIds = uniqueStrings([
    ...(options.pluginIds ?? []),
    ...(source?.pluginId ? [source.pluginId] : []),
  ]);
  const aliases = uniqueStrings([
    ...(options.aliases ?? []),
    ...(geepilot?.selector ? [geepilot.selector] : []),
  ]);
  const model = normalizeInheritedValue(options.model ?? frontmatter.model);
  const instructions = nonEmptyString(definition.body.trim()) ?? frontmatter.initialPrompt;
  const sourceDefaultReadOnly = source?.kind === "plugin";
  const readOnly = options.readOnly ?? source?.readOnly ?? sourceDefaultReadOnly;

  return AgentProfileMetadataSchema.parse({
    id: options.id ?? profileIdFromNameOrContent(name, definition),
    name,
    description: frontmatter.description,
    aliases,
    harnessId: options.harnessId ?? geepilot?.harness,
    providerProfileId: options.providerProfileId ?? geepilot?.provider,
    model,
    instructions,
    tools: frontmatter.tools,
    disallowedTools: frontmatter.disallowedTools,
    skills: frontmatter.skills,
    mcpServers: stringListFromUnknown(frontmatter.mcpServers),
    permissionMode: frontmatter.permissionMode,
    maxTurns: frontmatter.maxTurns,
    memory: frontmatter.memory,
    effort: frontmatter.effort,
    background: frontmatter.background,
    isolation: frontmatter.isolation,
    color: frontmatter.color,
    enabled: options.enabled ?? geepilot?.enabled ?? true,
    readOnly,
    pluginIds,
    source,
    order: options.order,
  });
}

export function parseAgentProfileMetadata(input: unknown): AgentProfileMetadata {
  return AgentProfileMetadataSchema.parse(input);
}

function splitMarkdownFrontmatter(markdown: string): {
  frontmatterText: string;
  body: string;
} {
  const match = markdown.match(/^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*\r?\n?([\s\S]*)$/);
  if (!match) return { frontmatterText: "", body: markdown };
  return { frontmatterText: match[1] ?? "", body: match[2] ?? "" };
}

function parseFrontmatterYaml(frontmatterText: string): Record<string, unknown> {
  if (!frontmatterText.trim()) return {};
  let parsed: unknown;
  try {
    parsed = parseYaml(frontmatterText);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid agent definition frontmatter: ${message}`);
  }
  if (parsed === null || parsed === undefined) return {};
  if (!isObjectRecord(parsed)) {
    throw new Error("Agent definition frontmatter must be a YAML object.");
  }
  return parsed;
}

function compactFrontmatter(input: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(input).filter(([, value]) => {
      if (value === undefined) return false;
      if (Array.isArray(value) && value.length === 0) return false;
      return true;
    }),
  );
}

function stringListFromUnknown(value: unknown): string[] | undefined {
  if (value === undefined || value === null || value === "") return undefined;
  if (typeof value === "string") {
    return value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  return value as never;
}

function nonEmptyString(value: string | undefined): string | undefined {
  return value && value.length > 0 ? value : undefined;
}

function normalizeInheritedValue(value: string | undefined): string | undefined {
  if (!value) return undefined;
  return value === "inherit" ? undefined : value;
}

function profileIdFromNameOrContent(name: string, definition: AgentDefinitionDocument): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  if (slug) return slug;
  return `agent_${stableHash(stableJson(definition))}`;
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Agent profile records must not contain inline secret field "${issue.key}".`,
    });
  }
}

function findInlineSecrets(
  value: unknown,
  path: Array<string | number> = [],
): Array<{ key: string; path: Array<string | number> }> {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => findInlineSecrets(item, [...path, index]));
  }
  if (!isObjectRecord(value)) return [];

  const issues: Array<{ key: string; path: Array<string | number> }> = [];
  for (const [key, child] of Object.entries(value)) {
    if (isForbiddenSecretKey(key) && child !== REDACTED_VALUE) {
      issues.push({ key, path: [...path, key] });
    }
    issues.push(...findInlineSecrets(child, [...path, key]));
  }
  return issues;
}

function isForbiddenSecretKey(key: string): boolean {
  const normalizedKey = key.toLowerCase().replace(/[^a-z0-9_]/g, "");
  return (
    FORBIDDEN_SECRET_KEY_PATTERN.test(key) && !ALLOWED_SECRET_REFERENCE_KEYS.has(normalizedKey)
  );
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

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))];
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
