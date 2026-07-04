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
  /(api[_-]?key|access[_-]?token|bearer|password|passwd|secret|credential|private[_-]?key|smtp[_-]?password|telemetry[_-]?token|ingest[_-]?token|cluster[_-]?password)/i;
const HEX_SHA256_PATTERN = /^[a-fA-F0-9]{64}$/;
const DEPENDENCY_ID_PATTERN = /^[a-z][a-z0-9_.-]*$/;
const PLATFORM_KEY_PATTERN = /^[a-z0-9][a-z0-9_.-]*$/;
const INSTALL_ROOT_PATTERN =
  /^(?:\$[A-Z][A-Z0-9_]*(?:\/.*)?|\$\{[A-Z][A-Z0-9_]*\}(?:\/.*)?|settings\.[A-Za-z][A-Za-z0-9_.-]*(?:\/.*)?)$/;

export const DependencyKindSchema = z.enum([
  "python-project",
  "python-script",
  "desktop-node-project",
  "system-prerequisite",
  "managed-binary",
  "managed-installer",
  "external-harness-cli",
  "benchmark-asset",
]);

export const DependencyOwnerSchema = z.enum([
  "desktop",
  "server",
  "harness",
  "tuqiao",
  "benchmark",
  "dev",
]);

export const DependencyVersionSourceSchema = z.enum(["fixed", "lockfile", "external-vendor"]);

export const DependencyDetectionStatusSchema = z.enum([
  "detected",
  "missing",
  "installed",
  "failed",
]);

export const DependencyDetectionSourceSchema = z.enum([
  "env",
  "path",
  "managed",
  "missing",
  "lockfile",
  "system",
  "vendor",
  "user",
]);

export const DependencyInstallActionSchema = z.enum([
  "use_existing",
  "install_managed",
  "requires_user_action",
  "unavailable",
]);

export const ManagedDependencyPlatformSchema = z.preprocess(
  normalizePlatformInput,
  z
    .object({
      url: z.string().min(1).optional(),
      sha256: z.string().regex(HEX_SHA256_PATTERN).optional(),
      archiveMember: z.string().min(1).optional(),
      targetName: z.string().min(1).optional(),
      executable: z.boolean().optional(),
      signatureUrl: z.string().min(1).optional(),
      signaturePolicy: z.string().min(1).optional(),
      packageManager: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ManagedDependencySchema = z.preprocess(
  normalizeDependencyInput,
  z
    .object({
      id: z.string().regex(DEPENDENCY_ID_PATTERN),
      kind: DependencyKindSchema,
      owner: DependencyOwnerSchema,
      version: z.string().min(1),
      versionSource: DependencyVersionSourceSchema,
      installRoot: z.string().min(1).optional(),
      platforms: z
        .record(z.string().regex(PLATFORM_KEY_PATTERN), ManagedDependencyPlatformSchema)
        .default({}),
      license: z.string().min(1).optional(),
      homepage: z.string().min(1).optional(),
      minVersion: z.string().min(1).optional(),
      maxVersion: z.string().min(1).optional(),
      postInstallVersionCommand: z.array(z.string().min(1)).optional(),
      trustModel: z.string().min(1).optional(),
      notes: z.string().optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const ManagedDependencyManifestSchema = z.preprocess(
  normalizeManifestInput,
  z
    .object({
      schemaVersion: z.literal(1).default(1),
      dependencies: z.array(ManagedDependencySchema).default([]),
    })
    .passthrough()
    .superRefine((manifest, ctx) => {
      const seen = new Set<string>();
      for (const [index, dependency] of manifest.dependencies.entries()) {
        if (seen.has(dependency.id)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["dependencies", index, "id"],
            message: `Duplicate dependency id "${dependency.id}".`,
          });
        }
        seen.add(dependency.id);
      }
      addSecretIssues(manifest, ctx);
    }),
);

export const DependencyDetectionResultSchema = z.preprocess(
  normalizeDetectionInput,
  z
    .object({
      dependencyId: z.string().regex(DEPENDENCY_ID_PATTERN),
      status: DependencyDetectionStatusSchema,
      source: DependencyDetectionSourceSchema,
      path: z.string().min(1).optional(),
      version: z.string().min(1).optional(),
      sha256: z.string().regex(HEX_SHA256_PATTERN).optional(),
      message: z.string().optional(),
    })
    .passthrough()
    .superRefine(addSecretIssues),
);

export const DependencyInstallReceiptSchema = z.preprocess(
  normalizeReceiptInput,
  z
    .object({
      dependencyId: z.string().regex(DEPENDENCY_ID_PATTERN),
      version: z.string().min(1),
      platform: z.string().regex(PLATFORM_KEY_PATTERN),
      sourceUrl: z.string().min(1),
      sha256: z.string().regex(HEX_SHA256_PATTERN),
      installedPath: z.string().min(1),
      installedAt: z.string().min(1),
      installerVersion: z.string().min(1).optional(),
      swarmxVersion: z.string().min(1).optional(),
    })
    .passthrough()
    .superRefine((receipt, ctx) => {
      addSecretIssues(receipt, ctx);
      addUrlCredentialIssues(receipt.sourceUrl, ctx, ["sourceUrl"]);
    }),
);

export const DependencyInstallPlanSchema = z
  .object({
    dependencyId: z.string().regex(DEPENDENCY_ID_PATTERN),
    action: DependencyInstallActionSchema,
    reason: z.string().min(1),
    platform: z.string().regex(PLATFORM_KEY_PATTERN).optional(),
    selectedPlatform: ManagedDependencyPlatformSchema.optional(),
    detection: DependencyDetectionResultSchema.optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export type DependencyKind = z.infer<typeof DependencyKindSchema>;
export type DependencyOwner = z.infer<typeof DependencyOwnerSchema>;
export type DependencyVersionSource = z.infer<typeof DependencyVersionSourceSchema>;
export type DependencyDetectionStatus = z.infer<typeof DependencyDetectionStatusSchema>;
export type DependencyDetectionSource = z.infer<typeof DependencyDetectionSourceSchema>;
export type DependencyInstallAction = z.infer<typeof DependencyInstallActionSchema>;
export type ManagedDependencyPlatform = z.infer<typeof ManagedDependencyPlatformSchema>;
export type ManagedDependency = z.infer<typeof ManagedDependencySchema>;
export type ManagedDependencyManifest = z.infer<typeof ManagedDependencyManifestSchema>;
export type DependencyDetectionResult = z.infer<typeof DependencyDetectionResultSchema>;
export type DependencyInstallReceipt = z.infer<typeof DependencyInstallReceiptSchema>;
export type DependencyInstallPlan = z.infer<typeof DependencyInstallPlanSchema>;

export interface PlanDependencyActionOptions {
  platformKey?: string;
  detection?: DependencyDetectionResult;
}

export function parseManagedDependencyManifest(input: unknown): ManagedDependencyManifest {
  const manifest = ManagedDependencyManifestSchema.parse(input);
  for (const dependency of manifest.dependencies) {
    validateManagedDependencyPolicy(dependency);
  }
  return manifest;
}

export function parseManagedDependency(input: unknown): ManagedDependency {
  return validateManagedDependencyPolicy(ManagedDependencySchema.parse(input));
}

export function parseDependencyDetectionResult(input: unknown): DependencyDetectionResult {
  return DependencyDetectionResultSchema.parse(input);
}

export function parseDependencyInstallReceipt(input: unknown): DependencyInstallReceipt {
  return DependencyInstallReceiptSchema.parse(input);
}

export function resolveManagedDependency(
  manifestInput: unknown,
  dependencyId: string,
): ManagedDependency {
  const manifest = parseManagedDependencyManifest(manifestInput);
  const matches = manifest.dependencies.filter((dependency) => dependency.id === dependencyId);
  if (matches.length === 1) return matches[0] as ManagedDependency;
  if (matches.length > 1) throw new Error(`Ambiguous dependency id "${dependencyId}".`);
  throw new Error(`Unknown dependency id "${dependencyId}".`);
}

export function selectDependencyPlatform(
  dependencyInput: unknown,
  platformKey: string,
): ManagedDependencyPlatform {
  const dependency = ManagedDependencySchema.parse(dependencyInput);
  const platform = dependency.platforms[platformKey];
  if (!platform) {
    throw new Error(`Dependency "${dependency.id}" does not declare platform "${platformKey}".`);
  }
  return platform;
}

export function validateManagedDependencyPolicy(input: unknown): ManagedDependency {
  const dependency = ManagedDependencySchema.parse(input);
  const issues = dependencyPolicyIssues(dependency);
  if (issues.length > 0) {
    throw new Error(`Dependency "${dependency.id}" is invalid: ${issues.join("; ")}`);
  }
  return dependency;
}

export function planDependencyAction(
  dependencyInput: unknown,
  options: PlanDependencyActionOptions = {},
): DependencyInstallPlan {
  const dependency = validateManagedDependencyPolicy(dependencyInput);
  const detection = options.detection
    ? DependencyDetectionResultSchema.parse(options.detection)
    : undefined;

  if (detection && detection.dependencyId !== dependency.id) {
    throw new Error(
      `Detection result "${detection.dependencyId}" does not match dependency "${dependency.id}".`,
    );
  }

  if (detection?.status === "detected" || detection?.status === "installed") {
    return DependencyInstallPlanSchema.parse({
      dependencyId: dependency.id,
      action: "use_existing",
      reason: `Dependency "${dependency.id}" was detected from ${detection.source}.`,
      detection,
    });
  }

  if (detection?.status === "failed") {
    return DependencyInstallPlanSchema.parse({
      dependencyId: dependency.id,
      action: "unavailable",
      reason: detection.message ?? `Detection failed for dependency "${dependency.id}".`,
      detection,
    });
  }

  if (dependency.kind === "managed-binary") {
    const platformKey = options.platformKey ?? firstPlatformKey(dependency);
    if (!platformKey) {
      return DependencyInstallPlanSchema.parse({
        dependencyId: dependency.id,
        action: "unavailable",
        reason: `Dependency "${dependency.id}" has no platform entry for managed installation.`,
        detection,
      });
    }
    return DependencyInstallPlanSchema.parse({
      dependencyId: dependency.id,
      action: "install_managed",
      reason: `Dependency "${dependency.id}" can be installed from its managed manifest entry.`,
      platform: platformKey,
      selectedPlatform: selectDependencyPlatform(dependency, platformKey),
      detection,
    });
  }

  if (
    dependency.kind === "managed-installer" ||
    dependency.kind === "external-harness-cli" ||
    dependency.kind === "system-prerequisite" ||
    dependency.kind === "python-project" ||
    dependency.kind === "python-script" ||
    dependency.kind === "desktop-node-project"
  ) {
    return DependencyInstallPlanSchema.parse({
      dependencyId: dependency.id,
      action: "requires_user_action",
      reason: userActionReason(dependency),
      detection,
    });
  }

  return DependencyInstallPlanSchema.parse({
    dependencyId: dependency.id,
    action: "unavailable",
    reason: `Dependency "${dependency.id}" is a benchmark-only asset and is not part of product startup.`,
    detection,
  });
}

function dependencyPolicyIssues(dependency: ManagedDependency): string[] {
  const issues: string[] = [];

  if (hasUrlCredentials(dependency.homepage)) {
    issues.push("homepage must not contain embedded URL credentials");
  }

  if (dependency.kind === "benchmark-asset" && dependency.owner !== "benchmark") {
    issues.push("benchmark assets must be owned by benchmark");
  }

  if (dependency.kind === "managed-binary" || dependency.kind === "managed-installer") {
    if (dependency.versionSource !== "fixed") {
      issues.push("managed downloads must use fixed version_source");
    }
    if (!isExactPinnedVersion(dependency.version)) {
      issues.push("managed downloads must pin an exact version");
    }
    if (!dependency.installRoot) {
      issues.push("managed downloads must declare install_root");
    } else if (!INSTALL_ROOT_PATTERN.test(dependency.installRoot)) {
      issues.push("install_root must use a documented env var or settings key");
    }
    if (Object.keys(dependency.platforms).length === 0) {
      issues.push("managed downloads must declare at least one platform entry");
    }
    for (const [platformKey, platform] of Object.entries(dependency.platforms)) {
      issues.push(...platformPolicyIssues(dependency, platformKey, platform));
    }
  }

  if (
    dependency.kind === "external-harness-cli" &&
    dependency.versionSource !== "external-vendor" &&
    !dependency.trustModel
  ) {
    issues.push("external harness CLIs require external-vendor version_source or trust_model");
  }

  return issues;
}

function platformPolicyIssues(
  dependency: ManagedDependency,
  platformKey: string,
  platform: ManagedDependencyPlatform,
): string[] {
  const issues: string[] = [];

  if (!platform.url) {
    issues.push(`platform "${platformKey}" must declare url`);
  } else {
    if (!isHttpsUrl(platform.url)) {
      issues.push(`platform "${platformKey}" url must use HTTPS`);
    }
    if (hasUrlCredentials(platform.url)) {
      issues.push(`platform "${platformKey}" url must not contain embedded credentials`);
    }
  }

  if (!platform.sha256) {
    issues.push(`platform "${platformKey}" must declare sha256`);
  }

  if (platform.signatureUrl) {
    if (!isHttpsUrl(platform.signatureUrl)) {
      issues.push(`platform "${platformKey}" signature_url must use HTTPS`);
    }
    if (hasUrlCredentials(platform.signatureUrl)) {
      issues.push(`platform "${platformKey}" signature_url must not contain embedded credentials`);
    }
  }

  if (platform.archiveMember && !isSafeArchiveMember(platform.archiveMember)) {
    issues.push(`platform "${platformKey}" archive_member must be a safe relative archive path`);
  }

  if (platform.targetName && !isSafeTargetName(platform.targetName)) {
    issues.push(`platform "${platformKey}" target_name must be a safe file name`);
  }

  if (dependency.kind === "managed-binary" && !platform.archiveMember && !platform.targetName) {
    issues.push(`platform "${platformKey}" must name an archive_member or target_name`);
  }

  return issues;
}

function firstPlatformKey(dependency: ManagedDependency): string | undefined {
  return Object.keys(dependency.platforms).sort()[0];
}

function userActionReason(dependency: ManagedDependency): string {
  if (dependency.kind === "managed-installer") {
    return `Dependency "${dependency.id}" is a managed installer and requires an explicit user action before execution.`;
  }
  if (dependency.kind === "external-harness-cli") {
    return `Dependency "${dependency.id}" is installed by an external harness vendor or package manager.`;
  }
  if (dependency.kind === "system-prerequisite") {
    return `Dependency "${dependency.id}" is a system prerequisite and must be provided by the user or OS.`;
  }
  if (dependency.kind === "desktop-node-project") {
    return `Dependency "${dependency.id}" is governed by the desktop package lockfile.`;
  }
  if (dependency.kind === "python-project") {
    return `Dependency "${dependency.id}" is governed by the Python project lockfile.`;
  }
  return `Dependency "${dependency.id}" is governed by its owning script metadata.`;
}

function isExactPinnedVersion(version: string): boolean {
  const value = version.trim().toLowerCase();
  if (!value || value === "latest" || value === "next" || value === "stable") return false;
  if (/[<>=~^*|]/.test(value)) return false;
  if (/\b(x|latest)\b/.test(value)) return false;
  return true;
}

function isHttpsUrl(value: string): boolean {
  try {
    return new URL(value).protocol === "https:";
  } catch {
    return false;
  }
}

function hasUrlCredentials(value: string | undefined): boolean {
  if (!value) return false;
  try {
    const url = new URL(value);
    return !!url.username || !!url.password;
  } catch {
    return false;
  }
}

function isSafeArchiveMember(value: string): boolean {
  if (!value || value.startsWith("/") || value.startsWith("\\") || value.includes("\\")) {
    return false;
  }
  return !value.split("/").some((part) => part === "" || part === "." || part === "..");
}

function isSafeTargetName(value: string): boolean {
  return (
    !!value && !value.includes("/") && !value.includes("\\") && value !== "." && value !== ".."
  );
}

function normalizeManifestInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { schema, schemaVersion, dependencies, ...rest } = input;
  return {
    ...rest,
    schemaVersion: schemaVersion ?? schema,
    dependencies: Array.isArray(dependencies)
      ? dependencies.map((dependency) => normalizeDependencyInput(dependency))
      : dependencies,
  };
}

function normalizeDependencyInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const {
    version_source,
    install_root,
    min_version,
    max_version,
    post_install_version_command,
    trust_model,
    platforms,
    ...rest
  } = input;
  return {
    ...rest,
    versionSource: rest.versionSource ?? version_source,
    installRoot: rest.installRoot ?? install_root,
    minVersion: rest.minVersion ?? min_version,
    maxVersion: rest.maxVersion ?? max_version,
    postInstallVersionCommand: rest.postInstallVersionCommand ?? post_install_version_command,
    trustModel: rest.trustModel ?? trust_model,
    platforms: isObjectRecord(platforms)
      ? Object.fromEntries(
          Object.entries(platforms).map(([key, value]) => [key, normalizePlatformInput(value)]),
        )
      : platforms,
  };
}

function normalizePlatformInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { archive_member, target_name, signature_url, signature_policy, package_manager, ...rest } =
    input;
  return {
    ...rest,
    archiveMember: rest.archiveMember ?? archive_member,
    targetName: rest.targetName ?? target_name,
    signatureUrl: rest.signatureUrl ?? signature_url,
    signaturePolicy: rest.signaturePolicy ?? signature_policy,
    packageManager: rest.packageManager ?? package_manager,
  };
}

function normalizeDetectionInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const { dependency_id, ...rest } = input;
  return {
    ...rest,
    dependencyId: rest.dependencyId ?? dependency_id,
  };
}

function normalizeReceiptInput(input: unknown): unknown {
  if (!isObjectRecord(input)) return input;
  const {
    dependency_id,
    source_url,
    installed_path,
    installed_at,
    installer_version,
    swarmx_version,
    ...rest
  } = input;
  return {
    ...rest,
    dependencyId: rest.dependencyId ?? dependency_id,
    sourceUrl: rest.sourceUrl ?? source_url,
    installedPath: rest.installedPath ?? installed_path,
    installedAt: rest.installedAt ?? installed_at,
    installerVersion: rest.installerVersion ?? installer_version,
    swarmxVersion: rest.swarmxVersion ?? swarmx_version,
  };
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecrets(value)) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Dependency metadata must not contain inline secret field "${issue.key}".`,
    });
  }
}

function addUrlCredentialIssues(
  value: string | undefined,
  ctx: z.RefinementCtx,
  path: Array<string | number>,
): void {
  if (!hasUrlCredentials(value)) return;
  ctx.addIssue({
    code: z.ZodIssueCode.custom,
    path,
    message: "Dependency metadata URLs must not contain embedded credentials.",
  });
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

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}
