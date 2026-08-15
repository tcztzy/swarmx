import { z } from "zod";

export const SandboxStrategySchema = z.enum(["native_allowed", "protected_required"]);
export type SandboxStrategy = z.infer<typeof SandboxStrategySchema>;

export const SandboxModeSchema = z.enum(["native", "protected"]);
export type SandboxMode = z.infer<typeof SandboxModeSchema>;

const EnvironmentNameSchema = z
  .string()
  .regex(/^[A-Z_][A-Z0-9_]*$/, "Environment names must use ASCII shell-name characters.");

export const ProtectedSandboxMountSchema = z
  .object({
    source: z.enum(["project", "temporary", "credential"]),
    access: z.enum(["ro", "rw"]),
  })
  .strict();

export type ProtectedSandboxMount = z.infer<typeof ProtectedSandboxMountSchema>;

export const ProtectedSandboxResourcesSchema = z
  .object({
    cpu: z.number().finite().positive().max(64),
    memoryMiB: z.number().int().positive().max(131_072),
    temporaryMiB: z.number().int().positive().max(131_072),
  })
  .strict();

export type ProtectedSandboxResources = z.infer<typeof ProtectedSandboxResourcesSchema>;

export const ProtectedSandboxProfileSchema = z
  .object({
    id: z
      .string()
      .trim()
      .min(1)
      .max(160)
      .regex(/^[a-z0-9][a-z0-9._-]*$/),
    image: z.string().trim().min(1).max(512),
    imageDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
    command: z.array(z.string().min(1).max(4_096)).min(1).max(128),
    environmentAllowlist: z.array(EnvironmentNameSchema).max(256),
    mounts: z.array(ProtectedSandboxMountSchema).min(2).max(8),
    network: z.literal("none"),
    resources: ProtectedSandboxResourcesSchema,
  })
  .strict()
  .superRefine((profile, context) => {
    const projectMounts = profile.mounts.filter((mount) => mount.source === "project");
    const temporaryMounts = profile.mounts.filter((mount) => mount.source === "temporary");
    if (projectMounts.length !== 1 || projectMounts[0]?.access !== "rw") {
      context.addIssue({
        code: "custom",
        path: ["mounts"],
        message: "A protected profile must grant exactly one writable Project mount.",
      });
    }
    if (temporaryMounts.length !== 1 || temporaryMounts[0]?.access !== "rw") {
      context.addIssue({
        code: "custom",
        path: ["mounts"],
        message: "A protected profile must grant exactly one writable temporary mount.",
      });
    }
    const sources = profile.mounts.map((mount) => mount.source);
    if (new Set(sources).size !== sources.length) {
      context.addIssue({
        code: "custom",
        path: ["mounts"],
        message: "Each protected mount source may be declared at most once.",
      });
    }
    if (new Set(profile.environmentAllowlist).size !== profile.environmentAllowlist.length) {
      context.addIssue({
        code: "custom",
        path: ["environmentAllowlist"],
        message: "The protected environment allowlist must not contain duplicates.",
      });
    }
  });

export type ProtectedSandboxProfile = z.infer<typeof ProtectedSandboxProfileSchema>;

export interface SandboxResolution {
  strategy: SandboxStrategy;
  mode: SandboxMode;
  ready: boolean;
  profileId?: string;
  profile?: ProtectedSandboxProfile;
  runtimeId?: string;
  note?: string;
}

export interface SandboxResolutionInput {
  strategy: SandboxStrategy;
  profileId?: string;
  runtimeReady: boolean;
  runtimeId?: string;
  note?: string;
}

/**
 * Registry owned by the host process. Extension metadata is never accepted as
 * a profile source; callers must pass profiles assembled by trusted host code.
 */
export class ProtectedSandboxRegistry {
  readonly #profiles = new Map<string, ProtectedSandboxProfile>();

  constructor(profiles: readonly ProtectedSandboxProfile[] = []) {
    for (const profile of profiles) this.register(profile);
  }

  register(input: ProtectedSandboxProfile): ProtectedSandboxProfile {
    const profile = ProtectedSandboxProfileSchema.parse(input);
    if (this.#profiles.has(profile.id)) {
      throw new Error(`Protected sandbox profile "${profile.id}" is already registered.`);
    }
    const frozen = freezeProfile(profile);
    this.#profiles.set(frozen.id, frozen);
    return frozen;
  }

  get(id: string): ProtectedSandboxProfile | undefined {
    return this.#profiles.get(id);
  }

  ids(): string[] {
    return [...this.#profiles.keys()].sort();
  }

  resolve(input: SandboxResolutionInput): SandboxResolution {
    const strategy = SandboxStrategySchema.parse(input.strategy);
    if (strategy === "native_allowed") {
      return {
        strategy,
        mode: "native",
        ready: true,
        ...(input.profileId ? { profileId: input.profileId } : {}),
        note: "Native execution is allowed by the host sandbox policy.",
      };
    }

    const profile = input.profileId ? this.#profiles.get(input.profileId) : undefined;
    if (input.profileId && !profile) {
      return {
        strategy,
        mode: "protected",
        ready: false,
        profileId: input.profileId,
        note: `No host-registered protected sandbox profile exists for "${input.profileId}".`,
      };
    }
    if (!profile && this.#profiles.size === 0) {
      return {
        strategy,
        mode: "protected",
        ready: false,
        note: "No host-registered protected sandbox profile is available.",
      };
    }
    if (!input.runtimeReady) {
      return {
        strategy,
        mode: "protected",
        ready: false,
        ...(input.profileId ? { profileId: input.profileId } : {}),
        ...(profile ? { profile } : {}),
        ...(input.runtimeId ? { runtimeId: input.runtimeId } : {}),
        note: input.note ?? "The protected sandbox runtime is not ready.",
      };
    }
    return {
      strategy,
      mode: "protected",
      ready: true,
      ...(input.profileId ? { profileId: input.profileId } : {}),
      ...(profile ? { profile } : {}),
      ...(input.runtimeId ? { runtimeId: input.runtimeId } : {}),
      note: "Protected execution is ready; native fallback is disabled.",
    };
  }
}

function freezeProfile(profile: ProtectedSandboxProfile): ProtectedSandboxProfile {
  Object.freeze(profile.environmentAllowlist);
  Object.freeze(profile.command);
  for (const mount of profile.mounts) Object.freeze(mount);
  Object.freeze(profile.mounts);
  Object.freeze(profile.resources);
  return Object.freeze(profile);
}
