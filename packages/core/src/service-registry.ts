import { z } from "zod";
import type { AgentContextEngine } from "./context-engine.js";
import type { LocalTool } from "./local-tool-contracts.js";
import type {
  GlobalMemorySnapshot,
  MemoryReflectionDecision,
  PersonalMemorySnapshot,
} from "./personal-memory.js";
import type { SkillInstructionDelivery } from "./skill-delivery.js";

const VARIANT_ID_PATTERN = /^[A-Za-z][A-Za-z0-9_-]{0,127}$/u;

export const AgentServiceSeamSchema = z.enum(["context_engine", "memory", "skill_evolution"]);
export type AgentServiceSeam = z.infer<typeof AgentServiceSeamSchema>;

export const ServiceVariantIdSchema = z.string().regex(VARIANT_ID_PATTERN);
export type ServiceVariantId = z.infer<typeof ServiceVariantIdSchema>;

export const AblationProfileSchema = z
  .object({
    schemaVersion: z.literal(1),
    profileId: z.string().regex(VARIANT_ID_PATTERN),
    variants: z
      .object({
        context_engine: ServiceVariantIdSchema,
        memory: ServiceVariantIdSchema,
        skill_evolution: ServiceVariantIdSchema,
      })
      .strict(),
  })
  .strict();
export type AblationProfile = z.infer<typeof AblationProfileSchema>;

export const ServiceTopologySchema = z
  .object({
    swarmPath: z.array(z.string().min(1).max(128)).max(32),
    nodeId: z.string().min(1).max(128),
    rootNodeId: z.string().min(1).max(128),
    agentName: z.string().min(1).max(128),
  })
  .strict();
export type ServiceTopology = z.infer<typeof ServiceTopologySchema>;

export const ServiceActivationEntrySchema = z
  .object({
    seam: AgentServiceSeamSchema,
    variantId: ServiceVariantIdSchema,
  })
  .strict();
export type ServiceActivationEntry = z.infer<typeof ServiceActivationEntrySchema>;

export const ServiceActivationReceiptSchema = z
  .object({
    schemaVersion: z.literal(1),
    profileId: z.string().regex(VARIANT_ID_PATTERN),
    topology: ServiceTopologySchema,
    entries: z.array(ServiceActivationEntrySchema).length(3),
  })
  .strict()
  .superRefine((receipt, context) => {
    const seen = new Set(receipt.entries.map((entry) => entry.seam));
    for (const seam of AgentServiceSeamSchema.options) {
      if (!seen.has(seam)) {
        context.addIssue({
          code: "custom",
          path: ["entries"],
          message: `Service activation receipt is missing ${seam}.`,
        });
      }
    }
  });
export type ServiceActivationReceipt = z.infer<typeof ServiceActivationReceiptSchema>;

export const AblationRunReceiptSchema = z
  .object({
    schemaVersion: z.literal(1),
    profileId: z.string().regex(VARIANT_ID_PATTERN),
    variants: AblationProfileSchema.shape.variants,
    activations: z.array(ServiceActivationReceiptSchema).max(10_000),
  })
  .strict()
  .superRefine((receipt, context) => {
    for (const [index, activation] of receipt.activations.entries()) {
      if (activation.profileId !== receipt.profileId) {
        context.addIssue({
          code: "custom",
          path: ["activations", index, "profileId"],
          message: "Service activation profile does not match the run profile.",
        });
      }
      for (const entry of activation.entries) {
        if (receipt.variants[entry.seam] !== entry.variantId) {
          context.addIssue({
            code: "custom",
            path: ["activations", index, "entries"],
            message: `Service activation for ${entry.seam} does not match the run variant.`,
          });
        }
      }
    }
  });
export type AblationRunReceipt = z.infer<typeof AblationRunReceiptSchema>;

export interface AgentServiceInputs {
  contextEngine?: AgentContextEngine;
  personalMemory?: PersonalMemorySnapshot;
  globalMemory?: GlobalMemorySnapshot;
  memoryReflection?: MemoryReflectionDecision;
  memoryTools?: readonly LocalTool[];
  skillInstructions?: readonly SkillInstructionDelivery[];
}

export type ResolvedAgentServices = AgentServiceInputs;

export interface ServiceVariantResolutionInput {
  topology: Readonly<ServiceTopology>;
  services: Readonly<AgentServiceInputs>;
}

export type AgentServiceContribution =
  | { contextEngine?: AgentContextEngine }
  | {
      personalMemory?: PersonalMemorySnapshot;
      globalMemory?: GlobalMemorySnapshot;
      memoryReflection?: MemoryReflectionDecision;
      memoryTools?: readonly LocalTool[];
    }
  | { skillInstructions?: readonly SkillInstructionDelivery[] };

export interface AgentServiceVariant {
  seam: AgentServiceSeam;
  variantId: ServiceVariantId;
  resolve(input: ServiceVariantResolutionInput): AgentServiceContribution;
}

export interface AgentServiceActivation {
  services: Readonly<ResolvedAgentServices>;
  receipt: ServiceActivationReceipt;
}

const SERVICE_SEAMS = AgentServiceSeamSchema.options;

const defaultAblationProfile = AblationProfileSchema.parse({
  schemaVersion: 1,
  profileId: "production",
  variants: {
    context_engine: "production",
    memory: "production",
    skill_evolution: "production",
  },
});
Object.freeze(defaultAblationProfile.variants);
export const DEFAULT_ABLATION_PROFILE: AblationProfile = Object.freeze(defaultAblationProfile);

/** Synchronous startup registry for whole-service built-in Harness variants. */
export class AgentServiceRegistry {
  private readonly variants = new Map<string, AgentServiceVariant>();
  private locked = false;

  register(variantInput: AgentServiceVariant): this {
    if (this.locked) {
      throw new Error("Agent service registry is locked after successful activation assertion.");
    }
    const seam = AgentServiceSeamSchema.parse(variantInput.seam);
    const variantId = ServiceVariantIdSchema.parse(variantInput.variantId);
    const key = serviceKey(seam, variantId);
    if (this.variants.has(key)) {
      throw new Error(`Agent service variant "${key}" is already registered.`);
    }
    this.variants.set(key, { ...variantInput, seam, variantId });
    return this;
  }

  assertEntriesActivated(profileInput: AblationProfile): void {
    const profile = AblationProfileSchema.parse(profileInput);
    const missing = SERVICE_SEAMS.map((seam) => serviceKey(seam, profile.variants[seam])).filter(
      (key) => !this.variants.has(key),
    );
    if (missing.length > 0) {
      throw new Error(
        `Ablation profile "${profile.profileId}" has unresolved services: ${missing.join(", ")}.`,
      );
    }
    this.locked = true;
  }

  activate(
    profileInput: AblationProfile,
    topologyInput: ServiceTopology,
    services: AgentServiceInputs,
  ): AgentServiceActivation {
    const profile = AblationProfileSchema.parse(profileInput);
    const topology = freezeTopology(ServiceTopologySchema.parse(topologyInput));
    this.assertEntriesActivated(profile);

    const resolved: ResolvedAgentServices = {};
    const sourceServices = Object.freeze({ ...services });
    const entries: ServiceActivationEntry[] = [];
    for (const seam of SERVICE_SEAMS) {
      const variantId = profile.variants[seam];
      const variant = this.variants.get(serviceKey(seam, variantId));
      if (!variant) {
        throw new Error(
          `Agent service variant "${serviceKey(seam, variantId)}" disappeared during activation.`,
        );
      }
      applyContribution(resolved, seam, variant.resolve({ topology, services: sourceServices }));
      entries.push({ seam, variantId });
    }

    return {
      services: Object.freeze(resolved),
      receipt: ServiceActivationReceiptSchema.parse({
        schemaVersion: 1,
        profileId: profile.profileId,
        topology,
        entries,
      }),
    };
  }
}

export function createBuiltinAgentServiceRegistry(): AgentServiceRegistry {
  return new AgentServiceRegistry()
    .register({
      seam: "context_engine",
      variantId: "production",
      resolve: ({ services }) => ({ contextEngine: services.contextEngine }),
    })
    .register({ seam: "context_engine", variantId: "baseline", resolve: () => ({}) })
    .register({
      seam: "memory",
      variantId: "production",
      resolve: ({ services }) => ({
        personalMemory: services.personalMemory,
        globalMemory: services.globalMemory,
        memoryReflection: services.memoryReflection,
        memoryTools: services.memoryTools,
      }),
    })
    .register({ seam: "memory", variantId: "baseline", resolve: () => ({}) })
    .register({
      seam: "skill_evolution",
      variantId: "production",
      resolve: ({ services }) => ({ skillInstructions: services.skillInstructions }),
    })
    .register({ seam: "skill_evolution", variantId: "baseline", resolve: () => ({}) });
}

export function createAblationRunReceipt(
  profileInput: AblationProfile,
  activations: readonly ServiceActivationReceipt[],
): AblationRunReceipt {
  const profile = AblationProfileSchema.parse(profileInput);
  return AblationRunReceiptSchema.parse({
    schemaVersion: 1,
    profileId: profile.profileId,
    variants: profile.variants,
    activations,
  });
}

function serviceKey(seam: AgentServiceSeam, variantId: ServiceVariantId): string {
  return `${seam}:${variantId}`;
}

function applyContribution(
  target: ResolvedAgentServices,
  seam: AgentServiceSeam,
  contribution: AgentServiceContribution,
): void {
  if (!contribution || typeof contribution !== "object" || Array.isArray(contribution)) {
    throw new Error(`Agent service variant for "${seam}" returned an invalid contribution.`);
  }
  const allowedKeys: Record<AgentServiceSeam, readonly (keyof AgentServiceInputs)[]> = {
    context_engine: ["contextEngine"],
    memory: ["personalMemory", "globalMemory", "memoryReflection", "memoryTools"],
    skill_evolution: ["skillInstructions"],
  };
  const unexpected = Object.keys(contribution).filter(
    (key) => !allowedKeys[seam].includes(key as keyof AgentServiceInputs),
  );
  if (unexpected.length > 0) {
    throw new Error(
      `Agent service variant for "${seam}" returned cross-seam fields: ${unexpected.join(", ")}.`,
    );
  }
  Object.assign(target, contribution);
}

function freezeTopology(topology: ServiceTopology): Readonly<ServiceTopology> {
  Object.freeze(topology.swarmPath);
  return Object.freeze(topology);
}
