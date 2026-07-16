import { z } from "zod";
import { HARNESSES } from "./harness.js";
import { ModelApiSchema } from "./model-api.js";
import type { ModelApi } from "./model-api.js";

export const ModelReasoningControlSchema = z.enum([
  "none",
  "effort_enum",
  "token_budget",
  "adaptive",
  "unknown",
]);

export const ModelCapabilitySourceSchema = z.object({
  url: z.string().url(),
  checkedAt: z.string().date(),
  applicability: z.string().min(1),
  version: z.string().min(1),
});

export const ModelReasoningParameterMappingSchema = z.object({
  api: z.string().min(1),
  path: z.string().min(1),
});

const ModelCapabilityShape = {
  id: z.string().min(1),
  apiProtocol: ModelApiSchema,
  modelIds: z.array(z.string().min(1)).min(1),
  reasoningControl: ModelReasoningControlSchema,
  supportedEfforts: z.array(z.string().min(1)).default([]),
  defaultEffort: z.string().min(1).optional(),
  parameterMapping: ModelReasoningParameterMappingSchema.optional(),
  effortAliases: z.record(z.string().min(1)).default({}),
  source: ModelCapabilitySourceSchema,
};

export const ModelCapabilitySchema = z
  .object(ModelCapabilityShape)
  .superRefine((capability, ctx) => {
    const uniqueModels = new Set(capability.modelIds);
    if (uniqueModels.size !== capability.modelIds.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["modelIds"],
        message: "Model capability model identifiers must be unique.",
      });
    }

    const uniqueEfforts = new Set(capability.supportedEfforts);
    if (uniqueEfforts.size !== capability.supportedEfforts.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["supportedEfforts"],
        message: "Model capability effort values must be unique.",
      });
    }

    if (capability.reasoningControl === "effort_enum") {
      if (capability.supportedEfforts.length === 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["supportedEfforts"],
          message: "Enum reasoning controls must declare supported effort values.",
        });
      }
      if (!capability.parameterMapping) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["parameterMapping"],
          message: "Enum reasoning controls must declare an API parameter mapping.",
        });
      }
      if (
        capability.defaultEffort &&
        !capability.supportedEfforts.includes(capability.defaultEffort)
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["defaultEffort"],
          message: "The default effort must be one of the supported effort values.",
        });
      }
      for (const [alias, canonical] of Object.entries(capability.effortAliases)) {
        if (!capability.supportedEfforts.includes(canonical)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ["effortAliases", alias],
            message: "Effort aliases must map to a supported canonical value.",
          });
        }
      }
      return;
    }

    if (capability.supportedEfforts.length > 0 || capability.defaultEffort) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["supportedEfforts"],
        message: "Only enum reasoning controls may declare effort values.",
      });
    }
  });

export const ModelCapabilityLookupSchema = z.preprocess(
  normalizeModelLookup,
  z
    .object({
      modelId: z.string().min(1),
      apiProtocol: ModelApiSchema.optional(),
    })
    .strict(),
);

export const ModelReasoningSelectionSchema = z.preprocess(
  normalizeModelLookup,
  z
    .object({
      harnessId: z.string().min(1),
      modelId: z.string().min(1),
      apiProtocol: ModelApiSchema.optional(),
      effort: z.string().optional(),
    })
    .strict(),
);

export const ResolvedModelReasoningCapabilitySchema = z.object({
  capabilityId: z.string().min(1),
  harnessId: z.string().min(1),
  modelId: z.string().min(1),
  apiProtocol: ModelApiSchema,
  reasoningControl: z.literal("effort_enum"),
  supportedEfforts: z.array(z.string().min(1)).min(1),
  defaultEffort: z.string().min(1).optional(),
  parameterMapping: ModelReasoningParameterMappingSchema,
  effortAliases: z.record(z.string().min(1)).default({}),
  source: ModelCapabilitySourceSchema,
});

export const ModelSchema = z
  .object({
    id: z.string().min(1),
    label: z.string().min(1).optional(),
    runtimeModel: z.string().min(1),
    apiProtocols: z.array(ModelApiSchema).min(1),
    capabilityIds: z.array(z.string().min(1)).default([]),
    reasoningCapabilities: z.array(ModelCapabilitySchema).default([]),
    harnessRuntimeModels: z.record(z.string().min(1)).default({}),
    enabled: z.boolean().optional(),
    readOnly: z.boolean().optional(),
  })
  .passthrough()
  .superRefine((model, ctx) => {
    if (new Set(model.apiProtocols).size !== model.apiProtocols.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["apiProtocols"],
        message: "Model API protocols must be unique.",
      });
    }
    for (const [index, capability] of model.reasoningCapabilities.entries()) {
      if (!capability.modelIds.includes(model.id)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["reasoningCapabilities", index, "modelIds"],
          message: "A Model reasoning capability must include the Model id.",
        });
      }
    }
  });

export const ModelSupplyApiCompatibilitySchema = z
  .object({
    mode: z.enum(["auto", "native", "bridge"]).default("auto"),
    targetApi: ModelApiSchema.optional(),
    baseUrl: z.string().min(1).optional(),
  })
  .default({});

export const ModelSupplySchema = z.preprocess(
  normalizeModelSupply,
  z
    .object({
      id: z.string().min(1),
      modelId: z.string().min(1),
      providerProfileId: z.string().min(1),
      runtimeModel: z.string().min(1).optional(),
      apiCompatibility: ModelSupplyApiCompatibilitySchema,
      providerGroup: z.string().min(1).optional(),
      harnessIds: z.array(z.string().min(1)).min(1).optional(),
      reasoningCapabilities: z.array(ModelCapabilitySchema).default([]),
      enabled: z.boolean().optional(),
      readOnly: z.boolean().optional(),
    })
    .passthrough(),
);

export const ModelProviderLabelSchema = z
  .object({
    id: z.string().min(1),
    label: z.string().min(1).optional(),
    runtimeReady: z.boolean().optional(),
    runtimeNote: z.string().optional(),
    enabled: z.boolean().optional(),
  })
  .passthrough();

export const ModelHarnessInventoryEntrySchema = z
  .object({
    id: z.string().min(1),
    runtimeHarnessId: z.string().min(1).optional(),
    modelControl: z.enum(["direct", "session", "unsupported"]),
    modelCompatibility: z.enum(["declared_apis", "any"]),
    supportedModelApis: z.array(ModelApiSchema).default([]),
    requiresExplicitModelRoute: z.boolean().default(false),
    enabled: z.boolean().optional(),
  })
  .passthrough();

export const ResolvedModelSupplySchema = z.object({
  id: z.string().min(1),
  providerProfileId: z.string().min(1),
  providerLabel: z.string().min(1).optional(),
  providerKind: ModelApiSchema.optional(),
  providerGroup: z.string().min(1).optional(),
  runtimeModel: z.string().min(1),
  apiProtocol: ModelApiSchema,
  harnessIds: z.array(z.string().min(1)).min(1).optional(),
  runtimeReady: z.boolean().optional(),
  runtimeNote: z.string().optional(),
  apiCompatibility: ModelSupplyApiCompatibilitySchema,
  reasoning: ResolvedModelReasoningCapabilitySchema.optional(),
});

export const ResolvedHarnessModelSchema = z.object({
  agentId: z.string().min(1),
  harnessId: z.string().min(1),
  modelId: z.string().min(1),
  modelLabel: z.string().min(1),
  runtimeModel: z.string().min(1),
  apiProtocol: ModelApiSchema,
  modelControl: z.enum(["direct", "session"]),
  supplies: z.array(ResolvedModelSupplySchema).default([]),
  reasoning: ResolvedModelReasoningCapabilitySchema.optional(),
});

export type ModelReasoningControl = z.infer<typeof ModelReasoningControlSchema>;
export type ModelCapabilitySource = z.infer<typeof ModelCapabilitySourceSchema>;
export type ModelReasoningParameterMapping = z.infer<typeof ModelReasoningParameterMappingSchema>;
export type ModelCapability = z.infer<typeof ModelCapabilitySchema>;
export type ModelCapabilityLookup = z.infer<typeof ModelCapabilityLookupSchema>;
export type ModelReasoningSelection = z.infer<typeof ModelReasoningSelectionSchema>;
export type Model = z.infer<typeof ModelSchema>;
export type ModelSupplyApiCompatibility = z.infer<typeof ModelSupplyApiCompatibilitySchema>;
export type ModelSupply = z.infer<typeof ModelSupplySchema>;
export type ModelProviderLabel = z.infer<typeof ModelProviderLabelSchema>;
export type ResolvedModelReasoningCapability = z.infer<
  typeof ResolvedModelReasoningCapabilitySchema
>;
export type ModelHarnessInventoryEntry = z.infer<typeof ModelHarnessInventoryEntrySchema>;
export type ResolvedModelSupply = z.infer<typeof ResolvedModelSupplySchema>;
export type ResolvedHarnessModel = z.infer<typeof ResolvedHarnessModelSchema>;

export interface ResolveHarnessModelInventoryOptions {
  harnessId: string;
  models?: readonly unknown[];
  supplies?: readonly unknown[];
  providers?: readonly unknown[];
  harnesses?: readonly unknown[];
  registry?: readonly unknown[];
}

const OPENAI_MODEL_SOURCE = {
  checkedAt: "2026-07-11",
  version: "OpenAI API model documentation current on 2026-07-11",
} as const;

const ANTHROPIC_EFFORT_SOURCE = {
  url: "https://platform.claude.com/docs/en/build-with-claude/effort",
  checkedAt: "2026-07-11",
  version: "Claude API effort documentation current on 2026-07-11",
} as const;

const MODEL_CAPABILITY_DATA = [
  {
    id: "openai-gpt-5-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5", "gpt-5-2025-08-07"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["minimal", "low", "medium", "high"],
    defaultEffort: "medium",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5 aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5", "gpt-5-2025-08-07"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["minimal", "low", "medium", "high"],
    defaultEffort: "medium",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5 aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.1-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.1", "gpt-5.1-2025-11-13"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.1",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.1 aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.1-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.1", "gpt-5.1-2025-11-13"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.1",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.1 aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.2-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.2", "gpt-5.2-2025-12-11"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.2",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.2 aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.2-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.2", "gpt-5.2-2025-12-11"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.2",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.2 aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.4-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.4", "gpt-5.4-2026-03-05"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.4-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.4", "gpt-5.4-2026-03-05"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.4-mini-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 mini aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.4-mini-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 mini aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.4-nano-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.4-nano", "gpt-5.4-nano-2026-03-17"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4-nano",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 nano aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.4-nano-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.4-nano", "gpt-5.4-nano-2026-03-17"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "none",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.4-nano",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.4 nano aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.5-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["gpt-5.5", "gpt-5.5-2026-04-23"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "medium",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.5",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.5 aliases listed on the model page; Chat Completions API",
    },
  },
  {
    id: "openai-gpt-5.5-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.5", "gpt-5.5-2026-04-23"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh"],
    defaultEffort: "medium",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/models/gpt-5.5",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.5 aliases listed on the model page; Responses API",
    },
  },
  {
    id: "openai-gpt-5.6-responses-effort",
    apiProtocol: "openai_responses",
    modelIds: ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["none", "low", "medium", "high", "xhigh", "max"],
    defaultEffort: "medium",
    parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
    source: {
      url: "https://developers.openai.com/api/docs/guides/latest-model",
      ...OPENAI_MODEL_SOURCE,
      applicability: "GPT-5.6 aliases explicitly listed in OpenAI model guidance; Responses API",
    },
  },
  {
    id: "anthropic-opus-4.5-effort",
    apiProtocol: "anthropic",
    modelIds: ["claude-opus-4-5"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["low", "medium", "high"],
    defaultEffort: "high",
    parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
    source: {
      ...ANTHROPIC_EFFORT_SOURCE,
      applicability: "Claude Opus 4.5 alias listed in the effort documentation",
    },
  },
  {
    id: "anthropic-4.6-effort",
    apiProtocol: "anthropic",
    modelIds: ["claude-opus-4-6", "claude-sonnet-4-6"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["low", "medium", "high", "max"],
    defaultEffort: "high",
    parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
    source: {
      ...ANTHROPIC_EFFORT_SOURCE,
      applicability: "Claude Opus 4.6 and Sonnet 4.6 aliases listed in the effort documentation",
    },
  },
  {
    id: "anthropic-latest-enum-effort",
    apiProtocol: "anthropic",
    modelIds: [
      "claude-fable-5",
      "claude-mythos-5",
      "claude-opus-4-7",
      "claude-opus-4-8",
      "claude-sonnet-5",
    ],
    reasoningControl: "effort_enum",
    supportedEfforts: ["low", "medium", "high", "xhigh", "max"],
    defaultEffort: "high",
    parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
    source: {
      ...ANTHROPIC_EFFORT_SOURCE,
      applicability:
        "Claude Fable 5, Mythos 5, Opus 4.7, Opus 4.8, and Sonnet 5 aliases listed in the effort documentation",
    },
  },
  {
    id: "deepseek-v4-pro-chat-effort",
    apiProtocol: "openai_chat",
    modelIds: ["deepseek-v4-pro"],
    reasoningControl: "effort_enum",
    supportedEfforts: ["high", "max"],
    defaultEffort: "high",
    parameterMapping: { api: "openai.chat.completions", path: "reasoning_effort" },
    effortAliases: { low: "high", medium: "high", xhigh: "max" },
    source: {
      url: "https://api-docs.deepseek.com/guides/thinking_mode",
      checkedAt: "2026-07-11",
      applicability:
        "DeepSeek API thinking-mode contract for the deepseek-v4-pro identifier; canonical effort values only",
      version: "DeepSeek API thinking-mode documentation current on 2026-07-11",
    },
  },
  {
    id: "deepseek-thinking-effort-unknown",
    apiProtocol: "openai_chat",
    modelIds: ["deepseek-chat", "deepseek-reasoner"],
    reasoningControl: "unknown",
    supportedEfforts: [],
    source: {
      url: "https://api-docs.deepseek.com/guides/thinking_mode",
      checkedAt: "2026-07-11",
      applicability:
        "Legacy DeepSeek aliases not covered by the documented deepseek-v4-pro effort contract",
      version: "DeepSeek API thinking-mode documentation current on 2026-07-11",
    },
  },
] as const;

export const MODEL_CAPABILITIES: readonly ModelCapability[] = Object.freeze(
  MODEL_CAPABILITY_DATA.map((capability) => ModelCapabilitySchema.parse(capability)),
);

export const MODELS: readonly Model[] = Object.freeze(buildModelCatalog(MODEL_CAPABILITIES));

export function parseModelCapability(input: unknown): ModelCapability {
  return ModelCapabilitySchema.parse(input);
}

export function parseModel(input: unknown): Model {
  return ModelSchema.parse(input);
}

export function parseModelSupply(input: unknown): ModelSupply {
  return ModelSupplySchema.parse(input);
}

export function modelCapabilityRegistry(
  modelsInput: readonly unknown[],
  baseRegistry: readonly unknown[] = MODEL_CAPABILITIES,
): ModelCapability[] {
  const models = modelsInput.map(parseModel);
  const advertised = models.flatMap((model) => model.reasoningCapabilities);
  const capabilities = new Map<string, ModelCapability>();
  for (const capability of baseRegistry) {
    const parsed = parseModelCapability(capability);
    const advertisedModelIds = new Set(
      advertised
        .filter((candidate) => candidate.apiProtocol === parsed.apiProtocol)
        .flatMap((candidate) => candidate.modelIds),
    );
    const remainingModelIds = parsed.modelIds.filter((modelId) => !advertisedModelIds.has(modelId));
    if (remainingModelIds.length > 0) {
      capabilities.set(parsed.id, parseModelCapability({ ...parsed, modelIds: remainingModelIds }));
    }
  }
  for (const capability of advertised) {
    capabilities.set(capability.id, capability);
  }
  return [...capabilities.values()];
}

export function findModelCapability(
  lookupInput: unknown,
  registryInput: readonly unknown[] = MODEL_CAPABILITIES,
): ModelCapability | undefined {
  const lookup = ModelCapabilityLookupSchema.parse(lookupInput);
  const registry = registryInput.map(parseModelCapability);
  const matches = registry.filter(
    (capability) =>
      capability.modelIds.includes(lookup.modelId) &&
      (!lookup.apiProtocol || capability.apiProtocol === lookup.apiProtocol),
  );

  if (matches.length > 1) {
    throw new Error(
      `Ambiguous reasoning capability for model "${lookup.modelId}"; select an API protocol.`,
    );
  }
  return matches[0];
}

export function resolveModelReasoningCapability(
  selectionInput: unknown,
  registryInput: readonly unknown[] = MODEL_CAPABILITIES,
): ResolvedModelReasoningCapability | undefined {
  const selection = ModelReasoningSelectionSchema.parse(selectionInput);
  const model = MODELS.find((candidate) => candidate.id === selection.modelId);
  const apiProtocol =
    selection.apiProtocol ??
    (model ? modelApiForHarness(selection.harnessId, model, builtInHarnessInventory()) : undefined);
  if (!apiProtocol) return undefined;
  const capability = findModelCapability(
    { modelId: selection.modelId, apiProtocol },
    registryInput,
  );
  if (
    !capability ||
    capability.reasoningControl !== "effort_enum" ||
    !capability.parameterMapping
  ) {
    return undefined;
  }

  return ResolvedModelReasoningCapabilitySchema.parse({
    capabilityId: capability.id,
    harnessId: selection.harnessId,
    modelId: selection.modelId,
    apiProtocol,
    reasoningControl: capability.reasoningControl,
    supportedEfforts: capability.supportedEfforts,
    defaultEffort: capability.defaultEffort,
    parameterMapping: capability.parameterMapping,
    effortAliases: capability.effortAliases,
    source: capability.source,
  });
}

export function normalizeModelReasoningEffort(
  selectionInput: unknown,
  registryInput: readonly unknown[] = MODEL_CAPABILITIES,
): string | undefined {
  const selection = ModelReasoningSelectionSchema.parse(selectionInput);
  const capability = resolveModelReasoningCapability(selection, registryInput);
  if (!capability) return undefined;

  const requested = selection.effort?.trim().toLowerCase();
  if (requested && requested !== "default") {
    const canonical = capability.effortAliases[requested] ?? requested;
    if (capability.supportedEfforts.includes(canonical)) return canonical;
  }
  return capability.defaultEffort;
}

export function modelReasoningRequestParameters(
  selectionInput: unknown,
  registryInput: readonly unknown[] = MODEL_CAPABILITIES,
): Record<string, unknown> {
  const selection = ModelReasoningSelectionSchema.parse(selectionInput);
  const capability = resolveModelReasoningCapability(selection, registryInput);
  const effort = normalizeModelReasoningEffort(selection, registryInput);
  if (!capability || !effort) return {};
  return valueAtPath(capability.parameterMapping.path, effort);
}

export function resolveHarnessModelInventory(
  options: ResolveHarnessModelInventoryOptions,
): ResolvedHarnessModel[] {
  const harnesses = (options.harnesses ?? builtInHarnessInventory()).map((harness) =>
    ModelHarnessInventoryEntrySchema.parse(harness),
  );
  const matches = harnesses.filter((harness) => harness.id === options.harnessId);
  if (matches.length !== 1) {
    const reason = matches.length === 0 ? "Unknown" : "Ambiguous";
    throw new Error(`${reason} harness id "${options.harnessId}".`);
  }
  const harness = matches[0] as ModelHarnessInventoryEntry;
  if (harness.enabled === false) return [];
  if (harness.modelControl === "unsupported") return [];

  const models = (options.models ?? MODELS).map(parseModel);
  const supplies = (options.supplies ?? []).map(parseModelSupply);
  const providers = (options.providers ?? []).map((provider) =>
    ModelProviderLabelSchema.parse(provider),
  );
  const providersById = new Map(providers.map((provider) => [provider.id, provider]));
  const registry = options.registry ?? modelCapabilityRegistry(models);
  const resolved: ResolvedHarnessModel[] = [];
  for (const model of models) {
    if (model.enabled === false) continue;
    const adapterId = runtimeHarnessId(harness);
    const fixedRuntimeModel = model.harnessRuntimeModels[adapterId];
    const allModelSupplies = supplies.filter(
      (supply) => supply.enabled !== false && supply.modelId === model.id,
    );
    const modelSupplies = allModelSupplies.filter((supply) =>
      supplySupportsHarness(supply, harness),
    );
    if (allModelSupplies.length > 0 && modelSupplies.length === 0 && !fixedRuntimeModel) continue;
    const apiProtocol = modelApiForHarness(
      harness.id,
      model,
      harnesses,
      modelSupplies,
      providersById,
    );
    if (!apiProtocol) continue;
    const resolvedSupplies = modelSupplies.flatMap((supply) => {
      const provider = providersById.get(supply.providerProfileId);
      const supplyApi = compatibleModelApi(harness, model, [supply], providersById);
      if (!supplyApi) return [];
      const supplyRegistry =
        supply.reasoningCapabilities.length > 0
          ? modelCapabilityRegistry([
              {
                ...model,
                reasoningCapabilities: [
                  ...model.reasoningCapabilities,
                  ...supply.reasoningCapabilities,
                ],
              },
            ])
          : registry;
      return [
        ResolvedModelSupplySchema.parse({
          id: supply.id,
          providerProfileId: supply.providerProfileId,
          providerLabel: provider?.label,
          providerKind: modelApiProperty(provider, "kind"),
          providerGroup: supply.providerGroup,
          runtimeModel: supply.runtimeModel ?? model.runtimeModel,
          apiProtocol: supplyApi,
          harnessIds: supply.harnessIds,
          runtimeReady: provider?.enabled === false ? false : provider?.runtimeReady,
          runtimeNote: provider?.runtimeNote,
          apiCompatibility: supply.apiCompatibility,
          reasoning: resolveModelReasoningCapability(
            { harnessId: harness.id, modelId: model.id, apiProtocol: supplyApi },
            supplyRegistry,
          ),
        }),
      ];
    });
    resolved.push(
      ResolvedHarnessModelSchema.parse({
        agentId: `${harness.id}:${model.id}`,
        harnessId: harness.id,
        modelId: model.id,
        modelLabel: model.label ?? model.id,
        runtimeModel: fixedRuntimeModel ?? model.runtimeModel,
        apiProtocol,
        modelControl: harness.modelControl,
        supplies: resolvedSupplies,
        reasoning: resolveModelReasoningCapability(
          { harnessId: harness.id, modelId: model.id, apiProtocol },
          registry,
        ),
      }),
    );
  }
  return resolved;
}

function builtInHarnessInventory(): ModelHarnessInventoryEntry[] {
  return Object.entries(HARNESSES).map(([id, harness]) => ({
    id,
    runtimeHarnessId: id,
    modelControl: harness.modelControl,
    modelCompatibility: harness.modelCompatibility,
    supportedModelApis: [...harness.supportedModelApis],
    requiresExplicitModelRoute: harness.requiresExplicitModelRoute ?? false,
    enabled: harness.enabled,
  }));
}

export function isHarnessModelCompatible(harnessInput: unknown, modelInput: unknown): boolean {
  const harness = ModelHarnessInventoryEntrySchema.parse(harnessInput);
  const model = ModelSchema.parse(modelInput);
  return !!compatibleModelApi(harness, model);
}

function modelApiForHarness(
  harnessId: string,
  model: Model,
  harnesses: readonly ModelHarnessInventoryEntry[],
  supplies: readonly ModelSupply[] = [],
  providersById: ReadonlyMap<string, ModelProviderLabel> = new Map(),
): ModelApi | undefined {
  const harness = harnesses.find((candidate) => candidate.id === harnessId);
  return harness ? compatibleModelApi(harness, model, supplies, providersById) : undefined;
}

function compatibleModelApi(
  harness: ModelHarnessInventoryEntry,
  model: Model,
  supplies: readonly ModelSupply[] = [],
  providersById: ReadonlyMap<string, ModelProviderLabel> = new Map(),
): ModelApi | undefined {
  if (harness.enabled === false || harness.modelControl === "unsupported") return undefined;
  const orderedSupplies = [...supplies].sort((left, right) => {
    const leftProvider = providersById.get(left.providerProfileId);
    const rightProvider = providersById.get(right.providerProfileId);
    return (
      preferredNativeSupplyRank(left, leftProvider) -
        preferredNativeSupplyRank(right, rightProvider) || left.id.localeCompare(right.id)
    );
  });
  if (
    harness.modelControl === "session" &&
    harness.requiresExplicitModelRoute &&
    !model.harnessRuntimeModels[runtimeHarnessId(harness)] &&
    orderedSupplies.length === 0
  ) {
    return undefined;
  }
  if (harness.modelCompatibility === "any") {
    for (const supply of orderedSupplies) {
      const nativeApi = nativeSupplyApi(supply, providersById.get(supply.providerProfileId));
      if (nativeApi && model.apiProtocols.includes(nativeApi)) return nativeApi;
    }
    return model.apiProtocols[0];
  }
  const compatibleApis = harness.supportedModelApis.filter((api) =>
    model.apiProtocols.includes(api),
  );
  if (compatibleApis.length === 0) return undefined;

  for (const supply of orderedSupplies) {
    const nativeApi = nativeSupplyApi(supply, providersById.get(supply.providerProfileId));
    if (nativeApi && compatibleApis.includes(nativeApi)) return nativeApi;
  }
  for (const supply of orderedSupplies) {
    const targetApi = supply.apiCompatibility.targetApi;
    if (targetApi && compatibleApis.includes(targetApi)) return targetApi;
  }
  return compatibleApis[0];
}

function supplySupportsHarness(supply: ModelSupply, harness: ModelHarnessInventoryEntry): boolean {
  if (harness.modelControl === "direct") return true;
  const adapterId = runtimeHarnessId(harness);
  if (harness.requiresExplicitModelRoute) {
    return Boolean(supply.runtimeModel) && supply.harnessIds?.includes(adapterId) === true;
  }
  return !supply.harnessIds || supply.harnessIds.includes(adapterId);
}

function runtimeHarnessId(harness: ModelHarnessInventoryEntry): string {
  return harness.runtimeHarnessId ?? harness.id;
}

function preferredNativeSupplyRank(
  supply: ModelSupply,
  provider: ModelProviderLabel | undefined,
): number {
  const providerKind = modelApiProperty(provider, "kind");
  return providerKind && supply.apiCompatibility.targetApi === providerKind ? 0 : 1;
}

function modelApiProperty(record: object | undefined, key: string): ModelApi | undefined {
  const value = record ? (record as Record<string, unknown>)[key] : undefined;
  const parsed = ModelApiSchema.safeParse(value);
  return parsed.success ? parsed.data : undefined;
}

function nativeSupplyApi(
  supply: ModelSupply,
  provider: ModelProviderLabel | undefined,
): ModelApi | undefined {
  if (supply.apiCompatibility.mode === "bridge") return undefined;
  const targetApi = supply.apiCompatibility.targetApi;
  if (supply.apiCompatibility.mode === "native" && targetApi) return targetApi;

  const providerRecord = provider as Record<string, unknown> | undefined;
  const providerKind = ModelApiSchema.safeParse(providerRecord?.kind);
  const entrypoints = providerRecord?.apiEntrypoints;
  if (targetApi && providerSupportsApi(providerKind, entrypoints, targetApi)) return targetApi;
  return providerKind.success ? providerKind.data : undefined;
}

function providerSupportsApi(
  providerKind: ReturnType<typeof ModelApiSchema.safeParse>,
  entrypoints: unknown,
  targetApi: ModelApi,
): boolean {
  if (providerKind.success && providerKind.data === targetApi) return true;
  if (!entrypoints || typeof entrypoints !== "object" || Array.isArray(entrypoints)) return false;
  return typeof (entrypoints as Record<string, unknown>)[targetApi] === "string";
}

function buildModelCatalog(capabilities: readonly ModelCapability[]): Model[] {
  const entries = new Map<string, { apiProtocols: ModelApi[]; capabilityIds: string[] }>();
  for (const capability of capabilities) {
    for (const modelId of capability.modelIds) {
      const entry = entries.get(modelId) ?? { apiProtocols: [], capabilityIds: [] };
      if (!entry.apiProtocols.includes(capability.apiProtocol)) {
        entry.apiProtocols.push(capability.apiProtocol);
      }
      entry.capabilityIds.push(capability.id);
      entries.set(modelId, entry);
    }
  }
  return [...entries.entries()].map(([id, entry]) =>
    ModelSchema.parse({
      id,
      label: id,
      runtimeModel: id,
      apiProtocols: entry.apiProtocols,
      capabilityIds: entry.capabilityIds,
      ...(id === "deepseek-v4-pro"
        ? { harnessRuntimeModels: { claude_code: "deepseek-v4-pro[1m]" } }
        : {}),
      readOnly: true,
    }),
  );
}

function normalizeModelLookup(input: unknown): unknown {
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const record = input as Record<string, unknown>;
  const { model, ...rest } = record;
  return { ...rest, modelId: rest.modelId ?? model };
}

function normalizeModelSupply(input: unknown): unknown {
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const record = input as Record<string, unknown>;
  const compatibility =
    record.apiCompatibility ?? record.api_compatibility ?? ({} as Record<string, unknown>);
  const normalizedCompatibility =
    compatibility && typeof compatibility === "object" && !Array.isArray(compatibility)
      ? {
          ...(compatibility as Record<string, unknown>),
          targetApi:
            (compatibility as Record<string, unknown>).targetApi ??
            (compatibility as Record<string, unknown>).target_api ??
            (compatibility as Record<string, unknown>).targetKind ??
            (compatibility as Record<string, unknown>).target_kind,
          baseUrl:
            (compatibility as Record<string, unknown>).baseUrl ??
            (compatibility as Record<string, unknown>).base_url,
        }
      : compatibility;
  return {
    ...record,
    modelId: record.modelId ?? record.model_id,
    providerProfileId: record.providerProfileId ?? record.provider_profile_id,
    runtimeModel: record.runtimeModel ?? record.runtime_model,
    providerGroup: record.providerGroup ?? record.provider_group,
    harnessIds: record.harnessIds ?? record.harness_ids,
    reasoningCapabilities: record.reasoningCapabilities ?? record.reasoning_capabilities,
    apiCompatibility: normalizedCompatibility,
  };
}

function valueAtPath(path: string, value: string): Record<string, unknown> {
  const segments = path.split(".").filter(Boolean);
  if (segments.length === 0) return {};
  return segments.reduceRight<Record<string, unknown>>(
    (nested, segment, index) => ({ [segment]: index === segments.length - 1 ? value : nested }),
    {},
  );
}
