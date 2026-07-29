import { z } from "zod";

export const BUILTIN_TOOL_CONTRACT_REVISION = 1 as const;

export const BuiltinToolStyleSchema = z.enum(["claude_code", "codex", "kimi_code"]);
export const BuiltinToolStylePreferenceSchema = z.enum([
  "auto",
  "claude_code",
  "codex",
  "kimi_code",
]);
export const BuiltinToolStyleResolutionSourceSchema = z.enum(["settings", "model", "fallback"]);

export const SessionBuiltinToolBindingSchema = z
  .object({
    style: BuiltinToolStyleSchema,
    revision: z.literal(BUILTIN_TOOL_CONTRACT_REVISION),
    source: BuiltinToolStyleResolutionSourceSchema,
  })
  .strict();

export type BuiltinToolStyle = z.infer<typeof BuiltinToolStyleSchema>;
export type BuiltinToolStylePreference = z.infer<typeof BuiltinToolStylePreferenceSchema>;
export type BuiltinToolStyleResolutionSource = z.infer<
  typeof BuiltinToolStyleResolutionSourceSchema
>;
export type SessionBuiltinToolBinding = z.infer<typeof SessionBuiltinToolBindingSchema>;

export interface ResolveBuiltinToolStyleOptions {
  configuredStyle: BuiltinToolStylePreference;
  modelPreferredStyle?: BuiltinToolStyle;
  sessionBinding?: SessionBuiltinToolBinding;
}

export function resolveBuiltinToolStyle(
  options: ResolveBuiltinToolStyleOptions,
): SessionBuiltinToolBinding {
  if (options.sessionBinding) {
    return SessionBuiltinToolBindingSchema.parse(options.sessionBinding);
  }
  if (options.configuredStyle !== "auto") {
    return {
      style: options.configuredStyle,
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "settings",
    };
  }
  if (options.modelPreferredStyle) {
    return {
      style: options.modelPreferredStyle,
      revision: BUILTIN_TOOL_CONTRACT_REVISION,
      source: "model",
    };
  }
  return {
    style: "codex",
    revision: BUILTIN_TOOL_CONTRACT_REVISION,
    source: "fallback",
  };
}
