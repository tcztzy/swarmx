import { z } from "zod";

/** Wire protocols through which an independent Model can be invoked. */
export const ModelApiSchema = z.enum(["anthropic", "openai_chat", "openai_responses", "ollama"]);

export type ModelApi = z.infer<typeof ModelApiSchema>;

/** Transport behavior layered on top of a Model API wire protocol. */
export const ModelApiModeSchema = z.enum(["standard", "codex_responses"]);

export type ModelApiMode = z.infer<typeof ModelApiModeSchema>;
