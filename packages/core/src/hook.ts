import { type HookConfig, HookConfigSchema } from "./types.js";

export class Hook {
  onStart?: string;
  onEnd?: string;
  onHandoff?: string;
  onChunk?: string;

  constructor(config?: HookConfig) {
    if (config) {
      const parsed = HookConfigSchema.parse(config);
      this.onStart = parsed.onStart;
      this.onEnd = parsed.onEnd;
      this.onHandoff = parsed.onHandoff;
      this.onChunk = parsed.onChunk;
    }
  }
}
