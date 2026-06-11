import { evaluate } from "cel-js";
import { type EdgeConfig, EdgeConfigSchema } from "./types.js";

export class Edge {
  source: string;
  target: string;
  condition?: string;

  constructor(config: EdgeConfig) {
    const parsed = EdgeConfigSchema.parse(config);
    this.source = parsed.source;
    this.target = parsed.target;
    this.condition = parsed.condition;
  }

  evaluate(context: Record<string, unknown>): boolean {
    if (!this.condition) return true;
    try {
      const result = evaluate(this.condition, context);
      return Boolean(result);
    } catch {
      return true;
    }
  }

  resolveTargets(context: Record<string, unknown>): string[] {
    try {
      const result = evaluate(this.target, context);
      if (result === null || result === undefined) return [];
      if (typeof result === "string") return [result];
      if (Array.isArray(result)) {
        return result.filter((v): v is string => typeof v === "string").map(String);
      }
      if (typeof result === "object" && result !== null) {
        const obj = result as Record<string, unknown>;
        if (typeof obj.destination === "string") return [obj.destination];
        if (Array.isArray(obj.destinations)) {
          return obj.destinations.filter((v): v is string => typeof v === "string").map(String);
        }
      }
      return [String(result)];
    } catch {
      return [this.target];
    }
  }
}
