import {
  type BuiltinToolStyle,
  type DesktopBuiltinToolSettings,
  DesktopBuiltinToolSettingsSchema,
  type Model,
  resolveBuiltinToolStyle,
  type SessionBuiltinToolBinding,
  type SessionData,
} from "@swarmx/core";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

export class BuiltinToolSettingsService {
  readonly #settings: DesktopSettingsStoreLike;

  constructor(settings: DesktopSettingsStoreLike) {
    this.#settings = settings;
  }

  async get(): Promise<DesktopBuiltinToolSettings> {
    return (await this.#settings.read()).runtime.builtinTools;
  }

  async save(input: unknown): Promise<DesktopBuiltinToolSettings> {
    const builtinTools = DesktopBuiltinToolSettingsSchema.parse(input);
    const settings = await this.#settings.update((current) => ({
      ...current,
      runtime: {
        ...current.runtime,
        builtinTools,
      },
    }));
    return settings.runtime.builtinTools;
  }
}

export interface ResolveRunBuiltinToolsOptions {
  settings: DesktopBuiltinToolSettings;
  model?: Pick<Model, "preferredBuiltinToolStyle">;
  session?: Pick<SessionData, "builtinTools">;
}

export function resolveRunBuiltinTools(
  options: ResolveRunBuiltinToolsOptions,
): SessionBuiltinToolBinding {
  return resolveBuiltinToolStyle({
    configuredStyle: options.settings.style,
    ...(options.model?.preferredBuiltinToolStyle
      ? { modelPreferredStyle: options.model.preferredBuiltinToolStyle }
      : {}),
    ...(options.session?.builtinTools ? { sessionBinding: options.session.builtinTools } : {}),
  });
}

export function preferredBuiltinToolStyleForProvider(
  baseUrl: string | undefined,
  catalogAdapter: string | undefined,
): BuiltinToolStyle | undefined {
  if (catalogAdapter === "codex_app_server") return "codex";
  if (!baseUrl) return undefined;
  let hostname: string;
  try {
    hostname = new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return undefined;
  }
  if (
    hostname === "api.moonshot.cn" ||
    hostname === "api.moonshot.ai" ||
    hostname === "api.kimi.com"
  ) {
    return "kimi_code";
  }
  if (hostname === "api.anthropic.com") return "claude_code";
  return undefined;
}
