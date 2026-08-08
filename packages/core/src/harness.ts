import type { ModelApi } from "./model-api.js";
import type { AgentBackend } from "./types.js";

export type HarnessModelControl = "direct" | "session" | "unsupported";
export type HarnessModelCompatibility = "declared_apis" | "any";

export interface HarnessModelLaunchOptions {
  modelId: string;
  runtimeModel?: string;
  effort?: string;
  env?: Readonly<Record<string, string | undefined>>;
}

export interface HarnessSoftware {
  name: string;
  version?: string;
  runner?: string;
  command?: string[];
}

export interface HarnessConfig {
  label: string;
  icon: string;
  software: HarnessSoftware;
  /** How a request-scoped model is applied. */
  modelControl: HarnessModelControl;
  /** Whether compatibility uses Model API metadata or the harness runtime catalog. */
  modelCompatibility: HarnessModelCompatibility;
  /** Wire protocols accepted directly by this harness. Never provider identities. */
  supportedModelApis: ModelApi[];
  /** Session Harnesses require a fixed Model route or an explicitly compatible ModelSupply. */
  requiresExplicitModelRoute?: boolean;
  passthroughEnv: string[];
  backend: AgentBackend;
  /** False when a built-in entry is documented but not executable. */
  enabled?: boolean;
}

const CODEX_ACP_VERSION = "1.1.2";
const CLAUDE_AGENT_ACP_VERSION = "0.58.1";
const PI_ACP_VERSION = "0.0.31";
const CLAUDE_DEEPSEEK_PRO_MODEL = "deepseek-v4-pro[1m]";
const CLAUDE_DEEPSEEK_FLASH_MODEL = "deepseek-v4-flash";

function npxSoftware(
  packageName: string,
  version: string,
): HarnessSoftware & { runner: string; command: string[] } {
  return {
    name: packageName.split("/").at(-1) ?? packageName,
    version,
    runner: "npx",
    command: ["--yes", `${packageName}@${version}`],
  };
}

const CLAUDE_AGENT_ACP_SOFTWARE = npxSoftware(
  "@agentclientprotocol/claude-agent-acp",
  CLAUDE_AGENT_ACP_VERSION,
);
const CODEX_ACP_SOFTWARE = npxSoftware("@agentclientprotocol/codex-acp", CODEX_ACP_VERSION);
const PI_ACP_SOFTWARE = npxSoftware("pi-acp", PI_ACP_VERSION);

export const HARNESSES: Record<string, HarnessConfig> = {
  swarmx: {
    label: "SwarmX",
    icon: "swarmx",
    software: { name: "swarmx" },
    modelControl: "direct",
    modelCompatibility: "declared_apis",
    supportedModelApis: ["anthropic", "openai_responses", "openai_chat"],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: { type: "swarmx" },
  },
  claude_code: {
    label: "Claude Code",
    icon: "claude",
    software: CLAUDE_AGENT_ACP_SOFTWARE,
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["anthropic", "openai_chat", "openai_responses", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: CLAUDE_AGENT_ACP_SOFTWARE.runner,
      args: CLAUDE_AGENT_ACP_SOFTWARE.command,
    },
  },
  codex: {
    label: "Codex",
    icon: "codex",
    software: CODEX_ACP_SOFTWARE,
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["openai_responses", "openai_chat", "anthropic", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: [
      "PATH",
      "HOME",
      "LANG",
      "USER",
      "SHELL",
      "TERM",
      "CODEX_HOME",
      "APP_SERVER_LOGS",
    ],
    backend: {
      type: "custom",
      program: CODEX_ACP_SOFTWARE.runner,
      args: CODEX_ACP_SOFTWARE.command,
    },
  },
  pi: {
    label: "Pi",
    icon: "pi",
    software: PI_ACP_SOFTWARE,
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["anthropic", "openai_chat", "openai_responses", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: [
      "PATH",
      "HOME",
      "LANG",
      "USER",
      "SHELL",
      "TERM",
      "PI_CODING_AGENT_DIR",
      "PI_CODING_AGENT_SESSION_DIR",
      "PI_PACKAGE_DIR",
      "PI_OFFLINE",
      "PI_SKIP_VERSION_CHECK",
      "PI_CACHE_RETENTION",
    ],
    backend: {
      type: "custom",
      program: PI_ACP_SOFTWARE.runner,
      args: PI_ACP_SOFTWARE.command,
    },
  },
  kimi: {
    label: "Kimi Code",
    icon: "kimi",
    software: { name: "kimi", runner: "kimi", command: ["acp"] },
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["anthropic", "openai_chat", "openai_responses", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: [
      "PATH",
      "HOME",
      "LANG",
      "USER",
      "SHELL",
      "TERM",
      "KIMI_CODE_HOME",
      "KIMI_DISABLE_TELEMETRY",
      "KIMI_CODE_NO_AUTO_UPDATE",
    ],
    backend: {
      type: "custom",
      program: "kimi",
      args: ["acp"],
    },
  },
  opencode: {
    label: "OpenCode",
    icon: "opencode",
    software: { name: "opencode", runner: "opencode", command: ["acp"] },
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["anthropic", "openai_chat", "openai_responses", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "opencode",
      args: ["acp"],
    },
  },
  hermes: {
    label: "Hermes",
    icon: "hermes",
    software: { name: "hermes", runner: "hermes", command: ["acp"] },
    modelControl: "session",
    modelCompatibility: "any",
    supportedModelApis: ["openai_chat", "openai_responses", "anthropic", "ollama"],
    requiresExplicitModelRoute: true,
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "hermes",
      args: ["acp"],
    },
  },
  openclaw: {
    label: "OpenClaw",
    icon: "openclaw",
    software: { name: "openclaw", runner: "openclaw", command: ["acp"] },
    modelControl: "unsupported",
    modelCompatibility: "any",
    supportedModelApis: [],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "openclaw",
      args: ["acp"],
    },
  },
};

export function getHarness(name: string): HarnessConfig | undefined {
  const harness = HARNESSES[name];
  return harness?.enabled === false ? undefined : harness;
}

export function getHarnessList(): HarnessConfig[] {
  return Object.values(HARNESSES).filter((harness) => harness.enabled !== false);
}

/** Build request-scoped vendor bootstrap only; this never edits global harness config. */
export function harnessModelRuntimeEnv(
  harnessId: string,
  options: HarnessModelLaunchOptions,
): Record<string, string> {
  const harness = getHarness(harnessId);
  if (!harness) throw new Error(`Unknown harness: ${harnessId}`);
  if (harness.modelControl === "unsupported") {
    throw new Error(`Harness "${harnessId}" does not support request-scoped model selection.`);
  }
  const runtimeModel = harnessModelRuntimeModel(harnessId, options);
  switch (harnessId) {
    case "claude_code":
      if (options.modelId === "deepseek-v4-pro") {
        const apiKey = options.env?.DEEPSEEK_API_KEY;
        if (!apiKey) {
          throw new Error(
            'Harness "claude_code" with Model "deepseek-v4-pro" requires env secret "DEEPSEEK_API_KEY".',
          );
        }
        return {
          ANTHROPIC_BASE_URL: "https://api.deepseek.com/anthropic",
          ANTHROPIC_AUTH_TOKEN: apiKey,
          ANTHROPIC_MODEL: CLAUDE_DEEPSEEK_PRO_MODEL,
          ANTHROPIC_DEFAULT_OPUS_MODEL: CLAUDE_DEEPSEEK_PRO_MODEL,
          ANTHROPIC_DEFAULT_SONNET_MODEL: CLAUDE_DEEPSEEK_PRO_MODEL,
          ANTHROPIC_DEFAULT_HAIKU_MODEL: CLAUDE_DEEPSEEK_FLASH_MODEL,
          CLAUDE_CODE_SUBAGENT_MODEL: CLAUDE_DEEPSEEK_FLASH_MODEL,
          CLAUDE_CODE_EFFORT_LEVEL: options.effort ?? "max",
          CLAUDE_MODEL_CONFIG: JSON.stringify({ availableModels: [CLAUDE_DEEPSEEK_PRO_MODEL] }),
        };
      }
      return {
        ANTHROPIC_MODEL: runtimeModel,
        ANTHROPIC_CUSTOM_MODEL_OPTION: runtimeModel,
        CLAUDE_MODEL_CONFIG: JSON.stringify({ availableModels: [runtimeModel] }),
      };
    case "codex":
      return {
        CODEX_CONFIG: JSON.stringify({
          model: runtimeModel,
          ...(options.effort ? { model_reasoning_effort: options.effort } : {}),
        }),
      };
    case "opencode":
      return {
        OPENCODE_CONFIG_CONTENT: JSON.stringify({ model: runtimeModel }),
      };
    default:
      return {};
  }
}

/** Resolve the fixed runtime model alias for one Harness x Model pair. */
export function harnessModelRuntimeModel(
  harnessId: string,
  options: Pick<HarnessModelLaunchOptions, "modelId" | "runtimeModel">,
): string {
  if (harnessId === "claude_code" && options.modelId === "deepseek-v4-pro") {
    return CLAUDE_DEEPSEEK_PRO_MODEL;
  }
  return options.runtimeModel ?? options.modelId;
}
