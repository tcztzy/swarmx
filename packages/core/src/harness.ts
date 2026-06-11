import type { AgentBackend } from "./types.js";

export interface HarnessConfig {
  label: string;
  icon: string;
  compatibleProviders: ProviderKind[];
  passthroughEnv: string[];
  backend: AgentBackend;
}

export type ProviderKind = "anthropic" | "openai_chat" | "openai_responses" | "ollama";

export const HARNESSES: Record<string, HarnessConfig> = {
  swarmx: {
    label: "SwarmX",
    icon: "swarmx",
    compatibleProviders: ["anthropic", "openai_chat", "ollama"],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: { type: "swarmx" },
  },
  claude_code: {
    label: "Claude Code",
    icon: "claude",
    compatibleProviders: ["anthropic"],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "bun",
      args: ["x", "--silent", "@agentclientprotocol/claude-agent-acp"],
    },
  },
  codex: {
    label: "Codex",
    icon: "codex",
    compatibleProviders: ["openai_responses", "openai_chat"],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "bun",
      args: ["x", "--silent", "@agentclientprotocol/codex-acp"],
    },
  },
  opencode: {
    label: "OpenCode",
    icon: "opencode",
    compatibleProviders: ["anthropic", "openai_chat", "ollama"],
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
    compatibleProviders: ["openai_chat", "ollama"],
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
    compatibleProviders: ["anthropic"],
    passthroughEnv: ["PATH", "HOME", "LANG", "USER", "SHELL", "TERM"],
    backend: {
      type: "custom",
      program: "openclaw",
      args: ["acp"],
    },
  },
};

export interface ModelProvider {
  kind: ProviderKind;
  apiKey?: string;
  baseUrl?: string;
  model: string;
}

export function providerEnvVars(provider: ModelProvider): Record<string, string> {
  const env: Record<string, string> = {};
  if (provider.model) {
    env.OPENAI_MODEL = provider.model;
  }
  switch (provider.kind) {
    case "anthropic":
      if (provider.apiKey) env.ANTHROPIC_API_KEY = provider.apiKey;
      if (provider.baseUrl) env.ANTHROPIC_BASE_URL = provider.baseUrl;
      break;
    case "openai_chat":
    case "openai_responses":
      if (provider.apiKey) env.OPENAI_API_KEY = provider.apiKey;
      if (provider.baseUrl) env.OPENAI_BASE_URL = provider.baseUrl;
      break;
    case "ollama":
      if (provider.baseUrl) env.OLLAMA_HOST = provider.baseUrl;
      break;
  }
  return env;
}

export function getHarness(name: string): HarnessConfig | undefined {
  return HARNESSES[name];
}

export function getHarnessList(): HarnessConfig[] {
  return Object.values(HARNESSES);
}
