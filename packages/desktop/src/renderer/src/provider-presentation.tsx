import { KeyRound } from "lucide-react";
import type { ExtensionCapabilityInventory } from "../../shared/desktop-api.js";

type ExtensionProvider = ExtensionCapabilityInventory["providers"][number];

export function ProviderBrandIcon({
  label,
  sourceId,
  provider,
}: {
  label: string;
  sourceId: string;
  provider?: ExtensionProvider;
}) {
  const normalizedLabel = label.toLowerCase();
  const normalizedUrl = provider?.baseUrl?.toLowerCase() ?? "";
  const iconUrl =
    sourceId === "codex" || normalizedLabel === "codex"
      ? "./harness-icons/codex.svg"
      : isOpenAIProvider(provider)
        ? "./harness-icons/codex.svg"
        : isDeepSeekProvider(provider)
          ? "./provider-icons/deepseek.svg"
          : normalizedLabel.includes("packy") || normalizedUrl.includes("packyapi.com")
            ? "./provider-icons/packy.svg"
            : provider?.usageAdapter === "new_api"
              ? "./provider-icons/new-api.png"
              : undefined;
  return (
    <span className="settings-provider-matrix__icon" aria-hidden="true">
      {iconUrl ? <img src={iconUrl} alt="" /> : <KeyRound />}
    </span>
  );
}

export function isDeepSeekProvider(provider: ExtensionProvider | undefined): boolean {
  return !!provider && isDeepSeekProviderUrl(provider.baseUrl ?? "");
}

export function isOpenAIProvider(provider: ExtensionProvider | undefined): boolean {
  if (!provider?.baseUrl) return false;
  try {
    return new URL(provider.baseUrl).hostname.toLowerCase() === "api.openai.com";
  } catch {
    return provider.baseUrl.toLowerCase().includes("api.openai.com");
  }
}

export function isDeepSeekProviderUrl(value: string): boolean {
  try {
    return new URL(value).hostname.toLowerCase() === "api.deepseek.com";
  } catch {
    return value.toLowerCase().includes("api.deepseek.com");
  }
}

export function isOpenCodeGoProviderUrl(value: string): boolean {
  try {
    const url = new URL(value);
    const pathname = url.pathname.replace(/\/+$/, "");
    return (
      url.protocol === "https:" &&
      url.hostname.toLowerCase() === "opencode.ai" &&
      !url.port &&
      (pathname === "/zen/go" || pathname === "/zen/go/v1")
    );
  } catch {
    return false;
  }
}

export function providerProtocolLabel(value: string): string {
  if (value === "anthropic") return "Anthropic";
  if (value === "openai_chat") return "OpenAI Chat";
  if (value === "openai_responses") return "OpenAI Responses";
  if (value === "ollama") return "Ollama";
  return value.replaceAll("_", " ");
}
