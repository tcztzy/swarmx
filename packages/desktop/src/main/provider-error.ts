import type { MessageChunk } from "@swarmx/core";

export type ProviderErrorCode = "overloaded" | "rate_limited" | "temporarily_unavailable";

export interface ProviderErrorNotice {
  type: "provider_error";
  code: ProviderErrorCode;
  title: string;
  message: string;
  retryable: true;
}

interface ProviderErrorRule {
  code: ProviderErrorCode;
  patterns: readonly RegExp[];
  title: string;
  message: string;
}

const PROVIDER_ERROR_RULES: readonly ProviderErrorRule[] = [
  {
    code: "overloaded",
    patterns: [
      /\b(?:our|the) servers? (?:are|is) (?:currently )?overloaded\b/i,
      /\boverloaded_error\b/i,
      /\b(?:provider|upstream|api) (?:is )?(?:currently )?overloaded\b/i,
    ],
    title: "Provider is temporarily busy",
    message:
      "The selected Provider is overloaded right now. Try again in a moment, or choose another model.",
  },
  {
    code: "rate_limited",
    patterns: [
      /\brate[_ -]?limit(?:ed| exceeded)?\b/i,
      /\btoo many requests\b/i,
      /\b(?:http|status(?: code)?)\s*[:=]?\s*429\b/i,
    ],
    title: "Provider rate limit reached",
    message:
      "This Provider is limiting requests right now. Wait for the limit to reset, or choose another model.",
  },
  {
    code: "temporarily_unavailable",
    patterns: [
      /\bservice (?:is )?temporarily unavailable\b/i,
      /\b(?:provider|upstream) (?:is )?(?:temporarily )?unavailable\b/i,
      /\b(?:upstream|gateway) (?:request )?(?:timed out|timeout)\b/i,
      /\b(?:http|status(?: code)?)\s*[:=]?\s*(?:502|503|504)\b/i,
    ],
    title: "Provider is temporarily unavailable",
    message:
      "The selected Provider is not responding right now. Try again shortly, or choose another model.",
  },
];

export function classifyProviderError(error: unknown): ProviderErrorNotice | null {
  const raw = error instanceof Error ? error.message : String(error);
  const rule = PROVIDER_ERROR_RULES.find((candidate) =>
    candidate.patterns.some((pattern) => pattern.test(raw)),
  );
  if (!rule) return null;
  return {
    type: "provider_error",
    code: rule.code,
    title: rule.title,
    message: rule.message,
    retryable: true,
  };
}

export function providerErrorMessage(error: unknown): MessageChunk | null {
  const notice = classifyProviderError(error);
  if (!notice) return null;
  return {
    role: "system",
    kind: "message",
    content: notice.message,
    structuredContent: notice,
    render: {
      source: "provider",
      status: "failed",
    },
  };
}
