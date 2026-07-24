import {
  Bot,
  Code2,
  Hammer,
  type LucideIcon,
  Sparkles,
  Terminal as TerminalIcon,
  Workflow,
} from "lucide-react";
import { useState } from "react";
import { APP_ICON_URL } from "./app-brand.js";
import { PACKAGED_HARNESS_ICON_URLS } from "./harness-icon-data.js";

export interface HarnessOption {
  id: string;
  label: string;
  icon: LucideIcon;
  modelControl: "direct" | "session" | "unsupported";
  disabled?: boolean;
  disabledReason?: string;
}

export const HARNESSES: HarnessOption[] = [
  { id: "swarmx", label: "SwarmX", icon: Workflow, modelControl: "direct" },
  { id: "claude_code", label: "Claude Code", icon: Hammer, modelControl: "session" },
  { id: "codex", label: "Codex", icon: TerminalIcon, modelControl: "session" },
  { id: "pi", label: "Pi", icon: Bot, modelControl: "session" },
  { id: "kimi", label: "Kimi Code", icon: Bot, modelControl: "session" },
  { id: "opencode", label: "OpenCode", icon: Code2, modelControl: "session" },
  { id: "hermes", label: "Hermes", icon: Sparkles, modelControl: "session" },
  {
    id: "openclaw",
    label: "OpenClaw",
    icon: Bot,
    modelControl: "unsupported",
    disabled: true,
    disabledReason: "Model switching is not configured.",
  },
];

export function HarnessBrandIcon({ harness }: { harness: HarnessOption }) {
  const [failedIconUrl, setFailedIconUrl] = useState<string | null>(null);
  const Fallback = harness.icon;
  const iconUrl = harness.id === "swarmx" ? APP_ICON_URL : PACKAGED_HARNESS_ICON_URLS[harness.id];
  if (!iconUrl || failedIconUrl === iconUrl) {
    return <Fallback aria-hidden="true" data-harness-icon-fallback={harness.id} />;
  }
  return (
    <img
      className="harness-brand-icon"
      src={iconUrl}
      alt=""
      aria-hidden="true"
      data-harness-icon={harness.id}
      onError={() => setFailedIconUrl(iconUrl)}
    />
  );
}

export function harnessOption(id: string, label: string): HarnessOption {
  return (
    HARNESSES.find((harness) => harness.id === id) ?? {
      id,
      label,
      icon: Bot,
      modelControl: "session",
    }
  );
}
