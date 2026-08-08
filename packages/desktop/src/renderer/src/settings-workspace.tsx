import type {
  BuiltinToolStylePreference,
  DesktopBuiltinToolSettings,
  HarnessPermissionMode,
  HarnessPermissionPolicyLayer,
  SessionPermissionMode,
} from "@swarmx/core";
import type { HarnessEnvironmentStatus } from "@swarmx/runtime";
import {
  ArrowLeft,
  Bot,
  Check,
  ChevronDown,
  Clock3,
  Gauge,
  Hammer,
  KeyRound,
  Loader2,
  type LucideIcon,
  Package,
  Plus,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Terminal as TerminalIcon,
  Trash2,
  User,
  X,
  XCircle,
} from "lucide-react";
import type React from "react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type {
  DesktopPermissionStatus,
  ExtensionCapabilityInventory,
  ModelApiProtocol,
  ModelCatalogSummary,
  ProviderKeyUsageSummary,
  ProviderUsageEntry,
  ProviderUsageMeter,
  ProviderUsageSnapshot,
  ProviderUsageTarget,
  UserProviderInput,
} from "../../shared/desktop-api.js";
import {
  composerModelOptionId,
  groupComposerModels,
  resolveComposerModelOptions,
} from "./agent-picker.js";
import {
  formatSoftwareSummary,
  nativeAgentHostLabel,
  uniqueById,
} from "./extension-presentation.js";
import {
  isDeepSeekProvider,
  isDeepSeekProviderUrl,
  isOpenCodeGoProviderUrl,
  ProviderBrandIcon,
  providerProtocolLabel,
} from "./provider-presentation.js";
import { capitalize, errorMessage, formatTimestamp, lines, slugId } from "./text-utils.js";
import { Badge, Button, cx } from "./ui-primitives.js";

export type SettingsSection =
  | "general"
  | "profile"
  | "permissions"
  | "providers"
  | "extensions"
  | "agents"
  | "runtime";

type ExtensionProviderSummary = ExtensionCapabilityInventory["providers"][number];
type ExtensionAgentSummary = ExtensionCapabilityInventory["agents"][number];

export function SettingsSidebar({
  section,
  query,
  onQueryChange,
  onSectionChange,
  onBack,
}: {
  section: SettingsSection;
  query: string;
  onQueryChange: (query: string) => void;
  onSectionChange: (section: SettingsSection) => void;
  onBack: () => void;
}) {
  const normalizedQuery = query.trim().toLowerCase();
  const personalSections = [
    { id: "general" as const, label: "General", icon: Settings },
    { id: "profile" as const, label: "Profile", icon: User },
  ].filter((item) => item.label.toLowerCase().includes(normalizedQuery));
  const systemSections = [
    { id: "permissions" as const, label: "Advanced permissions", icon: ShieldCheck },
    { id: "providers" as const, label: "Providers", icon: KeyRound },
    { id: "extensions" as const, label: "Extensions", icon: Package },
    { id: "agents" as const, label: "Custom Agents", icon: Bot },
    { id: "runtime" as const, label: "Runtime", icon: TerminalIcon },
  ].filter((item) => item.label.toLowerCase().includes(normalizedQuery));
  const renderSections = (
    label: string,
    sections: Array<{ id: SettingsSection; label: string; icon: LucideIcon }>,
  ) =>
    sections.length > 0 ? (
      <>
        <span className="settings-sidebar__group-label">{label}</span>
        {sections.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              className={section === item.id ? "is-active" : undefined}
              aria-current={section === item.id ? "page" : undefined}
              onClick={() => onSectionChange(item.id)}
            >
              <Icon aria-hidden="true" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </>
    ) : null;

  return (
    <div className="settings-sidebar">
      <button type="button" className="settings-sidebar__back" onClick={onBack}>
        <ArrowLeft aria-hidden="true" />
        <span>Back to app</span>
      </button>
      <label className="sidebar-search settings-sidebar__search">
        <Search aria-hidden="true" />
        <input
          type="search"
          value={query}
          placeholder="Search settings"
          aria-label="Search settings"
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </label>
      <nav className="settings-sidebar__sections" aria-label="Settings sections">
        {renderSections("Personal", personalSections)}
        {renderSections("System", systemSections)}
        {personalSections.length === 0 && systemSections.length === 0 && (
          <span className="settings-sidebar__empty">No matching settings</span>
        )}
      </nav>
    </div>
  );
}

const PERMISSION_TOOL_SUGGESTIONS = [
  "Read",
  "Glob",
  "Grep",
  "LSP",
  "Edit",
  "Write",
  "apply_patch",
  "Bash",
  "exec_command",
  "Task",
  "WebFetch",
  "WebSearch",
];

const PERMISSION_MODE_OPTIONS: Array<{
  id: HarnessPermissionMode;
  label: string;
  description: string;
}> = [
  {
    id: "plan",
    label: "Plan only",
    description: "Read-only tools can run; writes and commands are denied.",
  },
  {
    id: "restricted",
    label: "Restricted",
    description: "Only read-only and explicitly pre-approved tools can run.",
  },
  {
    id: "default",
    label: "Ask for approval",
    description: "Read-only tools run; each write or command needs one-time approval.",
  },
  {
    id: "auto",
    label: "Auto-review",
    description: "Read and Project writes run; commands and control actions still ask once.",
  },
  {
    id: "trusted",
    label: "Full tool access",
    description: "Tools run without prompts, while the host OS sandbox still applies.",
  },
];

const GENERAL_PERMISSION_MODE_OPTIONS = [
  {
    id: "default" as const,
    label: "Default permissions",
    description:
      "By default, SwarmX can read and edit files in its Project. It asks for additional access when needed.",
  },
  {
    id: "auto" as const,
    label: "Auto-review",
    description:
      "SwarmX automatically approves lower-risk Project changes. Commands and control actions can still ask.",
  },
  {
    id: "trusted" as const,
    label: "Full access",
    description:
      "SwarmX can edit files and run commands without approval. This increases the risk of data loss or unexpected changes.",
  },
];

const GENERAL_BUILTIN_TOOL_STYLE_OPTIONS: Array<{
  id: BuiltinToolStylePreference;
  label: string;
  description: string;
}> = [
  {
    id: "auto",
    label: "Auto (recommended)",
    description: "Use the selected Model's declared native tool compatibility.",
  },
  {
    id: "claude_code",
    label: "Claude Code",
    description: "Use Claude Code-trained tool names, arguments, and result conventions.",
  },
  {
    id: "codex",
    label: "Codex",
    description: "Use Codex exec_command, write_stdin, and apply_patch contracts.",
  },
  {
    id: "kimi_code",
    label: "Kimi Code",
    description: "Use Kimi Code file, Bash, todo, and background-task contracts.",
  },
];

export function GeneralSettings({
  status,
  loading,
  error,
  builtinTools,
  builtinToolsLoading,
  builtinToolsError,
  onSaveProfiles,
  onSaveBuiltinTools,
}: {
  status?: DesktopPermissionStatus;
  loading: boolean;
  error: unknown;
  builtinTools?: DesktopBuiltinToolSettings;
  builtinToolsLoading: boolean;
  builtinToolsError: unknown;
  onSaveProfiles: (
    profileAvailability: DesktopPermissionStatus["profileAvailability"],
  ) => Promise<void>;
  onSaveBuiltinTools: (settings: DesktopBuiltinToolSettings) => Promise<void>;
}) {
  const [savingMode, setSavingMode] = useState<
    keyof DesktopPermissionStatus["profileAvailability"] | null
  >(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savingBuiltinTools, setSavingBuiltinTools] = useState(false);
  const [builtinToolsSaveError, setBuiltinToolsSaveError] = useState<string | null>(null);

  const toggleProfile = async (mode: keyof DesktopPermissionStatus["profileAvailability"]) => {
    if (!status || savingMode) return;
    setSavingMode(mode);
    setSaveError(null);
    try {
      await onSaveProfiles({
        ...status.profileAvailability,
        [mode]: !status.profileAvailability[mode],
      });
    } catch (saveFailure) {
      setSaveError(errorMessage(saveFailure));
    } finally {
      setSavingMode(null);
    }
  };

  const saveBuiltinToolStyle = async (style: BuiltinToolStylePreference) => {
    if (!builtinTools || savingBuiltinTools) return;
    setSavingBuiltinTools(true);
    setBuiltinToolsSaveError(null);
    try {
      await onSaveBuiltinTools({ style });
    } catch (saveFailure) {
      setBuiltinToolsSaveError(errorMessage(saveFailure));
    } finally {
      setSavingBuiltinTools(false);
    }
  };

  if (loading && !status) {
    return (
      <section className="settings-workspace general-settings" aria-label="General settings">
        <div className="settings-workspace__loading">
          <Loader2 className="is-spinning" aria-hidden="true" /> Loading General settings…
        </div>
      </section>
    );
  }

  return (
    <section className="settings-workspace general-settings" aria-label="General settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content general-settings__content">
          <div className="general-settings__heading">
            <h2>General</h2>
          </div>

          {Boolean(saveError || error || builtinToolsSaveError || builtinToolsError) && (
            <div className="settings-provider-error">
              {saveError ?? builtinToolsSaveError ?? errorMessage(error ?? builtinToolsError)}
            </div>
          )}

          <section className="general-settings__section" aria-labelledby="general-permissions">
            <h3 id="general-permissions">Permissions</h3>
            <fieldset className="general-permission-card" disabled={!status || Boolean(savingMode)}>
              <legend className="sr-only">Available permission profiles</legend>
              {GENERAL_PERMISSION_MODE_OPTIONS.map((option) => {
                const enabled = status?.profileAvailability[option.id] ?? false;
                return (
                  <label key={option.id}>
                    <input
                      type="checkbox"
                      role="switch"
                      aria-checked={enabled}
                      value={option.id}
                      checked={enabled}
                      onChange={() => void toggleProfile(option.id)}
                    />
                    <span className="general-permission-card__copy">
                      <strong>{option.label}</strong>
                      <small>{option.description}</small>
                    </span>
                    <span
                      className={cx(
                        "general-permission-card__switch",
                        enabled && "is-enabled",
                        savingMode === option.id && "is-saving",
                      )}
                      aria-hidden="true"
                    >
                      {savingMode === option.id ? <Loader2 className="is-spinning" /> : <span />}
                    </span>
                  </label>
                );
              })}
            </fieldset>
          </section>

          <section className="general-settings__section" aria-labelledby="general-agent-runtime">
            <h3 id="general-agent-runtime">Agent runtime</h3>
            <div className="general-runtime-card">
              <label htmlFor="general-builtin-tool-style">
                <span>
                  <strong>Built-in tool style</strong>
                  <small>
                    Applies to new direct SwarmX conversations. External ACP Harnesses keep their
                    native tools, and existing conversations keep their bound style.
                  </small>
                </span>
                <select
                  id="general-builtin-tool-style"
                  aria-label="Built-in tool style"
                  value={builtinTools?.style ?? "auto"}
                  disabled={!builtinTools || builtinToolsLoading || savingBuiltinTools}
                  onChange={(event) =>
                    void saveBuiltinToolStyle(event.target.value as BuiltinToolStylePreference)
                  }
                >
                  {GENERAL_BUILTIN_TOOL_STYLE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <p>
                {
                  GENERAL_BUILTIN_TOOL_STYLE_OPTIONS.find(
                    (option) => option.id === (builtinTools?.style ?? "auto"),
                  )?.description
                }
              </p>
            </div>
          </section>
        </div>
      </div>
    </section>
  );
}

export function PermissionsSettings({
  status,
  loading,
  error,
  projectName,
  agentName,
  onSave,
}: {
  status?: DesktopPermissionStatus;
  loading: boolean;
  error: unknown;
  projectName?: string;
  agentName?: string;
  onSave: (policy: unknown) => Promise<void>;
}) {
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [deniedTools, setDeniedTools] = useState<string[]>([]);
  const [mode, setMode] = useState<HarnessPermissionMode>("default");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  useEffect(() => {
    if (!status) return;
    setAllowedTools(status.personalPolicy.allowedTools);
    setDeniedTools(status.personalPolicy.deniedTools);
    setMode(status.personalPolicy.mode ?? status.defaultMode);
    setSaveError(null);
  }, [status]);

  const save = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      await onSave({
        mode,
        allowedTools,
        deniedTools,
      });
    } catch (saveFailure) {
      setSaveError(errorMessage(saveFailure));
    } finally {
      setSaving(false);
    }
  };

  if (loading && !status) {
    return (
      <section
        className="settings-workspace permission-settings"
        aria-label="Advanced permissions settings"
      >
        <div className="settings-workspace__loading">
          <Loader2 className="is-spinning" aria-hidden="true" /> Loading permission policy…
        </div>
      </section>
    );
  }

  if (error && !status) {
    return (
      <section
        className="settings-workspace permission-settings"
        aria-label="Advanced permissions settings"
      >
        <div className="settings-provider-error">{errorMessage(error)}</div>
      </section>
    );
  }

  const effective = status?.effective;
  const effectiveMode = effective?.policy.mode ?? "default";
  const effectiveModeLabel = permissionModeLabel(effectiveMode);

  return (
    <section
      className="settings-workspace permission-settings"
      aria-label="Advanced permissions settings"
    >
      <div className="settings-workspace__body">
        <div className="settings-workspace__content permission-settings__content">
          <div className="settings-content-heading permission-settings__heading">
            <span>
              <small>Exact rules and audit</small>
              <h2>Advanced permissions</h2>
              <p>
                Review effective authority, configure exact tool rules, and inspect one-call
                decisions. Profile availability lives in General.
              </p>
            </span>
            <button
              type="button"
              className="settings-primary-action"
              disabled={saving}
              onClick={() => void save()}
            >
              {saving ? "Saving…" : "Save policy"}
            </button>
          </div>

          {Boolean(saveError || error) && (
            <div className="settings-provider-error">{saveError ?? errorMessage(error)}</div>
          )}

          <section className="permission-fallback" aria-labelledby="permission-fallback-title">
            <span>
              <h3 id="permission-fallback-title">Inherited fallback</h3>
              <p>
                This is what <strong>Use default</strong> means for direct SwarmX conversations.
                Plan only and Restricted remain available here as conservative profiles.
              </p>
            </span>
            <label>
              <span>Default mode</span>
              <select
                value={mode}
                onChange={(event) => setMode(event.target.value as HarnessPermissionMode)}
              >
                {PERMISSION_MODE_OPTIONS.map((option) => (
                  <option
                    key={option.id}
                    value={option.id}
                    disabled={
                      (option.id === "default" ||
                        option.id === "auto" ||
                        option.id === "trusted") &&
                      !status?.profileAvailability[option.id]
                    }
                  >
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </section>

          <section
            className={cx(
              "permission-effective",
              status?.blocked && "permission-effective--blocked",
            )}
            aria-label="Effective permission policy"
          >
            <div className="permission-effective__icon">
              {status?.blocked ? (
                <XCircle aria-hidden="true" />
              ) : (
                <ShieldCheck aria-hidden="true" />
              )}
            </div>
            <div className="permission-effective__copy">
              <small>
                {status?.blocked
                  ? "Execution blocked"
                  : agentName
                    ? `Effective policy · ${agentName}`
                    : "Effective policy · before Agent"}
              </small>
              <h3>{status?.blocked ? "A policy source needs attention" : effectiveModeLabel}</h3>
              <p>
                {status?.blocked
                  ? "Malformed managed or Project policy fails closed until its source is corrected."
                  : permissionModeDescription(effectiveMode)}
              </p>
            </div>
            {!status?.blocked && effective && (
              <dl className="permission-effective__stats">
                <div>
                  <dt>Pre-approved</dt>
                  <dd>{effective.policy.allowedTools.length}</dd>
                </div>
                <div>
                  <dt>Denied</dt>
                  <dd>{effective.policy.deniedTools.length}</dd>
                </div>
                <div>
                  <dt>Sources</dt>
                  <dd>{effective.layers.length}</dd>
                </div>
              </dl>
            )}
          </section>

          <section className="permission-panel" aria-labelledby="personal-permission-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Editable on this device</small>
                <h3 id="personal-permission-heading">Exact tool rules</h3>
                <p>
                  Pre-approvals and denials combine with the default mode, managed, Project, and
                  Agent policy at execution time.
                </p>
              </span>
              <Badge tone="active">Personal</Badge>
            </div>

            <div className="permission-rule-grid">
              <PermissionToolRulesEditor
                label="Pre-approved tools"
                description="Exact tool names that may run without a prompt."
                values={allowedTools}
                blockedValues={deniedTools}
                onChange={setAllowedTools}
              />
              <PermissionToolRulesEditor
                label="Denied tools"
                description="Exact tool names that can never run. Denials always win."
                values={deniedTools}
                blockedValues={allowedTools}
                tone="danger"
                onChange={setDeniedTools}
              />
            </div>
          </section>

          <section className="permission-panel" aria-labelledby="permission-sources-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Effective authority stack</small>
                <h3 id="permission-sources-heading">Policy sources</h3>
                <p>
                  {projectName
                    ? `Project policy is resolved for ${projectName}.`
                    : "Choose a Project to include its repository policy."}
                </p>
              </span>
              <code>{status?.projectPolicyPath ?? ".swarmx/permissions.json"}</code>
            </div>
            <div className="permission-layer-list">
              {status?.layers.map((layer) => (
                <article
                  key={layer.id}
                  className={cx(
                    "permission-layer-card",
                    layer.error && "permission-layer-card--error",
                  )}
                >
                  <div className="permission-layer-card__heading">
                    <span className={`permission-source permission-source--${layer.source}`}>
                      {permissionSourceLabel(layer.source)}
                    </span>
                    <Badge tone={layer.error ? "danger" : layer.configured ? "active" : "neutral"}>
                      {layer.error ? "Invalid" : layer.configured ? "Applied" : "Not configured"}
                    </Badge>
                  </div>
                  <strong>{layer.label}</strong>
                  {layer.error ? (
                    <p>{layer.error}</p>
                  ) : layer.configured ? (
                    <dl>
                      <div>
                        <dt>Mode</dt>
                        <dd>{layer.mode ? permissionModeLabel(layer.mode) : "No ceiling"}</dd>
                      </div>
                      <div>
                        <dt>Allow</dt>
                        <dd>{layer.allowedTools.length}</dd>
                      </div>
                      <div>
                        <dt>Deny</dt>
                        <dd>{layer.deniedTools.length}</dd>
                      </div>
                    </dl>
                  ) : (
                    <p>No policy was found for this source.</p>
                  )}
                </article>
              ))}
            </div>
          </section>

          <section className="permission-panel" aria-labelledby="permission-history-heading">
            <div className="permission-panel__heading">
              <span>
                <small>Sanitized local audit trail</small>
                <h3 id="permission-history-heading">Approval history</h3>
                <p>Only the tool, decision, reviewer, source, and policy provenance are stored.</p>
              </span>
              <Badge>{status?.approvalReceipts.length ?? 0} receipts</Badge>
            </div>
            {(status?.approvalReceipts.length ?? 0) === 0 ? (
              <div className="permission-history-empty">
                <Clock3 aria-hidden="true" /> No approval decisions yet.
              </div>
            ) : (
              <div className="permission-history-list">
                {status?.approvalReceipts.map((receipt) => (
                  <article key={`${receipt.id}:${receipt.createdAt}`}>
                    <span
                      className={cx(
                        "permission-decision",
                        `permission-decision--${receipt.decision}`,
                      )}
                    >
                      {receipt.decision}
                    </span>
                    <span>
                      <strong>{receipt.toolName}</strong>
                      <small>
                        {receipt.source.toUpperCase()}
                        {receipt.decidedBy === "llm" ? " · AUTO REVIEW" : " · USER"}
                        {receipt.risk ? ` · ${receipt.risk} risk` : ""}
                        {receipt.toolKind ? ` · ${receipt.toolKind}` : ""}
                        {receipt.policySourceIds.length > 0
                          ? ` · ${receipt.policySourceIds.join(" + ")}`
                          : ""}
                      </small>
                    </span>
                    <time dateTime={receipt.createdAt}>
                      {formatPermissionTime(receipt.createdAt)}
                    </time>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </section>
  );
}

function PermissionToolRulesEditor({
  label,
  description,
  values,
  blockedValues,
  tone = "default",
  onChange,
}: {
  label: string;
  description: string;
  values: string[];
  blockedValues: string[];
  tone?: "default" | "danger";
  onChange: (values: string[]) => void;
}) {
  const listId = useId();
  const [draft, setDraft] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const add = () => {
    const toolName = draft.trim();
    if (!toolName) {
      setInputError("Enter an exact tool name.");
      return;
    }
    if (values.includes(toolName)) {
      setInputError(`${toolName} is already listed.`);
      return;
    }
    if (blockedValues.includes(toolName)) {
      setInputError(`${toolName} already has the opposite rule.`);
      return;
    }
    onChange([...values, toolName]);
    setDraft("");
    setInputError(null);
  };

  return (
    <div className={cx("permission-rule-editor", tone === "danger" && "is-danger")}>
      <div>
        <strong>{label}</strong>
        <p>{description}</p>
      </div>
      <div className="permission-rule-editor__input">
        <input
          list={listId}
          value={draft}
          aria-label={`${label} tool name`}
          placeholder="Type an exact tool name"
          onChange={(event) => {
            setDraft(event.target.value);
            setInputError(null);
          }}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            add();
          }}
        />
        <datalist id={listId}>
          {PERMISSION_TOOL_SUGGESTIONS.filter(
            (toolName) => !values.includes(toolName) && !blockedValues.includes(toolName),
          ).map((toolName) => (
            <option key={toolName} value={toolName} />
          ))}
        </datalist>
        <button type="button" onClick={add}>
          <Plus aria-hidden="true" /> Add
        </button>
      </div>
      {inputError && <small className="permission-rule-editor__error">{inputError}</small>}
      <div className="permission-rule-editor__chips" aria-label={`${label} rules`}>
        {values.length === 0 ? (
          <small>No exact-tool rules.</small>
        ) : (
          values.map((toolName) => (
            <span key={toolName}>
              <code>{toolName}</code>
              <button
                type="button"
                aria-label={`Remove ${toolName} from ${label}`}
                onClick={() => onChange(values.filter((value) => value !== toolName))}
              >
                <X aria-hidden="true" />
              </button>
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function permissionModeLabel(mode: HarnessPermissionMode): string {
  return PERMISSION_MODE_OPTIONS.find((option) => option.id === mode)?.label ?? mode;
}

function permissionModeDescription(mode: HarnessPermissionMode): string {
  return PERMISSION_MODE_OPTIONS.find((option) => option.id === mode)?.description ?? "";
}

function permissionSourceLabel(source: HarnessPermissionPolicyLayer["source"]): string {
  if (source === "project") return "Project";
  return `${source.slice(0, 1).toUpperCase()}${source.slice(1)}`;
}

function formatPermissionTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

const CONVERSATION_PERMISSION_OPTIONS: Array<{
  id: SessionPermissionMode;
  label: string;
  shortLabel: string;
  description: string;
}> = [
  {
    id: "inherit",
    label: "Use default",
    shortLabel: "Default",
    description: "Follow General and the selected Agent's default mode.",
  },
  {
    id: "default",
    label: "Ask for approval",
    shortLabel: "Ask",
    description: "Read-only tools run; writes and commands ask once.",
  },
  {
    id: "auto",
    label: "Approve for me",
    shortLabel: "Auto",
    description: "Review Project changes automatically; commands can still ask.",
  },
  {
    id: "trusted",
    label: "Full access",
    shortLabel: "Full access",
    description: "Run without prompts inside the unchanged Project sandbox.",
  },
  {
    id: "plan",
    label: "Plan only",
    shortLabel: "Plan",
    description: "Inspect and plan without writes or commands.",
  },
];

export function ConversationPermissionPicker({
  open,
  mode,
  supported,
  profileAvailability,
  disabled,
  onOpenChange,
  onChange,
}: {
  open: boolean;
  mode: SessionPermissionMode;
  supported: boolean;
  profileAvailability?: DesktopPermissionStatus["profileAvailability"];
  disabled: boolean;
  onOpenChange: (open: boolean) => void;
  onChange: (mode: SessionPermissionMode) => Promise<void>;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [savingMode, setSavingMode] = useState<SessionPermissionMode | null>(null);
  const descriptionId = useId();
  const availableOptions = CONVERSATION_PERMISSION_OPTIONS.filter((option) => {
    if (option.id === "inherit" || option.id === "plan") return true;
    return profileAvailability?.[option.id] ?? true;
  });
  const selected =
    availableOptions.find((option) => option.id === mode) ??
    CONVERSATION_PERMISSION_OPTIONS.find((option) => option.id === "plan") ??
    CONVERSATION_PERMISSION_OPTIONS[0];

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) onOpenChange(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onOpenChange(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onOpenChange, open]);

  const selectMode = async (nextMode: SessionPermissionMode) => {
    if (savingMode) return;
    setSavingMode(nextMode);
    try {
      await onChange(nextMode);
      onOpenChange(false);
    } catch {
      // The owning Composer surfaces the persistence error without closing this menu.
    } finally {
      setSavingMode(null);
    }
  };

  return (
    <div className="conversation-permission-picker" ref={rootRef}>
      <button
        type="button"
        className="conversation-permission-picker__trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-describedby={descriptionId}
        disabled={disabled || !supported}
        title={
          supported
            ? "Set permissions for this conversation"
            : "External ACP Harnesses keep their native permission controls."
        }
        onClick={() => onOpenChange(!open)}
      >
        <ShieldCheck aria-hidden="true" />
        <span>{supported ? selected?.shortLabel : "Harness managed"}</span>
        <ChevronDown aria-hidden="true" />
      </button>
      <span id={descriptionId} className="sr-only">
        {supported
          ? "This selection applies to this conversation only."
          : "External ACP Harnesses keep their native permission controls."}
      </span>
      {open && supported && (
        <section
          className="conversation-permission-picker__menu"
          role="menu"
          aria-label="Conversation permissions"
        >
          <div className="conversation-permission-picker__options">
            {availableOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                role="menuitemradio"
                aria-checked={mode === option.id}
                className={cx(
                  mode === option.id && "is-selected",
                  option.id === "trusted" && "is-trusted",
                  option.id === "plan" && "is-secondary",
                )}
                disabled={Boolean(savingMode)}
                onClick={() => void selectMode(option.id)}
              >
                <span className="conversation-permission-picker__check">
                  {savingMode === option.id ? (
                    <Loader2 className="is-spinning" aria-hidden="true" />
                  ) : mode === option.id ? (
                    <Check aria-hidden="true" />
                  ) : null}
                </span>
                <span>
                  <strong>{option.label}</strong>
                  <small>{option.description}</small>
                </span>
              </button>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export function CustomAgentsSettings({
  inventory,
  environment,
  onSave,
  onRemove,
  onSetupSoftware,
}: {
  inventory?: ExtensionCapabilityInventory;
  environment?: HarnessEnvironmentStatus;
  onSave: (input: unknown) => Promise<void>;
  onRemove: (id: string) => Promise<void>;
  onSetupSoftware: (harnessId: string) => Promise<void>;
}) {
  const agents = inventory?.agents ?? [];
  const customAgents = agents.filter((agent) => Boolean(agent.harnessRecipe) && !agent.readOnly);
  const nativeAgents = agents.filter(
    (agent) => !agent.harnessRecipe && Boolean(agent.definition?.host),
  );
  const extensionAgents = agents.filter((agent) => !agent.harnessRecipe && !agent.definition?.host);
  const customHarnessIds = new Set(
    agents.flatMap((agent) => (agent.harnessRecipe ? [agent.harnessRecipe.id] : [])),
  );
  const softwareOptions = (inventory?.harnesses ?? []).filter(
    (harness) => !customHarnessIds.has(harness.id),
  );
  const skills = uniqueById(inventory?.skills ?? []);
  const mcpServers = uniqueById(inventory?.mcpServers ?? []);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [softwareId, setSoftwareId] = useState(softwareOptions[0]?.id ?? "");
  const [modelOptionId, setModelOptionId] = useState("");
  const [instructions, setInstructions] = useState("");
  const [skillModes, setSkillModes] = useState<
    Record<string, { mode: "off" | "auto" | "required"; variantId?: string }>
  >({});
  const [selectedMcps, setSelectedMcps] = useState<ReadonlySet<string>>(new Set());
  const [contextPaths, setContextPaths] = useState("");
  const [instructionFiles, setInstructionFiles] = useState("AGENTS.md");
  const [permissionMode, setPermissionMode] = useState<HarnessPermissionMode>("default");
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [deniedTools, setDeniedTools] = useState<string[]>([]);
  const [unsupportedSkill, setUnsupportedSkill] = useState<"block" | "skip">("block");
  const [saving, setSaving] = useState(false);
  const [setupBusy, setSetupBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const modelOptions = useMemo(() => {
    if (!inventory || !softwareId) return [];
    try {
      return resolveComposerModelOptions(inventory, softwareId);
    } catch {
      return [];
    }
  }, [inventory, softwareId]);
  const modelGroups = useMemo(() => groupComposerModels(modelOptions, ""), [modelOptions]);
  const defaultModelOptionId = modelGroups[0]?.subgroups[0]?.models[0]?.id ?? "";
  const selectedModelOption = modelOptions.find((model) => model.id === modelOptionId);
  const editingAgent = customAgents.find((agent) => agent.id === editingId);
  const unavailableEditingModel =
    editingAgent?.modelId && !selectedModelOption
      ? {
          id: modelOptionId,
          label: `${editingAgent.modelId} (unavailable)`,
        }
      : undefined;

  useEffect(() => {
    if (!softwareId && softwareOptions[0]) {
      setSoftwareId(softwareOptions[0].id);
      return;
    }
    if (modelOptionId && modelOptions.some((model) => model.id === modelOptionId)) return;
    if (editingId && modelOptionId) return;
    setModelOptionId(defaultModelOptionId);
  }, [defaultModelOptionId, editingId, modelOptionId, modelOptions, softwareId, softwareOptions]);

  const reset = () => {
    setEditingId(null);
    setName("");
    setSoftwareId(softwareOptions[0]?.id ?? "");
    setModelOptionId("");
    setInstructions("");
    setSkillModes({});
    setSelectedMcps(new Set());
    setContextPaths("");
    setInstructionFiles("AGENTS.md");
    setPermissionMode("default");
    setAllowedTools([]);
    setDeniedTools([]);
    setUnsupportedSkill("block");
    setFormError(null);
  };
  const edit = (agent: ExtensionAgentSummary) => {
    const recipe = agent.harnessRecipe;
    if (!recipe) return;
    setEditingId(agent.id);
    setName(agent.name);
    setSoftwareId(recipe.softwareId);
    setModelOptionId(
      agent.modelId
        ? composerModelOptionId(recipe.softwareId, agent.modelId, agent.modelSupplyId)
        : "",
    );
    setInstructions(agent.instructions ?? "");
    setSkillModes(
      Object.fromEntries(
        recipe.skillBindings.map((binding) => [
          binding.skillId,
          { mode: binding.mode, variantId: binding.variantId },
        ]),
      ),
    );
    setSelectedMcps(new Set(recipe.mcpServerIds));
    setContextPaths(recipe.projectContext.paths.join("\n"));
    setInstructionFiles(recipe.projectContext.instructionFiles.join("\n"));
    setPermissionMode(recipe.permissions.mode);
    setAllowedTools(recipe.permissions.allowedTools);
    setDeniedTools(recipe.permissions.deniedTools);
    setUnsupportedSkill(recipe.delivery.unsupportedSkill);
    setFormError(null);
  };
  const selectedSoftware = softwareOptions.find((harness) => harness.id === softwareId);
  const softwareHealth = environment?.harnesses.find((harness) => harness.harnessId === softwareId);
  const selectedSkillRows = skills.filter((skill) => skillModes[skill.id]?.mode !== undefined);
  const contextTokenEstimate = selectedSkillRows.reduce((total, skill) => {
    const selectedVariant = skill.variants?.find(
      (variant) => variant.id === skillModes[skill.id]?.variantId,
    );
    return total + (selectedVariant?.tokenEstimate ?? skill.tokenEstimate ?? 0);
  }, 0);
  const unknownTokenCount = selectedSkillRows.filter((skill) => {
    const selectedVariant = skill.variants?.find(
      (variant) => variant.id === skillModes[skill.id]?.variantId,
    );
    return selectedVariant?.tokenEstimate === undefined && skill.tokenEstimate === undefined;
  }).length;

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedModelOption) {
      setFormError("Choose an available Model before saving this Agent.");
      return;
    }
    const id = editingId ?? slugId(name, "custom-agent");
    const harnessId = `${id}-harness`;
    setSaving(true);
    setFormError(null);
    try {
      await onSave({
        id,
        name: name.trim(),
        harnessId,
        harnessRecipe: {
          id: harnessId,
          revisionId: "draft",
          name: `${name.trim()} Harness`,
          softwareId,
          skillBindings: Object.entries(skillModes).map(([skillId, binding]) => ({
            skillId,
            mode: binding.mode,
            ...(binding.variantId ? { variantId: binding.variantId } : {}),
          })),
          mcpServerIds: [...selectedMcps],
          projectContext: {
            paths: lines(contextPaths),
            instructionFiles: lines(instructionFiles),
            includeWorkspaceRules: true,
          },
          delivery: {
            unsupportedSkill,
            requireContentDigest: true,
            allowHostNativePlugins: true,
          },
          permissions: {
            mode: permissionMode,
            allowedTools,
            deniedTools,
          },
        },
        modelId: selectedModelOption.modelId,
        ...(selectedModelOption.modelSupplyId
          ? { modelSupplyId: selectedModelOption.modelSupplyId }
          : {}),
        instructions: instructions.trim() || undefined,
        skills: Object.entries(skillModes)
          .filter(([, binding]) => binding.mode !== "off")
          .map(([skillId]) => skillId),
        mcpServers: [...selectedMcps],
        permissionMode,
        enabled: true,
        readOnly: false,
      });
      reset();
    } catch (error) {
      setFormError(errorMessage(error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section
      className="settings-workspace custom-agents-settings"
      aria-label="Custom Agents settings"
    >
      <div className="settings-workspace__body custom-agent-layout">
        <aside className="custom-agent-list" aria-label="Agent profiles">
          <div className="custom-agent-list__heading">
            <span>
              <small>Agent = Harness + Model</small>
              <strong>Custom Agents</strong>
            </span>
            <Button size="icon" onClick={reset} aria-label="New Custom Agent">
              <Plus aria-hidden="true" />
            </Button>
          </div>
          {customAgents.length === 0 && (
            <p className="custom-agent-list__empty">No custom agents yet.</p>
          )}
          {customAgents.map((agent) => (
            <button
              key={agent.id}
              type="button"
              className={editingId === agent.id ? "is-active" : undefined}
              onClick={() => edit(agent)}
            >
              <Bot aria-hidden="true" />
              <span>
                <strong>{agent.name}</strong>
                <small>
                  {agent.harnessRecipe?.softwareId} + {agent.modelId}
                </small>
              </span>
            </button>
          ))}
          {nativeAgents.length > 0 && (
            <div className="custom-agent-list__readonly">
              <small>Native definitions · read-only</small>
              {nativeAgents.map((agent) => (
                <span key={agent.id} title={agent.definition?.path}>
                  <Bot aria-hidden="true" />
                  {nativeAgentHostLabel(agent.definition?.host)} · {agent.name} ·{" "}
                  {agent.modelId ?? agent.nativeModel ?? "Model unresolved"}
                </span>
              ))}
            </div>
          )}
          {extensionAgents.length > 0 && (
            <div className="custom-agent-list__readonly">
              <small>Extension profiles · read-only</small>
              {extensionAgents.map((agent) => (
                <span key={agent.id}>
                  <Package aria-hidden="true" />
                  {agent.name}
                </span>
              ))}
            </div>
          )}
        </aside>

        <form className="custom-agent-editor" onSubmit={(event) => void submit(event)}>
          <div className="settings-content-heading">
            <span>
              <small>Reusable composition</small>
              <h2>{editingId ? `Edit ${name}` : "New Custom Agent"}</h2>
              <p>
                Build a versioned Harness from Software, Skills, MCPs, and policy; then choose its
                Model.
              </p>
            </span>
            <div>
              {editingId && (
                <button
                  type="button"
                  className="settings-secondary-action is-danger"
                  onClick={() =>
                    void onRemove(editingId)
                      .then(reset)
                      .catch((error) => setFormError(errorMessage(error)))
                  }
                >
                  <Trash2 aria-hidden="true" />
                  Delete
                </button>
              )}
              <button type="submit" className="settings-primary-action" disabled={saving}>
                {saving ? "Saving…" : "Save Agent"}
              </button>
            </div>
          </div>
          {formError && <div className="settings-provider-error">{formError}</div>}

          <div className="custom-agent-identity-grid">
            <label>
              <span>Name</span>
              <input
                required
                value={name}
                placeholder="Research agent"
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label>
              <span>Model</span>
              <select
                required
                value={modelOptionId}
                onChange={(event) => setModelOptionId(event.target.value)}
              >
                <option value="" disabled>
                  Select a Model
                </option>
                {unavailableEditingModel && (
                  <option value={unavailableEditingModel.id} disabled>
                    {unavailableEditingModel.label}
                  </option>
                )}
                {modelGroups.map((group) => (
                  <optgroup key={group.id} label={group.label}>
                    {group.subgroups.flatMap((subgroup) =>
                      subgroup.models.map((model) => (
                        <option key={model.id} value={model.id}>
                          {subgroup.label ? `${subgroup.label} · ${model.label}` : model.label}
                        </option>
                      )),
                    )}
                  </optgroup>
                ))}
              </select>
            </label>
          </div>

          <section className="harness-recipe-card">
            <div className="harness-recipe-card__heading">
              <span>
                <small>Harness recipe</small>
                <h3>Software + Skills + MCP + Context + Policy</h3>
              </span>
              <Badge tone={softwareHealth?.status === "ready" ? "active" : "danger"}>
                {softwareHealth?.status === "ready" ? "Software ready" : "Setup needed"}
              </Badge>
            </div>
            <label className="harness-software-picker">
              <span>Software</span>
              <select
                required
                value={softwareId}
                onChange={(event) => {
                  setSoftwareId(event.target.value);
                  setModelOptionId("");
                }}
              >
                <option value="" disabled>
                  Select harness software
                </option>
                {softwareOptions.map((software) => (
                  <option key={software.id} value={software.id}>
                    {software.label} · {formatSoftwareSummary(software.software)}
                  </option>
                ))}
              </select>
              <small>
                {softwareHealth?.note ??
                  selectedSoftware?.software?.version ??
                  "Version detected locally"}
              </small>
              {softwareHealth && softwareHealth.status !== "ready" && (
                <Button
                  type="button"
                  size="sm"
                  disabled={setupBusy}
                  onClick={() => {
                    setSetupBusy(true);
                    void onSetupSoftware(softwareId)
                      .catch((error) => setFormError(errorMessage(error)))
                      .finally(() => setSetupBusy(false));
                  }}
                >
                  <Hammer aria-hidden="true" />
                  {setupBusy ? "Setting up…" : "Set up software"}
                </Button>
              )}
            </label>

            <section className="harness-permission-section" aria-label="Agent permission policy">
              <div className="harness-permission-section__heading">
                <ShieldCheck aria-hidden="true" />
                <span>
                  <strong>Agent permission policy</strong>
                  <small>
                    This is one layer. Managed, Project, and personal restrictions can reduce it.
                  </small>
                </span>
              </div>
              <label className="harness-permission-mode">
                <span>Mode</span>
                <select
                  aria-label="Permission mode"
                  value={permissionMode}
                  onChange={(event) =>
                    setPermissionMode(event.target.value as HarnessPermissionMode)
                  }
                >
                  {PERMISSION_MODE_OPTIONS.map((option) => (
                    <option key={option.id} value={option.id}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <small>{permissionModeDescription(permissionMode)}</small>
              </label>
              <div className="permission-rule-grid">
                <PermissionToolRulesEditor
                  label="Pre-approved tools"
                  description="Exact tool names that skip a prompt for this Agent."
                  values={allowedTools}
                  blockedValues={deniedTools}
                  onChange={setAllowedTools}
                />
                <PermissionToolRulesEditor
                  label="Denied tools"
                  description="Exact tool names this Agent can never use."
                  values={deniedTools}
                  blockedValues={allowedTools}
                  tone="danger"
                  onChange={setDeniedTools}
                />
              </div>
            </section>

            <div className="harness-recipe-columns">
              <fieldset>
                <legend>Skills</legend>
                <p>Only add constraints this Agent/Model actually needs.</p>
                <div className="harness-capability-list">
                  {skills.map((skill) => {
                    const binding = skillModes[skill.id];
                    return (
                      <div key={skill.id} className="harness-capability-row">
                        <label>
                          <input
                            type="checkbox"
                            checked={binding !== undefined}
                            onChange={(event) =>
                              setSkillModes((current) => {
                                const next = { ...current };
                                if (event.target.checked) next[skill.id] = { mode: "auto" };
                                else delete next[skill.id];
                                return next;
                              })
                            }
                          />
                          <span>
                            <strong>{skill.name ?? skill.id}</strong>
                            <small>{skill.id}</small>
                          </span>
                        </label>
                        {binding && (
                          <>
                            <select
                              aria-label={`${skill.name ?? skill.id} binding mode`}
                              value={binding.mode}
                              onChange={(event) =>
                                setSkillModes((current) => ({
                                  ...current,
                                  [skill.id]: {
                                    ...current[skill.id],
                                    mode: event.target.value as "off" | "auto" | "required",
                                  },
                                }))
                              }
                            >
                              <option value="auto">Auto</option>
                              <option value="required">Required</option>
                              <option value="off">Off</option>
                            </select>
                            {(skill.variants?.length ?? 0) > 1 && (
                              <select
                                aria-label={`${skill.name ?? skill.id} variant`}
                                value={binding.variantId ?? ""}
                                onChange={(event) =>
                                  setSkillModes((current) => ({
                                    ...current,
                                    [skill.id]: {
                                      ...current[skill.id],
                                      variantId: event.target.value || undefined,
                                    },
                                  }))
                                }
                              >
                                <option value="">Resolve for Agent + Model</option>
                                {skill.variants?.map((variant) => (
                                  <option key={variant.id} value={variant.id}>
                                    {variant.id}
                                  </option>
                                ))}
                              </select>
                            )}
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="harness-context-cost">
                  <Gauge aria-hidden="true" />
                  <span>
                    <strong>~{contextTokenEstimate.toLocaleString()} tokens</strong>
                    <small>
                      {unknownTokenCount > 0
                        ? `${unknownTokenCount} unknown estimates`
                        : "Resolved Skill context"}
                    </small>
                  </span>
                </div>
              </fieldset>

              <fieldset>
                <legend>MCP servers</legend>
                <p>Attach only tools required by this Harness.</p>
                <div className="harness-capability-list">
                  {mcpServers.map((server) => (
                    <label key={server.id} className="harness-simple-capability">
                      <input
                        type="checkbox"
                        checked={selectedMcps.has(server.id)}
                        onChange={(event) =>
                          setSelectedMcps((current) => {
                            const next = new Set(current);
                            if (event.target.checked) next.add(server.id);
                            else next.delete(server.id);
                            return next;
                          })
                        }
                      />
                      <span>
                        <strong>{server.name ?? server.id}</strong>
                        <small>{server.scope ?? "MCP"}</small>
                      </span>
                    </label>
                  ))}
                </div>
              </fieldset>
            </div>

            <div className="harness-policy-grid">
              <label>
                <span>Project context paths</span>
                <textarea
                  value={contextPaths}
                  placeholder="docs/\nsrc/"
                  onChange={(event) => setContextPaths(event.target.value)}
                />
              </label>
              <label>
                <span>Instruction files</span>
                <textarea
                  value={instructionFiles}
                  placeholder="AGENTS.md"
                  onChange={(event) => setInstructionFiles(event.target.value)}
                />
              </label>
              <label>
                <span>Unsupported Skill delivery</span>
                <select
                  value={unsupportedSkill}
                  onChange={(event) => setUnsupportedSkill(event.target.value as "block" | "skip")}
                >
                  <option value="block">Block the Agent</option>
                  <option value="skip">Skip with warning</option>
                </select>
              </label>
            </div>
          </section>

          <label className="custom-agent-instructions">
            <span>Agent instructions</span>
            <textarea
              value={instructions}
              placeholder="Describe the role, priorities, and stopping conditions."
              onChange={(event) => setInstructions(event.target.value)}
            />
          </label>
        </form>
      </div>
    </section>
  );
}

export function SettingsWorkspace({
  providers,
  modelCatalog,
  modelCatalogRefreshing,
  modelCatalogError,
  providerUsage,
  providerUsageRefreshing,
  providerUsageRefreshingIds,
  providerUsageError,
  onRefreshModels,
  onRefreshUsage,
  onSaveProvider,
  onRemoveProvider,
  onResetProviderKey,
}: {
  providers: ExtensionProviderSummary[];
  modelCatalog?: ModelCatalogSummary;
  modelCatalogRefreshing: boolean;
  modelCatalogError: string | null;
  providerUsage: ProviderUsageSnapshot | null;
  providerUsageRefreshing: boolean;
  providerUsageRefreshingIds: ReadonlySet<string>;
  providerUsageError: string | null;
  onRefreshModels: () => Promise<void>;
  onRefreshUsage: (target?: ProviderUsageTarget) => Promise<void>;
  onSaveProvider: (input: UserProviderInput) => Promise<void>;
  onRemoveProvider: (providerId: string) => Promise<void>;
  onResetProviderKey: (providerId: string, keyId: string) => Promise<void>;
}) {
  const [providerFormOpen, setProviderFormOpen] = useState(false);
  const [editingProviderId, setEditingProviderId] = useState<string | null>(null);
  const [providerLabel, setProviderLabel] = useState("");
  const [providerKind, setProviderKind] = useState<ModelApiProtocol>("anthropic");
  const [providerBaseUrl, setProviderBaseUrl] = useState("https://api.anthropic.com");
  const [providerAuthMode, setProviderAuthMode] = useState<"api_key" | "auth_token">("api_key");
  const [providerUsageAdapter, setProviderUsageAdapter] = useState<"automatic" | "new_api">(
    "automatic",
  );
  const [providerSecret, setProviderSecret] = useState("");
  const [providerAccountAccessToken, setProviderAccountAccessToken] = useState("");
  const [providerAccountUserId, setProviderAccountUserId] = useState("");
  const [providerClearAccountAccess, setProviderClearAccountAccess] = useState(false);
  const [providerAdditionalApiKeys, setProviderAdditionalApiKeys] = useState("");
  const [providerRemovedApiKeyIds, setProviderRemovedApiKeyIds] = useState<string[]>([]);
  const [providerSaving, setProviderSaving] = useState(false);
  const [providerError, setProviderError] = useState<string | null>(null);
  const userProviderIds = new Set(modelCatalog?.userProviderIds ?? []);
  const providerUsageById = new Map(
    (providerUsage?.providers ?? []).map((entry) => [entry.providerProfileId, entry]),
  );
  const codexCatalogProvider = providers.find(
    (provider) => provider.catalogAdapter === "codex_app_server",
  );
  const managedProviders = providers.filter(
    (provider) => provider.catalogAdapter !== "codex_app_server",
  );
  const deepSeekForm = isDeepSeekProviderUrl(providerBaseUrl);
  const openCodeGoForm = isOpenCodeGoProviderUrl(providerBaseUrl);
  const editingProvider = providers.find((provider) => provider.id === editingProviderId);

  const resetProviderForm = () => {
    setProviderFormOpen(false);
    setEditingProviderId(null);
    setProviderLabel("");
    setProviderKind("anthropic");
    setProviderBaseUrl("https://api.anthropic.com");
    setProviderAuthMode("api_key");
    setProviderUsageAdapter("automatic");
    setProviderSecret("");
    setProviderAccountAccessToken("");
    setProviderAccountUserId("");
    setProviderClearAccountAccess(false);
    setProviderAdditionalApiKeys("");
    setProviderRemovedApiKeyIds([]);
    setProviderError(null);
  };
  const beginAddProvider = () => {
    resetProviderForm();
    setProviderFormOpen(true);
  };
  const beginEditProvider = (provider: ExtensionProviderSummary) => {
    setEditingProviderId(provider.id);
    setProviderLabel(provider.label);
    setProviderKind(
      ["anthropic", "openai_chat", "openai_responses", "ollama"].includes(provider.kind)
        ? (provider.kind as ModelApiProtocol)
        : "anthropic",
    );
    setProviderBaseUrl(provider.baseUrl ?? "");
    setProviderAuthMode(provider.authMode ?? "api_key");
    setProviderUsageAdapter(provider.usageAdapter === "new_api" ? "new_api" : "automatic");
    setProviderSecret("");
    setProviderAccountAccessToken("");
    setProviderAccountUserId(provider.newApiAccountUserId ?? "");
    setProviderClearAccountAccess(false);
    setProviderAdditionalApiKeys("");
    setProviderRemovedApiKeyIds([]);
    setProviderError(null);
    setProviderFormOpen(true);
  };
  const submitProvider = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setProviderSaving(true);
    setProviderError(null);
    try {
      await onSaveProvider({
        ...(editingProviderId ? { id: editingProviderId } : {}),
        label: providerLabel,
        kind: providerKind,
        baseUrl: providerBaseUrl,
        authMode: providerAuthMode,
        ...(!openCodeGoForm && providerUsageAdapter === "new_api"
          ? { usageAdapter: "new_api" as const }
          : {}),
        ...(providerSecret.trim() ? { secret: providerSecret } : {}),
        ...(!openCodeGoForm &&
        providerUsageAdapter === "new_api" &&
        !providerClearAccountAccess &&
        providerAccountAccessToken.trim()
          ? { accountAccessToken: providerAccountAccessToken }
          : {}),
        ...(!openCodeGoForm &&
        providerUsageAdapter === "new_api" &&
        !providerClearAccountAccess &&
        providerAccountUserId.trim()
          ? { accountUserId: providerAccountUserId.trim() }
          : {}),
        ...(!openCodeGoForm && providerUsageAdapter === "new_api" && providerClearAccountAccess
          ? { clearAccountAccess: true }
          : {}),
        ...(openCodeGoForm && providerAdditionalApiKeys.trim()
          ? {
              additionalApiKeys: providerAdditionalApiKeys
                .split(/\r?\n/)
                .map((value) => value.trim())
                .filter(Boolean)
                .map((value) => ({ value })),
            }
          : {}),
        ...(openCodeGoForm && providerRemovedApiKeyIds.length > 0
          ? { removeApiKeyIds: providerRemovedApiKeyIds }
          : {}),
      });
      resetProviderForm();
    } catch (error) {
      setProviderError(errorMessage(error));
    } finally {
      setProviderSaving(false);
    }
  };
  const removeUserProvider = async (providerId: string) => {
    setProviderError(null);
    try {
      await onRemoveProvider(providerId);
      if (editingProviderId === providerId) resetProviderForm();
    } catch (error) {
      setProviderError(errorMessage(error));
    }
  };

  return (
    <section className="settings-workspace" aria-label="Settings">
      <div className="settings-workspace__body">
        <div className="settings-workspace__content">
          <section className="settings-providers" aria-labelledby="settings-providers-title">
            <div className="settings-content-heading">
              <span>
                <small>Connections and usage</small>
                <h2 id="settings-providers-title">Providers</h2>
                <p>Model access, account limits, and balances in one place.</p>
              </span>
              <div>
                <button
                  type="button"
                  className="settings-secondary-action"
                  disabled={providerUsageRefreshing || providerUsageRefreshingIds.size > 0}
                  onClick={() => void onRefreshUsage()}
                >
                  <RefreshCw
                    className={providerUsageRefreshing ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  {providerUsageRefreshing ? "Refreshing…" : "Refresh all"}
                </button>
                <button
                  type="button"
                  className="settings-secondary-action"
                  disabled={modelCatalogRefreshing}
                  onClick={() => void onRefreshModels()}
                >
                  <RefreshCw
                    className={modelCatalogRefreshing ? "is-spinning" : undefined}
                    aria-hidden="true"
                  />
                  {modelCatalogRefreshing ? "Refreshing…" : "Refresh Models"}
                </button>
                <button
                  type="button"
                  className="settings-primary-action"
                  onClick={beginAddProvider}
                >
                  <Plus aria-hidden="true" />
                  Add Provider
                </button>
              </div>
            </div>
            {(providerError || modelCatalogError) && (
              <div className="settings-provider-error" role="alert">
                {providerError ?? modelCatalogError}
              </div>
            )}
            {providerUsageError && (
              <div className="settings-provider-error" role="alert">
                {providerUsageError}
              </div>
            )}
            {providerFormOpen && (
              <form
                className="settings-provider-form"
                aria-label={editingProviderId ? "Edit Provider" : "Add Provider"}
                onSubmit={(event) => void submitProvider(event)}
              >
                <div className="settings-provider-form__heading">
                  <span>
                    <small>{editingProviderId ? "Existing connection" : "New connection"}</small>
                    <strong>{editingProviderId ? "Edit Provider" : "Add Provider"}</strong>
                  </span>
                  <button
                    type="button"
                    onClick={resetProviderForm}
                    aria-label="Close Provider form"
                  >
                    <X aria-hidden="true" />
                  </button>
                </div>
                <div className="settings-provider-form__grid">
                  <label>
                    <span>Provider name</span>
                    <input
                      required
                      value={providerLabel}
                      placeholder="My Anthropic gateway"
                      onChange={(event) => setProviderLabel(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>
                      {deepSeekForm || openCodeGoForm ? "Preferred API protocol" : "API protocol"}
                    </span>
                    <select
                      aria-label={
                        deepSeekForm || openCodeGoForm ? "Preferred API protocol" : "API protocol"
                      }
                      value={providerKind}
                      onChange={(event) => setProviderKind(event.target.value as ModelApiProtocol)}
                    >
                      <option value="anthropic">Anthropic</option>
                      <option value="openai_responses">OpenAI Responses</option>
                      <option value="openai_chat">OpenAI Chat</option>
                      {!openCodeGoForm && <option value="ollama">Ollama</option>}
                    </select>
                    {deepSeekForm && (
                      <small className="settings-provider-form__helper">
                        DeepSeek supports native OpenAI and Anthropic APIs. This selects the
                        preferred route while keeping them in one Provider.
                      </small>
                    )}
                    {openCodeGoForm && (
                      <small className="settings-provider-form__helper">
                        The official Go endpoint routes each documented model through its native
                        Anthropic, OpenAI Chat, or OpenAI Responses API. Verified compatibility
                        exceptions may expose more than one route. Models are loaded from
                        /zen/go/v1/models.
                      </small>
                    )}
                  </label>
                  <label className="settings-provider-form__wide">
                    <span>Base URL</span>
                    <input
                      required
                      type="url"
                      value={providerBaseUrl}
                      placeholder="https://api.example.com"
                      onChange={(event) => {
                        const value = event.target.value;
                        setProviderBaseUrl(value);
                        if (
                          isOpenCodeGoProviderUrl(value) &&
                          !["anthropic", "openai_chat", "openai_responses"].includes(providerKind)
                        ) {
                          setProviderKind("openai_chat");
                        }
                      }}
                    />
                  </label>
                  <label>
                    <span>Authentication</span>
                    <select
                      value={providerAuthMode}
                      onChange={(event) =>
                        setProviderAuthMode(event.target.value as "api_key" | "auth_token")
                      }
                    >
                      <option value="api_key">API Key</option>
                      <option value="auth_token">Auth Token</option>
                    </select>
                  </label>
                  {openCodeGoForm ? (
                    <div className="settings-provider-form__section">
                      <strong>Local usage tracking</strong>
                      <small>
                        No usage endpoint is queried. Quota errors cool and rotate keys.
                      </small>
                    </div>
                  ) : (
                    <label>
                      <span>Usage API</span>
                      <select
                        value={providerUsageAdapter}
                        onChange={(event) =>
                          setProviderUsageAdapter(event.target.value as "automatic" | "new_api")
                        }
                      >
                        <option value="automatic">Automatic</option>
                        <option value="new_api">New API</option>
                      </select>
                    </label>
                  )}
                  <label className="settings-provider-form__wide">
                    <span>
                      {openCodeGoForm
                        ? "Primary API key"
                        : providerUsageAdapter === "new_api"
                          ? "Primary API token"
                          : providerAuthMode === "auth_token"
                            ? "Auth token"
                            : "API key"}
                    </span>
                    <input
                      aria-label={
                        openCodeGoForm
                          ? "Primary API key"
                          : providerUsageAdapter === "new_api"
                            ? "Primary API token"
                            : providerAuthMode === "auth_token"
                              ? "Auth token"
                              : "API key"
                      }
                      required={!editingProviderId}
                      type="password"
                      autoComplete="new-password"
                      value={providerSecret}
                      placeholder={
                        editingProviderId
                          ? "Leave blank to keep current"
                          : "Stored in ~/.swarmx/provider-auth.json"
                      }
                      onChange={(event) => setProviderSecret(event.target.value)}
                    />
                    {!openCodeGoForm && providerUsageAdapter === "new_api" && (
                      <small className="settings-provider-form__helper">
                        Used for Model requests and its own /api/usage/token quota.
                      </small>
                    )}
                  </label>
                  {openCodeGoForm && (
                    <>
                      <label className="settings-provider-form__wide">
                        <span>Additional API keys</span>
                        <textarea
                          aria-label="Additional API keys"
                          rows={4}
                          value={providerAdditionalApiKeys}
                          placeholder="One API key per line"
                          onChange={(event) => setProviderAdditionalApiKeys(event.target.value)}
                        />
                        <small className="settings-provider-form__helper">
                          Saved in the local Provider auth file. Edit that file directly when
                          needed.
                        </small>
                      </label>
                      {editingProvider?.runtimeKeyUsage &&
                        editingProvider.runtimeKeyUsage.length > 0 && (
                          <div className="settings-provider-form__wide settings-provider-key-editor">
                            <strong>Saved keys</strong>
                            {editingProvider.runtimeKeyUsage.map((key) => {
                              const removed = providerRemovedApiKeyIds.includes(key.id);
                              return (
                                <span key={key.id} className={cx(removed && "is-removed")}>
                                  <span>
                                    <strong>{key.label}</strong>
                                    <small>
                                      {removed
                                        ? "Will be removed"
                                        : `${capitalize(key.status)} · ${key.totalTokens.toLocaleString()} tokens`}
                                    </small>
                                  </span>
                                  {key.id !== "primary" && (
                                    <button
                                      type="button"
                                      onClick={() =>
                                        setProviderRemovedApiKeyIds((current) =>
                                          current.includes(key.id)
                                            ? current.filter((id) => id !== key.id)
                                            : [...current, key.id],
                                        )
                                      }
                                    >
                                      {removed ? "Keep" : "Remove"}
                                    </button>
                                  )}
                                </span>
                              );
                            })}
                          </div>
                        )}
                    </>
                  )}
                  {!openCodeGoForm && providerUsageAdapter === "new_api" && (
                    <>
                      <div className="settings-provider-form__section settings-provider-form__wide">
                        <strong>Account usage</strong>
                        <small>
                          Optional high-privilege management credential for /api/user/self and the
                          masked /api/token listing. It is never used for Model requests.
                        </small>
                      </div>
                      <label>
                        <span>New API user ID</span>
                        <input
                          inputMode="numeric"
                          value={providerAccountUserId}
                          placeholder="User ID"
                          disabled={providerClearAccountAccess}
                          onChange={(event) => setProviderAccountUserId(event.target.value)}
                        />
                      </label>
                      <label>
                        <span>Account access token</span>
                        <input
                          type="password"
                          autoComplete="new-password"
                          value={providerAccountAccessToken}
                          placeholder={
                            editingProviderId && !providerClearAccountAccess
                              ? "Leave blank to keep current"
                              : "Optional account credential"
                          }
                          disabled={providerClearAccountAccess}
                          onChange={(event) => setProviderAccountAccessToken(event.target.value)}
                        />
                      </label>
                      {editingProviderId &&
                        providers.find((provider) => provider.id === editingProviderId)
                          ?.accountAccessReady && (
                          <label className="settings-provider-form__checkbox settings-provider-form__wide">
                            <input
                              type="checkbox"
                              checked={providerClearAccountAccess}
                              onChange={(event) =>
                                setProviderClearAccountAccess(event.target.checked)
                              }
                            />
                            <span>Remove saved account access</span>
                          </label>
                        )}
                    </>
                  )}
                </div>
                <p className="settings-provider-form__security">
                  Credentials are stored as plaintext in ~/.swarmx/provider-auth.json with
                  restrictive file permissions. The Renderer never reads this file; the Main process
                  uses credentials only for the configured Provider operation.
                </p>
                <div className="settings-provider-form__actions">
                  <button type="button" onClick={resetProviderForm}>
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={
                      providerSaving ||
                      !providerLabel.trim() ||
                      !providerBaseUrl.trim() ||
                      (!editingProviderId && !providerSecret.trim())
                    }
                  >
                    {providerSaving ? "Saving…" : "Save Provider"}
                  </button>
                </div>
              </form>
            )}
            <section className="settings-provider-matrix" aria-label="Provider usage matrix">
              <div className="settings-provider-matrix__header">
                <span>Provider</span>
                <span>5-hour</span>
                <span>7-day</span>
                <span>Credit &amp; balance</span>
                <span>Resets</span>
                <span>Updated</span>
                <span>Actions</span>
              </div>
              {(providerUsage?.toolAccounts.length ? providerUsage.toolAccounts : [undefined]).map(
                (entry) => {
                  const sourceId = entry?.sourceId ?? "codex";
                  const label = entry?.label ?? "Codex";
                  const target = { source: "tool_account" as const, sourceId };
                  const catalogProvider = sourceId === "codex" ? codexCatalogProvider : undefined;
                  const discovery = catalogProvider
                    ? modelCatalog?.providers.find(
                        (status) => status.providerProfileId === catalogProvider.id,
                      )
                    : undefined;
                  return (
                    <ProviderMatrixRow
                      key={`tool-account:${sourceId}`}
                      label={label}
                      source="tool_account"
                      sourceId={sourceId}
                      provider={catalogProvider}
                      discovery={discovery}
                      entry={entry}
                      loading={
                        providerUsageRefreshing ||
                        providerUsageRefreshingIds.has(providerUsageTargetKey(target))
                      }
                      updatedFallback={providerUsage?.fetchedAt}
                      onRefresh={() => onRefreshUsage(target)}
                    />
                  );
                },
              )}
              {managedProviders.map((provider) => {
                const discovery = modelCatalog?.providers.find(
                  (status) => status.providerProfileId === provider.id,
                );
                const entry = providerUsageById.get(provider.id);
                const target = { source: "provider" as const, sourceId: provider.id };
                const userManaged = userProviderIds.has(provider.id) && provider.readOnly !== true;
                return (
                  <ProviderMatrixRow
                    key={`provider:${provider.id}`}
                    label={provider.label}
                    source="provider"
                    sourceId={provider.id}
                    provider={provider}
                    discovery={discovery}
                    entry={entry}
                    loading={
                      providerUsageRefreshing ||
                      providerUsageRefreshingIds.has(providerUsageTargetKey(target))
                    }
                    updatedFallback={providerUsage?.fetchedAt}
                    userManaged={userManaged}
                    saving={providerSaving}
                    onRefresh={() => onRefreshUsage(target)}
                    onEdit={userManaged ? () => beginEditProvider(provider) : undefined}
                    onRemove={userManaged ? () => void removeUserProvider(provider.id) : undefined}
                    onResetProviderKey={(keyId) => onResetProviderKey(provider.id, keyId)}
                  />
                );
              })}
            </section>
          </section>
        </div>
      </div>
    </section>
  );
}

function ProviderMatrixRow({
  label,
  source,
  sourceId,
  provider,
  discovery,
  entry,
  loading,
  updatedFallback,
  userManaged = false,
  saving = false,
  onRefresh,
  onEdit,
  onRemove,
  onResetProviderKey,
}: {
  label: string;
  source: ProviderUsageEntry["source"];
  sourceId: string;
  provider?: ExtensionProviderSummary;
  discovery?: ModelCatalogSummary["providers"][number];
  entry?: ProviderUsageEntry;
  loading: boolean;
  updatedFallback?: string;
  userManaged?: boolean;
  saving?: boolean;
  onRefresh: () => Promise<void>;
  onEdit?: () => void;
  onRemove?: () => void;
  onResetProviderKey?: (keyId: string) => Promise<void>;
}) {
  const fiveHour = findUsageWindow(entry, "five_hour");
  const weekly = findUsageWindow(entry, "weekly");
  const reset = entry?.meters.find(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "credit" }> =>
      meter.kind === "credit" && /reset/i.test(meter.label),
  );
  const updatedAt = entry?.fetchedAt ?? (entry ? updatedFallback : undefined);
  const status = providerUsageStatus(entry, loading);
  const deepSeek = provider ? isDeepSeekProvider(provider) : false;
  const openCodeGo = provider ? isOpenCodeGoProviderUrl(provider.baseUrl ?? "") : false;
  const modelCount =
    discovery?.status === "ready" || discovery?.status === "cached"
      ? `${discovery.modelCount} model${discovery.modelCount === 1 ? "" : "s"}`
      : provider?.runtimeReady === false
        ? "Needs attention"
        : "Configured";
  const providerMeta =
    source === "tool_account"
      ? `OpenAI official · Local account · ${
          entry?.plan ? `${capitalize(entry.plan)} plan` : "Codex app-server"
        } · ${modelCount}`
      : deepSeek && provider
        ? `${providerProtocolLabel(provider.kind)} + ${
            provider.kind === "anthropic" ? "OpenAI" : "Anthropic"
          } · Preferred ${providerProtocolLabel(provider.kind)} · ${modelCount}`
        : openCodeGo
          ? `OpenCode Go · ${entry?.keys?.length ?? provider?.runtimeKeySlots?.length ?? 0} keys · Local usage · ${modelCount}`
          : provider?.usageAdapter === "new_api"
            ? `New API · ${modelCount}`
            : `${providerProtocolLabel(provider?.kind ?? "Provider")} · ${modelCount}`;

  return (
    <article
      className={cx("settings-provider-matrix__row", loading && "is-loading")}
      aria-label={`${label} Provider`}
    >
      <div className="settings-provider-matrix__provider" data-label="Provider">
        <ProviderBrandIcon label={label} sourceId={sourceId} provider={provider} />
        <span className="settings-provider-matrix__identity">
          <span>
            <strong>{label}</strong>
            <small
              className={cx("settings-provider-status", `is-${status.tone}`)}
              title={entry?.detail}
            >
              {status.label}
            </small>
          </span>
          <small>{provider?.baseUrl ?? "Local official connection"}</small>
          <span>{providerMeta}</span>
        </span>
      </div>
      <ProviderWindowCell label="5-hour" meter={fiveHour} loading={loading && !entry} />
      <ProviderWindowCell label="7-day" meter={weekly} loading={loading && !entry} />
      <ProviderFinanceCell
        id={`${source}-${sourceId}`}
        label={label}
        entry={entry}
        loading={loading && !entry}
      />
      <div className="settings-provider-matrix__metric" data-label="Resets">
        {loading && !entry ? (
          <span className="settings-provider-matrix__skeleton" aria-label="Loading resets" />
        ) : reset ? (
          <>
            <strong>{reset.remaining}</strong>
            <small>{reset.unit}</small>
          </>
        ) : (
          <NotProvided />
        )}
      </div>
      <div className="settings-provider-matrix__updated" data-label="Updated">
        {loading && <Loader2 className="is-spinning" aria-hidden="true" />}
        <span>{updatedAt ? formatTimestamp(updatedAt) : "Not checked"}</span>
      </div>
      <div className="settings-provider-matrix__actions" data-label="Actions">
        <button
          type="button"
          aria-label={`Refresh ${label} usage`}
          title={`Refresh ${label} usage`}
          disabled={loading}
          onClick={() => void onRefresh()}
        >
          <RefreshCw className={loading ? "is-spinning" : undefined} aria-hidden="true" />
        </button>
        {userManaged && onEdit && (
          <button type="button" aria-label={`Edit Provider ${label}`} onClick={onEdit}>
            Edit
          </button>
        )}
        {userManaged && onRemove && (
          <button
            type="button"
            aria-label={`Remove Provider ${label}`}
            disabled={saving}
            onClick={onRemove}
          >
            <Trash2 aria-hidden="true" />
          </button>
        )}
      </div>
      {(provider?.usageAdapter === "new_api" || entry?.account) && (
        <NewApiAccountDetails provider={provider} entry={entry} onManage={onEdit} />
      )}
      {openCodeGo && (
        <ProviderKeyPoolDetails
          keys={entry?.keys ?? provider?.runtimeKeyUsage ?? []}
          onReset={onResetProviderKey}
        />
      )}
    </article>
  );
}

function ProviderWindowCell({
  label,
  meter,
  loading,
}: {
  label: "5-hour" | "7-day";
  meter?: Extract<ProviderUsageMeter, { kind: "window" }>;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="settings-provider-matrix__metric" data-label={label}>
        <span
          className="settings-provider-matrix__skeleton"
          aria-label={`Loading ${label} quota`}
        />
      </div>
    );
  }
  if (!meter) {
    return (
      <div className="settings-provider-matrix__metric" data-label={label}>
        <NotProvided />
      </div>
    );
  }
  const remaining = Math.max(0, Math.min(100, meter.remainingPercent));
  return (
    <div className="settings-provider-matrix__metric" data-label={label}>
      <strong>{formatUsagePercent(meter.remainingPercent)} left</strong>
      <span
        className="settings-provider-matrix__track"
        role="progressbar"
        aria-label={`${label} remaining`}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(remaining)}
        tabIndex={0}
      >
        <span style={{ width: `${remaining}%` }} />
      </span>
      <small>
        {meter.resetsAt ? `Resets ${formatTimestamp(meter.resetsAt)}` : "Reset not provided"}
      </small>
    </div>
  );
}

function ProviderFinanceCell({
  id,
  label,
  entry,
  loading,
}: {
  id: string;
  label: string;
  entry?: ProviderUsageEntry;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="settings-provider-matrix__metric" data-label="Credit & balance">
        <span
          className="settings-provider-matrix__skeleton"
          aria-label="Loading credit and balance"
        />
      </div>
    );
  }
  const finance = providerFinanceSummary(entry);
  if (!finance) {
    return (
      <div className="settings-provider-matrix__metric" data-label="Credit & balance">
        <NotProvided />
      </div>
    );
  }
  const tooltipId = `provider-finance-${id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  return (
    <div className="settings-provider-matrix__metric" data-label="Credit & balance">
      <button
        type="button"
        className="settings-provider-finance"
        aria-label={`${label} credit and balance: ${finance.primary}. Focus for breakdown.`}
        aria-describedby={tooltipId}
      >
        <strong>{finance.primary}</strong>
        <small>{finance.caption}</small>
        <span id={tooltipId} className="settings-provider-finance__popup" role="tooltip">
          <strong>Credit &amp; balance</strong>
          {finance.lines.map((line) => (
            <span key={line}>{line}</span>
          ))}
        </span>
      </button>
    </div>
  );
}

function NewApiAccountDetails({
  provider,
  entry,
  onManage,
}: {
  provider?: ExtensionProviderSummary;
  entry?: ProviderUsageEntry;
  onManage?: () => void;
}) {
  const account = entry?.account;
  const tokenCount = account?.totalTokens ?? 0;
  const accountLabel = account
    ? `${tokenCount} API token${tokenCount === 1 ? "" : "s"}`
    : provider?.accountAccessReady
      ? "Refresh to load account"
      : "Account access not configured";
  return (
    <details className="settings-provider-account">
      <summary>
        <span>Account &amp; API tokens</span>
        <small>{accountLabel}</small>
      </summary>
      <div className="settings-provider-account__content">
        <div className="settings-provider-account__summary">
          <span>
            <strong>{account?.displayName ?? "New API account"}</strong>
            <small>{account?.group ? `Group: ${account.group}` : "Account-level usage"}</small>
          </span>
          <small className={cx("settings-provider-status", `is-${account?.status ?? "muted"}`)}>
            {account ? capitalize(account.status) : "Not configured"}
          </small>
        </div>
        {account?.detail && <p>{account.detail}</p>}
        {!account && !provider?.accountAccessReady && (
          <div className="settings-provider-account__connect">
            <p>Connect account access to see wallet and API tokens.</p>
            {onManage && (
              <button type="button" onClick={onManage}>
                Manage account access
              </button>
            )}
          </div>
        )}
        {account &&
          (account.tokens.length ? (
            <section className="settings-provider-token-list" aria-label="New API tokens">
              <div className="settings-provider-token-list__header">
                <span>Token</span>
                <span>Status</span>
                <span>Remaining</span>
                <span>Used</span>
                <span>Expires</span>
              </div>
              {account.tokens.map((token) => (
                <div className="settings-provider-token-list__row" key={token.id}>
                  <span data-label="Token">
                    <strong>{token.name}</strong>
                    <small>{maskProviderTokenId(token.id)}</small>
                  </span>
                  <span data-label="Status" className={`is-${token.status}`}>
                    {capitalize(token.status)}
                  </span>
                  <span data-label="Remaining">{token.remaining}</span>
                  <span data-label="Used">{token.used ?? "—"}</span>
                  <span data-label="Expires">
                    {token.expiresAt ? formatTimestamp(token.expiresAt) : "No expiry"}
                  </span>
                </div>
              ))}
            </section>
          ) : (
            <p>No API token summaries returned for this account.</p>
          ))}
      </div>
    </details>
  );
}

function ProviderKeyPoolDetails({
  keys,
  onReset,
}: {
  keys: ProviderKeyUsageSummary[];
  onReset?: (keyId: string) => Promise<void>;
}) {
  const [resettingKeyId, setResettingKeyId] = useState<string | null>(null);
  const ready = keys.filter((key) => key.status === "ready").length;
  return (
    <details className="settings-provider-account settings-provider-key-pool">
      <summary>
        <span>API key pool</span>
        <small>
          {ready}/{keys.length} ready · local counters
        </small>
      </summary>
      <div className="settings-provider-account__content">
        <p>
          Keys rotate only after an explicit quota-exhausted response and only before any output or
          tool event has been emitted.
        </p>
        <section className="settings-provider-key-list" aria-label="OpenCode Go API keys">
          {keys.map((key) => (
            <div className="settings-provider-key-list__row" key={key.id}>
              <span>
                <strong>{key.label}</strong>
                <small>{key.id === "primary" ? "Primary" : "Additional key"}</small>
              </span>
              <span className={`is-${key.status}`}>{capitalize(key.status)}</span>
              <span>
                <strong>{key.totalTokens.toLocaleString()}</strong>
                <small>{key.requestCount.toLocaleString()} requests</small>
              </span>
              <span>
                {key.cooldownUntil
                  ? `Retry ${formatTimestamp(key.cooldownUntil)}`
                  : key.lastUsedAt
                    ? `Used ${formatTimestamp(key.lastUsedAt)}`
                    : "Not used yet"}
              </span>
              {key.status === "cooling" && onReset ? (
                <button
                  type="button"
                  disabled={resettingKeyId !== null}
                  onClick={() => {
                    setResettingKeyId(key.id);
                    void onReset(key.id)
                      .catch(() => undefined)
                      .finally(() => setResettingKeyId(null));
                  }}
                >
                  {resettingKeyId === key.id ? "Resetting…" : "Reset cooldown"}
                </button>
              ) : (
                <span />
              )}
            </div>
          ))}
        </section>
      </div>
    </details>
  );
}

function NotProvided() {
  return (
    <span className="settings-provider-matrix__missing">
      <strong>—</strong>
      <small>Not provided</small>
    </span>
  );
}

export function providerUsageTargetKey(target: ProviderUsageTarget): string {
  return `${target.source}:${target.sourceId}`;
}

export function mergeProviderUsageSnapshot(
  current: ProviderUsageSnapshot | null,
  next: ProviderUsageSnapshot,
): ProviderUsageSnapshot {
  if (!current) return next;
  return {
    fetchedAt: next.fetchedAt,
    providers: mergeProviderUsageEntries(current.providers, next.providers),
    toolAccounts: mergeProviderUsageEntries(current.toolAccounts, next.toolAccounts),
  };
}

function mergeProviderUsageEntries(
  current: ProviderUsageEntry[],
  next: ProviderUsageEntry[],
): ProviderUsageEntry[] {
  const merged = new Map(current.map((entry) => [entry.sourceId, entry]));
  for (const entry of next) merged.set(entry.sourceId, entry);
  return [...merged.values()];
}

function findUsageWindow(
  entry: ProviderUsageEntry | undefined,
  slot: "five_hour" | "weekly",
): Extract<ProviderUsageMeter, { kind: "window" }> | undefined {
  return entry?.meters.find((meter): meter is Extract<ProviderUsageMeter, { kind: "window" }> => {
    if (meter.kind !== "window") return false;
    const identity = `${meter.id} ${meter.label}`.toLowerCase();
    return slot === "five_hour"
      ? /five[_ -]?hour|5[_ -]?hour/.test(identity)
      : /weekly|7[_ -]?day/.test(identity);
  });
}

function providerUsageStatus(
  entry: ProviderUsageEntry | undefined,
  loading: boolean,
): { label: string; tone: "ready" | "loading" | "muted" | "warning" | "error" } {
  if (loading) return { label: entry ? "Updating" : "Checking", tone: "loading" };
  if (!entry) return { label: "Not checked", tone: "muted" };
  if (entry.status === "ready") {
    return entry.meters.length > 0 || entry.account
      ? { label: "Ready", tone: "ready" }
      : { label: "No data", tone: "muted" };
  }
  if (entry.status === "unsupported") return { label: "Not supported", tone: "muted" };
  if (entry.status === "unavailable") return { label: "Unavailable", tone: "warning" };
  return { label: "Query failed", tone: "error" };
}

function providerFinanceSummary(
  entry: ProviderUsageEntry | undefined,
): { primary: string; caption: string; lines: string[] } | undefined {
  const accountBalance = entry?.account?.balance;
  const balances = (entry?.meters ?? []).filter(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "balance" }> => meter.kind === "balance",
  );
  const credits = (entry?.meters ?? []).filter(
    (meter): meter is Extract<ProviderUsageMeter, { kind: "credit" }> =>
      meter.kind === "credit" && !/reset/i.test(meter.label),
  );
  if (!accountBalance && balances.length === 0 && credits.length === 0) return undefined;

  const lines: string[] = [];
  if (accountBalance) {
    lines.push(
      `Account available: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.remaining)}`,
      `Account used: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.used)}`,
      `Account total: ${formatProviderUnitAmount(accountBalance.unit, accountBalance.total)}`,
    );
  }
  for (const balance of balances) {
    if (!(accountBalance && balance.label.toLowerCase() === "account balance")) {
      lines.push(`${balance.label}: ${formatProviderAmount(balance.currency, balance.total)}`);
    }
    if (balance.granted !== undefined) {
      lines.push(`Granted: ${formatProviderAmount(balance.currency, balance.granted)}`);
    }
    if (balance.toppedUp !== undefined) {
      lines.push(`Paid: ${formatProviderAmount(balance.currency, balance.toppedUp)}`);
    }
  }
  for (const credit of credits) {
    lines.push(`${credit.label}: ${credit.remaining} ${credit.unit}`);
  }

  if (accountBalance) {
    return {
      primary: formatProviderUnitAmount(accountBalance.unit, accountBalance.remaining),
      caption: "Account available",
      lines,
    };
  }
  const balance = balances[0];
  if (balance) {
    return {
      primary: formatProviderAmount(balance.currency, balance.total),
      caption: balances.length > 1 ? `${balances.length} balances` : balance.label,
      lines,
    };
  }
  const credit = credits[0];
  if (!credit) return undefined;
  return {
    primary: credit.remaining,
    caption: credit.label,
    lines,
  };
}

function formatProviderUnitAmount(unit: string, value: string): string {
  return /^[A-Z]{3}$/.test(unit) ? formatProviderAmount(unit, value) : `${value} ${unit}`;
}

function maskProviderTokenId(value: string): string {
  return value.length > 10 ? `${value.slice(0, 4)}…${value.slice(-4)}` : value;
}

function formatProviderAmount(currency: string, value: string): string {
  const amount = Number(value);
  if (!Number.isFinite(amount)) return `${currency} ${value}`;
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency,
      maximumFractionDigits: 6,
    }).format(amount);
  } catch {
    return `${currency} ${value}`;
  }
}

function formatUsagePercent(value: number): string {
  return `${Math.max(0, Math.min(200, value)).toFixed(value % 1 === 0 ? 0 : 1)}%`;
}
