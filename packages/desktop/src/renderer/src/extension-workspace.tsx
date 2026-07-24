import { Package, Plus, RefreshCw, XCircle } from "lucide-react";
import { type FormEvent, useMemo, useState } from "react";
import type {
  ExtensionCapabilityInventory,
  ExtensionManagementState,
} from "../../shared/desktop-api.js";
import {
  agentPlanChips,
  agentPlanTone,
  capabilityCount,
  extensionComponentRows,
  extensionSkillChips,
  extensionUiContributionChips,
  formatComponentCounts,
  formatSoftwareSummary,
  modelApiLabel,
  planBlockedTitle,
  uniqueById,
} from "./extension-presentation.js";
import { HarnessBrandIcon, harnessOption } from "./harness-presentation.js";
import { capitalize, errorMessage, slugId } from "./text-utils.js";
import { Badge, Button } from "./ui-primitives.js";

interface ExtensionMarketplaceSourceSummary {
  id: string;
  name: string;
  host?: string;
  kind?: string;
  path?: string;
  url?: string;
  package?: string;
  enabled?: boolean;
  readOnly?: boolean;
  trust?: string;
}

interface ExtensionPluginCatalogEntrySummary {
  id: string;
  name: string;
  version?: string;
  marketplaceSourceId?: string;
  bundleId?: string;
  hosts?: string[];
  trust?: string;
  installState?: string;
  updateState?: string;
  providesHarness?: boolean;
  componentCounts?: Record<string, number>;
  readOnly?: boolean;
  description?: string;
}

export function ExtensionWorkspace({
  inventory,
  management,
  loading,
  error,
  selectedAgentId,
  onSelectAgent,
  onSaveSource,
  onRefreshSource,
  onRemoveSource,
  onApplyAction,
  onSaveEvolutionPolicy,
}: {
  inventory?: ExtensionCapabilityInventory;
  management?: ExtensionManagementState;
  loading: boolean;
  error: unknown;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  onSaveSource: (input: unknown) => Promise<void>;
  onRefreshSource: (id: string) => Promise<void>;
  onRemoveSource: (id: string) => Promise<void>;
  onApplyAction: (input: unknown) => Promise<{
    status: "applied" | "rejected" | "failed";
    message: string;
  }>;
  onSaveEvolutionPolicy: (input: {
    enabled: boolean;
    promotionGate: "human" | "policy";
  }) => Promise<void>;
}) {
  const [sourceFormOpen, setSourceFormOpen] = useState(false);
  const [sourceName, setSourceName] = useState("");
  const [sourceLocation, setSourceLocation] = useState("");
  const [sourceKind, setSourceKind] = useState<"remote_catalog" | "registry" | "local_path">(
    "remote_catalog",
  );
  const [managementBusy, setManagementBusy] = useState<string | null>(null);
  const [managementError, setManagementError] = useState<string | null>(null);
  const bundles = inventory?.bundles ?? [];
  const harnesses = inventory?.harnesses ?? [];
  const agents = inventory?.agents ?? [];
  const models = inventory?.models ?? [];
  const modelSupplies = inventory?.modelSupplies ?? [];
  const providers = inventory?.providers ?? [];
  const skills = inventory?.skills ?? [];
  const mcpServers = inventory?.mcpServers ?? [];
  const appConnectors = inventory?.appConnectors ?? [];
  const uiContributions = inventory?.uiContributions ?? [];
  const agentPlans = inventory?.agentPlans ?? [];
  const pluginComponents = extensionComponentRows(inventory);
  const marketplaceSources: ExtensionMarketplaceSourceSummary[] = [
    ...(management?.sources.map((source) => ({
      ...source,
      host: "swarmx",
      ...(source.kind === "local_path" ? { path: source.location } : { url: source.location }),
    })) ?? []),
    ...(inventory?.marketplaceSources ?? []).filter(
      (source) => !management?.sources.some((managed) => managed.id === source.id),
    ),
  ];
  const candidateById = new Map(
    (management?.candidates ?? []).map((candidate) => [candidate.pluginId, candidate]),
  );
  const pluginCatalog = uniqueById<ExtensionPluginCatalogEntrySummary>([
    ...(management?.candidates ?? []).map<ExtensionPluginCatalogEntrySummary>((candidate) => ({
      id: candidate.pluginId,
      name: candidate.name,
      version: candidate.revision.version,
      marketplaceSourceId: candidate.revision.sourceId,
      trust: candidate.trust,
      installState: "available",
      updateState: "unknown",
      description: candidate.description,
    })),
    ...(inventory?.pluginCatalog ?? []),
  ]);
  const warnings = inventory?.warnings ?? [];
  const installedById = new Map(
    (management?.installed ?? []).map((plugin) => [plugin.pluginId, plugin]),
  );
  const planByAgentId = useMemo(
    () => new Map(agentPlans.map((plan) => [plan.agentProfileId ?? plan.agentId, plan] as const)),
    [agentPlans],
  );

  const runManagement = async (id: string, action: () => Promise<void>) => {
    setManagementBusy(id);
    setManagementError(null);
    try {
      await action();
    } catch (error) {
      setManagementError(errorMessage(error));
    } finally {
      setManagementBusy(null);
    }
  };
  const submitSource = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const id = slugId(sourceName || sourceLocation, "source");
    await runManagement(`source:${id}`, async () => {
      await onSaveSource({
        id,
        name: sourceName.trim(),
        kind: sourceKind,
        location: sourceLocation.trim(),
        trust: sourceKind === "local_path" ? "local" : "untrusted",
      });
      setSourceFormOpen(false);
      setSourceName("");
      setSourceLocation("");
    });
  };
  const applyCatalogAction = async (
    entry: ExtensionPluginCatalogEntrySummary,
    action: "install" | "update" | "enable" | "disable" | "rollback" | "uninstall",
  ) => {
    if (
      ["install", "update", "rollback", "uninstall"].includes(action) &&
      !window.confirm(`${capitalize(action)} ${entry.name}?`)
    ) {
      return;
    }
    await runManagement(`plugin:${entry.id}`, async () => {
      const version = entry.version ?? "0.0.0";
      const managedCandidate = candidateById.get(entry.id);
      const result = await onApplyAction({
        action,
        pluginId: entry.id,
        confirmed: true,
        ...(["install", "update"].includes(action)
          ? {
              candidate: managedCandidate ?? {
                pluginId: entry.id,
                name: entry.name,
                trust: entry.trust ?? "untrusted",
                revision: {
                  revisionId: `${entry.id}@${version}`,
                  version,
                  contentDigest: `catalog:${entry.id}@${version}`,
                  sourceId: entry.marketplaceSourceId ?? "local",
                },
              },
            }
          : {}),
      });
      if (result.status !== "applied") throw new Error(result.message);
    });
  };

  return (
    <section className="extension-workspace" aria-label="Extension inventory">
      <div className="extension-topbar">
        <div className="extension-title">
          <Package aria-hidden="true" />
          <div>
            <h2>Extensions</h2>
            <span>
              {loading
                ? "Loading inventory"
                : `${bundles.length} bundles / ${marketplaceSources.length} sources / ${harnesses.length} harnesses`}
            </span>
          </div>
        </div>
        <div className="extension-stats" aria-label="Extension counts">
          <Badge tone="neutral">{pluginCatalog.length} plugins</Badge>
          <Badge tone="neutral">{agents.length} agents</Badge>
          <Badge tone="neutral">{skills.length} skills</Badge>
          <Badge tone="neutral">{mcpServers.length} MCPs</Badge>
          <Badge tone="neutral">{uiContributions.length} UI</Badge>
          <Badge tone="neutral">{pluginComponents.length} components</Badge>
          {warnings.length > 0 && <Badge tone="danger">{warnings.length} warnings</Badge>}
          <button
            type="button"
            className="settings-primary-action"
            onClick={() => setSourceFormOpen((open) => !open)}
          >
            <Plus aria-hidden="true" />
            Add source
          </button>
        </div>
      </div>

      {Boolean(managementError || error) && (
        <div className="settings-provider-error" role="alert">
          {managementError ?? errorMessage(error)}
        </div>
      )}

      {sourceFormOpen && (
        <form className="extension-source-form" onSubmit={(event) => void submitSource(event)}>
          <label>
            <span>Source name</span>
            <input
              required
              value={sourceName}
              placeholder="Official marketplace"
              onChange={(event) => setSourceName(event.target.value)}
            />
          </label>
          <label>
            <span>Source type</span>
            <select
              value={sourceKind}
              onChange={(event) =>
                setSourceKind(event.target.value as "remote_catalog" | "registry" | "local_path")
              }
            >
              <option value="remote_catalog">Remote catalog</option>
              <option value="registry">Registry</option>
              <option value="local_path">Local path</option>
            </select>
          </label>
          <label className="is-wide">
            <span>{sourceKind === "local_path" ? "Path" : "HTTPS URL"}</span>
            <input
              required
              type={sourceKind === "local_path" ? "text" : "url"}
              value={sourceLocation}
              placeholder={
                sourceKind === "local_path"
                  ? "/Users/me/extensions"
                  : "https://plugins.example.com/catalog.json"
              }
              onChange={(event) => setSourceLocation(event.target.value)}
            />
          </label>
          <div className="extension-source-form__actions">
            <Button type="button" variant="ghost" onClick={() => setSourceFormOpen(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={managementBusy !== null}>
              Save source
            </Button>
          </div>
        </form>
      )}

      <div className="extension-layout">
        <section className="extension-section" aria-label="Plugin bundles">
          <div className="extension-section__header">
            <h3>Plugin bundles</h3>
            <span>{bundles.length}</span>
          </div>
          {bundles.length === 0 ? (
            <div className="extension-empty">No bundles</div>
          ) : (
            <ul className="extension-list">
              {bundles.map((bundle) => (
                <li key={bundle.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{bundle.name}</strong>
                    <span>{bundle.id}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{bundle.version}</span>
                    <span>{bundle.trust ?? "local"}</span>
                    {bundle.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    <span>{capabilityCount(bundle, "harnesses")} harnesses</span>
                    <span>{capabilityCount(bundle, "agents")} agents</span>
                    <span>{capabilityCount(bundle, "skills")} skills</span>
                    <span>{capabilityCount(bundle, "mcpServers")} MCPs</span>
                    <span>{capabilityCount(bundle, "commands")} commands</span>
                    <span>{capabilityCount(bundle, "lspServers")} LSPs</span>
                    <span>{capabilityCount(bundle, "hooks")} hooks</span>
                    <span>{capabilityCount(bundle, "uiContributions")} UI</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Marketplace sources">
          <div className="extension-section__header">
            <h3>Marketplace sources</h3>
            <span>{marketplaceSources.length}</span>
          </div>
          {marketplaceSources.length === 0 ? (
            <div className="extension-empty">No marketplace sources</div>
          ) : (
            <ul className="extension-list">
              {marketplaceSources.map((source) => (
                <li key={source.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{source.name}</strong>
                    <span>{source.id}</span>
                    {management?.sources.some((managed) => managed.id === source.id) && (
                      <span className="extension-item__actions">
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={managementBusy === `source:${source.id}`}
                          onClick={() =>
                            void runManagement(`source:${source.id}`, () =>
                              onRefreshSource(source.id),
                            )
                          }
                        >
                          <RefreshCw aria-hidden="true" />
                          Refresh
                        </Button>
                        {!source.readOnly && (
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={managementBusy === `source:${source.id}`}
                            onClick={() =>
                              void runManagement(`source:${source.id}`, () =>
                                onRemoveSource(source.id),
                              )
                            }
                          >
                            Remove
                          </Button>
                        )}
                      </span>
                    )}
                  </div>
                  <div className="extension-item__meta">
                    <span>{source.host ?? "custom"}</span>
                    <span>{source.kind ?? "local_path"}</span>
                    <span>{source.trust ?? "local"}</span>
                    {source.enabled === false && <span>disabled</span>}
                    {source.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    <span>{source.path ?? source.url ?? source.package ?? "host-native"}</span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Plugin catalog">
          <div className="extension-section__header">
            <h3>Plugin catalog</h3>
            <span>{pluginCatalog.length}</span>
          </div>
          {pluginCatalog.length === 0 ? (
            <div className="extension-empty">No plugin catalog entries</div>
          ) : (
            <ul className="extension-list">
              {pluginCatalog.map((entry) => {
                const installed = installedById.get(entry.id);
                const updateAvailable = Boolean(
                  installed &&
                    entry.version &&
                    installed.currentRevision?.version !== entry.version,
                );
                return (
                  <li key={entry.id} className="extension-item">
                    <div className="extension-item__main">
                      <strong>{entry.name}</strong>
                      <span>{entry.id}</span>
                      <span className="extension-item__actions">
                        {!installed ? (
                          <Button
                            size="sm"
                            disabled={managementBusy === `plugin:${entry.id}`}
                            onClick={() => void applyCatalogAction(entry, "install")}
                          >
                            Install
                          </Button>
                        ) : (
                          <>
                            <Button
                              size="sm"
                              variant="secondary"
                              disabled={managementBusy === `plugin:${entry.id}`}
                              onClick={() =>
                                void applyCatalogAction(
                                  entry,
                                  installed.enabled ? "disable" : "enable",
                                )
                              }
                            >
                              {installed.enabled ? "Disable" : "Enable"}
                            </Button>
                            {updateAvailable && (
                              <Button
                                size="sm"
                                disabled={managementBusy === `plugin:${entry.id}`}
                                onClick={() => void applyCatalogAction(entry, "update")}
                              >
                                Update
                              </Button>
                            )}
                            {installed.previousRevisions.length > 0 && (
                              <Button
                                size="sm"
                                variant="ghost"
                                disabled={managementBusy === `plugin:${entry.id}`}
                                onClick={() => void applyCatalogAction(entry, "rollback")}
                              >
                                Roll back
                              </Button>
                            )}
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={managementBusy === `plugin:${entry.id}`}
                              onClick={() => void applyCatalogAction(entry, "uninstall")}
                            >
                              Uninstall
                            </Button>
                          </>
                        )}
                      </span>
                    </div>
                    <div className="extension-item__meta">
                      {entry.version && <span>{entry.version}</span>}
                      <span>{installed?.state ?? entry.installState ?? "available"}</span>
                      <span>{entry.updateState ?? "unknown"}</span>
                      <span>{entry.trust ?? "local"}</span>
                      {entry.providesHarness && <span>runnable harness</span>}
                      {entry.readOnly && <span>read-only</span>}
                    </div>
                    <div className="extension-item__chips">
                      {(entry.hosts ?? []).map((host) => (
                        <span key={`${entry.id}:${host}`}>{host}</span>
                      ))}
                      {entry.marketplaceSourceId && <span>{entry.marketplaceSourceId}</span>}
                      {entry.bundleId && <span>{entry.bundleId}</span>}
                      {formatComponentCounts(entry.componentCounts).map((item) => (
                        <span key={`${entry.id}:${item}`}>{item}</span>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section
          className="extension-section extension-section--evolution"
          aria-label="Skill evolution"
        >
          <div className="extension-section__header">
            <h3>Skill evolution</h3>
            <Badge tone={management?.skillEvolutionEnabled ? "active" : "neutral"}>
              {management?.skillEvolutionEnabled ? "Enabled" : "Off"}
            </Badge>
          </div>
          <p className="extension-section__description">
            Generate agent/model-specific candidate variants, evaluate them against the active
            baseline, and keep promotion gated with immutable lineage and rollback.
          </p>
          <div className="extension-evolution-controls">
            <label>
              <input
                type="checkbox"
                checked={management?.skillEvolutionEnabled ?? false}
                onChange={(event) =>
                  void runManagement("evolution", () =>
                    onSaveEvolutionPolicy({
                      enabled: event.target.checked,
                      promotionGate: management?.skillPromotionGate ?? "human",
                    }),
                  )
                }
              />
              <span>Allow candidate generation</span>
            </label>
            <label>
              <span>Promotion gate</span>
              <select
                value={management?.skillPromotionGate ?? "human"}
                onChange={(event) =>
                  void runManagement("evolution", () =>
                    onSaveEvolutionPolicy({
                      enabled: management?.skillEvolutionEnabled ?? false,
                      promotionGate: event.target.value as "human" | "policy",
                    }),
                  )
                }
              >
                <option value="human">Human approval</option>
                <option value="policy">Evaluation policy</option>
              </select>
            </label>
          </div>
        </section>

        <section className="extension-section" aria-label="Plugin components">
          <div className="extension-section__header">
            <h3>Plugin components</h3>
            <span>{pluginComponents.length}</span>
          </div>
          {pluginComponents.length === 0 ? (
            <div className="extension-empty">No plugin components</div>
          ) : (
            <ul className="extension-list">
              {pluginComponents.map((component) => (
                <li key={`${component.kind}:${component.id}`} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{component.title}</strong>
                    <span>{component.kind}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{component.id}</span>
                    {component.detail && <span>{component.detail}</span>}
                  </div>
                  <div className="extension-item__chips">
                    {component.chips.map((chip) => (
                      <span key={`${component.kind}:${component.id}:${chip}`}>{chip}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="GUI contributions">
          <div className="extension-section__header">
            <h3>GUI contributions</h3>
            <span>{uiContributions.length}</span>
          </div>
          {uiContributions.length === 0 ? (
            <div className="extension-empty">No GUI contributions</div>
          ) : (
            <ul className="extension-list">
              {uiContributions.map((contribution) => (
                <li key={contribution.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{contribution.name}</strong>
                    <span>{contribution.kind}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{contribution.id}</span>
                    <span>{contribution.placement}</span>
                    {contribution.route && <span>{contribution.route}</span>}
                    {contribution.componentRef && <span>{contribution.componentRef}</span>}
                    {contribution.readOnly && <span>read-only</span>}
                  </div>
                  <div className="extension-item__chips">
                    {extensionUiContributionChips(contribution).map((chip) => (
                      <span key={`${contribution.id}:${chip}`}>{chip}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Harnesses">
          <div className="extension-section__header">
            <h3>Harnesses</h3>
            <span>{harnesses.length}</span>
          </div>
          <ul className="extension-list">
            {harnesses.map((harness) => (
              <li key={harness.id} className="extension-item">
                <div className="extension-item__main">
                  <strong>{harness.label}</strong>
                  <span>{harness.id}</span>
                </div>
                <div className="extension-item__meta">
                  <span>{formatSoftwareSummary(harness.software)}</span>
                  {harness.readOnly && <span>read-only</span>}
                </div>
                <div className="extension-item__chips">
                  <span>{harness.modelControl}</span>
                  {(harness.supportedModelApis ?? []).map((api) => (
                    <span key={`${harness.id}:${api}`}>{modelApiLabel(api)}</span>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="extension-section" aria-label="Agent profiles">
          <div className="extension-section__header">
            <h3>Agent profiles</h3>
            <span>{agents.length}</span>
          </div>
          {agents.length === 0 ? (
            <div className="extension-empty">No agent profiles</div>
          ) : (
            <ul className="extension-list">
              {agents.map((agent) => {
                const plan = planByAgentId.get(agent.id);
                const canUseAgent = !plan || plan.status === "ready";
                return (
                  <li key={agent.id} className="extension-item">
                    <div className="extension-item__main">
                      <strong>{agent.name}</strong>
                      <span>{agent.id}</span>
                      {plan && <Badge tone={agentPlanTone(plan)}>{plan.status}</Badge>}
                      <Button
                        size="sm"
                        variant={selectedAgentId === agent.id ? "secondary" : "default"}
                        onClick={() => onSelectAgent(agent.id)}
                        aria-label={`Use agent profile ${agent.name}`}
                        disabled={!canUseAgent}
                        title={
                          canUseAgent ? `Use agent profile ${agent.name}` : planBlockedTitle(plan)
                        }
                      >
                        {selectedAgentId === agent.id ? "Selected" : "Use"}
                      </Button>
                    </div>
                    <div className="extension-item__meta">
                      <span>{plan?.canonicalSelector ?? agent.selector ?? agent.id}</span>
                      <span>{plan?.harnessLabel ?? agent.harnessId ?? "no harness"}</span>
                      <span>{plan?.modelId ?? agent.modelId ?? "no model"}</span>
                      {(plan?.modelSupplyId ?? agent.modelSupplyId) && (
                        <span>supply {plan?.modelSupplyId ?? agent.modelSupplyId}</span>
                      )}
                      {agent.permissionMode && <span>permission {agent.permissionMode}</span>}
                      {agent.memory && <span>memory {agent.memory}</span>}
                    </div>
                    <div className="extension-item__chips">
                      {agentPlanChips(plan, agent).map((chip) => (
                        <span key={`${agent.id}:${chip}`}>{chip}</span>
                      ))}
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Models and supplies">
          <div className="extension-section__header">
            <h3>Models and supplies</h3>
            <span>{models.length + modelSupplies.length}</span>
          </div>
          <ul className="extension-list extension-list--compact">
            {models.map((model) => (
              <li key={`model:${model.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{model.label ?? model.id}</strong>
                  <span>{model.id}</span>
                </div>
                <div className="extension-item__meta">
                  <span>{model.runtimeModel}</span>
                  {model.apiProtocols.map((api) => (
                    <span key={`${model.id}:${api}`}>{modelApiLabel(api)}</span>
                  ))}
                </div>
              </li>
            ))}
            {modelSupplies.map((supply) => (
              <li key={`supply:${supply.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{supply.id}</strong>
                  <span>
                    {supply.modelId} → {supply.providerProfileId}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        <section className="extension-section" aria-label="Providers">
          <div className="extension-section__header">
            <h3>Providers</h3>
            <span>{providers.length}</span>
          </div>
          {providers.length === 0 ? (
            <div className="extension-empty">No provider profiles</div>
          ) : (
            <ul className="extension-list">
              {providers.map((provider) => (
                <li key={provider.id} className="extension-item">
                  <div className="extension-item__main">
                    <strong>{provider.label}</strong>
                    <span>{provider.id}</span>
                  </div>
                  <div className="extension-item__meta">
                    <span>{provider.kind}</span>
                    {provider.runtimeReady === false && <span>not ready</span>}
                    {provider.runtimeNote && <span>{provider.runtimeNote}</span>}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="extension-section" aria-label="Skills and MCP">
          <div className="extension-section__header">
            <h3>Skills and MCP</h3>
            <span>{skills.length + mcpServers.length + appConnectors.length}</span>
          </div>
          <ul className="extension-list extension-list--compact">
            {skills.map((skill) => (
              <li key={`skill:${skill.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{skill.name ?? skill.id}</strong>
                  <span>{skill.path ?? skill.id}</span>
                </div>
                <div className="extension-item__chips">
                  {extensionSkillChips(skill).map((chip) => (
                    <span key={`${skill.id}:${chip}`}>{chip}</span>
                  ))}
                </div>
              </li>
            ))}
            {mcpServers.map((server) => (
              <li key={`mcp:${server.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{server.name ?? server.id}</strong>
                  <span>{server.scope ?? "MCP server"}</span>
                </div>
              </li>
            ))}
            {appConnectors.map((connector) => (
              <li key={`connector:${connector.id}`} className="extension-item">
                <div className="extension-item__main">
                  <strong>{connector.name}</strong>
                  <span>{connector.kind}</span>
                </div>
              </li>
            ))}
          </ul>
        </section>

        {(error || warnings.length > 0) && (
          <section
            className="extension-section extension-section--alerts"
            aria-label="Extension alerts"
          >
            <div className="extension-section__header">
              <h3>Alerts</h3>
              <span>{warnings.length + (error ? 1 : 0)}</span>
            </div>
            <ul className="extension-list extension-list--compact">
              {error ? (
                <li className="extension-alert">
                  <XCircle aria-hidden="true" />
                  <span>{errorMessage(error)}</span>
                </li>
              ) : null}
              {warnings.map((warning) => (
                <li key={`${warning.source}:${warning.message}`} className="extension-alert">
                  <XCircle aria-hidden="true" />
                  <span>
                    {warning.source}: {warning.message}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </section>
  );
}
