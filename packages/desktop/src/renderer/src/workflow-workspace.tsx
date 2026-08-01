import type { EdgeConfig, SwarmConfig, SwarmNodeConfig } from "@swarmx/core";
import {
  Bot,
  CircleCheck,
  Loader2,
  type LucideIcon,
  Maximize2,
  MessageSquarePlus,
  Minus,
  Play,
  Plus,
  Upload,
  Workflow,
  Wrench,
  XCircle,
} from "lucide-react";
import { useRef } from "react";
import type { DesktopMessageChunk as MessageChunk } from "../../shared/desktop-api.js";
import { AppBrandIcon } from "./app-brand.js";
import { messageKey } from "./conversation-messages.js";
import { harnessOption } from "./harness-presentation.js";
import { errorMessage, isRecord } from "./text-utils.js";
import { Badge, Button, cx } from "./ui-primitives.js";

export interface HarnessDescriptor {
  software?: {
    name?: string;
    version?: string;
    runner?: string;
    command?: string[];
  };
  mcps?: Array<{ name?: string; transport?: string; scope?: string } | string>;
  skills?: string[];
  projectFiles?: string[];
}

interface WorkflowGraphNode {
  id: string;
  kind: SwarmNodeConfig["kind"];
  displayKind: "trigger" | SwarmNodeConfig["kind"];
  title: string;
  detail: string;
  isRoot: boolean;
  harnessId?: string;
  harnessLabel?: string;
  harness?: HarnessDescriptor;
  softwareLabel?: string;
  mcpsLabel?: string;
  skillsLabel?: string;
  projectFilesLabel?: string;
  model?: string;
}

export interface WorkflowParseResult {
  config: SwarmConfig | null;
  error: string | null;
  nodes: WorkflowGraphNode[];
  edges: EdgeConfig[];
}

export interface WorkflowImportStatus {
  kind: "success" | "error";
  message: string;
  warnings: string[];
}

export function WorkflowWorkspace({
  workflowJson,
  onWorkflowJsonChange,
  workflowEnabled,
  onWorkflowEnabledChange,
  workflowImportStatus,
  workflowState,
  input,
  onInputChange,
  onExecute,
  onImportN8nFile,
  loading,
  messages,
  activeWorkflowConfig,
}: {
  workflowJson: string;
  onWorkflowJsonChange: (value: string) => void;
  workflowEnabled: boolean;
  onWorkflowEnabledChange: (value: boolean) => void;
  workflowImportStatus: WorkflowImportStatus | null;
  workflowState: WorkflowParseResult;
  input: string;
  onInputChange: (value: string) => void;
  onExecute: () => void;
  onImportN8nFile: (file: File) => void;
  loading: boolean;
  messages: MessageChunk[];
  activeWorkflowConfig: SwarmConfig | null;
}) {
  const importInputRef = useRef<HTMLInputElement>(null);
  const workflowName = workflowState.config?.name ?? "Workflow";
  const selectedNode =
    workflowState.nodes.find((node) => node.id === "writer_agent") ?? workflowState.nodes.at(-1);
  const importNoticeRole = workflowImportStatus?.kind === "error" ? "alert" : "status";

  return (
    <section className="workflow-workspace" aria-label="Workflow editor">
      <div className="workflow-topbar">
        <div className="workflow-breadcrumb">
          <Workflow aria-hidden="true" />
          <span>Personal</span>
          <span>/</span>
          <strong>{workflowName}</strong>
        </div>
        <div className="workflow-view-tabs" role="tablist" aria-label="Workflow views">
          <button
            type="button"
            role="tab"
            aria-selected="true"
            className="workflow-view-tab is-active"
          >
            Editor
          </button>
          <button type="button" role="tab" aria-selected="false" className="workflow-view-tab">
            Executions
          </button>
          <button type="button" role="tab" aria-selected="false" className="workflow-view-tab">
            JSON
          </button>
        </div>
        <div className="workflow-topbar__actions">
          <input
            ref={importInputRef}
            type="file"
            accept=".json,application/json"
            hidden
            aria-label="n8n workflow JSON file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) onImportN8nFile(file);
            }}
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={() => importInputRef.current?.click()}
            title="Import n8n workflow JSON"
          >
            <Upload data-icon="inline-start" aria-hidden="true" />
            Import n8n
          </Button>
          <label className="workflow-toggle">
            <input
              type="checkbox"
              checked={workflowEnabled}
              onChange={(event) => onWorkflowEnabledChange(event.target.checked)}
            />
            <span>Use workflow</span>
          </label>
          <Badge
            tone={activeWorkflowConfig ? "active" : workflowState.error ? "danger" : "neutral"}
          >
            {activeWorkflowConfig ? "Saved" : workflowState.error ? "Invalid" : "Draft"}
          </Badge>
        </div>
      </div>

      <div className="workflow-editor-shell">
        <nav className="workflow-rail" aria-label="Workflow navigation">
          <button type="button" className="workflow-rail__brand" aria-label="Workflows">
            <AppBrandIcon className="workflow-rail__logo" />
          </button>
          <button type="button" className="workflow-rail__create" aria-label="Add node">
            <MessageSquarePlus aria-hidden="true" />
          </button>
          <span>Overview</span>
          <span className="is-active">Workflows</span>
          <span>Agents</span>
          <span>Tools</span>
          <span>MCP</span>
          <span>Runs</span>
        </nav>

        <WorkflowCanvas
          workflowState={workflowState}
          onExecute={onExecute}
          loading={loading}
          input={input}
        />

        <aside className="workflow-inspector" aria-label="Workflow inspector">
          <div className="workflow-inspector__header">
            <div>
              <span>
                {selectedNode?.kind === "agent"
                  ? "harness = software + MCPs + skills + project files"
                  : (selectedNode?.displayKind ?? "workflow")}
              </span>
              <strong>{selectedNode?.title ?? workflowName}</strong>
              {selectedNode?.harnessLabel && selectedNode.model && (
                <em>
                  {selectedNode.harnessLabel} / {selectedNode.softwareLabel ?? "software"} /{" "}
                  {selectedNode.model}
                </em>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onExecute}
              disabled={loading || !input.trim()}
              title="Run selected node"
              aria-label="Run selected node"
            >
              {loading ? (
                <Loader2 data-icon aria-hidden="true" />
              ) : (
                <Play data-icon aria-hidden="true" />
              )}
            </Button>
          </div>

          <div className="workflow-inspector__tabs" role="tablist" aria-label="Inspector tabs">
            <button type="button" role="tab" aria-selected="true">
              Parameters
            </button>
            <button type="button" role="tab" aria-selected="false">
              Settings
            </button>
            <button type="button" role="tab" aria-selected="false">
              Notes
            </button>
          </div>

          <label className="workflow-run-input">
            <span>Run input</span>
            <textarea
              value={input}
              onChange={(event) => onInputChange(event.target.value)}
              placeholder="Message SwarmX"
              rows={3}
              disabled={loading}
            />
          </label>

          <label className="workflow-editor">
            <span>Workflow JSON</span>
            <textarea
              aria-label="Workflow JSON"
              value={workflowJson}
              onChange={(event) => onWorkflowJsonChange(event.target.value)}
              spellCheck={false}
            />
          </label>

          {workflowState.error && (
            <div className="workflow-panel__error" role="alert">
              <XCircle aria-hidden="true" />
              <span>{workflowState.error}</span>
            </div>
          )}

          {workflowImportStatus && (
            <div
              className={cx(
                "workflow-panel__notice",
                `workflow-panel__notice--${workflowImportStatus.kind}`,
              )}
              role={importNoticeRole}
            >
              {workflowImportStatus.kind === "error" ? (
                <XCircle aria-hidden="true" />
              ) : (
                <CircleCheck aria-hidden="true" />
              )}
              <div>
                <span>{workflowImportStatus.message}</span>
                {workflowImportStatus.warnings.length > 0 && (
                  <ul>
                    {workflowImportStatus.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          <div className="workflow-output">
            <div className="workflow-output__header">
              <span>Execution Log</span>
              <Badge tone={messages.length > 0 ? "active" : "neutral"}>
                {messages.length} events
              </Badge>
            </div>
            <div className="workflow-output__list">
              {messages.length === 0 ? (
                <div className="workflow-output__empty">No run output yet</div>
              ) : (
                messages.slice(-5).map((message) => (
                  <div key={messageKey(message)} className="workflow-output__event">
                    <span>{message.agent ?? message.role}</span>
                    <p>{message.content}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
}

function WorkflowCanvas({
  workflowState,
  onExecute,
  loading,
  input,
}: {
  workflowState: WorkflowParseResult;
  onExecute: () => void;
  loading: boolean;
  input: string;
}) {
  if (!workflowState.config) {
    return (
      <div className="workflow-canvas workflow-canvas--empty" aria-label="Workflow canvas">
        <div className="workflow-canvas__empty-message">No workflow</div>
        <Button
          className="workflow-execute"
          onClick={onExecute}
          disabled={loading || !input.trim()}
        >
          {loading ? (
            <Loader2 data-icon="inline-start" aria-hidden="true" />
          ) : (
            <Play data-icon="inline-start" aria-hidden="true" />
          )}
          Execute workflow
        </Button>
      </div>
    );
  }

  const nodes = workflowState.nodes.map((node, index) => ({
    ...node,
    layout: workflowNodeLayout(node.id, index),
  }));
  const nodeLayouts = new Map(nodes.map((node) => [node.id, node.layout]));

  return (
    <div className="workflow-canvas" aria-label="Workflow canvas">
      <svg className="workflow-connectors" viewBox="0 0 804 620" aria-label="Workflow connectors">
        <title>Workflow connectors</title>
        {workflowState.edges.map((edge) => {
          const source = nodeLayouts.get(edge.source);
          const target = nodeLayouts.get(edge.target);
          if (!source || !target) return null;
          return (
            <path
              key={`${edge.source}:${edge.target}:${edge.condition ?? ""}`}
              aria-label={`Workflow connector ${edge.source} to ${edge.target}`}
              className="workflow-connector"
              d={connectorPath(source, target)}
            />
          );
        })}
      </svg>

      <ul className="workflow-canvas__nodes" aria-label="Workflow nodes">
        {nodes.map((node) => {
          const NodeIcon = nodeIcon(node.displayKind);
          const nodeHarnessId = node.harnessId ?? node.displayKind;
          const nodeModel = node.model ?? "";
          return (
            <li
              key={node.id}
              aria-label={`Workflow node ${node.id} ${nodeHarnessId}${nodeModel ? ` ${nodeModel}` : ""}${node.isRoot ? " root" : ""}`}
              className={cx(
                "workflow-node",
                `workflow-node--${node.displayKind}`,
                node.id === "writer_agent" && "is-selected",
              )}
              style={{ left: node.layout.x, top: node.layout.y }}
            >
              <span className="workflow-port workflow-port--in" aria-hidden="true" />
              <span className="workflow-port workflow-port--out" aria-hidden="true" />
              <div className="workflow-node__topline">
                <span className="workflow-node__icon">
                  <NodeIcon aria-hidden="true" />
                </span>
                <span className="workflow-node__kind">
                  {node.kind === "agent" ? "ACP Agent" : node.displayKind}
                </span>
                <span className="workflow-node__status">
                  <CircleCheck aria-hidden="true" />
                </span>
              </div>
              <div className="workflow-node__name">{node.title}</div>
              <div className="workflow-node__detail">{node.detail}</div>
              {node.kind === "agent" && (
                <div className="workflow-node__identity">
                  {node.softwareLabel && <span>Software {node.softwareLabel}</span>}
                  <span>Harness {node.harnessLabel ?? "SwarmX"}</span>
                  <span>{node.model ? `Model ${node.model}` : "Model negotiated by harness"}</span>
                  {node.mcpsLabel && <span>MCPs {node.mcpsLabel}</span>}
                  {node.skillsLabel && <span>Skills {node.skillsLabel}</span>}
                  {node.projectFilesLabel && <span>Project files {node.projectFilesLabel}</span>}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <ul className="workflow-edges" aria-label="Workflow edges">
        {workflowState.edges.map((edge) => (
          <li
            key={`${edge.source}:${edge.target}:${edge.condition ?? ""}`}
            aria-label={`Workflow edge ${edge.source} to ${edge.target}`}
            className="workflow-edge"
          >
            <span>{edge.source}</span>
            <span aria-hidden="true">-&gt;</span>
            <span>{edge.target}</span>
          </li>
        ))}
      </ul>

      <div className="workflow-canvas__add workflow-canvas__add--top" aria-hidden="true">
        +
      </div>
      <div className="workflow-canvas__add workflow-canvas__add--bottom" aria-hidden="true">
        +
      </div>

      <div className="workflow-canvas-controls" aria-label="Canvas controls">
        <button type="button" aria-label="Fit workflow">
          <Maximize2 aria-hidden="true" />
        </button>
        <button type="button" aria-label="Zoom out">
          <Minus aria-hidden="true" />
        </button>
        <span>100%</span>
        <button type="button" aria-label="Zoom in">
          <Plus aria-hidden="true" />
        </button>
        <button type="button">Tidy</button>
      </div>

      <Button className="workflow-execute" onClick={onExecute} disabled={loading || !input.trim()}>
        {loading ? (
          <Loader2 data-icon="inline-start" aria-hidden="true" />
        ) : (
          <Play data-icon="inline-start" aria-hidden="true" />
        )}
        Execute workflow
      </Button>
    </div>
  );
}

export function parseWorkflowJson(source: string): WorkflowParseResult {
  const emptyResult: WorkflowParseResult = { config: null, error: null, nodes: [], edges: [] };
  const trimmed = source.trim();
  if (!trimmed) {
    return { ...emptyResult, error: "Workflow JSON is empty." };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    return {
      ...emptyResult,
      error: `Workflow JSON parse error: ${errorMessage(error)}`,
    };
  }

  if (!isRecord(parsed)) {
    return { ...emptyResult, error: "Workflow JSON must be an object." };
  }

  const name = parsed.name;
  if (typeof name !== "string" || name.trim() === "") {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty name." };
  }

  const root = parsed.root;
  if (typeof root !== "string" || root.trim() === "") {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty root." };
  }

  const rawNodes = parsed.nodes;
  if (!isRecord(rawNodes) || Object.keys(rawNodes).length === 0) {
    return { ...emptyResult, error: "Workflow JSON needs a non-empty nodes object." };
  }

  if (!Object.hasOwn(rawNodes, root)) {
    return { ...emptyResult, error: `Workflow JSON root "${root}" is not in nodes.` };
  }

  const nodes: WorkflowGraphNode[] = [];
  for (const [id, value] of Object.entries(rawNodes)) {
    if (!isRecord(value)) {
      return { ...emptyResult, error: `Workflow node "${id}" must be an object.` };
    }

    const kind = value.kind;
    if (kind !== "agent" && kind !== "tool" && kind !== "swarm") {
      return {
        ...emptyResult,
        error: `Workflow node "${id}" needs kind "agent", "tool", or "swarm".`,
      };
    }

    if (kind === "agent" && !isRecord(value.agent)) {
      return { ...emptyResult, error: `Workflow agent node "${id}" needs agent config.` };
    }
    if (kind === "tool" && !isRecord(value.tool)) {
      return { ...emptyResult, error: `Workflow tool node "${id}" needs tool config.` };
    }
    if (kind === "swarm" && !isRecord(value.swarm)) {
      return { ...emptyResult, error: `Workflow swarm node "${id}" needs swarm config.` };
    }

    nodes.push(toWorkflowGraphNode(id, value as SwarmNodeConfig, root));
  }

  const rawEdges = parsed.edges;
  if (!Array.isArray(rawEdges)) {
    return { ...emptyResult, error: "Workflow JSON needs an edges array." };
  }

  const edges: EdgeConfig[] = [];
  for (const [index, value] of rawEdges.entries()) {
    if (!isRecord(value)) {
      return { ...emptyResult, error: `Workflow edge ${index + 1} must be an object.` };
    }
    if (typeof value.source !== "string" || value.source.trim() === "") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} needs source.` };
    }
    if (typeof value.target !== "string" || value.target.trim() === "") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} needs target.` };
    }
    if (value.condition !== undefined && typeof value.condition !== "string") {
      return { ...emptyResult, error: `Workflow edge ${index + 1} condition must be a string.` };
    }

    edges.push({
      source: value.source,
      target: value.target,
      condition: value.condition,
    });
  }

  return {
    config: parsed as SwarmConfig,
    error: null,
    nodes,
    edges,
  };
}

function toWorkflowGraphNode(id: string, node: SwarmNodeConfig, root: string): WorkflowGraphNode {
  const source = node.kind === "agent" ? node.agent : node.kind === "tool" ? node.tool : node.swarm;
  const title = readString(source, "name") || id;
  const detail = readString(source, "description") || readString(source, "instructions") || title;
  const harness = node.kind === "agent" ? readHarnessDescriptor(source) : undefined;
  const harnessId =
    node.kind === "agent"
      ? harnessIdFromBackend(isRecord(source) ? source.backend : undefined)
      : undefined;
  const softwareLabel = formatSoftwareLabel(harness);
  const mcpsLabel = formatNamedList(harness?.mcps);
  const skillsLabel = formatStringList(harness?.skills);
  const projectFilesLabel = formatStringList(harness?.projectFiles);
  return {
    id,
    kind: node.kind,
    displayKind: id.includes("trigger") ? "trigger" : node.kind,
    title,
    detail,
    harnessId,
    harnessLabel: harnessId ? harnessOption(harnessId, harnessId).label : undefined,
    harness,
    softwareLabel,
    mcpsLabel,
    skillsLabel,
    projectFilesLabel,
    model: readString(source, "model"),
    isRoot: id === root,
  };
}

function workflowNodeLayout(id: string, fallbackIndex: number): { x: number; y: number } {
  const layout: Record<string, { x: number; y: number }> = {
    triage_agent: { x: 36, y: 210 },
    researcher_agent: { x: 292, y: 128 },
    writer_agent: { x: 548, y: 210 },
  };
  return layout[id] ?? { x: 36 + fallbackIndex * 256, y: 210 };
}

function connectorPath(source: { x: number; y: number }, target: { x: number; y: number }): string {
  const sourceX = source.x + 220;
  const sourceY = source.y + 112;
  const targetX = target.x;
  const targetY = target.y + 112;
  const gap = Math.max(24, Math.abs(targetX - sourceX));
  const curve = Math.min(92, Math.max(24, gap * 0.5));
  return `M ${sourceX} ${sourceY} C ${sourceX + curve} ${sourceY}, ${targetX - curve} ${targetY}, ${targetX} ${targetY}`;
}

function nodeIcon(kind: WorkflowGraphNode["displayKind"]): LucideIcon {
  switch (kind) {
    case "trigger":
      return Play;
    case "tool":
      return Wrench;
    case "swarm":
      return Workflow;
    default:
      return Bot;
  }
}

function readString(source: unknown, key: string): string {
  if (!isRecord(source)) return "";
  const value = source[key];
  return typeof value === "string" ? value : "";
}

function readHarnessDescriptor(source: unknown): HarnessDescriptor | undefined {
  if (!isRecord(source) || !isRecord(source.parameters)) return undefined;
  const harness = source.parameters.harness;
  if (!isRecord(harness)) return undefined;

  const software = isRecord(harness.software)
    ? {
        name: readString(harness.software, "name") || undefined,
        version: readString(harness.software, "version") || undefined,
        runner: readString(harness.software, "runner") || undefined,
        command: readStringArray(harness.software.command),
      }
    : undefined;

  return {
    software,
    mcps: readMcpList(harness.mcps),
    skills: readStringArray(harness.skills),
    projectFiles: readStringArray(harness.projectFiles),
  };
}

function formatSoftwareLabel(harness: HarnessDescriptor | undefined): string {
  const software = harness?.software;
  if (!software?.name) return "";
  return software.version ? `${software.name}@${software.version}` : software.name;
}

function formatNamedList(items: HarnessDescriptor["mcps"]): string {
  if (!items || items.length === 0) return "";
  return items
    .map((item) => (typeof item === "string" ? item : item.name))
    .filter((item): item is string => Boolean(item))
    .join(", ");
}

function formatStringList(items: string[] | undefined): string {
  return items?.filter(Boolean).join(", ") ?? "";
}

function readStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.filter((item): item is string => typeof item === "string" && item !== "");
  return items.length > 0 ? items : undefined;
}

function readMcpList(value: unknown): HarnessDescriptor["mcps"] | undefined {
  if (!Array.isArray(value)) return undefined;
  type HarnessMcp = NonNullable<HarnessDescriptor["mcps"]>[number];
  const items = value.flatMap((item): HarnessMcp[] => {
    if (typeof item === "string" && item !== "") return [item];
    if (!isRecord(item)) return [];
    const name = readString(item, "name");
    if (!name) return [];
    return [
      {
        name,
        transport: readString(item, "transport") || undefined,
        scope: readString(item, "scope") || undefined,
      },
    ];
  });
  return items.length > 0 ? items : undefined;
}

function harnessIdFromBackend(backend: unknown): string {
  if (!isRecord(backend)) return "swarmx";

  const type = readString(backend, "type");
  if (type === "swarmx" || type === "claude_code") return type;

  const program = readString(backend, "program");
  if (
    program === "kimi" ||
    program === "opencode" ||
    program === "hermes" ||
    program === "openclaw"
  ) {
    return program;
  }

  const args = Array.isArray(backend.args)
    ? backend.args.filter((arg): arg is string => typeof arg === "string")
    : [];
  const commandLine = [program, ...args].join(" ");
  if (commandLine.includes("@agentclientprotocol/codex-acp")) return "codex";
  if (commandLine.includes("@agentclientprotocol/claude-agent-acp")) return "claude_code";
  if (commandLine.includes("pi-acp")) return "pi";

  return type || program || "custom";
}
