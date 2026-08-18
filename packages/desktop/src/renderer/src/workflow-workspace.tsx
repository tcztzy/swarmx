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
    <section
      className="workflow-workspace [height:100%] [min-width:0] [min-height:0] [overflow:hidden] [display:grid] [grid-template-rows:56px_minmax(0,_1fr)] [background:#101114] max-680:[grid-template-rows:auto_minmax(0,_1fr)]"
      aria-label="Workflow editor"
    >
      <div className="workflow-topbar [min-width:0] [display:grid] [grid-template-columns:minmax(220px,_1fr)_auto_minmax(220px,_1fr)] [align-items:center] [gap:16px] [padding:0_16px] [color:var(--foreground)] [background:#17181c] [border-bottom:1px_solid_rgba(255,_255,_255,_0.08)] max-860:[grid-template-columns:minmax(0,_1fr)_auto] max-680:[gap:10px] max-680:[padding:10px_12px]">
        <div className="workflow-breadcrumb [gap:9px] [color:#bec3cc] [font-size:13px] [white-space:nowrap] [min-width:0] [display:flex] [align-items:center] [&_svg]:[flex:0_0_auto] [&_svg]:[width:15px] [&_svg]:[height:15px] [&_svg]:[color:#f36f5b] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[color:#f4f6fa] [&_strong]:[font-weight:650] max-680:[width:100%]">
          <Workflow aria-hidden="true" />
          <span>Personal</span>
          <span>/</span>
          <strong>{workflowName}</strong>
        </div>
        <div
          className="workflow-view-tabs [justify-self:center] [height:32px] [padding:2px] [gap:2px] [background:#121317] [border:1px_solid_rgba(255,_255,_255,_0.12)] [border-radius:7px] [min-width:0] [display:flex] [align-items:center] max-860:[justify-self:end] max-680:[width:100%] max-680:[grid-column:1_/_-1]"
          role="tablist"
          aria-label="Workflow views"
        >
          <button
            type="button"
            role="tab"
            aria-selected="true"
            className="workflow-view-tab is-active [height:26px] [min-width:92px] [padding:0_14px] [color:#a9afba] [background:transparent] [border:0] [border-radius:5px] [font-size:13px] [font-weight:560] [cursor:pointer] max-680:[min-width:0] max-680:[flex:1_1_0] max-680:[padding:0_8px]"
          >
            Editor
          </button>
          <button
            type="button"
            role="tab"
            aria-selected="false"
            className="workflow-view-tab [height:26px] [min-width:92px] [padding:0_14px] [color:#a9afba] [background:transparent] [border:0] [border-radius:5px] [font-size:13px] [font-weight:560] [cursor:pointer] max-680:[min-width:0] max-680:[flex:1_1_0] max-680:[padding:0_8px]"
          >
            Executions
          </button>
          <button
            type="button"
            role="tab"
            aria-selected="false"
            className="workflow-view-tab [height:26px] [min-width:92px] [padding:0_14px] [color:#a9afba] [background:transparent] [border:0] [border-radius:5px] [font-size:13px] [font-weight:560] [cursor:pointer] max-680:[min-width:0] max-680:[flex:1_1_0] max-680:[padding:0_8px]"
          >
            JSON
          </button>
        </div>
        <div className="workflow-topbar__actions [justify-content:flex-end] [gap:8px] [min-width:0] [display:flex] [align-items:center] max-860:[grid-column:1_/_-1] max-860:[justify-content:flex-start]">
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
          <label className="workflow-toggle [height:28px] [gap:7px] [padding:0_9px] [color:#c8cdd6] [background:#202127] [border:1px_solid_rgba(255,_255,_255,_0.1)] [border-radius:6px] [font-size:12px] [font-weight:620] [white-space:nowrap] [min-width:0] [display:flex] [align-items:center] [&_input]:[width:13px] [&_input]:[height:13px] [&_input]:[margin:0] [&_input]:[accent-color:var(--accent)]">
            <input
              type="checkbox"
              checked={workflowEnabled}
              onChange={(event) => onWorkflowEnabledChange(event.target.checked)}
            />
            <span>Use workflow</span>
          </label>
          <Badge
            tone={activeWorkflowConfig ? "success" : workflowState.error ? "danger" : "neutral"}
          >
            {activeWorkflowConfig ? "Saved" : workflowState.error ? "Invalid" : "Draft"}
          </Badge>
        </div>
      </div>

      <div className="workflow-editor-shell [min-width:0] [min-height:0] [display:grid] [grid-template-columns:72px_minmax(0,_1fr)_324px] [overflow:hidden] max-860:[grid-template-columns:64px_minmax(0,_1fr)] max-680:[grid-template-columns:1fr] max-680:[grid-template-rows:auto_360px_auto] max-680:[overflow-y:auto]">
        <nav
          className="workflow-rail [min-width:0] [display:flex] [flex-direction:column] [align-items:center] [gap:13px] [padding:16px_10px_12px] [color:#969ca8] [background:#15161a] [border-right:1px_solid_rgba(255,_255,_255,_0.08)] [font-size:11px] [font-weight:560] [&_span]:[width:100%] [&_span]:[padding:8px_0] [&_span]:[text-align:center] [&_span]:[border-radius:8px] [&_span.is-active]:[color:#ffffff] [&_span.is-active]:[background:#24262c] [&_button]:[display:grid] [&_button]:[place-items:center] [&_button]:[border:0] [&_button]:[cursor:pointer] [&_svg]:[width:21px] [&_svg]:[height:21px] max-680:[min-height:54px] max-680:[flex-direction:row] max-680:[overflow-x:auto] max-680:[padding:8px_10px] max-680:[border-right:0] max-680:[border-bottom:1px_solid_rgba(255,_255,_255,_0.08)] max-680:[&_span]:[width:auto] max-680:[&_span]:[min-width:max-content] max-680:[&_span]:[padding:7px_9px]"
          aria-label="Workflow navigation"
        >
          <button
            type="button"
            className="workflow-rail__brand [width:46px] [height:46px] [color:#ffffff] [background:#f36f5b] [border-radius:10px] [box-shadow:0_10px_22px_rgba(243,_111,_91,_0.22)] max-680:[flex:0_0_auto] max-680:[width:38px] max-680:[height:38px]"
            aria-label="Workflows"
          >
            <AppBrandIcon className="workflow-rail__logo [width:32px] [height:32px] [object-fit:contain]" />
          </button>
          <button
            type="button"
            className="workflow-rail__create [width:46px] [height:46px] [color:#ffffff] [background:#f36f5b] [border-radius:10px] [box-shadow:0_10px_22px_rgba(243,_111,_91,_0.22)] max-680:[flex:0_0_auto] max-680:[width:38px] max-680:[height:38px]"
            aria-label="Add node"
          >
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

        <aside
          className="workflow-inspector [min-width:0] [min-height:0] [overflow-y:auto] [background:linear-gradient(180deg,_#24262c,_#1d1f24)] [border-left:1px_solid_rgba(255,_255,_255,_0.1)] [box-shadow:-18px_0_48px_rgba(0,_0,_0,_0.18)] max-860:[grid-column:1_/_-1] max-860:[min-height:280px] max-860:[border-left:0] max-860:[border-top:1px_solid_rgba(255,_255,_255,_0.1)] max-860:[box-shadow:none]"
          aria-label="Workflow inspector"
        >
          <div className="workflow-inspector__header [height:72px] [padding:14px_14px] [display:flex] [align-items:center] [justify-content:space-between] [gap:10px] [border-bottom:1px_solid_rgba(255,_255,_255,_0.08)] [&_div]:[min-width:0] [&_div]:[display:flex] [&_div]:[flex-direction:column] [&_div]:[gap:2px] [&_span]:[color:#a8aeba] [&_span]:[font-size:11px] [&_span]:[font-weight:760] [&_span]:[text-transform:uppercase] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[color:#ffffff] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_strong]:[font-size:15px] [&_strong]:[font-weight:680] [&_em]:[min-width:0] [&_em]:[overflow:hidden] [&_em]:[color:#d1d6df] [&_em]:[text-overflow:ellipsis] [&_em]:[white-space:nowrap] [&_em]:[font-family:var(--font-mono)] [&_em]:[font-size:11px] [&_em]:[font-style:normal]">
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

          <div
            className="workflow-inspector__tabs [height:42px] [display:grid] [grid-template-columns:repeat(3,_1fr)] [border-bottom:1px_solid_rgba(255,_255,_255,_0.08)] [&_button]:[color:#a9afba] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-bottom:2px_solid_transparent] [&_button]:[font-size:12px] [&_button]:[font-weight:620] [&_[aria-selected='true']]:[color:#ffffff] [&_[aria-selected='true']]:[border-bottom-color:#f36f5b]"
            role="tablist"
            aria-label="Inspector tabs"
          >
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

          <label className="workflow-run-input [margin:14px] [min-width:0] [display:flex] [flex-direction:column] [gap:7px] [&_>_span]:[color:#dfe4ed] [&_>_span]:[font-size:12px] [&_>_span]:[font-weight:740] [&_textarea]:[width:100%] [&_textarea]:[resize:vertical] [&_textarea]:[padding:10px_11px] [&_textarea]:[color:#e4e7ec] [&_textarea]:[background:#15171c] [&_textarea]:[border:1px_solid_rgba(255,_255,_255,_0.12)] [&_textarea]:[border-radius:7px] [&_textarea]:[box-shadow:inset_0_1px_0_rgba(255,_255,_255,_0.035)] [&_textarea]:[font-size:12px] [&_textarea]:[line-height:1.48] [&_textarea]:[tab-size:2] [&_textarea]:[min-height:74px] [&_textarea]:[font-family:var(--font-sans)]">
            <span>Run input</span>
            <textarea
              value={input}
              onChange={(event) => onInputChange(event.target.value)}
              placeholder="Message SwarmX"
              rows={3}
              disabled={loading}
            />
          </label>

          <label className="workflow-editor [margin:14px] [min-width:0] [display:flex] [flex-direction:column] [gap:7px] [&_>_span]:[color:#dfe4ed] [&_>_span]:[font-size:12px] [&_>_span]:[font-weight:740] [&_textarea]:[width:100%] [&_textarea]:[resize:vertical] [&_textarea]:[padding:10px_11px] [&_textarea]:[color:#e4e7ec] [&_textarea]:[background:#15171c] [&_textarea]:[border:1px_solid_rgba(255,_255,_255,_0.12)] [&_textarea]:[border-radius:7px] [&_textarea]:[box-shadow:inset_0_1px_0_rgba(255,_255,_255,_0.035)] [&_textarea]:[font-family:var(--font-mono)] [&_textarea]:[font-size:12px] [&_textarea]:[line-height:1.48] [&_textarea]:[tab-size:2] [&_textarea]:[min-height:194px] [&_textarea]:[max-height:260px] max-680:[&_textarea]:[min-height:170px]">
            <span>Workflow JSON</span>
            <textarea
              aria-label="Workflow JSON"
              value={workflowJson}
              onChange={(event) => onWorkflowJsonChange(event.target.value)}
              spellCheck={false}
            />
          </label>

          {workflowState.error && (
            <div
              className="workflow-panel__error [margin:14px] [min-width:0] [padding:8px_9px] [display:flex] [align-items:flex-start] [gap:8px] [color:var(--danger)] [background:var(--danger-muted)] [border:1px_solid_rgba(248,_113,_113,_0.26)] [border-radius:var(--radius)] [font-size:12px] [line-height:1.35] [&_svg]:[flex:0_0_auto] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[margin-top:1px]"
              role="alert"
            >
              <XCircle aria-hidden="true" />
              <span>{workflowState.error}</span>
            </div>
          )}

          {workflowImportStatus && (
            <div
              className={cx(
                String.raw`workflow-panel__notice [margin:14px] [min-width:0] [padding:8px_9px] [display:flex] [align-items:flex-start] [gap:8px] [color:var(--success)] [background:var(--success-muted)] [border:1px_solid_rgba(52,_211,_153,_0.26)] [border-radius:var(--radius)] [font-size:12px] [line-height:1.35] [&_svg]:[flex:0_0_auto] [&_svg]:[width:14px] [&_svg]:[height:14px] [&_svg]:[margin-top:1px] [&_div]:[min-width:0] [&_span]:[display:block] [&_span]:[overflow-wrap:anywhere] [&_ul]:[margin:6px_0_0] [&_ul]:[padding-left:16px] [&_ul]:[color:#c8cdd6] [&.workflow-panel\_\_notice--error]:[color:var(--danger)] [&.workflow-panel\_\_notice--error]:[background:var(--danger-muted)] [&.workflow-panel\_\_notice--error]:[border-color:rgba(248,_113,_113,_0.26)]`,
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

          <div className="workflow-output [margin:14px] [min-width:0] [display:flex] [flex-direction:column] [gap:7px]">
            <div className="workflow-output__header [display:flex] [align-items:center] [justify-content:space-between] [gap:10px] [color:#dfe4ed] [font-size:12px] [font-weight:740]">
              <span>Execution Log</span>
              <Badge tone={messages.length > 0 ? "success" : "neutral"}>
                {messages.length} events
              </Badge>
            </div>
            <div className="workflow-output__list [display:flex] [flex-direction:column] [gap:8px]">
              {messages.length === 0 ? (
                <div className="workflow-output__empty [padding:9px_10px] [color:#a9afba] [background:#17191e] [border:1px_solid_rgba(255,_255,_255,_0.08)] [border-radius:7px] [font-size:12px] [line-height:1.35]">
                  No run output yet
                </div>
              ) : (
                messages.slice(-5).map((message) => (
                  <div
                    key={messageKey(message)}
                    className="workflow-output__event [padding:9px_10px] [color:#a9afba] [background:#17191e] [border:1px_solid_rgba(255,_255,_255,_0.08)] [border-radius:7px] [font-size:12px] [line-height:1.35] [&_span]:[display:block] [&_span]:[margin-bottom:4px] [&_span]:[color:#69d991] [&_span]:[font-size:10px] [&_span]:[font-weight:760] [&_span]:[text-transform:uppercase] [&_p]:[margin:0] [&_p]:[display:-webkit-box] [&_p]:[overflow:hidden] [&_p]:[-webkit-box-orient:vertical] [&_p]:[-webkit-line-clamp:3]"
                  >
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
      <div
        className="workflow-canvas workflow-canvas--empty [position:relative] [min-width:0] [overflow:hidden] [background-color:#111317] [background-image:radial-gradient(circle,_rgba(255,_255,_255,_0.18)_1px,_transparent_1px)] [background-position:0_0] [background-size:18px_18px] [display:grid] [place-items:center] [color:#9da3ae] max-860:[min-height:0] max-680:[min-height:360px] max-680:[overflow:hidden]"
        aria-label="Workflow canvas"
      >
        <div className="workflow-canvas__empty-message">No workflow</div>
        <Button
          className="workflow-execute [position:absolute] [left:50%] [bottom:18px] [transform:translateX(-50%)] [min-width:214px] [height:42px] [color:#ffffff] [background:linear-gradient(180deg,_#ff826d,_#f15f4d)] [border-color:rgba(255,_155,_136,_0.72)] max-680:[right:14px] max-680:[bottom:14px] max-680:[left:auto] max-680:[min-width:182px] max-680:[height:38px] max-680:[transform:none] max-680:[font-size:12px]"
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
    <div
      className="workflow-canvas [position:relative] [min-width:0] [overflow:hidden] [background-color:#111317] [background-image:radial-gradient(circle,_rgba(255,_255,_255,_0.18)_1px,_transparent_1px)] [background-position:0_0] [background-size:18px_18px] max-860:[min-height:0] max-680:[min-height:360px] max-680:[overflow:hidden]"
      aria-label="Workflow canvas"
    >
      <svg
        className="workflow-connectors [position:absolute] [inset:0] [width:804px] [height:620px] [pointer-events:none] max-680:[transform:scale(0.47)] max-680:[transform-origin:top_left]"
        viewBox="0 0 804 620"
        aria-label="Workflow connectors"
      >
        <title>Workflow connectors</title>
        {workflowState.edges.map((edge) => {
          const source = nodeLayouts.get(edge.source);
          const target = nodeLayouts.get(edge.target);
          if (!source || !target) return null;
          return (
            <path
              key={`${edge.source}:${edge.target}:${edge.condition ?? ""}`}
              aria-label={`Workflow connector ${edge.source} to ${edge.target}`}
              className="workflow-connector [fill:none] [stroke:#c7ccd6] [stroke-width:2] [stroke-linecap:round] [opacity:0.84]"
              d={connectorPath(source, target)}
            />
          );
        })}
      </svg>

      <ul
        className="workflow-canvas__nodes [position:absolute] [inset:0] [width:804px] [height:620px] [margin:0] [padding:0] [list-style:none] max-680:[transform:scale(0.47)] max-680:[transform-origin:top_left]"
        aria-label="Workflow nodes"
      >
        {nodes.map((node) => {
          const NodeIcon = nodeIcon(node.displayKind);
          const nodeHarnessId = node.harnessId ?? node.displayKind;
          const nodeModel = node.model ?? "";
          return (
            <li
              key={node.id}
              aria-label={`Workflow node ${node.id} ${nodeHarnessId}${nodeModel ? ` ${nodeModel}` : ""}${node.isRoot ? " root" : ""}`}
              className={cx(
                "workflow-node [position:absolute] [width:220px] [min-height:226px] [padding:14px_14px_12px] [display:flex] [flex-direction:column] [gap:8px] [color:#f4f6fa] [background:#202228] [border:1px_solid_#4a4f5a] [border-radius:8px] [box-shadow:0_18px_38px_rgba(0,_0,_0,_0.28),_inset_0_1px_0_rgba(255,_255,_255,_0.05)]",
                `workflow-node--${node.displayKind}`,
                node.id === "writer_agent" && "is-selected",
              )}
              style={{ left: node.layout.x, top: node.layout.y }}
            >
              <span
                className="workflow-port workflow-port--in [position:absolute] [top:106px] [width:12px] [height:12px] [background:#2c3038] [border:2px_solid_#c7ccd6] [border-radius:999px] [left:-7px]"
                aria-hidden="true"
              />
              <span
                className="workflow-port workflow-port--out [position:absolute] [top:106px] [width:12px] [height:12px] [background:#2c3038] [border:2px_solid_#c7ccd6] [border-radius:999px] [right:-7px]"
                aria-hidden="true"
              />
              <div className="workflow-node__topline [display:flex] [align-items:center] [gap:8px]">
                <span className="workflow-node__icon [width:24px] [height:24px] [display:grid] [place-items:center] [border-radius:7px] [color:#111317] [background:#7ee0a1] [font-size:16px] [font-weight:760] [&_svg]:[width:14px] [&_svg]:[height:14px] [.workflow-node--agent_&]:[color:#ecf4ff] [.workflow-node--agent_&]:[background:#4d7dff] [.workflow-node--tool_&]:[color:#111317] [.workflow-node--tool_&]:[background:#f2d45c] [.workflow-node--trigger_&]:[color:#102018] [.workflow-node--trigger_&]:[background:#69d991]">
                  <NodeIcon aria-hidden="true" />
                </span>
                <span className="workflow-node__kind [color:#a5abb6] [font-size:10px] [font-weight:760] [line-height:1.2] [text-transform:uppercase]">
                  {node.kind === "agent" ? "Harness Agent" : node.displayKind}
                </span>
                <span className="workflow-node__status [margin-left:auto] [color:#69d991] [line-height:0] [&_svg]:[width:14px] [&_svg]:[height:14px]">
                  <CircleCheck aria-hidden="true" />
                </span>
              </div>
              <div className="workflow-node__name [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap] [font-size:14px] [font-weight:650] [line-height:1.2]">
                {node.title}
              </div>
              <div className="workflow-node__detail [min-width:0] [display:-webkit-box] [overflow:hidden] [color:#a9afba] [font-size:12px] [line-height:1.35] [-webkit-box-orient:vertical] [-webkit-line-clamp:2]">
                {node.detail}
              </div>
              {node.kind === "agent" && (
                <div className="workflow-node__identity [margin-top:auto] [min-width:0] [display:flex] [flex-direction:column] [gap:4px] [&_span]:[min-width:0] [&_span]:[overflow:hidden] [&_span]:[padding:3px_6px] [&_span]:[color:#dce1ea] [&_span]:[background:rgba(255,_255,_255,_0.055)] [&_span]:[border:1px_solid_rgba(255,_255,_255,_0.07)] [&_span]:[border-radius:5px] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap] [&_span]:[font-family:var(--font-mono)] [&_span]:[font-size:10.5px] [&_span]:[line-height:1.2]">
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

      <ul
        className="workflow-edges [position:absolute] [width:1px] [height:1px] [overflow:hidden] [opacity:0] [pointer-events:none] [margin:0] [padding:0] [list-style:none]"
        aria-label="Workflow edges"
      >
        {workflowState.edges.map((edge) => (
          <li
            key={`${edge.source}:${edge.target}:${edge.condition ?? ""}`}
            aria-label={`Workflow edge ${edge.source} to ${edge.target}`}
            className="workflow-edge [font-size:1px]"
          >
            <span>{edge.source}</span>
            <span aria-hidden="true">-&gt;</span>
            <span>{edge.target}</span>
          </li>
        ))}
      </ul>

      <div
        className="workflow-canvas__add workflow-canvas__add--top [position:absolute] [left:387px] [width:30px] [height:30px] [display:grid] [place-items:center] [color:#bdc3ce] [border:1px_dashed_rgba(255,_255,_255,_0.28)] [border-radius:7px] [font-size:22px] [top:82px] max-680:[display:none]"
        aria-hidden="true"
      >
        +
      </div>
      <div
        className="workflow-canvas__add workflow-canvas__add--bottom [position:absolute] [left:387px] [width:30px] [height:30px] [display:grid] [place-items:center] [color:#bdc3ce] [border:1px_dashed_rgba(255,_255,_255,_0.28)] [border-radius:7px] [font-size:22px] [bottom:112px] max-680:[display:none]"
        aria-hidden="true"
      >
        +
      </div>

      <div
        className="workflow-canvas-controls [position:absolute] [left:24px] [bottom:18px] [display:flex] [align-items:center] [gap:7px] [color:#c7ccd6] [&_button]:[height:34px] [&_button]:[min-width:38px] [&_button]:[padding:0_10px] [&_button]:[color:inherit] [&_button]:[background:#1d1f25] [&_button]:[border:1px_solid_rgba(255,_255,_255,_0.11)] [&_button]:[border-radius:7px] [&_button]:[font-size:13px] [&_button]:[font-weight:620] [&_span]:[height:34px] [&_span]:[min-width:38px] [&_span]:[padding:0_10px] [&_span]:[color:inherit] [&_span]:[background:#1d1f25] [&_span]:[border:1px_solid_rgba(255,_255,_255,_0.11)] [&_span]:[border-radius:7px] [&_span]:[font-size:13px] [&_span]:[font-weight:620] [&_button]:[display:grid] [&_button]:[place-items:center] [&_svg]:[width:15px] [&_svg]:[height:15px] max-680:[top:14px] max-680:[bottom:auto] max-680:[left:14px]"
        aria-label="Canvas controls"
      >
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

      <Button
        className="workflow-execute [position:absolute] [left:50%] [bottom:18px] [transform:translateX(-50%)] [min-width:214px] [height:42px] [color:#ffffff] [background:linear-gradient(180deg,_#ff826d,_#f15f4d)] [border-color:rgba(255,_155,_136,_0.72)] max-680:[right:14px] max-680:[bottom:14px] max-680:[left:auto] max-680:[min-width:182px] max-680:[height:38px] max-680:[transform:none] max-680:[font-size:12px]"
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
  if (program === "swarmx-codex") return "codex";
  if (commandLine.includes("@agentclientprotocol/codex-acp")) return "codex";
  if (commandLine.includes("@agentclientprotocol/claude-agent-acp")) return "claude_code";
  if (commandLine.includes("pi-acp")) return "pi";

  return type || program || "custom";
}
