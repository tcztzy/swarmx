import { ChevronDown, SquareTerminal } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

export interface AgentQuestionOption {
  label: string;
  description: string;
  preview?: string;
}

export interface AgentQuestion {
  question: string;
  header: string;
  options: AgentQuestionOption[];
  multiSelect: boolean;
}

export interface ToolApprovalOption {
  optionId: string;
  name: string;
  kind: "allow_once" | "allow_always" | "reject_once" | "reject_always";
}

export type AgentInteractionEvent =
  | {
      kind: "questions";
      requestId: string;
      interactionId: string;
      questions: AgentQuestion[];
    }
  | {
      kind: "plan_approval";
      requestId: string;
      interactionId: string;
      plan: string;
      filePath: string;
    }
  | {
      kind: "tool_approval";
      requestId: string;
      interactionId: string;
      title: string;
      toolKind?: string;
      source?: "direct" | "acp";
      policySourceIds?: string[];
      summary: string;
      options: ToolApprovalOption[];
    };

export type AgentInteractionResponse =
  | { kind: "questions"; answers: Record<string, string> }
  | { kind: "plan_approval"; approved: boolean; feedback?: string }
  | { kind: "tool_approval"; optionId: string };

interface AgentInteractionDialogProps {
  interaction: AgentInteractionEvent;
  resolving: boolean;
  error: string | null;
  onResolve: (response: AgentInteractionResponse) => void;
  onStop: () => void;
}

export function AgentInteractionDialog({
  interaction,
  resolving,
  error,
  onResolve,
  onStop,
}: AgentInteractionDialogProps) {
  if (interaction.kind === "questions") {
    return (
      <QuestionDialog
        interaction={interaction}
        resolving={resolving}
        error={error}
        onResolve={onResolve}
        onStop={onStop}
      />
    );
  }
  if (interaction.kind === "tool_approval") {
    return (
      <ToolApprovalDialog
        interaction={interaction}
        resolving={resolving}
        error={error}
        onResolve={onResolve}
        onStop={onStop}
      />
    );
  }
  return (
    <PlanApprovalDialog
      interaction={interaction}
      resolving={resolving}
      error={error}
      onResolve={onResolve}
      onStop={onStop}
    />
  );
}

function ToolApprovalDialog({
  interaction,
  resolving,
  error,
  onResolve,
}: AgentInteractionDialogProps & {
  interaction: Extract<AgentInteractionEvent, { kind: "tool_approval" }>;
}) {
  const rootRef = useRef<HTMLDialogElement>(null);
  const [expanded, setExpanded] = useState(false);
  const [allowMenuOpen, setAllowMenuOpen] = useState(false);
  const rejectOptions = interaction.options.filter((option) => option.kind.startsWith("reject"));
  const allowOptions = interaction.options.filter((option) => option.kind.startsWith("allow"));
  const primaryAllow =
    allowOptions.find((option) => option.kind === "allow_once") ?? allowOptions[0] ?? null;
  const approvalTitle =
    interaction.toolKind === "execute" && interaction.source === "direct"
      ? "Allow SwarmX to run this command?"
      : interaction.title;
  const hasLongSummary =
    interaction.summary.length > 180 || interaction.summary.split("\n").length > 3;

  useEffect(() => {
    if (!allowMenuOpen) return;
    const closeOnPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setAllowMenuOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAllowMenuOpen(false);
    };
    window.addEventListener("pointerdown", closeOnPointer);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeOnPointer);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [allowMenuOpen]);

  return (
    <dialog
      open
      ref={rootRef}
      className="agent-tool-approval [position:relative] [width:min(100%,_960px)] [min-height:218px] [margin:0_auto] [padding:20px_22px_18px] [display:flex] [flex-direction:column] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:22px] [box-shadow:var(--shadow),_var(--shadow-inset)] [animation:event-enter_var(--duration-med)_var(--ease-out)_both] [&_h2]:[margin:0] [&_h2]:[font-size:17px] [&_h2]:[font-weight:600] [&_h2]:[line-height:1.35] [&_h2]:[letter-spacing:-0.01em] max-680:[min-height:196px] max-680:[padding:17px_16px_15px] max-680:[border-radius:18px]"
      aria-labelledby="agent-tool-approval-title"
    >
      <header className="agent-tool-approval__header [display:grid] [gap:8px]">
        <span className="agent-tool-approval__kind [display:inline-flex] [align-items:center] [gap:9px] [color:var(--muted-foreground)] [font-size:13px] [font-weight:600] [&_svg]:[width:17px] [&_svg]:[height:17px]">
          <SquareTerminal aria-hidden="true" />
          {interaction.toolKind === "execute" ? "Terminal" : "Tool permission"}
        </span>
        <h2 id="agent-tool-approval-title">{approvalTitle}</h2>
      </header>
      <div className="agent-tool-approval__summary [margin:16px_6px_0] [display:grid] [justify-items:end] [gap:8px] [&_pre]:[width:100%] [&_pre]:[max-height:220px] [&_pre]:[margin:0] [&_pre]:[overflow:auto] [&_pre]:[color:var(--muted-foreground)] [&_pre]:[font:13px_/_1.48_var(--font-mono)] [&_pre]:[white-space:pre-wrap] [&_pre]:[overflow-wrap:anywhere] [&_pre.is-collapsed]:[max-height:4.65em] [&_pre.is-collapsed]:[overflow:hidden] [&_>_button]:[padding:2px_0] [&_>_button]:[color:var(--muted-foreground)] [&_>_button]:[background:transparent] [&_>_button]:[border:0] [&_>_button]:[font-size:12px] [&_>_button]:[font-weight:600] [&_>_button]:[cursor:pointer] max-680:[margin-top:16px]">
        <pre className={hasLongSummary && !expanded ? "is-collapsed" : undefined}>
          {interaction.summary}
        </pre>
        {hasLongSummary && (
          <button type="button" onClick={() => setExpanded((current) => !current)}>
            {expanded ? "Collapse" : "Expand"}
          </button>
        )}
      </div>
      {error && (
        <p
          className="agent-tool-approval__error [margin:12px_6px_0] [color:var(--danger)] [font-size:12px]"
          role="alert"
        >
          {error}
        </p>
      )}
      <footer className="agent-tool-approval__actions [position:relative] [margin-top:auto] [padding-top:12px] [display:flex] [align-items:center] [justify-content:flex-end] [gap:9px] [&_button]:[height:38px] [&_button]:[padding:0_16px] [&_button]:[color:var(--foreground)] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:999px] [&_button]:[font:inherit] [&_button]:[font-weight:600] [&_button]:[cursor:pointer]">
        {rejectOptions.map((option, index) => (
          <button
            type="button"
            className="agent-tool-approval__deny [background:transparent]"
            // biome-ignore lint/a11y/noAutofocus: Permission dialogs put keyboard focus on the safe default.
            autoFocus={index === 0}
            disabled={resolving}
            key={option.optionId}
            onClick={() => onResolve({ kind: "tool_approval", optionId: option.optionId })}
          >
            {option.kind === "reject_once" ? "Deny" : option.name}
          </button>
        ))}
        {primaryAllow && (
          <div
            className={String.raw`agent-tool-approval__allow [position:relative] [display:flex] [color:var(--primary-foreground)] [background:var(--primary)] [border-radius:999px] [&_.agent-tool-approval\_\_allow-primary]:[color:inherit] [&_.agent-tool-approval\_\_allow-primary]:[background:transparent] [&_.agent-tool-approval\_\_allow-primary]:[border-color:transparent] [&_.agent-tool-approval\_\_allow-menu-trigger]:[width:34px] [&_.agent-tool-approval\_\_allow-menu-trigger]:[padding:0_11px_0_3px] [&_.agent-tool-approval\_\_allow-menu-trigger]:[color:inherit] [&_.agent-tool-approval\_\_allow-menu-trigger]:[background:transparent] [&_.agent-tool-approval\_\_allow-menu-trigger]:[border-color:transparent] [&_.agent-tool-approval\_\_allow-menu-trigger]:[border-left-color:color-mix(in_srgb,_currentColor_20%,_transparent)] [&_.agent-tool-approval\_\_allow-menu-trigger]:[border-radius:0_999px_999px_0]`}
          >
            <button
              type="button"
              className="agent-tool-approval__allow-primary"
              disabled={resolving}
              onClick={() => onResolve({ kind: "tool_approval", optionId: primaryAllow.optionId })}
            >
              {resolving ? "Sending…" : primaryAllow.name}
            </button>
            {allowOptions.length > 1 && (
              <>
                <button
                  type="button"
                  className="agent-tool-approval__allow-menu-trigger [&_svg]:[width:15px] [&_svg]:[height:15px]"
                  aria-label="More allow options"
                  aria-haspopup="menu"
                  aria-expanded={allowMenuOpen}
                  disabled={resolving}
                  onClick={() => setAllowMenuOpen((current) => !current)}
                >
                  <ChevronDown aria-hidden="true" />
                </button>
                {allowMenuOpen && (
                  <div
                    className="agent-tool-approval__allow-menu [position:absolute] [z-index:44] [right:0] [bottom:calc(100%_+_8px)] [width:max-content] [min-width:224px] [padding:5px] [display:grid] [gap:2px] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:13px] [box-shadow:var(--shadow-soft),_var(--shadow-inset)] [&_button]:[width:100%] [&_button]:[min-height:42px] [&_button]:[padding:8px_12px] [&_button]:[color:inherit] [&_button]:[text-align:left] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:9px]"
                    role="menu"
                  >
                    {allowOptions.map((option) => (
                      <button
                        type="button"
                        role="menuitem"
                        key={option.optionId}
                        onClick={() => {
                          setAllowMenuOpen(false);
                          onResolve({ kind: "tool_approval", optionId: option.optionId });
                        }}
                      >
                        {option.name}
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </footer>
    </dialog>
  );
}

function QuestionDialog({
  interaction,
  resolving,
  error,
  onResolve,
  onStop,
}: AgentInteractionDialogProps & {
  interaction: Extract<AgentInteractionEvent, { kind: "questions" }>;
}) {
  const [selected, setSelected] = useState<Record<string, string[]>>({});
  const [otherEnabled, setOtherEnabled] = useState<Record<string, boolean>>({});
  const [otherText, setOtherText] = useState<Record<string, string>>({});
  const answers = useMemo(
    () => questionAnswers(interaction.questions, selected, otherEnabled, otherText),
    [interaction.questions, otherEnabled, otherText, selected],
  );
  const complete = Object.keys(answers).length === interaction.questions.length;

  const selectOption = (question: AgentQuestion, label: string, checked: boolean) => {
    setSelected((current) => {
      if (!question.multiSelect) return { ...current, [question.question]: checked ? [label] : [] };
      const values = new Set(current[question.question] ?? []);
      if (checked) values.add(label);
      else values.delete(label);
      return { ...current, [question.question]: [...values] };
    });
    if (!question.multiSelect && checked) {
      setOtherEnabled((current) => ({ ...current, [question.question]: false }));
    }
  };

  const selectOther = (question: AgentQuestion, checked: boolean) => {
    setOtherEnabled((current) => ({ ...current, [question.question]: checked }));
    if (!question.multiSelect && checked) {
      setSelected((current) => ({ ...current, [question.question]: [] }));
    }
  };

  return (
    <div className="agent-interaction-backdrop [position:fixed] [z-index:150] [inset:0] [padding:24px] [display:grid] [place-items:center] [background:rgba(0,_0,_0,_0.5)] [-webkit-backdrop-filter:blur(8px)] max-680:[padding:10px]">
      <dialog
        open
        className="agent-interaction-dialog [width:min(100%,_760px)] [max-height:min(88vh,_900px)] [margin:0] [padding:0] [overflow:auto] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:24px] [box-shadow:0_28px_100px_rgba(0,_0,_0,_0.48),_var(--shadow-inset)] [&_>_header]:[padding:26px_28px_18px] [&_>_header]:[border-bottom:1px_solid_var(--border)] [&_>_header_>_span]:[display:block] [&_>_header_>_span]:[margin-bottom:6px] [&_>_header_>_span]:[color:var(--muted-foreground)] [&_>_header_>_span]:[font-size:11px] [&_>_header_>_span]:[font-weight:700] [&_>_header_>_span]:[letter-spacing:0.08em] [&_>_header_>_span]:[text-transform:uppercase] [&_legend_>_span]:[display:block] [&_legend_>_span]:[margin-bottom:6px] [&_legend_>_span]:[color:var(--muted-foreground)] [&_legend_>_span]:[font-size:11px] [&_legend_>_span]:[font-weight:700] [&_legend_>_span]:[letter-spacing:0.08em] [&_legend_>_span]:[text-transform:uppercase] [&_h2]:[margin:0] [&_h2]:[font-size:25px] [&_h2]:[line-height:1.2] [&_>_header_p]:[margin:9px_0_0] [&_>_header_p]:[color:var(--muted-foreground)] [&_>_header_p]:[font-size:14px] [&_form]:[display:grid] [&_fieldset]:[margin:0] [&_fieldset]:[padding:22px_0] [&_fieldset]:[border:0] [&_fieldset_+_fieldset]:[border-top:1px_solid_var(--border)] [&_legend]:[width:100%] [&_legend]:[padding:0] [&_legend]:[font-size:16px] [&_legend]:[font-weight:650] [&_footer]:[padding:16px_28px_22px] [&_footer]:[display:flex] [&_footer]:[justify-content:flex-end] [&_footer]:[gap:10px] [&_footer]:[border-top:1px_solid_var(--border)] [&_footer_>_span]:[flex:1] [&_footer_button]:[min-width:96px] [&_footer_button]:[height:40px] [&_footer_button]:[padding:0_16px] [&_footer_button]:[color:var(--foreground)] [&_footer_button]:[background:transparent] [&_footer_button]:[border:1px_solid_var(--border)] [&_footer_button]:[border-radius:11px] [&_footer_button]:[font:inherit] [&_footer_button]:[font-weight:600] [&_footer_button]:[cursor:pointer] [&_footer_button.is-primary]:[color:var(--primary-foreground,_#09090b)] [&_footer_button.is-primary]:[background:var(--foreground)] [&_footer_button.is-primary]:[border-color:var(--foreground)] max-680:[&_footer]:[flex-wrap:wrap]"
        aria-modal="true"
        aria-labelledby="agent-interaction-title"
      >
        <header>
          <span>Claude needs your input</span>
          <h2 id="agent-interaction-title">Choose an answer</h2>
          <p>The task is paused until every question has an answer.</p>
        </header>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (complete) onResolve({ kind: "questions", answers });
          }}
        >
          <div className="agent-interaction-dialog__questions [padding:4px_28px_10px]">
            {interaction.questions.map((question, questionIndex) => (
              <fieldset key={question.question}>
                <legend>
                  <span>{question.header}</span>
                  {question.question}
                </legend>
                <div className="agent-interaction-dialog__options [margin-top:14px] [display:grid] [grid-template-columns:repeat(2,_minmax(0,_1fr))] [gap:10px] [&_>_label]:[min-width:0] [&_>_label]:[padding:13px] [&_>_label]:[display:flex] [&_>_label]:[align-items:flex-start] [&_>_label]:[gap:10px] [&_>_label]:[background:var(--card)] [&_>_label]:[border:1px_solid_var(--border)] [&_>_label]:[border-radius:14px] [&_>_label]:[cursor:pointer] [&_>_label.is-selected]:[background:var(--card-hover)] [&_>_label.is-selected]:[border-color:var(--ring)] [&_>_label.is-selected]:[box-shadow:0_0_0_2px_rgba(149,_233,_255,_0.08)] [&_>_label_>_input]:[margin-top:3px] [&_>_label_>_input]:[accent-color:var(--foreground)] [&_>_label_>_span]:[min-width:0] [&_>_label_>_span]:[display:grid] [&_>_label_>_span]:[gap:4px] [&_strong]:[font-size:14px] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:12px] [&_small]:[line-height:1.4] [&_pre]:[max-height:140px] [&_pre]:[margin:7px_0_0] [&_pre]:[padding:9px] [&_pre]:[overflow:auto] [&_pre]:[background:var(--input)] [&_pre]:[border-radius:8px] [&_pre]:[white-space:pre-wrap] [&_input[type='text']]:[width:100%] [&_input[type='text']]:[padding:9px_10px] [&_input[type='text']]:[color:var(--foreground)] [&_input[type='text']]:[background:var(--input)] [&_input[type='text']]:[border:1px_solid_var(--border)] [&_input[type='text']]:[border-radius:9px] [&_input[type='text']]:[font:inherit] [&_input[type='text']]:[outline:none] max-680:[grid-template-columns:1fr]">
                  {question.options.map((option, optionIndex) => {
                    const id = `${interaction.interactionId}-${questionIndex}-${optionIndex}`;
                    const checked = (selected[question.question] ?? []).includes(option.label);
                    return (
                      <label
                        key={option.label}
                        htmlFor={id}
                        className={checked ? "is-selected" : ""}
                      >
                        <input
                          id={id}
                          type={question.multiSelect ? "checkbox" : "radio"}
                          name={`question-${questionIndex}`}
                          checked={checked}
                          disabled={resolving}
                          onChange={(event) =>
                            selectOption(question, option.label, event.target.checked)
                          }
                        />
                        <span>
                          <strong>{option.label}</strong>
                          <small>{option.description}</small>
                          {option.preview && <pre>{option.preview}</pre>}
                        </span>
                      </label>
                    );
                  })}
                  <label className={otherEnabled[question.question] ? "is-selected" : ""}>
                    <input
                      type={question.multiSelect ? "checkbox" : "radio"}
                      name={`question-${questionIndex}`}
                      checked={otherEnabled[question.question] ?? false}
                      disabled={resolving}
                      onChange={(event) => selectOther(question, event.target.checked)}
                    />
                    <span>
                      <strong>Other</strong>
                      <small>Type a different answer</small>
                      <input
                        type="text"
                        aria-label={`${question.header} other answer`}
                        value={otherText[question.question] ?? ""}
                        disabled={resolving}
                        onFocus={() => selectOther(question, true)}
                        onChange={(event) => {
                          setOtherText((current) => ({
                            ...current,
                            [question.question]: event.target.value,
                          }));
                          selectOther(question, true);
                        }}
                      />
                    </span>
                  </label>
                </div>
              </fieldset>
            ))}
          </div>
          {error && (
            <p
              className="agent-interaction-dialog__error [margin:0_28px_14px] [color:var(--danger)] [font-size:13px]"
              role="alert"
            >
              {error}
            </p>
          )}
          <footer>
            <button type="button" disabled={resolving} onClick={onStop}>
              Stop task
            </button>
            <button type="submit" className="is-primary" disabled={resolving || !complete}>
              {resolving ? "Sending…" : "Continue"}
            </button>
          </footer>
        </form>
      </dialog>
    </div>
  );
}

function PlanApprovalDialog({
  interaction,
  resolving,
  error,
  onResolve,
  onStop,
}: AgentInteractionDialogProps & {
  interaction: Extract<AgentInteractionEvent, { kind: "plan_approval" }>;
}) {
  const [feedback, setFeedback] = useState("");
  return (
    <div className="agent-interaction-backdrop [position:fixed] [z-index:150] [inset:0] [padding:24px] [display:grid] [place-items:center] [background:rgba(0,_0,_0,_0.5)] [-webkit-backdrop-filter:blur(8px)] max-680:[padding:10px]">
      <dialog
        open
        className="agent-interaction-dialog agent-interaction-dialog--plan [width:min(100%,_760px)] [max-height:min(88vh,_900px)] [margin:0] [padding:0] [overflow:auto] [color:var(--foreground)] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:24px] [box-shadow:0_28px_100px_rgba(0,_0,_0,_0.48),_var(--shadow-inset)] [&_>_header]:[padding:26px_28px_18px] [&_>_header]:[border-bottom:1px_solid_var(--border)] [&_>_header_>_span]:[display:block] [&_>_header_>_span]:[margin-bottom:6px] [&_>_header_>_span]:[color:var(--muted-foreground)] [&_>_header_>_span]:[font-size:11px] [&_>_header_>_span]:[font-weight:700] [&_>_header_>_span]:[letter-spacing:0.08em] [&_>_header_>_span]:[text-transform:uppercase] [&_legend_>_span]:[display:block] [&_legend_>_span]:[margin-bottom:6px] [&_legend_>_span]:[color:var(--muted-foreground)] [&_legend_>_span]:[font-size:11px] [&_legend_>_span]:[font-weight:700] [&_legend_>_span]:[letter-spacing:0.08em] [&_legend_>_span]:[text-transform:uppercase] [&_h2]:[margin:0] [&_h2]:[font-size:25px] [&_h2]:[line-height:1.2] [&_>_header_p]:[margin:9px_0_0] [&_>_header_p]:[color:var(--muted-foreground)] [&_>_header_p]:[font-size:14px] [&_form]:[display:grid] [&_fieldset]:[margin:0] [&_fieldset]:[padding:22px_0] [&_fieldset]:[border:0] [&_fieldset_+_fieldset]:[border-top:1px_solid_var(--border)] [&_legend]:[width:100%] [&_legend]:[padding:0] [&_legend]:[font-size:16px] [&_legend]:[font-weight:650] [&_footer]:[padding:16px_28px_22px] [&_footer]:[display:flex] [&_footer]:[justify-content:flex-end] [&_footer]:[gap:10px] [&_footer]:[border-top:1px_solid_var(--border)] [&_footer_>_span]:[flex:1] [&_footer_button]:[min-width:96px] [&_footer_button]:[height:40px] [&_footer_button]:[padding:0_16px] [&_footer_button]:[color:var(--foreground)] [&_footer_button]:[background:transparent] [&_footer_button]:[border:1px_solid_var(--border)] [&_footer_button]:[border-radius:11px] [&_footer_button]:[font:inherit] [&_footer_button]:[font-weight:600] [&_footer_button]:[cursor:pointer] [&_footer_button.is-primary]:[color:var(--primary-foreground,_#09090b)] [&_footer_button.is-primary]:[background:var(--foreground)] [&_footer_button.is-primary]:[border-color:var(--foreground)] max-680:[&_footer]:[flex-wrap:wrap]"
        aria-modal="true"
        aria-labelledby="agent-plan-title"
      >
        <header>
          <span>Plan mode</span>
          <h2 id="agent-plan-title">Review Claude's plan</h2>
          <p>Approval exits read-only plan mode and allows implementation to begin.</p>
        </header>
        <div className="agent-interaction-dialog__plan [margin:22px_28px_16px] [display:grid] [gap:8px] [&_pre]:[max-height:46vh] [&_pre]:[margin:0] [&_pre]:[padding:18px] [&_pre]:[overflow:auto] [&_pre]:[color:var(--foreground)] [&_pre]:[background:var(--input)] [&_pre]:[border:1px_solid_var(--border)] [&_pre]:[border-radius:14px] [&_pre]:[font:13px_/_1.55_ui-monospace,_SFMono-Regular,_Menlo,_monospace] [&_pre]:[white-space:pre-wrap] [&_small]:[overflow:hidden] [&_small]:[color:var(--muted-foreground)] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap]">
          <pre>{interaction.plan}</pre>
          <small title={interaction.filePath}>{interaction.filePath}</small>
        </div>
        <label className="agent-interaction-dialog__feedback [margin:0_28px_20px] [display:grid] [gap:8px] [color:var(--muted-foreground)] [font-size:13px] [&_textarea]:[width:100%] [&_textarea]:[padding:9px_10px] [&_textarea]:[color:var(--foreground)] [&_textarea]:[background:var(--input)] [&_textarea]:[border:1px_solid_var(--border)] [&_textarea]:[border-radius:9px] [&_textarea]:[font:inherit] [&_textarea]:[outline:none] [&_textarea]:[min-height:78px] [&_textarea]:[resize:vertical]">
          Feedback for another planning pass (optional)
          <textarea
            value={feedback}
            disabled={resolving}
            onChange={(event) => setFeedback(event.target.value)}
          />
        </label>
        {error && (
          <p
            className="agent-interaction-dialog__error [margin:0_28px_14px] [color:var(--danger)] [font-size:13px]"
            role="alert"
          >
            {error}
          </p>
        )}
        <footer>
          <button type="button" disabled={resolving} onClick={onStop}>
            Stop task
          </button>
          <span />
          <button
            type="button"
            disabled={resolving}
            onClick={() =>
              onResolve({
                kind: "plan_approval",
                approved: false,
                ...(feedback.trim() ? { feedback: feedback.trim() } : {}),
              })
            }
          >
            Keep planning
          </button>
          <button
            type="button"
            className="is-primary"
            disabled={resolving}
            onClick={() => onResolve({ kind: "plan_approval", approved: true })}
          >
            {resolving ? "Sending…" : "Approve plan"}
          </button>
        </footer>
      </dialog>
    </div>
  );
}

function questionAnswers(
  questions: readonly AgentQuestion[],
  selected: Readonly<Record<string, string[]>>,
  otherEnabled: Readonly<Record<string, boolean>>,
  otherText: Readonly<Record<string, string>>,
): Record<string, string> {
  const answers: Record<string, string> = {};
  for (const question of questions) {
    const values = [...(selected[question.question] ?? [])];
    if (otherEnabled[question.question] && otherText[question.question]?.trim()) {
      values.push(otherText[question.question].trim());
    }
    if (values.length > 0) answers[question.question] = values.join(", ");
  }
  return answers;
}
