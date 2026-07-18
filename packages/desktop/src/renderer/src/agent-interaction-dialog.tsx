import type React from "react";
import { useMemo, useState } from "react";

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
  onStop,
}: AgentInteractionDialogProps & {
  interaction: Extract<AgentInteractionEvent, { kind: "tool_approval" }>;
}) {
  const rejectOptions = interaction.options.filter((option) => option.kind.startsWith("reject"));
  const allowOptions = interaction.options.filter((option) => option.kind.startsWith("allow"));
  return (
    <div className="agent-interaction-backdrop">
      <dialog
        open
        className="agent-interaction-dialog agent-interaction-dialog--tool"
        aria-modal="true"
        aria-labelledby="agent-tool-approval-title"
      >
        <header>
          <span>Permission request</span>
          <h2 id="agent-tool-approval-title">{interaction.title}</h2>
          <p>This action is paused until you choose one of the offered permissions.</p>
        </header>
        <div className="agent-interaction-dialog__plan">
          <pre>{interaction.summary}</pre>
          {interaction.toolKind && <small>Tool kind: {interaction.toolKind}</small>}
        </div>
        {error && (
          <p className="agent-interaction-dialog__error" role="alert">
            {error}
          </p>
        )}
        <footer>
          <button type="button" disabled={resolving} onClick={onStop}>
            Stop task
          </button>
          {rejectOptions.map((option) => (
            <button
              type="button"
              disabled={resolving}
              key={option.optionId}
              onClick={() => onResolve({ kind: "tool_approval", optionId: option.optionId })}
            >
              {option.name}
            </button>
          ))}
          {allowOptions.map((option) => (
            <button
              type="button"
              className="is-primary"
              disabled={resolving}
              key={option.optionId}
              onClick={() => onResolve({ kind: "tool_approval", optionId: option.optionId })}
            >
              {resolving ? "Sending…" : option.name}
            </button>
          ))}
        </footer>
      </dialog>
    </div>
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
    <div className="agent-interaction-backdrop">
      <dialog
        open
        className="agent-interaction-dialog"
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
          <div className="agent-interaction-dialog__questions">
            {interaction.questions.map((question, questionIndex) => (
              <fieldset key={question.question}>
                <legend>
                  <span>{question.header}</span>
                  {question.question}
                </legend>
                <div className="agent-interaction-dialog__options">
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
            <p className="agent-interaction-dialog__error" role="alert">
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
    <div className="agent-interaction-backdrop">
      <dialog
        open
        className="agent-interaction-dialog agent-interaction-dialog--plan"
        aria-modal="true"
        aria-labelledby="agent-plan-title"
      >
        <header>
          <span>Plan mode</span>
          <h2 id="agent-plan-title">Review Claude's plan</h2>
          <p>Approval exits read-only plan mode and allows implementation to begin.</p>
        </header>
        <div className="agent-interaction-dialog__plan">
          <pre>{interaction.plan}</pre>
          <small title={interaction.filePath}>{interaction.filePath}</small>
        </div>
        <label className="agent-interaction-dialog__feedback">
          Feedback for another planning pass (optional)
          <textarea
            value={feedback}
            disabled={resolving}
            onChange={(event) => setFeedback(event.target.value)}
          />
        </label>
        {error && (
          <p className="agent-interaction-dialog__error" role="alert">
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
