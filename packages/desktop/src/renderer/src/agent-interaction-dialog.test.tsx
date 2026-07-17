// @vitest-environment jsdom

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AgentInteractionDialog } from "./agent-interaction-dialog.js";

describe("V390 AgentInteractionDialog", () => {
  it("V386 collects single, multiple, and automatic Other answers", () => {
    const onResolve = vi.fn();
    render(
      <AgentInteractionDialog
        interaction={{
          kind: "questions",
          requestId: "request-1",
          interactionId: "interaction-1",
          questions: [
            {
              question: "Which runtime?",
              header: "Runtime",
              options: [
                { label: "Node", description: "Use Node.js" },
                { label: "Bun", description: "Use Bun" },
              ],
              multiSelect: false,
            },
            {
              question: "Which features?",
              header: "Features",
              options: [
                { label: "Tests", description: "Add tests", preview: "pnpm test" },
                { label: "Docs", description: "Add documentation" },
              ],
              multiSelect: true,
            },
          ],
        }}
        resolving={false}
        error={null}
        onResolve={onResolve}
        onStop={vi.fn()}
      />,
    );

    expect((screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    fireEvent.click(screen.getByRole("radio", { name: /Node/ }));
    fireEvent.click(screen.getByRole("checkbox", { name: /Tests/ }));
    fireEvent.change(screen.getByRole("textbox", { name: "Features other answer" }), {
      target: { value: "Telemetry" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));

    expect(onResolve).toHaveBeenCalledWith({
      kind: "questions",
      answers: {
        "Which runtime?": "Node",
        "Which features?": "Tests, Telemetry",
      },
    });
  });

  it("V389 returns explicit plan approval or rejection feedback", () => {
    const onResolve = vi.fn();
    const { rerender } = render(
      <AgentInteractionDialog
        interaction={{
          kind: "plan_approval",
          requestId: "request-1",
          interactionId: "plan-1",
          plan: "# Plan\n\n1. Implement.",
          filePath: "/private/plan.md",
        }}
        resolving={false}
        error={null}
        onResolve={onResolve}
        onStop={vi.fn()}
      />,
    );
    expect(screen.getByText(/1\. Implement/)).not.toBeNull();
    fireEvent.change(screen.getByLabelText(/Feedback for another planning pass/), {
      target: { value: "Add rollback steps" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Keep planning" }));
    expect(onResolve).toHaveBeenLastCalledWith({
      kind: "plan_approval",
      approved: false,
      feedback: "Add rollback steps",
    });

    rerender(
      <AgentInteractionDialog
        interaction={{
          kind: "plan_approval",
          requestId: "request-1",
          interactionId: "plan-2",
          plan: "# Revised plan",
          filePath: "/private/plan.md",
        }}
        resolving={false}
        error={null}
        onResolve={onResolve}
        onStop={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Approve plan" }));
    expect(onResolve).toHaveBeenLastCalledWith({ kind: "plan_approval", approved: true });
  });
});
