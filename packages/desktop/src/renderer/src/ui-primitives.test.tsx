/** @vitest-environment jsdom */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  Badge,
  Button,
  badgeVariants,
  doctorNoticeVariants,
  rightPanelVariants,
} from "./ui-primitives.js";

describe("Tailwind UI primitives", () => {
  it("maps semantic button variants to complete static utility classes", () => {
    render(
      <Button variant="destructive" size="sm">
        Delete
      </Button>,
    );

    const button = screen.getByRole("button", { name: "Delete" });
    expect([...button.classList]).toEqual(
      expect.arrayContaining(["bg-danger-muted", "text-danger", "h-[34px]", "px-[11px]"]),
    );
    expect([...button.classList]).not.toEqual(
      expect.arrayContaining(["button--destructive", "button--sm"]),
    );
  });

  it("keeps caller classes while applying a typed badge tone", () => {
    render(
      <Badge tone="success" className="test-hook">
        Ready
      </Badge>,
    );

    expect([...screen.getByText("Ready").classList]).toEqual(
      expect.arrayContaining(["test-hook", "bg-success-muted", "text-success"]),
    );
  });

  it("expresses loading badges through the same variant contract", () => {
    expect(badgeVariants({ tone: "loading" })).toContain(
      "[&_svg]:animate-[spin_900ms_linear_infinite]",
    );
  });

  it("keeps repeated panel and notice overrides inside typed variants", () => {
    expect(rightPanelVariants({ kind: "workspace" })).toContain("workspace-panel");
    expect(rightPanelVariants({ kind: "workspace" })).toContain("[padding:0]");
    expect(rightPanelVariants({ kind: "doctor" })).toContain("doctor-panel");
    expect(rightPanelVariants({ kind: "doctor" })).toContain("[gap:12px]");
    expect(doctorNoticeVariants({ tone: "error" })).toContain("doctor-notice--error");
    expect(doctorNoticeVariants({ tone: "error" })).toContain("[color:var(--danger)]");
  });
});
