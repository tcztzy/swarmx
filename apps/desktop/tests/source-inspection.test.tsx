// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { i18n } from "../src/renderer/i18n.js";
import { CopyButton, SourceInspection } from "../src/renderer/source-inspection.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

let gateway: BridgeHarness;
const copy = vi.fn();
beforeEach(async () => {
  await i18n.changeLanguage("en");
  gateway = installBridge();
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("clears copied feedback when the copied value changes", async () => {
  const { rerender } = render(<CopyButton value="old" label="Copy reference" />);
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  await screen.findByRole("button", { name: "Copied" });
  rerender(<CopyButton value="new" label="Copy reference" />);
  expect(screen.getByRole("button", { name: "Copy reference" })).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Copied" })).toBeNull();
  expect(copy).toHaveBeenCalledWith("old");
});

it("keeps external source references readable without invoking local domain operations", async () => {
  render(
    <SourceInspection
      source={{ resource: "sx:a/figure@1", title: "External source" }}
      onClose={() => {}}
    />,
  );
  expect((await screen.findByRole("alert")).textContent).toBe(
    "This source is managed by an external application. Copy its reference to inspect it there.",
  );
  expect(screen.getByText("External source")).toBeTruthy();
  expect(gateway.logsEvidence).not.toHaveBeenCalled();
  expect(gateway.tool).not.toHaveBeenCalled();
});
