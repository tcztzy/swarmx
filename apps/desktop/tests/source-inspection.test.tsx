// @vitest-environment jsdom
import { createResearchObject } from "@swarmx/science";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { i18n } from "../src/renderer/i18n.js";
import { ResearchPanel } from "../src/renderer/research.js";
import { CopyButton, SourceInspection } from "../src/renderer/source-inspection.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

const identity = {
  createdAt: 0,
  updatedAt: 0,
  revision: 1,
  provenance: { eventId: "event", journalSeq: 1, sessionId: "renderer" },
};
const input = {
  ...identity,
  id: "input",
  projectId: "project",
  kind: "dataset" as const,
  title: "Measured input",
  digest: `sha256:${"a".repeat(64)}`,
  mime: "text/csv",
  size: 10,
  creator: { kind: "session" as const, sessionId: "renderer" },
  runId: null,
  environment: {},
  license: null,
  sourceEntityIds: [],
};
const figure = {
  ...input,
  id: "figure",
  kind: "figure" as const,
  title: "Registered figure",
  mime: "image/png",
  sourceEntityIds: ["input"],
};
const code = {
  id: "code",
  kind: "code" as const,
  source: "print('measured output')",
  executionCount: 1,
  executionTimeMs: 10,
  inputArtifactIds: ["input"],
  outputArtifactIds: ["figure"],
  runtimeEnvironment: {},
  relatedClaimIds: [],
  relatedExperimentIds: [],
  outputs: [],
};
const notebook = {
  ...identity,
  id: "plot",
  projectId: "project",
  kind: "notebook" as const,
  title: "Plotting run",
  cells: [code],
};
const snapshot = {
  projects: [
    {
      ...identity,
      id: "project",
      kind: "project" as const,
      title: "Project",
    },
  ],
  artifacts: [input, figure],
  notebooks: [
    notebook,
    { ...notebook, id: "check", title: "Input check", cells: [{ ...code, outputArtifactIds: [] }] },
  ],
  documents: [],
  figures: [],
  records: [],
  relations: [],
  experiments: [],
  runs: [],
  exports: [],
};
const execution = {
  id: "execution",
  notebookId: "plot",
  cellId: "code",
  source: code.source,
  executionCount: 1,
  status: "succeeded" as const,
  stdout: { text: "measured output", truncated: false },
  stderr: { text: "", truncated: false },
  outputs: [],
  exitCode: 0,
  signal: null,
  durationMs: 10,
  environment: { pythonVersion: "3.13" },
  inputArtifactIds: ["input"],
  artifact: figure,
  provenance: identity.provenance,
};
const executions = [
  {
    ...execution,
    id: "check-execution",
    notebookId: "check",
    artifact: null,
    status: "failed" as const,
    exitCode: 1,
    stderr: { text: "Check failed", truncated: false },
  },
  execution,
  {
    ...execution,
    id: "unrelated",
    notebookId: "unrelated",
    inputArtifactIds: ["other-input"],
    artifact: null,
  },
];
const copy = vi.fn().mockResolvedValue(undefined);
let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("en");
  gateway = installBridge();
  gateway.scienceArtifactPreview.mockResolvedValue({
    kind: "image",
    artifactId: figure.id,
    digest: figure.digest,
    size: figure.size,
    mime: "image/png",
    dataUrl: "data:image/png;base64,AAAA",
  });
  gateway.scienceArtifactContent.mockResolvedValue({
    name: "Registered figure.png",
    mime: "image/png",
    bytes: new Uint8Array([1, 2, 3]),
  });
  URL.createObjectURL = vi.fn(() => "blob:swarmx");
  URL.revokeObjectURL = vi.fn();
  Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: copy } });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

it("inspects pinned output and shared-input computations without executing or changing records", async () => {
  render(
    <SourceInspection
      source={{ resource: "sx:a/figure@1", title: "Registered figure" }}
      snapshot={snapshot}
      executions={executions}
      onClose={vi.fn()}
    />,
  );
  await screen.findByRole("img", { name: "Scientific figure preview" });
  expect(screen.getByText("Input artifact").nextElementSibling?.textContent).toBe("input");
  fireEvent.click(screen.getByRole("button", { name: "Copy full digest" }));
  await waitFor(() => expect(copy).toHaveBeenCalledWith(input.digest));
  fireEvent.click(screen.getByRole("button", { name: "View original output" }));
  await waitFor(() =>
    expect(gateway.scienceArtifactContent).toHaveBeenCalledWith({ id: figure.id }),
  );
  expect(gateway.tool).not.toHaveBeenCalled();
  expect(screen.getAllByRole("button", { expanded: true })[0]?.textContent).toContain(
    "Plotting run",
  );
  expect(screen.getByText("measured output")).toBeTruthy();
  fireEvent.keyDown(screen.getByRole("tab", { name: "Code", exact: true }), { key: "Enter" });
  expect(screen.getByText(code.source)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
  await waitFor(() => expect(copy).toHaveBeenCalledWith(code.source));
  fireEvent.keyDown(screen.getByRole("tab", { name: "Environment", exact: true }), {
    key: "Enter",
  });
  expect(screen.getByRole("tabpanel").textContent).toContain("3.13");
  fireEvent.keyDown(screen.getByRole("tab", { name: "Environment", exact: true }), { key: "Home" });
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole("tab", { name: "Code", exact: true })),
  );
  expect(screen.getByRole("tab", { name: "Code", exact: true }).getAttribute("aria-selected")).toBe(
    "true",
  );
  fireEvent.click(screen.getByRole("button", { name: /Input check/ }));
  const check = screen.getByRole("button", { name: /Input check/ }).closest("article");
  if (!check) throw new Error("Missing computation card");
  expect(within(check).getByText("Failed")).toBeTruthy();
  expect(within(check).getByRole("tabpanel").textContent).toContain("Check failed");
  expect(screen.queryByText("unrelated")).toBeNull();
});

it("shows and copies the executed source after the current notebook changes", async () => {
  render(
    <SourceInspection
      source={{ resource: "sx:a/figure@1" }}
      snapshot={{
        ...snapshot,
        notebooks: [
          { ...notebook, cells: [{ ...code, source: "print('later unexecuted edit')" }] },
        ],
      }}
      executions={[execution]}
      onClose={vi.fn()}
    />,
  );
  await screen.findByRole("img", { name: "Scientific figure preview" });
  fireEvent.keyDown(screen.getByRole("tab", { name: "Code", exact: true }), { key: "Enter" });
  expect(screen.getByRole("tabpanel").textContent).toContain(code.source);
  expect(screen.getByRole("tabpanel").textContent).not.toContain("later unexecuted edit");
  fireEvent.click(screen.getByRole("button", { name: "Copy code" }));
  await waitFor(() => expect(copy).toHaveBeenCalledWith(code.source));
});

it("reports a changed pinned revision instead of displaying the current artifact", () => {
  render(
    <SourceInspection
      source={{ resource: "sx:a/figure@2" }}
      snapshot={snapshot}
      executions={executions}
      onClose={vi.fn()}
    />,
  );
  expect(screen.getByRole("alert").textContent).toContain("revision 2");
  expect(screen.queryByRole("button", { name: "View original output" })).toBeNull();
  expect(gateway.scienceArtifactPreview).not.toHaveBeenCalled();
});

it("requests the pinned artifact's producer beyond recent history and reloads when the source changes", async () => {
  gateway.scienceWorkspace.mockResolvedValue(snapshot);
  gateway.scienceResearchObject.mockResolvedValue(createResearchObject(snapshot, "project"));
  gateway.scienceNotebookExecutions.mockImplementation(async ({ includeArtifactId }) =>
    includeArtifactId === figure.id ? [execution] : [],
  );
  const { rerender } = render(
    <ResearchPanel
      mode="observe"
      target={{ source: { resource: "sx:a/figure@1" } }}
      onClose={vi.fn()}
    />,
  );
  await screen.findByText("measured output");
  fireEvent.keyDown(screen.getByRole("tab", { name: "Code", exact: true }), { key: "Enter" });
  expect(screen.getByText(code.source)).toBeTruthy();
  fireEvent.keyDown(screen.getByRole("tab", { name: "Environment", exact: true }), {
    key: "Enter",
  });
  expect(screen.getByRole("tabpanel").textContent).toContain("3.13");
  rerender(
    <ResearchPanel
      mode="observe"
      target={{ source: { resource: "sx:a/input@1" } }}
      onClose={vi.fn()}
    />,
  );
  await waitFor(() =>
    expect(gateway.scienceNotebookExecutions).toHaveBeenCalledWith({
      projectId: "project",
      includeArtifactId: input.id,
    }),
  );
  await screen.findByText("No recorded computations are linked to this source.");
  expect(gateway.tool).not.toHaveBeenCalled();
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

it("does not treat a logical reference as a pinned revision and translates its error", async () => {
  render(
    <SourceInspection
      source={{ resource: "sx:a/figure" }}
      snapshot={snapshot}
      executions={executions}
      onClose={vi.fn()}
    />,
  );
  expect(screen.getByRole("alert").textContent).toContain("pinned revision");
  expect(gateway.scienceArtifactPreview).not.toHaveBeenCalled();
  await i18n.changeLanguage("zh");
  await screen.findByText("此来源未指定固定版本，无法作为固定版本检查。");
});
