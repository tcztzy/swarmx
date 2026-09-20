// @vitest-environment jsdom
import { RO_CRATE_CONTEXT } from "@swarmx/science/types";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { strFromU8, unzipSync } from "fflate";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { download } from "../src/renderer/bridge.js";
import { i18n } from "../src/renderer/i18n.js";
import { readConceptResult, SavedConcept } from "../src/renderer/saved-concept.js";
import { SourceInspection } from "../src/renderer/source-inspection.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

vi.mock("../src/renderer/bridge.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../src/renderer/bridge.js")>()),
  download: vi.fn(),
}));

const stored = {
  id: "writing-fit.md",
  revision: `sha256:${"a".repeat(64)}`,
  metadata: {
    title: "Writing fit",
    status: "draft",
    sources: [],
    swarmx_evaluation: {
      kind: "judgment",
      task: "Research writing",
      criteria: "Preserve the requested meaning",
      limitations: "One task",
    },
  },
};
const crate = {
  metadata: {
    "@context": RO_CRATE_CONTEXT,
    "@graph": [
      {
        "@id": "ro-crate-metadata.json",
        "@type": "CreativeWork",
        about: { "@id": "./" },
        conformsTo: { "@id": "https://w3id.org/ro/crate/1.3" },
      },
      {
        "@id": "./",
        "@type": "Dataset",
        name: "Selected execution evidence",
        description: "Private original records",
        datePublished: "2026-09-20T00:00:00Z",
        license: "Private",
        hasPart: [{ "@id": "evidence/records.json" }, { "@id": "concept.md" }],
      },
    ],
  },
  files: [
    {
      path: "evidence/records.json",
      content: JSON.stringify({ event: { original: "原始文本\nwith Unicode and \u0000 bytes" } }),
    },
    { path: "concept.md", content: "---\ntitle: Writing fit\n---\nExact saved body.\n" },
  ],
};
let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("en");
  gateway = installBridge();
  vi.mocked(download).mockClear();
  gateway.tool.mockResolvedValue({ action: "export_evaluation", data: crate });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function showConcept(value: unknown = stored) {
  const concept = readConceptResult({ action: "read_memory", data: value });
  if (!concept) throw new Error("Missing concept");
  return render(<SavedConcept concept={concept} />);
}

it("exports the exact concept revision with all original file payloads in a local RO-Crate ZIP", async () => {
  const clock = vi.spyOn(Date, "now").mockReturnValue(Date.UTC(2020, 0, 1));
  showConcept();
  expect(
    screen.getByText("Includes selected private source text. Downloads locally only."),
  ).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Export RO-Crate evidence" }));
  await waitFor(() => expect(download).toHaveBeenCalledTimes(1));
  expect(gateway.tool).toHaveBeenCalledExactlyOnceWith({
    requestId: expect.any(String),
    name: "memory",
    args: {
      action: "export_evaluation",
      request: { id: stored.id, expectedRevision: stored.revision },
    },
  });
  const [filename, bytes, mime] = vi.mocked(download).mock.calls[0] ?? [];
  expect(filename).toBe("writing-fit-ro-crate.zip");
  expect(mime).toBe("application/zip");
  if (!bytes || typeof bytes === "string") throw new Error("Missing ZIP bytes");
  const files = Object.fromEntries(
    Object.entries(unzipSync(bytes)).map(([path, content]) => [path, strFromU8(content)]),
  );
  expect(Object.keys(files).sort()).toEqual(
    ["ro-crate-metadata.json", ...crate.files.map(({ path }) => path)].sort(),
  );
  const metadata = files["ro-crate-metadata.json"];
  if (!metadata) throw new Error("Missing RO-Crate metadata");
  expect(JSON.parse(metadata)).toEqual(crate.metadata);
  for (const file of crate.files) expect(files[file.path]).toBe(file.content);
  clock.mockReturnValue(Date.UTC(2026, 8, 20));
  fireEvent.click(screen.getByRole("button", { name: "Export RO-Crate evidence" }));
  await waitFor(() => expect(download).toHaveBeenCalledTimes(2));
  expect(vi.mocked(download).mock.calls[1]?.[1]).toEqual(bytes);
});

it("disables export while pending and leaves Host errors visible without a partial download", async () => {
  let reject!: (error: Error) => void;
  gateway.tool.mockReturnValue(
    new Promise((_resolve, fail) => {
      reject = fail;
    }),
  );
  showConcept({ ...stored, evaluation: { status: "unverified", reason: "Source unavailable." } });
  const button = screen.getByRole<HTMLButtonElement>("button", {
    name: "Export RO-Crate evidence",
  });
  fireEvent.click(button);
  expect(button.disabled).toBe(true);
  fireEvent.click(button);
  expect(gateway.tool).toHaveBeenCalledTimes(1);
  reject(new Error("REVISION_CONFLICT: read the latest concept before exporting."));
  expect((await screen.findByRole("alert")).textContent).toContain("REVISION_CONFLICT");
  expect(button.disabled).toBe(false);
  expect(download).not.toHaveBeenCalled();
});

it("does not offer evaluation export without structured evaluation metadata", () => {
  showConcept({
    ...stored,
    metadata: { title: "Legacy", status: "draft", tags: ["agent-selection"], sources: [] },
  });
  expect(screen.queryByRole("button", { name: "Export RO-Crate evidence" })).toBeNull();
});

it("exports a loaded review snapshot directly without requiring a Memory concept", async () => {
  const id = "11111111-1111-4111-8111-111111111111";
  const source = { resource: `urn:swarmx:execution:${id}` };
  gateway.logsEvidence.mockResolvedValue({
    records: [
      {
        schemaVersion: 1,
        seq: 1,
        id,
        observedAt: "2026-09-20T00:00:00Z",
        workspaceId: "workspace",
        sessionId: null,
        runId: null,
        causedBy: null,
        attributes: {},
        event: {
          type: "CUSTOM",
          name: "swarmx.memory.review.started",
          value: { snapshot: { evidence: { records: [] } } },
        },
      },
    ],
    runs: [],
    statistics: {
      recipe: "swarmx.execution.v1",
      scope: "cited executions",
      runIds: [],
      window: { startedAt: null, finishedAt: null },
      sampleCount: 0,
      completed: 0,
      error: 0,
      cancelled: 0,
      incomplete: 0,
      other: 0,
      elapsed: { sampleCount: 0, medianMs: null },
      usage: { sampleCount: 0, inputTokens: null, outputTokens: null },
      cost: { sampleCount: 0, usd: null },
    },
  });
  render(
    <SourceInspection source={source} snapshot={undefined} executions={[]} onClose={() => {}} />,
  );
  expect(screen.queryByRole("button", { name: "Export RO-Crate evidence" })).toBeNull();
  fireEvent.click(await screen.findByRole("button", { name: "Export RO-Crate evidence" }));
  await waitFor(() => expect(download).toHaveBeenCalledTimes(1));
  expect(gateway.tool).toHaveBeenCalledWith(
    expect.objectContaining({
      name: "memory",
      args: { action: "export_evaluation", request: { source: source.resource } },
    }),
  );
  expect(vi.mocked(download).mock.calls[0]?.[0]).toBe(`review-${id}-ro-crate.zip`);
});
