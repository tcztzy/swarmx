// @vitest-environment jsdom
import { RO_CRATE_CONTEXT, type RoCrateMetadataDocument } from "@swarmx/evidence";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { i18n } from "../src/renderer/i18n.js";
import { ObservePanel } from "../src/renderer/observe.js";
import { crateGraph } from "../src/renderer/research-graph.js";
import { SettingsPage } from "../src/renderer/settings.js";
import { DEFAULT_POLICY } from "../src/settings.js";
import { type BridgeHarness, installBridge } from "./bridge-support.js";

let gateway: BridgeHarness;
beforeEach(async () => {
  await i18n.changeLanguage("zh");
  gateway = installBridge();
  gateway.tool.mockImplementation((payload: { args: { action: string } }) =>
    Promise.resolve(
      payload.args.action === "memory_status"
        ? {
            action: "memory_status",
            data: {
              settings: {},
              note: { content: "", revision: `sha256:${"0".repeat(64)}`, limit: 1375 },
              pending: [],
              review: { state: "idle", message: "", sessionId: null },
            },
          }
        : { action: "graph_memory", data: { nodes: [], edges: [] } },
    ),
  );
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("saves active grants while preserving hidden inactive domain grants", async () => {
  const settings = {
    policy: {
      ...DEFAULT_POLICY,
      tools: ["memory.read", "memory.write", "science.read", "science.write"],
    },
    environment: null,
    cwd: "/work",
  };
  gateway.settingsRead.mockResolvedValue(settings);
  gateway.settingsUpdate.mockImplementation(async (policy: unknown) => ({ ...settings, policy }));
  render(<SettingsPage />);
  await screen.findByRole("button", { name: "保存权限" });
  expect(screen.queryByRole("heading", { name: "运行环境" })).toBeNull();
  expect(screen.queryByLabelText("CPU 核数")).toBeNull();
  expect(screen.queryByLabelText("读取领域引用")).toBeNull();
  expect(screen.queryByLabelText("旧版领域写入授权")).toBeNull();
  fireEvent.click(screen.getByLabelText("修改记忆"));
  fireEvent.click(screen.getByLabelText("允许通过 Swarm 委派任务"));
  fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
  await screen.findByText("权限已保存，对下一次执行生效。");
  expect(gateway.settingsUpdate).toHaveBeenCalledWith({
    ...DEFAULT_POLICY,
    tools: ["memory.read", "science.read", "science.write"],
    delegation: false,
  });
});

it("lets users enable and tighten generic project resource learning", async () => {
  const settings = {
    policy: { ...DEFAULT_POLICY, filesystem: "read-only" as const },
    environment: null,
    cwd: "/work",
  };
  gateway.settingsRead.mockResolvedValue(settings);
  gateway.settingsUpdate.mockImplementation(async (policy: unknown) => ({ ...settings, policy }));
  render(<SettingsPage />);
  const access = await screen.findByLabelText("项目提示词与技能文件权限");
  expect((access as HTMLSelectElement).value).toBe("read-only");
  fireEvent.change(access, { target: { value: "workspace-write" } });
  fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
  await screen.findByText("权限已保存，对下一次执行生效。");
  expect(gateway.settingsUpdate).toHaveBeenLastCalledWith({ ...DEFAULT_POLICY, delegation: true });
  fireEvent.change(access, { target: { value: "read-only" } });
  fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
  await screen.findByText("权限已保存，对下一次执行生效。");
  expect(gateway.settingsUpdate).toHaveBeenLastCalledWith({ ...settings.policy, delegation: true });
  expect(screen.queryByRole("heading", { name: "运行环境" })).toBeNull();
});

it("does not add inactive domain grants when saving memory permissions", async () => {
  const settings = {
    policy: { ...DEFAULT_POLICY, tools: ["memory.read"] },
    environment: null,
    cwd: "/work",
  };
  gateway.settingsRead.mockResolvedValue(settings);
  gateway.settingsUpdate.mockImplementation(async (policy: unknown) => ({ ...settings, policy }));
  render(<SettingsPage />);
  fireEvent.click(await screen.findByLabelText("修改记忆"));
  fireEvent.click(screen.getByRole("button", { name: "保存权限" }));
  await screen.findByText("权限已保存，对下一次执行生效。");
  expect(gateway.settingsUpdate).toHaveBeenCalledWith({
    ...DEFAULT_POLICY,
    tools: ["memory.read", "memory.write"],
    delegation: true,
  });
});

it("leaves a rejected policy save visible and retryable", async () => {
  gateway.settingsRead.mockResolvedValue({
    policy: DEFAULT_POLICY,
    environment: null,
    cwd: "/work",
  });
  gateway.settingsUpdate.mockRejectedValue(new Error("Policy could not be saved"));
  render(<SettingsPage />);
  const save = await screen.findByRole("button", { name: "保存权限" });
  fireEvent.click(save);
  expect((await screen.findByRole("alert")).textContent).toContain("Policy could not be saved");
  expect((save as HTMLButtonElement).disabled).toBe(false);
});

it("shows trace in a closable generic observation region", () => {
  const close = vi.fn();
  render(<ObservePanel trace={<p>Trace fixture</p>} onClose={close} />);
  expect(screen.getByRole("complementary", { name: "观测侧栏" })).toBeTruthy();
  expect(screen.getByText("Trace fixture")).toBeTruthy();
  expect(screen.queryByRole("tab", { name: "运行记录" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "关闭侧栏" }));
  expect(close).toHaveBeenCalledOnce();
});

it("preserves RO-Crate identities and relation names while filtering a selected neighborhood", () => {
  const document: RoCrateMetadataDocument = {
    "@context": RO_CRATE_CONTEXT,
    "@graph": [
      { "@id": "ro-crate-metadata.json", "@type": "CreativeWork", about: { "@id": "./" } },
      {
        "@id": "./",
        "@type": "Dataset",
        name: "Project",
        hasPart: [{ "@id": "urn:uuid:figure" }],
      },
      {
        "@id": "urn:uuid:figure",
        "@type": "ImageObject",
        name: "Response figure",
        isBasedOn: [{ "@id": "urn:uuid:data" }],
      },
      { "@id": "urn:uuid:data", "@type": "Dataset", name: "Measured data" },
      { "@id": "urn:uuid:other", "@type": "CreativeWork", name: "Unrelated note" },
    ],
  };
  const graph = crateGraph(document, "", "urn:uuid:figure", true);
  expect(graph.nodes.map((node) => node.id)).toEqual(["./", "urn:uuid:figure", "urn:uuid:data"]);
  expect(
    graph.edges.map(({ source, target, label }) => ({ source, target, label })),
  ).toContainEqual({ source: "urn:uuid:figure", target: "urn:uuid:data", label: "isBasedOn" });
  const searched = crateGraph(document, "MEASURED", "", false);
  expect(searched.nodes.map((node) => node.id)).toEqual(["urn:uuid:data"]);
  expect(searched.edges).toEqual([]);
});

it("bounds generic crate graphs to 200 entities without dangling edges", () => {
  const document: RoCrateMetadataDocument = {
    "@context": RO_CRATE_CONTEXT,
    "@graph": Array.from({ length: 205 }, (_, index) => ({
      "@id": `record:${index}`,
      "@type": "CreativeWork",
      name: `Record ${index}`,
      isBasedOn: { "@id": `record:${index + 1}` },
    })),
  };
  const graph = crateGraph(document, "", "", false);
  expect(graph.count).toBe(205);
  expect(graph.nodes).toHaveLength(200);
  expect(graph.edges).toHaveLength(199);
  expect(graph.nodes.every((node) => !node.deletable)).toBe(true);
  expect(crateGraph(document, "Record 204", "", false).nodes.map((node) => node.id)).toEqual([
    "record:204",
  ]);
});
