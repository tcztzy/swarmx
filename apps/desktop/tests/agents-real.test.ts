import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EventType } from "@ag-ui/core";
import { expect, it, vi } from "vitest";
import { loadAgent } from "../src/agent.js";
import type { Observer } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";

it.runIf(process.env.SWARMX_REAL_CODEX === "1")(
  "real upstream Codex ACP history and MCP execution provenance across turns",
  async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "swarmx-native-codex-")));
    const products = await ProductServices.create({
      productHome: join(root, "product"),
      workspace: { id: "live", label: "Live", root },
    });
    const host = await startHost({
      products,
      rendererRoot: root,
      workspace: products.options.workspace,
    });
    const invoke = products.callTool.bind(products);
    const statusCall = vi.spyOn(products, "callTool").mockImplementation((name, args, context) => {
      expect(name).toBe("swarm");
      expect(args).toEqual({ action: "status" });
      return invoke(name, args, context);
    });
    try {
      await products.attachAgents(host.internalUrl, host.internalToken, undefined, "codex");
      const id = await products.rootAgent.create();
      const output: string[] = [];
      const toolResults: unknown[] = [];
      const observer: Observer = {
        text: (_id, text, role = "assistant") => {
          if (role === "assistant") output.push(text);
        },
        tool(_id, _name, _input, result) {
          if (result !== undefined) toolResults.push(result);
        },
        raw() {},
        interact: async (request) => {
          expect(request.title).toBe('Allow the swarmx MCP server to run tool "swarm"?');
          expect(request.schema).toEqual({ type: "object", properties: {} });
          return {};
        },
      };
      await products.rootAgent.read(id, observer);
      expect(output).toEqual([]);
      await products.rootAgent.start(
        id,
        "Reply with exactly SWARMX_NATIVE_OK. Do not use tools or change files.",
        observer,
      );
      expect(output.join("")).toContain("SWARMX_NATIVE_OK");
      output.length = 0;
      await products.rootAgent.read(id, observer);
      expect(output.join("")).toContain("SWARMX_NATIVE_OK");
      await products.rootAgent.start(
        id,
        'Call the swarmx MCP tool "swarm" with {"action":"status"} exactly once, then reply SWARMX_MCP_OK. Do not use other tools or change files.',
        observer,
      );
      expect(output.join("")).toContain("SWARMX_MCP_OK");
      expect(toolResults.length).toBeGreaterThan(0);
      const records = products.journal.read({ session: id, limit: 1000 }).events;
      const runs = records.filter(({ event }) => event.type === EventType.RUN_STARTED);
      expect(runs).toHaveLength(2);
      const calls = records.filter(
        ({ event, attributes }) =>
          event.type === EventType.TOOL_CALL_START && attributes["swarmx.actor.id"] === id,
      );
      expect(calls).toHaveLength(1);
      expect(statusCall).toHaveBeenCalledOnce();
      expect(calls[0]).toMatchObject({ runId: runs[1]?.runId, event: { toolCallName: "swarm" } });
      expect(records.some(({ event }) => event.type === EventType.RUN_ERROR)).toBe(false);
    } finally {
      statusCall.mockRestore();
      await products.dispose();
      await host.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
  120_000,
);

it.runIf(Boolean(process.env.SWARMX_HERMES_PYTHON))(
  "real Hermes ACP session lifecycle (no model request)",
  async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "swarmx-native-hermes-")));
    const agent = await loadAgent("hermes", {
      cwd: root,
      mcp: { url: "http://127.0.0.1:1/mcp", headers: {} },
    });
    try {
      expect(Array.isArray(await agent.list())).toBe(true);
      const id = await agent.create();
      expect(id).toMatch(/^hermes:/);
      await agent.read(id, { text() {}, tool() {}, raw() {}, interact: async () => undefined });
      await agent.interrupt(id);
    } finally {
      await agent.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
  60_000,
);
