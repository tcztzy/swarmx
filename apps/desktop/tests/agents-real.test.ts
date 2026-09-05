import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { loadAgent } from "../src/agent.js";
import type { Observer } from "../src/agents/types.js";
import { ProductServices } from "../src/host/product-services.js";
import { startHost } from "../src/host/server.js";

it.runIf(process.env.SWARMX_REAL_CODEX === "1")(
  "real Codex App Server prompt and native history",
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
    try {
      await products.attachAgents(host.internalUrl, host.internalToken, undefined, "codex");
      const id = await products.rootAgent.create();
      const output: string[] = [];
      const observer: Observer = {
        text: (_id, text, role = "assistant") => {
          if (role === "assistant") output.push(text);
        },
        tool() {},
        raw() {},
        interact: async () => {
          throw new Error("The smoke prompt must not require permissions.");
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
    } finally {
      await products.dispose();
      await host.dispose();
      await rm(root, { recursive: true, force: true });
    }
  },
  60_000,
);

it.runIf(Boolean(process.env.SWARMX_HERMES_PYTHON))(
  "real Hermes native gateway session lifecycle (no model request)",
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
