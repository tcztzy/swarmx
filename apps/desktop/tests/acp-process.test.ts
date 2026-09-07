import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import * as acp from "@agentclientprotocol/sdk";
import { expect, it } from "vitest";
import { connectAcpProcess } from "../src/agents/acp-process.js";

const require = createRequire(new URL("../package.json", import.meta.url));
const fixture = `
import { Readable, Writable } from 'node:stream';
import * as acp from ${JSON.stringify(pathToFileURL(require.resolve("@agentclientprotocol/sdk")).href)};
const connection = acp.agent().onRequest('initialize', () => ({protocolVersion: acp.PROTOCOL_VERSION, agentCapabilities: {}}))
  .onRequest('session/new', () => { process.exit(7); })
  .connect(acp.ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)));
connection.closed.then(() => process.exit(0));
`;
it("uses official SDK stdio framing and rejects requests when the process exits", async () => {
  const peer = await connectAcpProcess(
    acp.client(),
    process.execPath,
    ["--input-type=module", "-e", fixture],
    process.cwd(),
  );
  try {
    await expect(
      peer.connection.agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
      }),
    ).resolves.toMatchObject({ protocolVersion: acp.PROTOCOL_VERSION });
    await expect(
      peer.connection.agent.request(acp.methods.agent.session.new, {
        cwd: process.cwd(),
        mcpServers: [],
      }),
    ).rejects.toThrow();
  } finally {
    await peer.close();
  }
});

it("reports a missing executable instead of falling back", async () => {
  await expect(
    connectAcpProcess(acp.client(), "/swarmx/nonexistent-acp", [], process.cwd()),
  ).rejects.toThrow("ENOENT");
});

it.runIf(process.env.SWARMX_REAL_ACP_HANDSHAKE === "1").each(["codex-acp", "claude-agent-acp"])(
  "initializes packaged upstream %s over stdio without a model request",
  async (pkg) => {
    const peer = await connectAcpProcess(
      acp.client(),
      process.execPath,
      [require.resolve(`@agentclientprotocol/${pkg}/dist/index.js`)],
      process.cwd(),
      { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
    );
    try {
      const result = await peer.connection.agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
      });
      expect(result.protocolVersion).toBe(acp.PROTOCOL_VERSION);
      expect(result.agentInfo?.name).toBeTruthy();
      expect(result.agentCapabilities?.sessionCapabilities?.list).toBeDefined();
    } finally {
      await peer.close();
    }
  },
  30000,
);
