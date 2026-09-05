import { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { ndJsonStream } from "@agentclientprotocol/sdk";
import { selectedAgent } from "./agent.js";
import { acpAgent } from "./host/acp.js";
import { startDesktopPlatform } from "./platform.js";

const { values } = parseArgs({ options: { agent: { type: "string" } } });
const platform = await startDesktopPlatform({
  workspaceRoot: process.env.SWARMX_WORKSPACE ?? process.cwd(),
  rendererRoot: fileURLToPath(new URL("./renderer", import.meta.url)),
  agentId: selectedAgent(values.agent),
});
process.stderr.write(`SwarmX A2A: ${platform.a2aUrl}\n`);
const connection = acpAgent(platform.agent, platform.workspaceRoot).connect(
  ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
);
process.once("SIGTERM", () => connection.close());
process.once("SIGINT", () => connection.close());
try {
  await connection.closed;
} finally {
  await platform.dispose();
}
