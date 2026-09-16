import { Readable, Writable } from "node:stream";
import { parseArgs } from "node:util";
import { ndJsonStream } from "@agentclientprotocol/sdk";
import { selectedAgent } from "./agent.js";
import { startDesktopPlatform } from "./platform.js";

const { values } = parseArgs({ options: { agent: { type: "string" } } });
const platform = await startDesktopPlatform({
  cwd: process.env.SWARMX_CWD ?? process.cwd(),
  agentId: selectedAgent(values.agent),
});
process.stderr.write(`SwarmX A2A: ${platform.a2aUrl}\n`);
const connection = platform.acp.connect(
  ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)),
);
process.once("SIGTERM", () => connection.close());
process.once("SIGINT", () => connection.close());
try {
  await connection.closed;
} finally {
  await platform.dispose();
}
