import { spawn } from "node:child_process";
import { once } from "node:events";
import { Readable, Writable } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";

/** The upstream executable owns harness translation; the official SDK owns the wire. */
export async function connectAcpProcess(
  client: acp.ClientApp,
  command: string,
  args: string[],
  cwd: string,
  env = process.env,
) {
  const child = spawn(command, args, { cwd, env, stdio: ["pipe", "pipe", "inherit"] });
  await once(child, "spawn");
  const exited = new Promise<void>((resolve) => child.once("exit", () => resolve()));
  const connection = client.connect(
    acp.ndJsonStream(Writable.toWeb(child.stdin), Readable.toWeb(child.stdout)),
  );
  let closing: Promise<void> | undefined;
  const close = () =>
    (closing ??= (async () => {
      connection.close();
      child.stdin.end();
      const timeout = setTimeout(() => child.kill("SIGKILL"), 3000);
      timeout.unref();
      try {
        await exited;
      } finally {
        clearTimeout(timeout);
      }
    })());
  return { connection, close };
}
