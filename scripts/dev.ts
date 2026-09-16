import { type ChildProcess, spawn } from "node:child_process";
import { once } from "node:events";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const desktopRequire = createRequire(new URL("../apps/desktop/package.json", import.meta.url));
let child: ChildProcess | undefined;
let stopping = false;
const stop = () => {
  stopping = true;
  child?.kill("SIGTERM");
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);

async function run(command: string, args: string[], env = process.env) {
  if (stopping) return;
  child = spawn(command, args, { stdio: "inherit", env });
  const [code, signal] = await once(child, "exit");
  child = undefined;
  if (!stopping && code !== 0) throw new Error(`${command} exited with ${signal ?? code}.`);
}

const vite = await createServer({
  configFile: fileURLToPath(new URL("../apps/desktop/vite.config.ts", import.meta.url)),
  server: { host: "127.0.0.1", port: 0 },
});
try {
  await run("pnpm", ["generate:codex"]);
  await run(process.execPath, [
    desktopRequire.resolve("typescript/bin/tsc"),
    "-b",
    "tsconfig.host.json",
  ]);
  await run("pnpm", ["--filter", "./packages/**", "-r", "bundle"]);
  await vite.listen();
  const url = vite.resolvedUrls?.local[0];
  if (url === undefined) throw new Error("The Vite development server has no local URL.");
  await run(
    desktopRequire("electron") as string,
    [
      fileURLToPath(new URL("../apps/desktop/dist/main.js", import.meta.url)),
      ...process.argv.slice(2),
    ],
    { ...process.env, SWARMX_DEV: "1", SWARMX_RENDERER_URL: url },
  );
} finally {
  await vite.close();
  process.off("SIGINT", stop);
  process.off("SIGTERM", stop);
}
