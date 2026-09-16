import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const destination = fileURLToPath(
  new URL("../apps/desktop/src/agents/generated/", import.meta.url),
);
const temporary = await mkdtemp(join(tmpdir(), "swarmx-codex-types-"));
try {
  execFileSync(
    process.env.CODEX_PATH ?? fileURLToPath(import.meta.resolve("@openai/codex/bin/codex.js")),
    ["app-server", "generate-ts", "--experimental", "--out", temporary],
    { stdio: "inherit" },
  );
  await rm(destination, { recursive: true, force: true });
  for (const entry of await readdir(temporary, { recursive: true, withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith(".ts")) continue;
    const source = join(entry.parentPath, entry.name);
    const target = join(destination, source.slice(temporary.length + 1).replace(/\.ts$/u, ".d.ts"));
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, await readFile(source));
  }
} finally {
  await rm(temporary, { recursive: true, force: true });
}
