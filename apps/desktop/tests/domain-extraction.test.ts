import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, it } from "vitest";
import { ProductServices } from "../src/host/product-services.js";
import { DEFAULT_POLICY } from "../src/settings.js";

it("preserves legacy scientific bytes and settings without activating domain services", async () => {
  const cwd = await mkdtemp(join(tmpdir(), "swarmx-domain-extraction-"));
  const productHome = join(cwd, "home");
  const legacy = join(productHome, "science", "legacy-artifact.bin");
  const bytes = Uint8Array.from([0, 1, 255, 10, 0, 42]);
  const environment = { imageId: `sha256:${"a".repeat(64)}`, domainMetadata: { retained: true } };
  await mkdir(join(productHome, "science"), { recursive: true });
  await writeFile(legacy, bytes);
  await writeFile(
    join(productHome, "settings.json"),
    JSON.stringify({ policy: DEFAULT_POLICY, environment }),
  );
  const services = await ProductServices.create({ cwd, productHome });
  try {
    expect(services.toolManifest.map(({ name }) => name).sort()).toEqual([
      "memory",
      "swarm",
      "work",
    ]);
    expect(services).not.toHaveProperty("science");
    expect(services).not.toHaveProperty("environment");
    services.updatePolicy({ ...services.settings.read().policy, timeoutSeconds: 301 });
    expect(services.settings.read().environment).toEqual(environment);
    expect(
      JSON.parse(await readFile(join(productHome, "settings.json"), "utf8")).environment,
    ).toEqual(environment);
  } finally {
    await services.dispose();
  }
  try {
    expect(new Uint8Array(await readFile(legacy))).toEqual(bytes);
  } finally {
    await rm(cwd, { recursive: true, force: true });
  }
});
