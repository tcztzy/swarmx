import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");

execFileSync(process.execPath, [join(root, "scripts", "summarize.mjs")], {
  cwd: root,
  stdio: ["ignore", "pipe", "inherit"],
});

assert.deepEqual((await readdir(join(root, "results"))).sort(), ["summary.json", "summary.md"]);

const manifest = JSON.parse(await readFile(join(root, "run", "manifest.json"), "utf8"));
for (const artifact of manifest.artifacts) {
  const bytes = await readFile(join(root, artifact.path));
  assert.equal(digest(bytes), artifact.sha256, `Digest mismatch: ${artifact.path}`);
}

const summary = JSON.parse(await readFile(join(root, "results", "summary.json"), "utf8"));
assert.equal(summary.input.rows, 6);
assert.deepEqual(summary.treatments.control, { replicates: 3, meanPercent: 79 });
assert.deepEqual(summary.treatments.primed, { replicates: 3, meanPercent: 87 });
assert.equal(summary.differencePercentagePoints, 8);

const markdown = await readFile(join(root, "results", "summary.md"), "utf8");
assert.match(markdown, /synthetic workflow fixture/);
assert.match(markdown, /not a biological inference/);

const receiptDigest = manifest.knowledgeAdmission.receipt.revision.replace(/^sha256:/, "");
assert.equal(
  digest(await readFile(join(root, "run", "memory-concept.md"))),
  receiptDigest,
  "memory concept does not match its owner receipt",
);

const crate = JSON.parse(await readFile(join(root, "ro-crate-metadata.json"), "utf8"));
assert.equal(crate["@context"], "https://w3id.org/ro/crate/1.3/context");
const crateRoot = crate["@graph"].find((entity) => entity["@id"] === "./");
assert(crateRoot, "RO-Crate root data entity is missing");
assert.deepEqual(crateRoot.hasPart.map((entity) => entity["@id"]).sort(), [
  "README.md",
  "SOURCES.md",
  "data/germination.csv",
  "results/summary.json",
  "results/summary.md",
  "run/manifest.json",
  "run/memory-concept.md",
  "scripts/summarize.mjs",
  "scripts/verify.mjs",
]);

const checksumLines = (await readFile(join(root, "MANIFEST.sha256"), "utf8")).trim().split("\n");
for (const line of checksumLines) {
  const match = /^([a-f0-9]{64}) {2}([^/].*)$/.exec(line);
  assert(match, `Invalid checksum line: ${line}`);
  const [, expected, path] = match;
  assert(!path.split("/").includes(".."), `Unsafe checksum path: ${path}`);
  assert.equal(digest(await readFile(join(root, path))), expected, `Checksum mismatch: ${path}`);
}

process.stdout.write(
  `${JSON.stringify({
    status: "pass",
    artifacts: manifest.artifacts.length,
    tasks: manifest.tasks.length,
    admission: manifest.knowledgeAdmission.status,
    checksums: checksumLines.length,
  })}\n`,
);
