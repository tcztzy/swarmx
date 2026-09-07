import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const directory = resolve(process.argv[2] ?? fileURLToPath(new URL("./evidence", import.meta.url)));
const read = (name) => readFile(resolve(directory, name));
const digest = (bytes) => "sha256:" + createHash("sha256").update(bytes).digest("hex");
const record = JSON.parse(await read("result.json"));
const csv = await read("germination.csv");
const groups = { control: [], primed: [] };
for (const line of csv.toString("utf8").trim().split("\n").slice(1)) {
  const [, treatment, germinated, total] = line.split(",");
  groups[treatment].push((100 * Number(germinated)) / Number(total));
}
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
assert.equal(groups.control.length, 3);
assert.equal(groups.primed.length, 3);
assert.deepEqual(record.result, {
  controlMeanPercent: mean(groups.control),
  primedMeanPercent: mean(groups.primed),
  differencePercentagePoints: mean(groups.primed) - mean(groups.control),
  replicatesPerGroup: 3,
});
assert.equal(record.dataKind, "synthetic");
assert.equal(digest(csv), record.inputArtifact.digest);
assert.equal(digest(await read("germination.png")), record.figureArtifact.digest);
assert.equal(record.executions.length, 2);
assert.ok(
  record.executions.every(
    (execution) => execution.status === "succeeded" && execution.exitCode === 0,
  ),
);
assert.equal(record.nativeRunCount, 3);
assert.equal(record.delegatedRunCount, 1);
const runs = JSON.parse(await read("native-runs.json"));
assert.equal(runs.length, record.nativeRunCount);
assert.ok(runs.every((run) => run.finished));
const children = runs.filter((run) => run.causedBy !== null);
assert.equal(children.length, record.delegatedRunCount);
assert.ok(runs.some((run) => run.runId === children[0].parentRunId));
assert.equal(record.codeUnmodified, true);
assert.equal(record.approval, "author approved through authenticated browser");
const concept = (await read("finding.md")).toString("utf8");
assert.equal(digest(Buffer.from(concept)), record.memory.revision);
for (const value of ["79", "87", "8", record.figureArtifact.id, record.inputArtifact.digest]) {
  assert.ok(concept.includes(value), "Finding is missing " + value);
}
for (const value of ["79", "87", "8"]) assert.ok(record.recall.includes(value));
const crate = JSON.parse(await read("ro-crate-metadata.json"));
const ids = new Set(crate["@graph"].map((entity) => entity["@id"]));
for (const entity of [record.projectId, record.inputArtifact.id, record.figureArtifact.id]) {
  assert.ok(ids.has("urn:uuid:" + entity), "RO-Crate is missing " + entity);
}
const manifest = (await read("MANIFEST.sha256")).toString("utf8").trim().split("\n");
for (const line of manifest) {
  const [expected, name] = line.split("  ");
  assert.match(name, /^[a-zA-Z0-9_.-]+$/);
  assert.equal(digest(await read(name)), "sha256:" + expected, "Checksum mismatch: " + name);
}
console.log(
  JSON.stringify({
    status: "pass",
    syntheticMeans: [mean(groups.control), mean(groups.primed)],
    nativeRuns: record.nativeRunCount,
    delegatedRuns: record.delegatedRunCount,
    successfulComputations: record.executions.length,
    checksums: manifest.length,
  }),
);
