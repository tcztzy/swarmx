import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const inputPath = join(root, "data", "germination.csv");
const resultsPath = join(root, "results");
const input = await readFile(inputPath, "utf8");
const [header, ...lines] = input.trim().split("\n");

if (header !== "replicate,treatment,germinated,total" || lines.length !== 6) {
  throw new Error("Expected the six-row germination fixture.");
}

const rows = lines.map((line) => {
  const [replicate, treatment, germinated, total] = line.split(",");
  const parsed = {
    replicate: Number(replicate),
    treatment,
    germinated: Number(germinated),
    total: Number(total),
  };
  if (
    !Number.isInteger(parsed.replicate) ||
    !["control", "primed"].includes(parsed.treatment) ||
    !Number.isInteger(parsed.germinated) ||
    !Number.isInteger(parsed.total) ||
    parsed.total <= 0 ||
    parsed.germinated < 0 ||
    parsed.germinated > parsed.total
  ) {
    throw new Error(`Invalid fixture row: ${line}`);
  }
  return parsed;
});

const meanPercent = (treatment) => {
  const selected = rows.filter((row) => row.treatment === treatment);
  return (
    (selected.reduce((sum, row) => sum + row.germinated / row.total, 0) / selected.length) * 100
  );
};

const control = meanPercent("control");
const primed = meanPercent("primed");
const result = {
  schema: "swarmx.softwarex.germination-summary.v1",
  input: {
    path: "data/germination.csv",
    rows: rows.length,
    sha256: createHash("sha256").update(input).digest("hex"),
  },
  treatments: {
    control: { replicates: 3, meanPercent: control },
    primed: { replicates: 3, meanPercent: primed },
  },
  differencePercentagePoints: primed - control,
  interpretation:
    "In this synthetic fixture, the primed rows have an 8 percentage-point higher mean germination rate than the control rows.",
};

if (control !== 79 || primed !== 87 || result.differencePercentagePoints !== 8) {
  throw new Error("The fixture no longer matches its declared expected result.");
}

await mkdir(resultsPath, { recursive: true });
await Promise.all([
  writeFile(join(resultsPath, "summary.json"), `${JSON.stringify(result, null, 2)}\n`),
  writeFile(
    join(resultsPath, "summary.md"),
    `# Germination fixture summary\n\n` +
      `The control mean is **${control}%** and the primed mean is **${primed}%**. ` +
      `The primed-minus-control difference is **${result.differencePercentagePoints} percentage points**.\n\n` +
      `This result describes a synthetic workflow fixture and is not a biological inference.\n\n` +
      `Input SHA-256: \`${result.input.sha256}\`\n`,
  ),
]);

process.stdout.write(`${JSON.stringify(result)}\n`);
