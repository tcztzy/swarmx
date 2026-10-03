#!/usr/bin/env node
import { isAbsolute } from "node:path";
import { parseArgs } from "node:util";
import { MemoryError } from "./errors.js";
import { importModelObservation, queryModelExperience } from "./model-experience.js";
import { MemoryVault } from "./vault.js";

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      vault: { type: "string" },
      query: { type: "string" },
      "include-body": { type: "boolean" },
      artifact: { type: "string" },
      "request-id": { type: "string" },
      title: { type: "string" },
    },
  });
  if (!values.vault || !isAbsolute(values.vault) || positionals.length !== 1)
    throw new Error("Usage: swarmx-memory query|import --vault /absolute/private/memory [options]");
  const vault = new MemoryVault({ root: values.vault, actor: "swarmx-memory-owner-local" });
  let result: unknown;
  if (positionals[0] === "query") {
    if (values.artifact || values["request-id"] || values.title)
      throw new Error("Import options are not valid for query.");
    result = await queryModelExperience(vault, {
      query: values.query ?? "",
      includeBody: values["include-body"] ?? false,
    });
  } else if (positionals[0] === "import") {
    if (values.query !== undefined || values["include-body"] !== undefined)
      throw new Error("Query options are not valid for import.");
    if (!values.artifact || !isAbsolute(values.artifact) || !values["request-id"] || !values.title)
      throw new Error(
        "Import requires --artifact /absolute/file.json --request-id UUID --title 'Observation title'.",
      );
    const saved = await importModelObservation(vault, {
      artifact: values.artifact,
      requestId: values["request-id"],
      title: values.title,
    });
    result = { id: saved.id, revision: saved.revision, provenance: "external-self-asserted" };
  } else throw new Error("Unknown command; use query or import.");
  process.stdout.write(`${JSON.stringify(result)}\n`);
} catch (error) {
  const code = error instanceof MemoryError ? error.code : "INVALID_REQUEST_OR_IO";
  process.stderr.write(
    `Memory command failed (${code}). Check query|import options, absolute canonical paths, artifact schema, permissions and current revisions. See docs/shared-model-experience.md.\n`,
  );
  process.exitCode = 1;
}
