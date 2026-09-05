import { runFaultBenchmark } from "../packages/core/swarm/src/verification-model.js";

const depth = Number(process.argv[2]);

if (!Number.isSafeInteger(depth) || depth < 0) {
  throw new Error("Model depth must be a non-negative integer");
}

process.stdout.write(`${JSON.stringify(runFaultBenchmark(depth), null, 2)}\n`);
