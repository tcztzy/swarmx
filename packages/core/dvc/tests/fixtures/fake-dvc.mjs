#!/usr/bin/env node
import { appendFileSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, parse, resolve } from "node:path";

function findRoot(start) {
  let current = resolve(start);
  const filesystemRoot = parse(current).root;
  while (true) {
    if (existsSync(join(current, ".dvc"))) return current;
    if (current === filesystemRoot) return null;
    current = dirname(current);
  }
}

function fixture(root, name) {
  const path = join(root, name);
  return existsSync(path) ? readFileSync(path, "utf8") : "{}";
}

const args = process.argv.slice(2);
if (args[0] === "--version") {
  process.stdout.write("3.99.0-test\n");
  process.exit(0);
}

const root = findRoot(process.cwd());
if (root === null) {
  process.stderr.write("not a DVC repository\n");
  process.exit(1);
}

if (args[0] === "root") {
  process.stdout.write(`${root}\n`);
} else if (args[0] === "data" && args[1] === "status" && args.includes("--json")) {
  process.stdout.write(fixture(root, ".fake-dvc-data.json"));
} else if (args[0] === "status" && args.includes("--json")) {
  process.stdout.write(fixture(root, ".fake-dvc-pipeline.json"));
} else if (args[0] === "cache" && args[1] === "dir") {
  const value = args.find((argument, index) => index > 1 && argument !== "--local");
  if (value === undefined) {
    process.stdout.write(`${join(root, ".dvc", "cache")}\n`);
  } else {
    appendFileSync(join(root, ".dvc", "config.local"), `\ncache = ${value}\n`);
  }
} else if (args[0] === "pull") {
  appendFileSync(join(root, ".fake-dvc-pulls"), `${JSON.stringify(args.slice(1))}\n`);
  process.stdout.write("pulled\n");
} else if (args[0] === "repro") {
  if (args.includes("fail")) {
    process.stderr.write(`failed at ${root} via https://user:secret@example.invalid/data\n`);
    process.exit(7);
  }
  writeFileSync(join(root, "generated.txt"), `${JSON.stringify(args.slice(1))}\n`);
  process.stdout.write("reproduced\n");
} else {
  process.stderr.write(`unsupported fake DVC invocation: ${JSON.stringify(args)}\n`);
  process.exit(2);
}
