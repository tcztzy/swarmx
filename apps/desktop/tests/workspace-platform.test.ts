import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { importLegacyProductState, WorkspaceAuthority } from "../src/runtime/index.js";

const roots: string[] = [];

function temporaryRoot(name: string): string {
  const root = mkdtempSync(join(tmpdir(), `${name}-`));
  roots.push(root);
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe("workspace authority", () => {
  it("mints opaque scopes for canonical roots and authorizes contained paths", () => {
    const root = temporaryRoot("swarmx-workspace");
    mkdirSync(join(root, "data"));
    const authority = new WorkspaceAuthority();
    const scope = authority.mint(root);

    const canonicalRoot = realpathSync(root);
    expect(scope.root).toBe(canonicalRoot);
    expect(scope.id).not.toContain(root);
    expect(authority.resolve(scope, "data/result.csv")).toBe(
      join(canonicalRoot, "data", "result.csv"),
    );
  });

  it("rejects forged scopes, traversal, absolute paths, and symlink escape", () => {
    const root = temporaryRoot("swarmx-workspace");
    const outside = temporaryRoot("swarmx-outside");
    symlinkSync(outside, join(root, "outside"));
    const authority = new WorkspaceAuthority();
    const scope = authority.mint(root);

    expect(() => authority.resolve({ ...scope, token: "forged" }, "result.txt")).toThrow(
      "Unknown workspace scope",
    );
    expect(() => authority.resolve(scope, "../secret.txt")).toThrow("relative path");
    expect(() => authority.resolve(scope, join(outside, "secret.txt"))).toThrow("relative path");
    expect(() => authority.resolve(scope, "outside/secret.txt")).toThrow("escapes workspace");
  });
});

describe("legacy product-state import", () => {
  it("copies only known product directories once, verifies them, and writes a marker", () => {
    const legacy = temporaryRoot("swarmx-legacy");
    const destination = temporaryRoot("swarmx-home");
    mkdirSync(join(legacy, "pkb"));
    mkdirSync(join(legacy, "science"));
    mkdirSync(join(legacy, "swarm"));
    mkdirSync(join(legacy, "sessions"));
    writeFileSync(join(legacy, "pkb", "index.md"), "knowledge\n");
    writeFileSync(join(legacy, "science", "science.sqlite"), "science\n");
    writeFileSync(join(legacy, "swarm", "swarm.sqlite"), "journal\n");
    writeFileSync(join(legacy, "sessions", "secret.jsonl"), "native transcript\n");

    expect(importLegacyProductState({ legacyHome: legacy, productHome: destination })).toBe(
      "imported",
    );
    expect(readFileSync(join(destination, "pkb", "index.md"), "utf8")).toBe("knowledge\n");
    expect(readFileSync(join(destination, "science", "science.sqlite"), "utf8")).toBe("science\n");
    expect(readFileSync(join(destination, "swarm", "swarm.sqlite"), "utf8")).toBe("journal\n");
    expect(() => readFileSync(join(destination, "sessions", "secret.jsonl"), "utf8")).toThrow();
    expect(importLegacyProductState({ legacyHome: legacy, productHome: destination })).toBe(
      "already_imported",
    );
  });

  it("does not merge legacy state into an initialized destination", () => {
    const legacy = temporaryRoot("swarmx-legacy");
    const destination = temporaryRoot("swarmx-home");
    mkdirSync(join(legacy, "pkb"));
    writeFileSync(join(legacy, "pkb", "index.md"), "legacy\n");
    writeFileSync(join(destination, "existing"), "owned\n");

    expect(importLegacyProductState({ legacyHome: legacy, productHome: destination })).toBe(
      "already_initialized",
    );
    expect(() => readFileSync(join(destination, "pkb", "index.md"), "utf8")).toThrow();
  });

  it("enforces one shared import budget across nested and peer product directories", () => {
    const legacy = temporaryRoot("swarmx-legacy");
    const destination = temporaryRoot("swarmx-home");
    mkdirSync(join(legacy, "pkb", "first"), { recursive: true });
    mkdirSync(join(legacy, "science", "second"), { recursive: true });
    writeFileSync(join(legacy, "pkb", "first", "one.txt"), "12");
    writeFileSync(join(legacy, "science", "second", "two.txt"), "34");

    expect(() =>
      importLegacyProductState({
        legacyHome: legacy,
        productHome: destination,
        limits: { maxEntries: 4, maxBytes: 3 },
      }),
    ).toThrow("exceeds the bounded import limit");
    expect(() => readFileSync(join(destination, "pkb", "first", "one.txt"), "utf8")).toThrow();
  });
});
