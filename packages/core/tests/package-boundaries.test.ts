import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createScanner, SyntaxKind } from "typescript/unstable/ast";
import { describe, expect, it } from "vitest";

const CORE_ROOT = fileURLToPath(new URL("..", import.meta.url));
const SOURCE_ROOT = path.join(CORE_ROOT, "src");

describe("Core package boundaries", () => {
  it("keeps local tool contracts browser-safe and dependency-free", () => {
    expect(importsFrom("local-tool-contracts.ts")).toEqual([]);
    expect(coreExports()["./local-tool-contracts"]).toEqual({
      types: "./dist/local-tool-contracts.d.ts",
      import: "./dist/local-tool-contracts.js",
    });
  });

  it("keeps request scope Node-only and protocol-adapter neutral", () => {
    expect(importsFrom("request-scope.ts").map((entry) => entry.module)).toEqual([
      "node:async_hooks",
    ]);
    expect(coreExports()["./request-scope"]).toEqual({
      types: "./dist/request-scope.d.ts",
      import: "./dist/request-scope.js",
    });
  });

  it("keeps the Project record contract browser-safe and separate from its registry", () => {
    expect(importsFrom("project-contracts.ts").map((entry) => entry.module)).toEqual(["zod"]);
    expect(coreExports()["./project-contracts"]).toEqual({
      types: "./dist/project-contracts.d.ts",
      import: "./dist/project-contracts.js",
    });
    expect(namedImportsFrom("project.ts", "./project-contracts.js")).toEqual(
      expect.arrayContaining(["ProjectData", "ProjectDataSchema"]),
    );
  });

  it("keeps the derived Session timeline browser-safe and separate from Session storage", () => {
    expect(importsFrom("session-timeline.ts").map((entry) => entry.module)).toEqual([
      "zod",
      "./canonical-json.js",
      "./types.js",
    ]);
    expect(coreExports()["./session-timeline"]).toEqual({
      types: "./dist/session-timeline.d.ts",
      import: "./dist/session-timeline.js",
    });
  });

  it("prevents domain modules from depending on MCP or ACP adapters for shared contracts", () => {
    for (const file of [
      "memory.ts",
      "reference-library.ts",
      "context-engine.ts",
      "context-evaluation.ts",
    ]) {
      expect(
        importsFrom(file).some((entry) => entry.module === "./mcp.js"),
        file,
      ).toBe(false);
    }
    for (const file of ["mcp.ts", "native-model.ts"]) {
      expect(
        importsFrom(file).some((entry) => entry.module === "./acp.js"),
        file,
      ).toBe(false);
    }
    expect(namedImportsFrom("agent.ts", "./mcp.js")).not.toEqual(
      expect.arrayContaining(["LocalTool", "LocalToolProgress"]),
    );
    expect(namedImportsFrom("extensions.ts", "./mcp.js")).not.toContain("LocalTool");
    expect(namedImportsFrom("native-model.ts", "./mcp.js")).not.toEqual(
      expect.arrayContaining(["LocalToolCallContext", "LocalToolProgress"]),
    );
  });
});

interface ImportEntry {
  module: string;
  names: string[];
}

function importsFrom(file: string): ImportEntry[] {
  const sourcePath = path.join(SOURCE_ROOT, file);
  const scanner = createScanner(true, undefined, readFileSync(sourcePath, "utf8"));
  const imports: ImportEntry[] = [];
  let token = scanner.scan();
  while (token === SyntaxKind.ImportKeyword) {
    token = scanner.scan();
    const names: string[] = [];
    let insideNamedImports = false;
    let module = "";
    while (token !== SyntaxKind.SemicolonToken && token !== SyntaxKind.EndOfFile) {
      if (token === SyntaxKind.OpenBraceToken) insideNamedImports = true;
      else if (token === SyntaxKind.CloseBraceToken) insideNamedImports = false;
      else if (insideNamedImports && token === SyntaxKind.Identifier) {
        names.push(scanner.getTokenValue());
      } else if (token === SyntaxKind.StringLiteral) {
        module = scanner.getTokenValue();
      }
      token = scanner.scan();
    }
    if (module) imports.push({ module, names });
    token = scanner.scan();
  }
  return imports;
}

function namedImportsFrom(file: string, module: string): string[] {
  return importsFrom(file).find((entry) => entry.module === module)?.names ?? [];
}

function coreExports(): Record<string, unknown> {
  return JSON.parse(readFileSync(path.join(CORE_ROOT, "package.json"), "utf8")).exports;
}
