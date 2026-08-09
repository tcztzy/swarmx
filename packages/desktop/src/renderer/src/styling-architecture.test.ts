/** @vitest-environment node */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const desktopRoot = new URL("../../../", import.meta.url);
const featureStylesRoot = new URL("src/renderer/src/assets/styles/", desktopRoot);
const MAX_RESIDUAL_CSS_LINES = 3_000;

function read(relativePath: string): string {
  return readFileSync(new URL(relativePath, desktopRoot), "utf8");
}

function readWorkspace(relativePath: string): string {
  return readFileSync(new URL(`../../${relativePath}`, desktopRoot), "utf8");
}

function rendererSourceEntries(
  directory = fileURLToPath(new URL("src/renderer/src", desktopRoot)),
): Array<{ name: string; source: string }> {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return rendererSourceEntries(path);
    if (!entry.name.match(/\.(?:ts|tsx)$/) || entry.name.includes(".test.")) return [];
    return [{ name: path, source: readFileSync(path, "utf8") }];
  });
}

function rendererSources(): string[] {
  return rendererSourceEntries().map((entry) => entry.source);
}

function featureStyles(): Array<{ name: string; source: string }> {
  return readdirSync(featureStylesRoot, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith(".css"))
    .map((entry) => ({
      name: entry.name,
      source: readFileSync(new URL(entry.name, featureStylesRoot), "utf8"),
    }));
}

function standaloneSimpleSelectors(source: string): string[] {
  const matches = [...source.matchAll(/^\s*(\.[a-zA-Z_][\w-]*)\s*\{/gm)];
  return matches
    .filter((match) => !source.slice(0, match.index).trimEnd().endsWith(","))
    .map((match) => match[1]);
}

function conflictingArbitraryUtilities(source: string, fileName: string): string[] {
  const conflicts: string[] = [];
  const stringLiterals = source.matchAll(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`/g);

  for (const literal of stringLiterals) {
    const text = literal[0].slice(1, -1);
    if (!text.includes("[")) continue;
    const properties = new Map<string, string>();
    for (const token of text.split(/\s+/)) {
      const match = /^(.*?)(?:!?)\[(-?[a-zA-Z][\w-]*):(.+)\]!?$/.exec(token);
      if (!match) continue;
      const key = `${match[1]}|${match[2]}`;
      const previous = properties.get(key);
      if (previous !== undefined && previous !== match[3]) {
        const line = source.slice(0, literal.index).split("\n").length;
        conflicts.push(`${fileName}:${line} ${key}: ${previous} <> ${match[3]}`);
      } else if (previous === undefined) {
        properties.set(key, match[3]);
      }
    }
  }
  return conflicts;
}

describe("renderer styling architecture", () => {
  it("compiles Tailwind through Vite and exposes the compiled stylesheet", () => {
    const packageJson = JSON.parse(read("package.json")) as {
      dependencies?: Record<string, string>;
      devDependencies?: Record<string, string>;
      exports?: Record<string, unknown>;
    };
    const viteConfig = read("vite.config.ts");
    const biomeConfig = JSON.parse(readWorkspace("biome.json")) as {
      css?: { parser?: { tailwindDirectives?: boolean } };
    };

    expect(packageJson.dependencies).toHaveProperty("class-variance-authority");
    expect(packageJson.devDependencies).toHaveProperty("tailwindcss");
    expect(packageJson.devDependencies).toHaveProperty("@tailwindcss/vite");
    expect(packageJson.exports?.["./styles.css"]).toBe("./out/renderer/assets/swarmx.css");
    expect(viteConfig).toMatch(/import tailwindcss from "@tailwindcss\/vite"/);
    expect(viteConfig).toMatch(/tailwindcss\(\)/);
    expect(viteConfig).toMatch(/cssCodeSplit:\s*false/);
    expect(viteConfig).toContain('? "assets/swarmx.css"');
    expect(biomeConfig.css?.parser?.tailwindDirectives).toBe(true);
  });

  it("uses explicit reset-free Tailwind layers and semantic tokens", () => {
    const styles = read("src/renderer/src/assets/styles.css");

    expect(styles).toContain("@layer theme, base, components, utilities;");
    expect(styles).toContain('@import "tailwindcss/theme.css" layer(theme);');
    expect(styles).toMatch(
      /@import "tailwindcss\/utilities\.css" layer\(utilities\) source\("\.\.\/"\);/,
    );
    const generatedUtilities = styles.indexOf('@import "tailwindcss/utilities.css"');
    const relationshipOverrides = styles.indexOf('@import "./styles/app-shell.css"');
    expect(generatedUtilities).toBeGreaterThan(-1);
    expect(relationshipOverrides).toBeGreaterThan(generatedUtilities);
    expect(styles).toContain('@import "./styles/app-shell.css" layer(utilities);');
    expect(styles).not.toContain("tailwindcss/preflight.css");
    expect(styles).not.toContain('@import "tailwindcss";');
    expect(styles).toContain("@theme inline");
    expect(styles).toContain("--color-background: var(--background);");
    expect(styles).toContain("--color-foreground: var(--foreground);");
    const responsiveVariants = ["max-1100", "max-860", "max-680", "max-520"];
    const responsiveVariantOffsets = responsiveVariants.map((variant) =>
      styles.indexOf(`@custom-variant ${variant}`),
    );
    expect(responsiveVariantOffsets.every((offset) => offset >= 0)).toBe(true);
    expect(responsiveVariantOffsets).toEqual([...responsiveVariantOffsets].sort((a, b) => a - b));
  });

  it("keeps visual variants on explicit CVA paths", () => {
    const sources = rendererSources().join("\n");

    expect(sources).not.toMatch(/\b(?:badge|button)--[a-z]/);
    expect(sources).not.toMatch(/(?:bg|border|fill|ring|stroke|text)-\$\{/);
    expect(sources).not.toMatch(/`(?:is-|permission-source--|run-event--|trace-card__)[^`]*\$\{/);
  });

  it("does not depend on ambiguous arbitrary-utility ordering", () => {
    const conflicts = rendererSourceEntries().flatMap((entry) =>
      conflictingArbitraryUtilities(entry.source, entry.name),
    );

    expect(conflicts).toEqual([]);
  });

  it("keeps responsive and escaped relationship variants scanner-safe", () => {
    for (const entry of rendererSourceEntries()) {
      expect(entry.source, entry.name).not.toContain("[@media(max-width:");
      expect(entry.source, entry.name).not.toContain("max-[");
      expect(entry.source, entry.name).not.toContain("[backdrop-filter:");

      for (const match of entry.source.matchAll(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g)) {
        expect(match[0], `${entry.name}:${match.index}`).not.toContain("\\\\_");
      }
      for (const match of entry.source.matchAll(/`(?:\\.|[^`\\])*`/g)) {
        expect(match[0], `${entry.name}:${match.index}`).not.toMatch(/\]:\s/);
        if (!match[0].includes("\\_")) continue;
        expect(
          entry.source
            .slice(Math.max(0, (match.index ?? 0) - 10), match.index)
            .endsWith("String.raw"),
        ).toBe(true);
        expect(match[0], `${entry.name}:${match.index}`).not.toContain("\\\\_");
      }
    }
  });

  it("keeps residual CSS bounded to selectors that utilities cannot express locally", () => {
    const styles = featureStyles();
    const lineCount = styles.reduce((total, file) => total + file.source.split("\n").length, 0);
    const componentStyles = styles.filter((file) => file.name !== "base.css");

    expect(lineCount).toBeLessThanOrEqual(MAX_RESIDUAL_CSS_LINES);
    for (const file of componentStyles) {
      expect(standaloneSimpleSelectors(file.source), file.name).toEqual([]);
      expect(file.source, file.name).not.toContain("@apply");
    }
  });

  it("keeps message actions available when the input device cannot hover", () => {
    const responsiveStyles = read("src/renderer/src/assets/styles/responsive.css");

    expect(responsiveStyles).toMatch(/@media \(hover: none\), \(pointer: coarse\)/);
    expect(responsiveStyles).toMatch(
      /\.conversation-turn > \.run-event > \.run-event__actions[\s\S]*opacity: 1/,
    );
    expect(responsiveStyles).toMatch(
      /\.conversation-turn > \.run-event > \.run-event__actions[\s\S]*pointer-events: auto/,
    );
  });
});
