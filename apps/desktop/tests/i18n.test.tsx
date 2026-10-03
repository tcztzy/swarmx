// @vitest-environment jsdom
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";
import { afterEach, describe, expect, it } from "vitest";
import { i18n, initialLanguage, t } from "../src/renderer/i18n.js";
import en from "../src/renderer/locales/en.json";

afterEach(async () => {
  await i18n.changeLanguage("zh");
});

describe("English and Chinese UI", () => {
  it("uses an explicit preference before the browser language and preserves interpolated content", async () => {
    expect(initialLanguage(null, "zh-TW")).toBe("zh");
    expect(initialLanguage(null, "de-DE")).toBe("en");
    expect(initialLanguage("en", "zh-CN")).toBe("en");
    expect(initialLanguage("invalid", "zh-CN")).toBe("zh");
    await i18n.changeLanguage("en");
    const title = "样本 <A> & β";
    expect(t("{{count}} 个执行样本", { count: 1 })).toBe("1 execution sample");
    expect(t("{{count}} 个执行样本", { count: 2 })).toBe("2 execution samples");
    expect(t("复盘结论与评价者")).toBe("Review conclusion and reviewer");
    expect(t("由 {{parent}} 委派", { parent: title })).toContain(title);
    await i18n.changeLanguage("zh");
    expect(t("{{count}} 个执行样本", { count: 2 })).toBe("2 个执行样本");
    expect(document.documentElement.lang).toBe("zh");
  });

  it("covers every literal UI translation key in the English catalog", () => {
    const missing = new Set<string>();
    const renderer = join(import.meta.dirname, "../src/renderer");
    for (const name of readdirSync(renderer, { recursive: true }).filter(
      (name) => typeof name === "string" && /\.tsx?$/u.test(name),
    )) {
      const path = join(renderer, String(name));
      const source = ts.createSourceFile(
        path,
        readFileSync(path, "utf8"),
        ts.ScriptTarget.Latest,
        true,
        ts.ScriptKind.TSX,
      );
      function visit(node: ts.Node) {
        if (
          ts.isCallExpression(node) &&
          node.expression.getText(source) === "t" &&
          node.arguments[0] &&
          ts.isStringLiteral(node.arguments[0]) &&
          !(node.arguments[0].text in en)
        )
          missing.add(node.arguments[0].text);
        ts.forEachChild(node, visit);
      }
      visit(source);
    }
    expect([...missing]).toEqual([]);
  });
});
