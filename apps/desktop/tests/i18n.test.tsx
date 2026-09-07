// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { join } from "node:path";
import ts from "typescript";
import { afterEach, describe, expect, it } from "vitest";
import { i18n, initialLanguage, t } from "../src/renderer/i18n.js";
import en from "../src/renderer/locales/en.json";

afterEach(async () => {
  await i18n.changeLanguage("zh");
});

describe("English and Chinese UI", () => {
  it("uses an explicit preference before the browser language and preserves interpolated research content", async () => {
    expect(initialLanguage(null, "zh-TW")).toBe("zh");
    expect(initialLanguage(null, "de-DE")).toBe("en");
    expect(initialLanguage("en", "zh-CN")).toBe("en");
    expect(initialLanguage("invalid", "zh-CN")).toBe("zh");
    await i18n.changeLanguage("en");
    expect(t("{{count}} 次执行 · v{{revision}}", { count: 1, revision: 2 })).toBe(
      "1 execution · v2",
    );
    expect(t("{{count}} 次执行 · v{{revision}}", { count: 2, revision: 3 })).toBe(
      "2 executions · v3",
    );
    const title = "样本 <A> & β";
    expect(
      t(
        "请修改科研成果「{{title}}」（artifactId: {{id}}），保留原始成果并生成新版本。我的修改要求：",
        { title, id: "original-id" },
      ),
    ).toContain(title);
    await i18n.changeLanguage("zh");
    expect(t("{{count}} 次执行 · v{{revision}}", { count: 1, revision: 2 })).toBe("1 次执行 · v2");
    expect(document.documentElement.lang).toBe("zh");
  });

  it("covers every literal UI translation key in the English catalog", () => {
    const missing = new Set<string>();
    for (const name of [
      "app",
      "projects",
      "chat",
      "agent-controls",
      "subagents",
      "trace",
      "settings",
      "memory",
      "research",
      "research-graph",
      "figure-studio",
      "artifact-preview",
    ]) {
      const path = join(import.meta.dirname, "../src/renderer", `${name}.tsx`);
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
