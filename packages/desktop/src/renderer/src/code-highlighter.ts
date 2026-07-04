const HIGHLIGHT_THEME = "github-dark";
const loadBashLanguage = () => import("shiki/langs/bash.mjs");
const loadCssLanguage = () => import("shiki/langs/css.mjs");
const loadDiffLanguage = () => import("shiki/langs/diff.mjs");
const loadHtmlLanguage = () => import("shiki/langs/html.mjs");
const loadJavascriptLanguage = () => import("shiki/langs/javascript.mjs");
const loadJsonLanguage = () => import("shiki/langs/json.mjs");
const loadJsoncLanguage = () => import("shiki/langs/jsonc.mjs");
const loadJsxLanguage = () => import("shiki/langs/jsx.mjs");
const loadMarkdownLanguage = () => import("shiki/langs/markdown.mjs");
const loadPythonLanguage = () => import("shiki/langs/python.mjs");
const loadTsxLanguage = () => import("shiki/langs/tsx.mjs");
const loadTypescriptLanguage = () => import("shiki/langs/typescript.mjs");
const loadYamlLanguage = () => import("shiki/langs/yaml.mjs");
const highlightLanguageLoaders = {
  bash: loadBashLanguage,
  css: loadCssLanguage,
  diff: loadDiffLanguage,
  html: loadHtmlLanguage,
  javascript: loadJavascriptLanguage,
  js: loadJavascriptLanguage,
  json: loadJsonLanguage,
  jsonc: loadJsoncLanguage,
  jsx: loadJsxLanguage,
  markdown: loadMarkdownLanguage,
  md: loadMarkdownLanguage,
  py: loadPythonLanguage,
  python: loadPythonLanguage,
  sh: loadBashLanguage,
  shell: loadBashLanguage,
  ts: loadTypescriptLanguage,
  tsx: loadTsxLanguage,
  typescript: loadTypescriptLanguage,
  yaml: loadYamlLanguage,
  yml: loadYamlLanguage,
};

type HighlightLanguage = keyof typeof highlightLanguageLoaders;
export interface HighlightedCodeToken {
  content: string;
  color?: string;
  fontStyle?: number;
}

export interface HighlightedCodeLine {
  tokens: HighlightedCodeToken[];
}

export interface HighlightedCodeBlock {
  lines: HighlightedCodeLine[];
}

type CodeHighlighter = {
  codeToTokens(
    code: string,
    options: { lang: string; theme: string },
  ): { tokens: Array<Array<HighlightedCodeToken>> };
};

const highlightedCodeCache = new Map<
  string,
  HighlightedCodeBlock | Promise<HighlightedCodeBlock | null> | null
>();
const codeHighlighterCache = new Map<HighlightLanguage, Promise<CodeHighlighter | null>>();

export async function highlightCodeBlock(
  codeText: string,
  language: string,
): Promise<HighlightedCodeBlock | null> {
  const normalizedLanguage = normalizeHighlightLanguage(language);
  if (!isHighlightLanguage(normalizedLanguage)) return null;

  const cacheKey = `${normalizedLanguage}\u0000${codeText}`;
  const cached = highlightedCodeCache.get(cacheKey);
  if (cached === null || (cached && !(cached instanceof Promise))) return cached;
  if (cached) return cached;

  const pending = loadCodeHighlighter(normalizedLanguage)
    .then((highlighter) => {
      if (!highlighter) return null;
      const highlighted = highlighter.codeToTokens(codeText, {
        lang: normalizedLanguage,
        theme: HIGHLIGHT_THEME,
      });
      const block = {
        lines: highlighted.tokens.map((tokens) => ({
          tokens: tokens.map((token) => ({
            color: token.color,
            content: token.content,
            fontStyle: token.fontStyle,
          })),
        })),
      };
      highlightedCodeCache.set(cacheKey, block);
      return block;
    })
    .catch(() => {
      highlightedCodeCache.set(cacheKey, null);
      return null;
    });

  highlightedCodeCache.set(cacheKey, pending);
  return pending;
}

async function loadCodeHighlighter(language: HighlightLanguage): Promise<CodeHighlighter | null> {
  const cached = codeHighlighterCache.get(language);
  if (cached) return cached;

  const loader = highlightLanguageLoaders[language];
  const pending = Promise.all([
    import("shiki/core"),
    import("shiki/engine/javascript"),
    import("shiki/themes/github-dark.mjs"),
    loader(),
  ])
    .then(async ([core, engine, theme, languageModule]) => {
      const createHighlighterCore = core.createHighlighterCore as unknown as (options: {
        engine: unknown;
        langs: unknown[];
        themes: unknown[];
      }) => Promise<CodeHighlighter>;
      return createHighlighterCore({
        engine: engine.createJavaScriptRegexEngine(),
        langs: [languageModule.default],
        themes: [theme.default],
      });
    })
    .catch(() => null);

  codeHighlighterCache.set(language, pending);
  return pending;
}

function normalizeHighlightLanguage(language: string): string {
  const normalized = language.toLowerCase().trim();
  if (!/^[a-z0-9_#+.-]+$/.test(normalized)) return "";
  return normalized;
}

function isHighlightLanguage(language: string): language is HighlightLanguage {
  return Object.prototype.hasOwnProperty.call(highlightLanguageLoaders, language);
}
