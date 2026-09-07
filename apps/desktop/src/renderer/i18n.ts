import i18next from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./locales/en.json";

export { useTranslation } from "react-i18next";
export function initialLanguage(saved: string | null, browser: string): "zh" | "en" {
  return saved === "zh" || saved === "en"
    ? saved
    : browser.toLowerCase().startsWith("zh")
      ? "zh"
      : "en";
}

export const i18n = i18next;
void i18n.use(initReactI18next).init({
  lng: initialLanguage(null, window.navigator.language),
  supportedLngs: ["zh", "en"],
  fallbackLng: "en",
  resources: {
    en: { translation: en },
    zh: { translation: Object.fromEntries(Object.keys(en).map((key) => [key, key])) },
  },
  keySeparator: false,
  nsSeparator: false,
  interpolation: { escapeValue: false },
  initAsync: false,
});
document.documentElement.lang = i18n.language;
i18n.on("languageChanged", (language) => {
  document.documentElement.lang = language;
});
export const t = i18n.t.bind(i18n);
