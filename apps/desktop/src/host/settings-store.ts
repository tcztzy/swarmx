import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { z } from "zod";
import { MemorySettingsSchema } from "../memory.js";
import { writePrivateJson } from "../private-json.js";
import { DEFAULT_POLICY, LanguageSchema, type Settings, SettingsSchema } from "../settings.js";

export class SettingsStore {
  private value: Settings;
  readonly path: string;
  private readonly languagePath: string;

  constructor(productHome: string) {
    this.path = join(productHome, "settings.json");
    this.languagePath = join(productHome, "language.json");
    try {
      const raw: unknown = JSON.parse(readFileSync(this.path, "utf8"));
      const legacy = z.object({ policy: z.object({ approval: z.string() }) }).safeParse(raw);
      if (legacy.success)
        throw new Error(
          `Legacy native permission policy in ${this.path}. Remove policy.approval and set policy.tools explicitly; legacy container fields are retained only for saved-setting compatibility. See docs/permissions.md.`,
        );
      this.value = SettingsSchema.parse(raw);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      this.value = { policy: { ...DEFAULT_POLICY }, environment: null };
    }
  }

  read(): Settings {
    return structuredClone(this.value);
  }

  readMemory() {
    try {
      return MemorySettingsSchema.parse(
        JSON.parse(readFileSync(join(dirname(this.path), "memory.json"), "utf8")),
      );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      return MemorySettingsSchema.parse({});
    }
  }

  writeMemory(raw: unknown) {
    const value = MemorySettingsSchema.parse(raw);
    writePrivateJson(join(dirname(this.path), "memory.json"), value);
    return value;
  }

  write(value: Settings): Settings {
    const parsed = SettingsSchema.parse(value);
    writePrivateJson(this.path, parsed);
    this.value = parsed;
    return this.read();
  }

  readLanguage(): "zh" | "en" | null {
    try {
      return LanguageSchema.parse(JSON.parse(readFileSync(this.languagePath, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      return null;
    }
  }

  writeLanguage(language: "zh" | "en"): void {
    writePrivateJson(this.languagePath, LanguageSchema.parse(language));
  }
}
