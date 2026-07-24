import {
  type DesktopComposerPreferenceUpdate,
  DesktopComposerPreferenceUpdateSchema,
  type DesktopComposerPreferences,
} from "@swarmx/core";
import type { DesktopSettingsStoreLike } from "./settings-store.js";

/** Persists the user's most recent routed Model choice without changing vendor config. */
export class ComposerPreferenceService {
  readonly #settings: DesktopSettingsStoreLike;

  constructor(settings: DesktopSettingsStoreLike) {
    this.#settings = settings;
  }

  async get(): Promise<DesktopComposerPreferences> {
    return (await this.#settings.read()).ui.composer;
  }

  async save(input: unknown): Promise<DesktopComposerPreferences> {
    const update = DesktopComposerPreferenceUpdateSchema.parse(input);
    const settings = await this.#settings.update((current) => ({
      ...current,
      ui: {
        ...current.ui,
        composer: updatedComposerPreferences(current.ui.composer, update),
      },
    }));
    return settings.ui.composer;
  }
}

function updatedComposerPreferences(
  current: DesktopComposerPreferences,
  update: DesktopComposerPreferenceUpdate,
): DesktopComposerPreferences {
  const selection = update.modelId
    ? {
        modelId: update.modelId,
        ...(update.modelSupplyId ? { modelSupplyId: update.modelSupplyId } : {}),
        ...(update.effort ? { effort: update.effort } : {}),
      }
    : current.selectionsByHarness[update.harnessId];
  return {
    lastHarnessId: update.harnessId,
    selectionsByHarness: {
      ...current.selectionsByHarness,
      ...(selection ? { [update.harnessId]: selection } : {}),
    },
  };
}
