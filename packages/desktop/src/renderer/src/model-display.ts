export interface ModelDisplayDescriptor {
  label: string;
  modelId: string;
  runtimeModel?: string;
}

export type ModelBrand = "gpt" | "claude" | "deepseek";

export interface ModelBrandPresentation {
  brand: ModelBrand;
  iconUrl: string;
  label: string;
}

export interface ModelReasoningPresentation {
  supportedEfforts: string[];
  defaultEffort?: string;
}

const MODEL_COLLATOR = new Intl.Collator("en", {
  numeric: true,
  sensitivity: "base",
});

const BRAND_PREFIXES: Record<ModelBrand, RegExp> = {
  gpt: /^gpt(?=$|[-_\s/])[-_\s/]*/i,
  claude: /^claude(?=$|[-_\s/])[-_\s/]*/i,
  deepseek: /^deepseek(?=$|[-_\s/])[-_\s/]*/i,
};

const BRAND_ICONS: Record<ModelBrand, string> = {
  gpt: "./harness-icons/codex.svg",
  claude: "./harness-icons/claude_code.svg",
  deepseek: "./provider-icons/deepseek.svg",
};

const GPT_VARIANT_RANK = new Map([
  ["sol", 0],
  ["terra", 1],
  ["luna", 2],
]);

const CLAUDE_FAMILY_RANK = new Map([
  ["mythos", 0],
  ["fable", 1],
  ["opus", 2],
  ["sonnet", 3],
  ["haiku", 4],
]);

export function compareModelDisplayOrder(
  left: ModelDisplayDescriptor,
  right: ModelDisplayDescriptor,
): number {
  const leftSort = modelSortDescriptor(left);
  const rightSort = modelSortDescriptor(right);

  if (leftSort?.brand === "gpt" && rightSort?.brand === "gpt") {
    return (
      compareNumericVersionsDescending(leftSort.version, rightSort.version) ||
      leftSort.variantRank - rightSort.variantRank ||
      compareModelIdentity(left, right)
    );
  }

  if (leftSort?.brand === "claude" && rightSort?.brand === "claude") {
    return (
      leftSort.familyRank - rightSort.familyRank ||
      compareNumericVersionsDescending(leftSort.version, rightSort.version) ||
      compareModelIdentity(left, right)
    );
  }

  return compareModelIdentity(left, right);
}

export function modelBrandPresentation(
  model: ModelDisplayDescriptor,
): ModelBrandPresentation | undefined {
  const brand = detectModelBrand(model);
  if (!brand) return undefined;

  const suffix =
    stripBrandPrefix(model.label, brand) ??
    stripBrandPrefix(model.modelId, brand) ??
    (model.runtimeModel ? stripBrandPrefix(model.runtimeModel, brand) : undefined);
  if (!suffix) return undefined;

  return {
    brand,
    iconUrl: BRAND_ICONS[brand],
    label: humanizeModelSuffix(suffix),
  };
}

export function selectableModelReasoning(
  reasoning: ModelReasoningPresentation | null | undefined,
): ModelReasoningPresentation | undefined {
  if (!reasoning) return undefined;
  const supportedEfforts = reasoning.supportedEfforts.filter(
    (effort) => !["none", "ultra"].includes(effort.toLowerCase()),
  );
  if (supportedEfforts.length === 0) return undefined;
  const defaultEffort =
    reasoning.defaultEffort && supportedEfforts.includes(reasoning.defaultEffort)
      ? reasoning.defaultEffort
      : undefined;
  return {
    supportedEfforts,
    ...(defaultEffort ? { defaultEffort } : {}),
  };
}

type ModelSortDescriptor =
  | {
      brand: "gpt";
      variantRank: number;
      version: number[];
    }
  | {
      brand: "claude";
      familyRank: number;
      version: number[];
    };

function modelSortDescriptor(model: ModelDisplayDescriptor): ModelSortDescriptor | undefined {
  const identity = [model.modelId, model.runtimeModel, model.label].find((value) =>
    value ? detectBrandFromValue(value) : undefined,
  );
  if (!identity) return undefined;

  const brand = detectBrandFromValue(identity);
  if (brand === "gpt") {
    const suffix = stripBrandPrefix(identity, brand);
    const versionMatch = suffix?.match(/^(\d+(?:\.\d+)*)/);
    if (!versionMatch) return undefined;
    const variant = suffix
      ?.slice(versionMatch[0].length)
      .split(/[-_\s/]+/)
      .find(Boolean)
      ?.toLowerCase();
    return {
      brand,
      version: versionMatch[1].split(".").map(Number),
      variantRank: variant ? (GPT_VARIANT_RANK.get(variant) ?? 100) : 100,
    };
  }

  if (brand === "claude") {
    const suffix = stripBrandPrefix(identity, brand) ?? "";
    const tokens = suffix
      .toLowerCase()
      .split(/[-_\s/.]+/)
      .filter(Boolean);
    const family = tokens.find((token) => CLAUDE_FAMILY_RANK.has(token));
    return {
      brand,
      familyRank: family ? (CLAUDE_FAMILY_RANK.get(family) ?? 100) : 100,
      version: [...suffix.matchAll(/\d+(?:\.\d+)*/g)].flatMap((match) =>
        match[0].split(".").map(Number),
      ),
    };
  }

  return undefined;
}

function detectModelBrand(model: ModelDisplayDescriptor): ModelBrand | undefined {
  for (const value of [model.modelId, model.label, model.runtimeModel]) {
    if (!value) continue;
    const brand = detectBrandFromValue(value);
    if (brand) return brand;
  }
  return undefined;
}

function detectBrandFromValue(value: string): ModelBrand | undefined {
  return (Object.keys(BRAND_PREFIXES) as ModelBrand[]).find((brand) =>
    BRAND_PREFIXES[brand].test(value.trim()),
  );
}

function stripBrandPrefix(value: string, brand: ModelBrand): string | undefined {
  const trimmed = value.trim();
  if (!BRAND_PREFIXES[brand].test(trimmed)) return undefined;
  return trimmed.replace(BRAND_PREFIXES[brand], "").trim() || undefined;
}

function humanizeModelSuffix(value: string): string {
  return value
    .replace(/[-_/]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((token) =>
      token === token.toLowerCase() ? `${token.charAt(0).toUpperCase()}${token.slice(1)}` : token,
    )
    .join(" ");
}

function compareNumericVersionsDescending(left: number[], right: number[]): number {
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const difference = (right[index] ?? -1) - (left[index] ?? -1);
    if (difference !== 0) return difference;
  }
  return 0;
}

function compareModelIdentity(left: ModelDisplayDescriptor, right: ModelDisplayDescriptor): number {
  return (
    MODEL_COLLATOR.compare(left.label, right.label) ||
    MODEL_COLLATOR.compare(left.modelId, right.modelId)
  );
}
