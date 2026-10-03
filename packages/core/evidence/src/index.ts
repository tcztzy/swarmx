import { z } from "zod";

export const RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.3/context" as const;
export const RO_CRATE_PROFILE = "https://w3id.org/ro/crate/1.3" as const;
export const RO_CRATE_FILENAME = "ro-crate-metadata.json" as const;
export const RO_CRATE_MEDIA_TYPE = "application/ld+json" as const;
export const RO_CRATE_FORMAT = "ro-crate@1.3" as const;

const roCrateIdSchema = z.string().trim().min(1).max(4_096);
const roCrateReferenceSchema = z.strictObject({ "@id": roCrateIdSchema });
const roCrateReferencesSchema = z.array(roCrateReferenceSchema).max(5_000);
const roCrateTypeSchema = z.union([
  z.string().trim().min(1).max(1_000),
  z
    .array(z.string().trim().min(1).max(1_000))
    .min(1)
    .max(16)
    .refine((types) => new Set(types).size === types.length, "RO-Crate types must be unique"),
]);

export const roCrateEntitySchema = z
  .object({
    "@id": roCrateIdSchema,
    "@type": roCrateTypeSchema,
    about: roCrateReferenceSchema.optional(),
    actionStatus: roCrateReferenceSchema.optional(),
    additionalType: z.union([z.string().max(1_000), roCrateReferenceSchema]).optional(),
    bestRating: z.number().finite().optional(),
    citation: roCrateReferencesSchema.optional(),
    conformsTo: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    contentSize: z.string().max(100).optional(),
    creativeWorkStatus: z.string().max(200).optional(),
    dateCreated: z.iso.datetime().optional(),
    dateModified: z.iso.datetime().optional(),
    datePublished: z.iso.datetime().optional(),
    description: z.string().max(20_000).optional(),
    encodingFormat: z.string().max(500).optional(),
    endTime: z.iso.datetime().optional(),
    hasPart: roCrateReferencesSchema.optional(),
    identifier: z.string().max(500).optional(),
    instrument: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    isBasedOn: roCrateReferencesSchema.optional(),
    isPartOf: roCrateReferenceSchema.optional(),
    itemReviewed: roCrateReferenceSchema.optional(),
    keywords: z.array(z.string().max(500)).max(100).optional(),
    license: z.union([z.string().max(2_000), roCrateReferenceSchema]).optional(),
    name: z.string().max(1_000).optional(),
    object: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    ratingValue: z.number().finite().optional(),
    result: z.union([roCrateReferenceSchema, roCrateReferencesSchema]).optional(),
    reviewRating: roCrateReferenceSchema.optional(),
    sha256: z
      .string()
      .regex(/^[0-9a-f]{64}$/u)
      .optional(),
    startTime: z.iso.datetime().optional(),
    text: z.string().max(100_000).optional(),
    version: z.union([z.string().max(100), z.number().finite()]).optional(),
    worstRating: z.number().finite().optional(),
  })
  .catchall(z.json());

function includesRoCrateType(entity: z.infer<typeof roCrateEntitySchema>, type: string): boolean {
  const types = entity["@type"];
  return Array.isArray(types) ? types.includes(type) : types === type;
}

function roCrateReferenceIds(
  value: z.infer<typeof roCrateReferenceSchema> | z.infer<typeof roCrateReferencesSchema>,
): readonly string[] {
  return (Array.isArray(value) ? value : [value]).map((reference) => reference["@id"]);
}

function validateRoCrateReferences(
  value: unknown,
  entityIds: ReadonlySet<string>,
  context: z.RefinementCtx,
  path: readonly (string | number)[],
): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => {
      validateRoCrateReferences(item, entityIds, context, [...path, index]);
    });
    return;
  }
  if (typeof value !== "object" || value === null) return;
  const object = value as Record<string, unknown>;
  if (typeof object["@id"] === "string") {
    if (Object.keys(object).length !== 1) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate entities must be flattened into @graph",
        path: [...path],
      });
      return;
    }
    const id = object["@id"];
    if ((id.startsWith("#") || id.startsWith("urn:uuid:")) && !entityIds.has(id)) {
      context.addIssue({
        code: "custom",
        message: `RO-Crate local reference '${id}' has no entity`,
        path: [...path, "@id"],
      });
    }
    return;
  }
  for (const [key, child] of Object.entries(object)) {
    if (key !== "@id" && key !== "@type") {
      validateRoCrateReferences(child, entityIds, context, [...path, key]);
    }
  }
}

export const roCrateMetadataDocumentSchema = z
  .strictObject({
    "@context": z.literal(RO_CRATE_CONTEXT),
    "@graph": z.array(roCrateEntitySchema).min(2).max(5_000),
  })
  .superRefine((document, context) => {
    const entities = new Map<string, z.infer<typeof roCrateEntitySchema>>();
    for (const [index, entity] of document["@graph"].entries()) {
      if (entities.has(entity["@id"])) {
        context.addIssue({
          code: "custom",
          message: `RO-Crate entity id '${entity["@id"]}' is duplicated`,
          path: ["@graph", index, "@id"],
        });
      }
      entities.set(entity["@id"], entity);
    }
    const descriptor = entities.get(RO_CRATE_FILENAME);
    if (
      !descriptor ||
      !includesRoCrateType(descriptor, "CreativeWork") ||
      !descriptor.about ||
      !descriptor.conformsTo ||
      !roCrateReferenceIds(descriptor.conformsTo).includes(RO_CRATE_PROFILE)
    ) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate Metadata Descriptor is missing or invalid",
        path: ["@graph"],
      });
      return;
    }
    const entityIds = new Set(entities.keys());
    for (const [index, entity] of document["@graph"].entries()) {
      for (const [key, value] of Object.entries(entity)) {
        if (key !== "@id" && key !== "@type") {
          validateRoCrateReferences(value, entityIds, context, ["@graph", index, key]);
        }
      }
    }
    const root = entities.get(descriptor.about["@id"]);
    if (
      !root ||
      !includesRoCrateType(root, "Dataset") ||
      !root.name ||
      !root.description ||
      !root.datePublished ||
      !root.license ||
      !root.hasPart
    ) {
      context.addIssue({
        code: "custom",
        message: "RO-Crate Root Data Entity is missing required metadata",
        path: ["@graph"],
      });
    }
  });

export type RoCrateEntity = z.infer<typeof roCrateEntitySchema>;
export type RoCrateMetadataDocument = z.infer<typeof roCrateMetadataDocumentSchema>;
