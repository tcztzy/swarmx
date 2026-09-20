import { roCrateMetadataDocumentSchema } from "@swarmx/science/types";
import { z } from "zod";

export const EvaluationCrateRequestSchema = z.union([
  z.strictObject({
    id: z.string().min(1).max(1024),
    expectedRevision: z.string().regex(/^sha256:[a-f0-9]{64}$/u),
  }),
  z.strictObject({ source: z.templateLiteral(["urn:swarmx:execution:", z.uuid()]) }),
]);

export const EvaluationCrateSchema = z.strictObject({
  metadata: roCrateMetadataDocumentSchema,
  files: z.array(
    z.strictObject({
      path: z
        .string()
        .regex(/^(?:[a-z0-9_-]+\/)*[a-z0-9_.-]+$/u)
        .refine((path) => !path.split("/").some((part) => part === "." || part === "..")),
      content: z.string(),
    }),
  ),
});
export type EvaluationCrate = z.infer<typeof EvaluationCrateSchema>;
