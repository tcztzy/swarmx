import { describe, expect, it } from "vitest";
import {
  RO_CRATE_CONTEXT,
  RO_CRATE_FILENAME,
  RO_CRATE_PROFILE,
  roCrateMetadataDocumentSchema,
} from "../src/index.js";

function crate() {
  return {
    "@context": RO_CRATE_CONTEXT,
    "@graph": [
      {
        "@id": RO_CRATE_FILENAME,
        "@type": "CreativeWork",
        about: { "@id": "./" },
        conformsTo: { "@id": RO_CRATE_PROFILE },
      },
      {
        "@id": "./",
        "@type": "Dataset",
        name: "Evaluation evidence",
        description: "Frozen inputs and observations",
        datePublished: "2026-10-02T00:00:00.000Z",
        license: "Private",
        hasPart: [{ "@id": "#observation" }],
      },
      { "@id": "#observation", "@type": "CreativeWork", name: "Observation" },
    ],
  };
}

describe("domain-neutral RO-Crate contract", () => {
  it("preserves an attached evidence document without Science objects", () => {
    const document = crate();
    expect(roCrateMetadataDocumentSchema.parse(document)).toEqual(document);
  });

  it("rejects duplicate entities, dangling local references and nested entities", () => {
    const document = crate();
    expect(
      roCrateMetadataDocumentSchema.safeParse({
        ...document,
        "@graph": [...document["@graph"], document["@graph"][2]],
      }).success,
    ).toBe(false);
    for (const citation of [
      [{ "@id": "#missing" }],
      [{ "@id": "urn:uuid:missing" }],
      [{ "@id": "#observation", name: "Nested entity" }],
    ]) {
      expect(
        roCrateMetadataDocumentSchema.safeParse({
          ...document,
          "@graph": document["@graph"].map((entity) => ({ ...entity, citation })),
        }).success,
      ).toBe(false);
    }
  });

  it("requires the metadata descriptor, context and root dataset metadata", () => {
    const document = crate();
    expect(
      roCrateMetadataDocumentSchema.safeParse({ ...document, "@context": "invalid" }).success,
    ).toBe(false);
    for (const index of [0, 1]) {
      expect(
        roCrateMetadataDocumentSchema.safeParse({
          ...document,
          "@graph": document["@graph"].filter((_, current) => current !== index),
        }).success,
      ).toBe(false);
    }
  });
});
