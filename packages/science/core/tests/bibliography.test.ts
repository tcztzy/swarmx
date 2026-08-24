import { describe, expect, it } from "vitest";
import { parseBibFile, serializeBibFile } from "../src/bibliography.js";

describe("provider-neutral BibTeX exchange", () => {
  it("round-trips nested values and strips path-bearing private fields", () => {
    const [entry] = parseBibFile(`
@article{ji2021dnabert,
  title = {{DNABERT}: pre-trained {DNA}-language model},
  author = {Ji, Yanrong and {Genome Consortium}},
  year = {2021},
  abstract = {A nested {DNA} abstract, with punctuation.},
  file = {/Users/researcher/private/paper.pdf},
  x-source = {zotero},
  x-source-id = {6FG7F8E3},
}
`);

    expect(entry).toMatchObject({
      type: "article",
      key: "ji2021dnabert",
      fields: {
        title: "{DNABERT}: pre-trained {DNA}-language model",
        year: "2021",
        "x-source": "zotero",
        "x-source-id": "6FG7F8E3",
      },
    });
    expect(entry?.fields.file).toBeUndefined();

    if (!entry) throw new Error("Expected one parsed BibTeX entry");
    const serialized = serializeBibFile([entry]);
    expect(serialized).toContain("@article{ji2021dnabert,");
    expect(serialized).toContain("x-source-id = {6FG7F8E3}");
    expect(serialized).not.toContain("/Users/researcher");
    expect(parseBibFile(serialized)).toEqual([entry]);
  });

  it("rejects malformed or duplicate citation keys", () => {
    expect(() => parseBibFile("@article{broken, title = {missing close}")).toThrowError(/BibTeX/u);
    expect(() => parseBibFile("@article{same,title={A}}\n@book{same,title={B}}")).toThrowError(
      /duplicate citation key/u,
    );
  });
});
