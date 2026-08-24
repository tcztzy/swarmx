import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LiteratureSearchRuntime, ZoteroBibliographySource } from "../src/literature.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function zoteroItem() {
  return {
    key: "6FG7F8E3",
    version: 14996,
    data: {
      key: "6FG7F8E3",
      itemType: "journalArticle",
      title: "DNABERT: pre-trained bidirectional representations for DNA-language",
      date: "2021",
      publicationTitle: "Bioinformatics",
      DOI: "10.1093/bioinformatics/btab083",
      url: "https://doi.org/10.1093/bioinformatics/btab083",
      abstractNote: "A genome foundation model.",
      citationKey: "ji2021dnabert",
      creators: [
        { firstName: "Yanrong", lastName: "Ji", creatorType: "author" },
        { firstName: "Zhihan", lastName: "Zhou", creatorType: "author" },
      ],
      tags: [{ tag: "genomics" }],
      extra: "Citation Key: ji2021dnabert\nLocal file: /Users/researcher/private.pdf",
    },
  };
}

describe("local Zotero literature search", () => {
  it("rejects any non-loopback bibliography source", () => {
    expect(() => new ZoteroBibliographySource({ baseUrl: "https://api.zotero.org" })).toThrowError(
      /loopback/u,
    );
  });

  it("normalizes candidates through an owner-only Bib file before ranking", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-literature-"));
    roots.push(root);
    const fetch = vi.fn(async (input: string | URL | Request) => {
      const url = new URL(String(input));
      expect(url.origin).toBe("http://127.0.0.1:23119");
      expect(url.pathname).toBe("/api/users/0/items/top");
      expect(url.searchParams.get("qmode")).toBe("everything");
      return new Response(JSON.stringify([zoteroItem()]), {
        headers: {
          "content-type": "application/json",
          "last-modified-version": "15002",
          "total-results": "1",
          "zotero-api-version": "3",
        },
      });
    });
    const runtime = new LiteratureSearchRuntime(root, new ZoteroBibliographySource({ fetch }));

    const result = await runtime.search(
      "workspace-key",
      { query: "genome foundation model", limit: 5 },
      new AbortController().signal,
    );

    expect(result).toMatchObject({
      source: "zotero",
      ranking: "zotero-local-v1",
      query: "genome foundation model",
      snapshot: {
        source: "zotero",
        format: "bibtex",
        entryCount: 1,
        sourceVersion: "15002",
        digest: expect.stringMatching(/^sha256:[a-f0-9]{64}$/u),
      },
      results: [
        {
          citationKey: "ji2021dnabert",
          sourceItemKey: "6FG7F8E3",
          title: "DNABERT: pre-trained bidirectional representations for DNA-language",
          authors: ["Yanrong Ji", "Zhihan Zhou"],
          year: 2021,
          doi: "10.1093/bioinformatics/btab083",
          matchedFields: expect.arrayContaining(["abstract"]),
          bibtex: expect.stringContaining("@article{ji2021dnabert,"),
        },
      ],
    });
    expect(result.results[0]?.bibtex).not.toContain("/Users/researcher");

    const snapshotPath = join(root, "literature", "workspace-key", "zotero.bib");
    expect(statSync(snapshotPath).mode & 0o777).toBe(0o600);
    const snapshot = readFileSync(snapshotPath, "utf8");
    expect(snapshot).toContain("x-source-id = {6FG7F8E3}");
    expect(snapshot).not.toContain("private.pdf");
    expect(fetch).toHaveBeenCalled();
  });

  it("makes cancellation and unavailable Zotero explicit", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-literature-"));
    roots.push(root);
    const unavailable = new LiteratureSearchRuntime(
      root,
      new ZoteroBibliographySource({
        fetch: vi.fn(() => Promise.reject(new TypeError("connection refused"))),
      }),
    );
    await expect(unavailable.search("workspace", { query: "genome" })).rejects.toMatchObject({
      code: "LITERATURE_SOURCE_UNAVAILABLE",
    });

    const controller = new AbortController();
    controller.abort();
    await expect(
      unavailable.search("workspace", { query: "genome" }, controller.signal),
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});
