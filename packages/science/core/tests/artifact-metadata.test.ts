import { createHash, randomUUID } from "node:crypto";
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { PDFDocument, PDFName } from "pdf-lib";
import { afterEach, describe, expect, it } from "vitest";
import type { ScienceError } from "../src/errors.js";
import {
  ARTIFACT_METADATA_KEYWORD,
  countPdfMetadataRecords,
  countPngMetadataChunks,
  countSvgMetadataRecords,
  extractArtifactMetadata,
  extractPdfMetadata,
  extractPngMetadata,
  extractSvgMetadata,
} from "../src/index.js";
import { createScienceFixture, type ScienceFixture } from "./fixture.js";

const MINIMAL_PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==",
  "base64",
);
const MINIMAL_SVG = Buffer.from(
  `<svg xmlns="http://www.w3.org/2000/svg" xmlns:dc="http://purl.org/dc/elements/1.1/" viewBox="0 0 10 10"><metadata id="author-metadata"><dc:title>Existing title</dc:title></metadata><circle cx="5" cy="5" r="4"/></svg>`,
);
const fixtures: ScienceFixture[] = [];

function xmpPacket(description: string): string {
  return [
    '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">',
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
    description,
    "</rdf:RDF>",
    "</x:xmpmeta>",
    '<?xpacket end="w"?>',
  ].join("");
}

async function minimalPdf(options: { readonly pdfa?: boolean; readonly signed?: boolean } = {}) {
  const document = await PDFDocument.create();
  document.addPage([72, 72]);
  const description = [
    '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"',
    options.pdfa ? ' xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">' : ">",
    '<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Existing title</rdf:li></rdf:Alt></dc:title>',
    options.pdfa ? "<pdfaid:part>2</pdfaid:part><pdfaid:conformance>B</pdfaid:conformance>" : "",
    "</rdf:Description>",
  ].join("");
  const metadata = document.context.stream(Buffer.from(xmpPacket(description)), {
    Type: PDFName.of("Metadata"),
    Subtype: PDFName.of("XML"),
  });
  document.catalog.set(PDFName.of("Metadata"), document.context.register(metadata));
  if (options.signed) {
    const signature = document.context.obj({
      Type: PDFName.of("Sig"),
      ByteRange: [0, 1, 2, 3],
    });
    document.catalog.set(PDFName.of("Perms"), document.context.obj({ DocMDP: signature }));
  }
  return Buffer.from(await document.save({ useObjectStreams: false }));
}

afterEach(async () => {
  await Promise.all(fixtures.splice(0).map((fixture) => fixture.dispose()));
});

async function fixture(embedArtifactMetadata = true): Promise<ScienceFixture> {
  const created = await createScienceFixture({ embedArtifactMetadata });
  fixtures.push(created);
  return created;
}

function objectPath(root: string, digest: string): string {
  const hex = digest.slice("sha256:".length);
  return join(root, "artifacts", "v1", "objects", hex.slice(0, 2), hex);
}

function project(current: ScienceFixture, sessionId = current.sessionA) {
  return current.context.science.createProject(sessionId, {
    requestId: randomUUID(),
    title: "Artifact provenance",
  });
}

describe("T36 reproducible artifact metadata", () => {
  it("V85/V86 stores one portable ggplot2 record without changing the workspace PNG", async () => {
    const current = await fixture();
    const created = project(current);
    const input = current.context.science.importArtifact(current.sessionA, {
      requestId: randomUUID(),
      projectId: created.id,
      name: "result.csv",
      dataBase64: Buffer.from("sample,value\nA,42\n").toString("base64"),
    });
    mkdirSync(join(current.workspaceA, "data"));
    writeFileSync(join(current.workspaceA, "data", "result.csv"), "sample,value\nA,42\n");
    writeFileSync(join(current.workspaceA, "result.png"), MINIMAL_PNG);
    const code = [
      'library("ggplot2")',
      'data <- read.csv("data/result.csv")',
      'ggsave("result.png", ggplot(data, aes(sample, value)) + geom_col())',
    ].join("\n");

    const request = {
      requestId: randomUUID(),
      projectId: created.id,
      relativePath: "result.png",
      kind: "figure",
      title: "Result",
      mime: "image/png",
      runId: null,
      environment: {
        R: "4.5.1",
        API_TOKEN: "do-not-embed",
        executable: join(current.workspaceA, "bin", "R"),
      },
      license: null,
      sourceEntityIds: [],
      reproducibilityMetadata: {
        library: "ggplot2",
        code,
        sources: [
          { kind: "workspace", relativePath: "data/result.csv" },
          { kind: "artifact", artifactId: input.id },
          {
            kind: "s3",
            uri: "s3://research-bucket/cohort/result.csv",
            versionId: "version-42",
          },
        ],
      },
    } as const;
    const artifact = await current.context.science.registerArtifact(current.sessionA, request);

    expect(readFileSync(join(current.workspaceA, "result.png"))).toEqual(MINIMAL_PNG);
    const stored = readFileSync(objectPath(current.root, artifact.digest));
    expect(countPngMetadataChunks(stored)).toBe(1);
    expect(stored.includes(Buffer.from(ARTIFACT_METADATA_KEYWORD))).toBe(true);
    expect(extractPngMetadata(stored)).toEqual({
      schema: "dsh-science.figure-provenance",
      version: 1,
      generationId: expect.any(String),
      generator: {
        library: "ggplot2",
        code,
        codeHash: `sha256:${createHash("sha256").update(code).digest("hex")}`,
      },
      sources: [
        { kind: "workspace", relativePath: "data/result.csv", digest: input.digest },
        { kind: "artifact", artifactId: input.id, digest: input.digest },
        {
          kind: "s3",
          uri: "s3://research-bucket/cohort/result.csv",
          versionId: "version-42",
        },
      ],
      environment: { R: "4.5.1", API_TOKEN: "[redacted]", executable: "[redacted]" },
    });
    expect(artifact.sourceEntityIds).toEqual([input.id]);
    expect(stored.includes(Buffer.from("do-not-embed"))).toBe(false);
    unlinkSync(join(current.workspaceA, "data", "result.csv"));
    await expect(
      current.context.science.registerArtifact(current.sessionA, request),
    ).resolves.toEqual(artifact);

    writeFileSync(join(current.workspaceA, "result.png"), stored);
    const revisedCode = code.replace("geom_col()", "geom_point()");
    const replaced = await current.context.science.registerArtifact(current.sessionA, {
      requestId: randomUUID(),
      projectId: created.id,
      relativePath: "result.png",
      kind: "figure",
      title: "Result revised",
      mime: "image/png",
      runId: null,
      environment: { R: "4.5.1" },
      license: null,
      sourceEntityIds: [input.id],
      reproducibilityMetadata: { library: "ggplot2", code: revisedCode, sources: [] },
    });
    const replacedBytes = readFileSync(objectPath(current.root, replaced.digest));
    expect(countPngMetadataChunks(replacedBytes)).toBe(1);
    expect(extractPngMetadata(replacedBytes)?.generator.code).toBe(revisedCode);
  });

  it("V85/V86 stores and replaces one SVG record while preserving unrelated metadata", async () => {
    const current = await fixture();
    const created = project(current);
    writeFileSync(join(current.workspaceA, "result.svg"), MINIMAL_SVG);
    const code = 'ggsave("result.svg", ggplot(data, aes(sample, value)) + geom_col())';
    const request = {
      requestId: randomUUID(),
      projectId: created.id,
      relativePath: "result.svg",
      kind: "figure",
      title: "SVG result",
      mime: "image/svg+xml",
      runId: null,
      environment: { R: "4.5.1" },
      license: null,
      sourceEntityIds: [],
      reproducibilityMetadata: {
        library: "ggplot2",
        code,
        sources: [
          {
            kind: "s3",
            uri: "s3://research-bucket/cohort/result.csv",
            versionId: "version-svg",
          },
        ],
      },
    } as const;

    const artifact = await current.context.science.registerArtifact(current.sessionA, request);
    expect(readFileSync(join(current.workspaceA, "result.svg"))).toEqual(MINIMAL_SVG);
    const stored = readFileSync(objectPath(current.root, artifact.digest));
    expect(countSvgMetadataRecords(stored)).toBe(1);
    expect(stored.toString()).toContain("Existing title");
    expect(extractSvgMetadata(stored)?.generator.code).toBe(code);
    expect(extractSvgMetadata(stored)?.sources).toEqual([
      {
        kind: "s3",
        uri: "s3://research-bucket/cohort/result.csv",
        versionId: "version-svg",
      },
    ]);
    await expect(extractArtifactMetadata(stored, "image/svg+xml")).resolves.toEqual(
      extractSvgMetadata(stored),
    );

    writeFileSync(join(current.workspaceA, "result.svg"), stored);
    const revisedCode = code.replace("geom_col", "geom_point");
    const replaced = await current.context.science.registerArtifact(current.sessionA, {
      ...request,
      requestId: randomUUID(),
      reproducibilityMetadata: { library: "ggplot2", code: revisedCode, sources: [] },
    });
    const replacedBytes = readFileSync(objectPath(current.root, replaced.digest));
    expect(countSvgMetadataRecords(replacedBytes)).toBe(1);
    expect(replacedBytes.toString()).toContain("Existing title");
    expect(extractSvgMetadata(replacedBytes)?.generator.code).toBe(revisedCode);
  });

  it("V85/V86 stores and replaces one PDF XMP record while preserving unrelated XMP", async () => {
    const current = await fixture();
    const created = project(current);
    const sourcePdf = await minimalPdf();
    writeFileSync(join(current.workspaceA, "result.pdf"), sourcePdf);
    const code = 'ggsave("result.pdf", ggplot(data, aes(sample, value)) + geom_col())';
    const request = {
      requestId: randomUUID(),
      projectId: created.id,
      relativePath: "result.pdf",
      kind: "figure",
      title: "PDF result",
      mime: "application/pdf",
      runId: null,
      environment: { R: "4.5.1" },
      license: null,
      sourceEntityIds: [],
      reproducibilityMetadata: {
        library: "ggplot2",
        code,
        sources: [
          {
            kind: "s3",
            uri: "s3://research-bucket/cohort/result.csv",
            versionId: "version-pdf",
          },
        ],
      },
    } as const;

    const artifact = await current.context.science.registerArtifact(current.sessionA, request);
    expect(readFileSync(join(current.workspaceA, "result.pdf"))).toEqual(sourcePdf);
    const stored = readFileSync(objectPath(current.root, artifact.digest));
    expect(await countPdfMetadataRecords(stored)).toBe(1);
    expect(stored.toString()).toContain("Existing title");
    expect((await extractPdfMetadata(stored))?.generator.code).toBe(code);
    expect((await extractPdfMetadata(stored))?.sources).toEqual([
      {
        kind: "s3",
        uri: "s3://research-bucket/cohort/result.csv",
        versionId: "version-pdf",
      },
    ]);
    await expect(extractArtifactMetadata(stored, "application/pdf")).resolves.toEqual(
      await extractPdfMetadata(stored),
    );

    writeFileSync(join(current.workspaceA, "result.pdf"), stored);
    const revisedCode = code.replace("geom_col", "geom_point");
    const replaced = await current.context.science.registerArtifact(current.sessionA, {
      ...request,
      requestId: randomUUID(),
      reproducibilityMetadata: { library: "ggplot2", code: revisedCode, sources: [] },
    });
    const replacedBytes = readFileSync(objectPath(current.root, replaced.digest));
    expect(await countPdfMetadataRecords(replacedBytes)).toBe(1);
    expect(replacedBytes.toString()).toContain("Existing title");
    expect((await extractPdfMetadata(replacedBytes))?.generator.code).toBe(revisedCode);
  });

  it("V87 auto-injects exact Notebook code and immutable input identity", async () => {
    const current = await fixture();
    const created = project(current);
    const notebook = current.context.science.createNotebook(current.sessionA, {
      requestId: randomUUID(),
      projectId: created.id,
      title: "Matplotlib output",
    });
    const input = current.context.science.importArtifact(current.sessionA, {
      requestId: randomUUID(),
      projectId: created.id,
      name: "result.csv",
      dataBase64: Buffer.from("sample,value\nA,42\n").toString("base64"),
    });
    const source = [
      "# matplotlib figure",
      "import base64",
      `open("result.png", "wb").write(base64.b64decode("${MINIMAL_PNG.toString("base64")}"))`,
    ].join("\n");

    const execution = await current.context.science.executeNotebookCell(current.sessionA, {
      requestId: randomUUID(),
      notebookId: notebook.id,
      inputArtifactIds: [input.id],
      source,
      outputArtifact: {
        relativePath: "result.png",
        kind: "figure",
        title: "Matplotlib result",
        mime: "image/png",
        license: null,
      },
    });

    const artifact = execution.artifact;
    expect(artifact).not.toBeNull();
    const stored = readFileSync(objectPath(current.root, artifact?.digest ?? ""));
    const metadata = extractPngMetadata(stored);
    expect(metadata?.generator).toMatchObject({ library: "matplotlib", code: source });
    expect(metadata?.sources).toEqual([
      { kind: "artifact", artifactId: input.id, digest: input.digest },
    ]);
    expect(artifact?.sourceEntityIds).toEqual([notebook.id, input.id]);
  });

  it("V87 auto-injects exact Notebook code into SVG and PDF Figures", async () => {
    const current = await fixture();
    const created = project(current);
    const pdf = await minimalPdf();
    for (const output of [
      {
        bytes: MINIMAL_SVG,
        mime: "image/svg+xml",
        name: "notebook.svg",
      },
      {
        bytes: pdf,
        mime: "application/pdf",
        name: "notebook.pdf",
      },
    ] as const) {
      const notebook = current.context.science.createNotebook(current.sessionA, {
        requestId: randomUUID(),
        projectId: created.id,
        title: output.name,
      });
      const source = [
        "# matplotlib figure",
        "import base64",
        `open("${output.name}", "wb").write(base64.b64decode("${output.bytes.toString("base64")}"))`,
      ].join("\n");
      const execution = await current.context.science.executeNotebookCell(current.sessionA, {
        requestId: randomUUID(),
        notebookId: notebook.id,
        source,
        outputArtifact: {
          relativePath: output.name,
          kind: "figure",
          title: output.name,
          mime: output.mime,
          license: null,
        },
      });
      const stored = readFileSync(objectPath(current.root, execution.artifact?.digest ?? ""));
      expect((await extractArtifactMetadata(stored, output.mime))?.generator).toMatchObject({
        library: "matplotlib",
        code: source,
      });
    }
  });

  it("V85 rejects a mutable relative source that changes during Notebook generation", async () => {
    const current = await fixture();
    const created = project(current);
    const notebook = current.context.science.createNotebook(current.sessionA, {
      requestId: randomUUID(),
      projectId: created.id,
      title: "Mutable source",
    });
    mkdirSync(join(current.workspaceA, "data"));
    writeFileSync(join(current.workspaceA, "data", "result.csv"), "sample,value\nA,42\n");
    const source = [
      "# matplotlib figure",
      "import base64",
      "from pathlib import Path",
      'Path("data/result.csv").write_text("sample,value\\nA,99\\n")',
      `Path("result.png").write_bytes(base64.b64decode("${MINIMAL_PNG.toString("base64")}"))`,
    ].join("\n");

    await expect(
      current.context.science.executeNotebookCell(current.sessionA, {
        requestId: randomUUID(),
        notebookId: notebook.id,
        source,
        outputArtifact: {
          relativePath: "result.png",
          kind: "figure",
          title: "Mutable result",
          mime: "image/png",
          license: null,
          reproducibilityMetadata: {
            library: "matplotlib",
            sources: [{ kind: "workspace", relativePath: "data/result.csv" }],
          },
        },
      }),
    ).rejects.toMatchObject({ code: "ARTIFACT_SOURCE_CHANGED" });
    expect(current.context.science.journalCount()).toBe(2);
    expect(current.context.science.getWorkspace(current.sessionA).artifacts).toEqual([]);
  });

  it("V87 preserves generator bytes for global and per-artifact opt-out", async () => {
    const pdf = await minimalPdf();
    for (const [globalEnabled, reproducibilityMetadata] of [
      [false, { library: "ggplot2", code: "ggplot()", sources: [] }],
      [true, false],
    ] as const) {
      for (const output of [
        { bytes: MINIMAL_PNG, mime: "image/png", name: "plain.png" },
        { bytes: MINIMAL_SVG, mime: "image/svg+xml", name: "plain.svg" },
        { bytes: pdf, mime: "application/pdf", name: "plain.pdf" },
      ] as const) {
        const current = await fixture(globalEnabled);
        const created = project(current);
        writeFileSync(join(current.workspaceA, output.name), output.bytes);
        const artifact = await current.context.science.registerArtifact(current.sessionA, {
          requestId: randomUUID(),
          projectId: created.id,
          relativePath: output.name,
          kind: "figure",
          title: "Plain",
          mime: output.mime,
          runId: null,
          environment: {},
          license: null,
          sourceEntityIds: [],
          reproducibilityMetadata,
        });
        const stored = readFileSync(objectPath(current.root, artifact.digest));
        expect(stored).toEqual(output.bytes);
        await expect(extractArtifactMetadata(stored, output.mime)).resolves.toBeUndefined();
      }
    }
    expect(countPngMetadataChunks(MINIMAL_PNG)).toBe(0);
  });

  it("V86 rejects metadata above the 1 MiB product limit before Journal commit", async () => {
    const current = await fixture();
    const created = project(current);
    writeFileSync(join(current.workspaceA, "oversize.png"), MINIMAL_PNG);
    const environment = Object.fromEntries(
      Array.from({ length: 64 }, (_, index) => [`runtime${index}`, "界".repeat(1_000)]),
    );
    const sources = Array.from({ length: 32 }, (_, index) => ({
      kind: "s3" as const,
      uri: `s3://research-bucket/${"界".repeat(3_900)}-${index}`,
    }));

    await expect(
      current.context.science.registerArtifact(current.sessionA, {
        requestId: randomUUID(),
        projectId: created.id,
        relativePath: "oversize.png",
        kind: "figure",
        title: "Oversize metadata",
        mime: "image/png",
        runId: null,
        environment,
        license: null,
        sourceEntityIds: [],
        reproducibilityMetadata: {
          library: "matplotlib",
          code: "界".repeat(200_000),
          sources,
        },
      }),
    ).rejects.toMatchObject(
      expect.objectContaining<Partial<ScienceError>>({
        code: "ARTIFACT_IO_FAILED",
        message: expect.stringContaining("1048576 byte limit"),
      }),
    );
    expect(current.context.science.journalCount()).toBe(1);
  });

  it("V85 rejects unsafe locators, foreign ids, absolute code paths, and malformed PNGs", async () => {
    const current = await fixture();
    const created = project(current);
    const foreignProject = project(current, current.sessionB);
    const foreign = current.context.science.importArtifact(current.sessionB, {
      requestId: randomUUID(),
      projectId: foreignProject.id,
      name: "foreign.csv",
      dataBase64: Buffer.from("private\n").toString("base64"),
    });
    writeFileSync(join(current.workspaceA, "invalid.png"), "not a PNG");
    const base = {
      requestId: randomUUID(),
      projectId: created.id,
      relativePath: "invalid.png",
      kind: "figure" as const,
      title: "Invalid",
      mime: "image/png",
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
    };
    writeFileSync(join(current.workspaceA, "source.csv"), "sample,value\nA,42\n");

    for (const reproducibilityMetadata of [
      {
        library: "matplotlib" as const,
        code: 'data = read_csv("/Users/alice/private/result.csv")',
        sources: [],
      },
      {
        library: "matplotlib" as const,
        code: "plot()",
        sources: [{ kind: "workspace" as const, relativePath: "../result.csv" }],
      },
      {
        library: "matplotlib" as const,
        code: "plot()",
        sources: [{ kind: "s3" as const, uri: "s3://user:secret@bucket/result.csv" }],
      },
    ]) {
      await expect(
        current.context.science.registerArtifact(current.sessionA, {
          ...base,
          requestId: randomUUID(),
          reproducibilityMetadata,
        }),
      ).rejects.toMatchObject(
        expect.objectContaining<Partial<ScienceError>>({ code: "INVALID_REQUEST" }),
      );
    }
    await expect(
      current.context.science.registerArtifact(current.sessionA, {
        ...base,
        requestId: randomUUID(),
        reproducibilityMetadata: {
          library: "matplotlib",
          code: "plot()",
          sources: [{ kind: "artifact", artifactId: foreign.id }],
        },
      }),
    ).rejects.toMatchObject(
      expect.objectContaining<Partial<ScienceError>>({ code: "PROVENANCE_ENTITY_NOT_FOUND" }),
    );
    await expect(
      current.context.science.registerArtifact(current.sessionA, {
        ...base,
        requestId: randomUUID(),
        reproducibilityMetadata: {
          library: "matplotlib",
          code: "plot()",
          sources: [
            {
              kind: "workspace",
              relativePath: "source.csv",
              digest: `sha256:${"0".repeat(64)}`,
            },
          ],
        },
      }),
    ).rejects.toMatchObject(
      expect.objectContaining<Partial<ScienceError>>({ code: "ARTIFACT_SOURCE_CHANGED" }),
    );
    await expect(
      current.context.science.registerArtifact(current.sessionA, {
        ...base,
        requestId: randomUUID(),
        reproducibilityMetadata: { library: "matplotlib", code: "plot()", sources: [] },
      }),
    ).rejects.toMatchObject(
      expect.objectContaining<Partial<ScienceError>>({ code: "ARTIFACT_IO_FAILED" }),
    );
    expect(current.context.science.journalCount()).toBe(3);
  });

  it("V86/V88 rejects unsupported targets, duplicate SVG records, and unsafe PDFs before commit", async () => {
    const current = await fixture();
    const created = project(current);
    const duplicateSvg = Buffer.from(
      `<svg xmlns="http://www.w3.org/2000/svg"><metadata id="${ARTIFACT_METADATA_KEYWORD}"/><metadata id="${ARTIFACT_METADATA_KEYWORD}"/></svg>`,
    );
    expect(() => extractSvgMetadata(duplicateSvg)).toThrowError(
      expect.objectContaining<Partial<ScienceError>>({ code: "ARTIFACT_IO_FAILED" }),
    );
    await expect(extractArtifactMetadata(MINIMAL_PNG, "image/jpeg")).rejects.toMatchObject({
      code: "ARTIFACT_IO_FAILED",
    });

    writeFileSync(join(current.workspaceA, "signed.pdf"), await minimalPdf({ signed: true }));
    writeFileSync(join(current.workspaceA, "archival.pdf"), await minimalPdf({ pdfa: true }));
    writeFileSync(join(current.workspaceA, "figure.jpg"), "not-a-jpeg");
    writeFileSync(join(current.workspaceA, "mismatch.svg"), MINIMAL_SVG);
    writeFileSync(
      join(current.workspaceA, "entity.svg"),
      '<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>',
    );
    const corruptPng = Buffer.from(MINIMAL_PNG);
    corruptPng[45] = (corruptPng[45] ?? 0) ^ 1;
    writeFileSync(join(current.workspaceA, "corrupt.png"), corruptPng);
    const metadata = { library: "matplotlib" as const, code: "plot()", sources: [] };
    const base = {
      projectId: created.id,
      kind: "figure" as const,
      title: "Unsafe PDF",
      runId: null,
      environment: {},
      license: null,
      sourceEntityIds: [],
      reproducibilityMetadata: metadata,
    };

    for (const target of [
      { relativePath: "signed.pdf", mime: "application/pdf" },
      { relativePath: "archival.pdf", mime: "application/pdf" },
      { relativePath: "figure.jpg", mime: "image/jpeg" },
      { relativePath: "mismatch.svg", mime: "image/png" },
      { relativePath: "entity.svg", mime: "image/svg+xml" },
      { relativePath: "corrupt.png", mime: "image/png" },
    ] as const) {
      await expect(
        current.context.science.registerArtifact(current.sessionA, {
          ...base,
          ...target,
          requestId: randomUUID(),
        }),
      ).rejects.toMatchObject(
        expect.objectContaining<Partial<ScienceError>>({
          code: target.mime === "image/jpeg" ? "INVALID_REQUEST" : "ARTIFACT_IO_FAILED",
        }),
      );
    }
    expect(current.context.science.journalCount()).toBe(1);
    expect(current.context.science.getWorkspace(current.sessionA).artifacts).toEqual([]);
  });
});
