# Reproducible artifact metadata

## Contract

`dsh-science` embeds one format-independent Figure generation record when a PNG, SVG, or PDF
enters the immutable Artifact Registry. The same canonical UTF-8 JSON is used by every adapter;
format containers only transport the record and never execute it.

| MIME | Owned container |
| --- | --- |
| `image/png` | uncompressed `iTXt` keyword `dsh-science.provenance` |
| `image/svg+xml` | never-rendered `<metadata id="dsh-science.provenance">` with a DSH namespaced child |
| `application/pdf` | Catalog XMP Metadata stream property `dsh:provenance` |

The decoded canonical JSON is capped at 1 MiB. Plotting code is capped at 200,000 characters. SVG
and PDF store the JSON as canonical base64 text so XML metacharacters in code cannot become markup.

```json
{
  "schema": "dsh-science.figure-provenance",
  "version": 1,
  "generationId": "e6ab9d32-10b0-4f08-9d0e-98689b01e2d1",
  "generator": {
    "library": "matplotlib",
    "code": "import pandas as pd\n...",
    "codeHash": "sha256:..."
  },
  "sources": [
    {
      "kind": "workspace",
      "relativePath": "data/result.csv",
      "digest": "sha256:..."
    },
    {
      "kind": "artifact",
      "artifactId": "file-object-id",
      "digest": "sha256:..."
    },
    {
      "kind": "s3",
      "uri": "s3://research-bucket/cohort/result.csv",
      "versionId": "3HL4kqtJlcpXroDTDmJ+rmSpXd3dIbrHY"
    }
  ],
  "environment": {
    "pythonVersion": "3.12.5",
    "packageSetHash": "sha256:..."
  }
}
```

Source references have three stable forms:

- `workspace` stores a traversal-free path relative to the live workspace. It never stores
  `/Users/...`, drive-letter paths, UNC paths, `..`, or a temporary materialization path. The Host
  verifies the live file and adds its SHA256; an optional caller digest is accepted only when it
  matches those bytes.
- `artifact` stores a DSH Science Artifact/File Object id. The Host resolves it inside the current
  workspace and adds the immutable digest; a caller cannot supply or spoof that digest.
- `s3` stores a credential-free `s3://bucket/key`. `versionId` and/or a SHA256 digest should be
  supplied when the exact historical object matters. Metadata injection performs no network call.

The output digest and Artifact id are deliberately absent: inserting either value changes the
output bytes and therefore changes the value again. `generationId` connects the portable record to
the idempotent generation request without creating this self-reference.

TypeScript consumers can call `await extractArtifactMetadata(bytes, mime)`, or the format-specific
`extractPngMetadata`, `extractSvgMetadata`, and async `extractPdfMetadata` exports.

## Generation flow

1. Code reads input through a workspace-relative path, an authorized materialized Artifact input,
   or application-owned S3 access.
2. matplotlib `savefig(...)`, seaborn/matplotlib, R `ggsave(...)`, or Plotly writes an ordinary PNG,
   SVG, or PDF inside the workspace.
3. `dsh-science` resolves Artifact sources and rejects unsafe relative/S3 references. Notebook
   output fingerprints relative sources before execution and verifies them again afterward.
4. The Artifact Store validates MIME against the actual file, replaces only its owned metadata,
   writes transformed bytes into owner-only staging, and computes SHA256 over the final bytes. The
   workspace file is never modified.
5. The transformed object is fsynced and content-addressed before the Journal commits metadata.

PNG injection validates the chunk structure and inserts one `iTXt` immediately before `IEND`. SVG
injection accepts bounded UTF-8 XML, preserves unrelated elements and metadata, and adds one
foreign-namespace record under the root. PDF injection preserves unrelated XMP values and writes
the DSH property into the document-level Catalog Metadata stream. Encrypted or signed PDFs are
rejected because rewriting them would require credentials or invalidate a signature. PDF/A inputs
must already declare the DSH custom XMP extension schema; no conformance claim is silently changed.

The postprocessor belongs at capture rather than inside matplotlib or ggplot2. Python and R
therefore share the schema, validation, opt-out behavior, and content hash.

## Python / matplotlib

For Notebook execution, declare a supported Figure output. The executed cell becomes
`generator.code`; input Artifact ids are added automatically.

```ts
await ctx.science.executeNotebookCell(sessionId, {
  requestId,
  notebookId,
  inputArtifactIds: [],
  source: pythonMatplotlibSource,
  outputArtifact: {
    relativePath: "figures/result.svg",
    kind: "figure",
    title: "Result",
    mime: "image/svg+xml",
    license: null,
    reproducibilityMetadata: {
      library: "matplotlib",
      sources: [{ kind: "workspace", relativePath: "data/result.csv" }]
    }
  }
});
```

## R / ggplot2

After `ggsave("figures/result.pdf", plot)`, register the generated PDF and pass the exact R source.

```ts
await ctx.science.registerArtifact(sessionId, {
  requestId,
  projectId,
  relativePath: "figures/result.pdf",
  kind: "figure",
  title: "Result",
  mime: "application/pdf",
  runId: null,
  environment: { R: "4.5.1", ggplot2: "4.0.0" },
  license: null,
  sourceEntityIds: [resultCsvArtifactId],
  reproducibilityMetadata: {
    library: "ggplot2",
    code: rGgplotSource,
    sources: [
      { kind: "artifact", artifactId: resultCsvArtifactId },
      { kind: "s3", uri: "s3://research-bucket/cohort/result.csv", versionId }
    ]
  }
});
```

## Disable injection

Set `embedArtifactMetadata: false` in the `dsh-science` service configuration to disable injection
for the installation. Set `reproducibilityMetadata: false` on one Figure output/registration to
disable only that artifact. Either switch stores the generator bytes unchanged and does not inspect,
remove, or rewrite existing owned metadata.

## Acceptance criteria

- matplotlib and ggplot2 PNG/SVG/PDF outputs pass through one generator-independent capture path.
- The stored object contains exactly one parseable owned record while the workspace source stays
  unchanged and unrelated SVG/PDF metadata values survive.
- Exact code, code hash, redacted runtime, and normalized sources round-trip from stored bytes.
- Absolute/traversal paths, credential-bearing S3 URIs, foreign Artifact ids, MIME mismatches,
  malformed inputs, duplicate/invalid owned records, signed/encrypted PDF, unsupported PDF/A, and
  metadata above 1 MiB are rejected before a Journal fact is committed.
- Global or per-request opt-out stores generator bytes unchanged.
- Idempotent retry does not rerun code or reread a removed source.

## Standards

- [W3C PNG Third Edition `iTXt`](https://www.w3.org/TR/png-3/#11iTXt)
- [W3C SVG 2 `metadata`](https://www.w3.org/TR/SVG/struct.html#MetadataElement)
- [PDF 2.0 object metadata streams](https://pdfa.org/resource/pdf-2-0-application-note-003-use-of-object-metadata-streams/)
- [Custom XMP metadata in PDF/PDF-A](https://pdfa.org/download-area/publications/Including-custom-metadata-structures-in-PDF.pdf)
