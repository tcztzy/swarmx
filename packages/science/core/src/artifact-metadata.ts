import { DOMParser, type Document, type Element, XMLSerializer } from "@xmldom/xmldom";
import {
  decodePDFRawStream,
  PDFArray,
  PDFDict,
  PDFDocument,
  PDFName,
  PDFRawStream,
  PDFRef,
  PDFStream,
} from "pdf-lib";
import {
  type FigureReproducibilityMetadata,
  figureReproducibilityMetadataSchema,
} from "./contracts.js";
import { ScienceError } from "./errors.js";

export const ARTIFACT_METADATA_KEYWORD = "dsh-science.provenance";
export const MAX_ARTIFACT_METADATA_BYTES = 1024 * 1024;
export const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

export const ARTIFACT_METADATA_MIMES = ["image/png", "image/svg+xml", "application/pdf"] as const;

export type ArtifactMetadataMime = (typeof ARTIFACT_METADATA_MIMES)[number];

const PNG_METADATA_KEYWORD_BYTES = Buffer.from(ARTIFACT_METADATA_KEYWORD, "latin1");
const PNG_CHUNK_TYPE = Buffer.from("iTXt", "ascii");
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const DSH_NAMESPACE = "https://dsh-science.local/ns/provenance/1.0/";
const RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const XMP_META_NAMESPACE = "adobe:ns:meta/";
const PDFA_ID_NAMESPACE = "http://www.aiim.org/pdfa/ns/id/";
const PDFA_SCHEMA_NAMESPACE = "http://www.aiim.org/pdfa/ns/schema#";
const XMLNS_NAMESPACE = "http://www.w3.org/2000/xmlns/";
const CANONICAL_BASE64 = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/u;
const FORBIDDEN_XML_DECLARATION = /<!DOCTYPE|<!ENTITY/iu;

interface PngChunk {
  readonly data: Buffer;
  readonly raw: Buffer;
  readonly type: string;
}

function pngError(message: string, cause?: unknown): ScienceError {
  return new ScienceError(
    message,
    "ARTIFACT_IO_FAILED",
    cause === undefined ? undefined : { cause },
  );
}

function crc32(content: Uint8Array): number {
  let crc = 0xffffffff;
  for (const byte of content) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function parseChunks(content: Uint8Array): PngChunk[] {
  const png = Buffer.from(content);
  if (
    png.length < PNG_SIGNATURE.length ||
    !png.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)
  ) {
    throw pngError("Artifact metadata injection requires a valid PNG signature");
  }
  const chunks: PngChunk[] = [];
  let offset = PNG_SIGNATURE.length;
  let sawIend = false;
  while (offset < png.length) {
    if (png.length - offset < 12) throw pngError("PNG chunk ended unexpectedly");
    const length = png.readUInt32BE(offset);
    if (length > 0x7fffffff || length > png.length - offset - 12) {
      throw pngError("PNG chunk length is invalid");
    }
    const type = png.subarray(offset + 4, offset + 8).toString("ascii");
    if (!/^[A-Za-z]{4}$/u.test(type)) throw pngError("PNG chunk type is invalid");
    const dataStart = offset + 8;
    const end = dataStart + length + 4;
    const data = png.subarray(dataStart, dataStart + length);
    const expectedCrc = png.readUInt32BE(dataStart + length);
    const actualCrc = crc32(Buffer.concat([png.subarray(offset + 4, offset + 8), data]));
    if (expectedCrc !== actualCrc) throw pngError("PNG chunk CRC is invalid");
    chunks.push({ data, raw: png.subarray(offset, end), type });
    offset = end;
    if (type === "IEND") {
      if (length !== 0 || offset !== png.length) throw pngError("PNG IEND chunk is invalid");
      sawIend = true;
      break;
    }
  }
  if (chunks[0]?.type !== "IHDR" || !chunks.some((chunk) => chunk.type === "IDAT") || !sawIend) {
    throw pngError("PNG requires IHDR, IDAT, and IEND chunks");
  }
  return chunks;
}

function isOwnedTextData(data: Uint8Array): boolean {
  return (
    data.length > PNG_METADATA_KEYWORD_BYTES.length &&
    Buffer.from(data.subarray(0, PNG_METADATA_KEYWORD_BYTES.length)).equals(
      PNG_METADATA_KEYWORD_BYTES,
    ) &&
    data[PNG_METADATA_KEYWORD_BYTES.length] === 0
  );
}

export function isPngMetadataKeywordPrefix(prefix: Uint8Array, chunkLength: number): boolean {
  return chunkLength >= PNG_METADATA_KEYWORD_BYTES.length + 1 && isOwnedTextData(prefix);
}

/** Encode one deterministic uncompressed UTF-8 iTXt chunk, including length/type/CRC. */
export function createPngMetadataChunk(metadata: FigureReproducibilityMetadata): Buffer {
  const parsed = figureReproducibilityMetadataSchema.parse(metadata);
  const text = Buffer.from(JSON.stringify(parsed), "utf8");
  if (text.length > MAX_ARTIFACT_METADATA_BYTES) {
    throw pngError(`Artifact metadata exceeds the ${MAX_ARTIFACT_METADATA_BYTES} byte limit`);
  }
  const data = Buffer.concat([PNG_METADATA_KEYWORD_BYTES, Buffer.from([0, 0, 0, 0, 0]), text]);
  const crcInput = Buffer.concat([PNG_CHUNK_TYPE, data]);
  const chunk = Buffer.allocUnsafe(12 + data.length);
  chunk.writeUInt32BE(data.length, 0);
  PNG_CHUNK_TYPE.copy(chunk, 4);
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(crcInput), 8 + data.length);
  return chunk;
}

export function countPngMetadataChunks(content: Uint8Array): number {
  return parseChunks(content).filter(
    (chunk) => chunk.type === "iTXt" && isOwnedTextData(chunk.data),
  ).length;
}

/** Decode the single owned metadata record for verification and portable tooling. */
export function validatePngMetadataData(data: Uint8Array): FigureReproducibilityMetadata {
  if (!isOwnedTextData(data)) throw pngError("PNG dsh-science metadata keyword is invalid");
  let offset = PNG_METADATA_KEYWORD_BYTES.length + 1;
  if (data.length < offset + 4 || data[offset] !== 0 || data[offset + 1] !== 0) {
    throw pngError("PNG dsh-science metadata must use uncompressed iTXt");
  }
  offset += 2;
  const languageEnd = data.indexOf(0, offset);
  if (languageEnd < 0) throw pngError("PNG iTXt language tag is invalid");
  offset = languageEnd + 1;
  const translatedEnd = data.indexOf(0, offset);
  if (translatedEnd < 0) throw pngError("PNG iTXt translated keyword is invalid");
  offset = translatedEnd + 1;
  const text = data.subarray(offset);
  if (text.length > MAX_ARTIFACT_METADATA_BYTES) {
    throw pngError(`Artifact metadata exceeds the ${MAX_ARTIFACT_METADATA_BYTES} byte limit`);
  }
  try {
    const json = new TextDecoder("utf-8", { fatal: true }).decode(text);
    return figureReproducibilityMetadataSchema.parse(JSON.parse(json));
  } catch (error) {
    throw pngError("PNG dsh-science metadata is invalid", error);
  }
}

export function extractPngMetadata(content: Uint8Array): FigureReproducibilityMetadata | undefined {
  const owned = parseChunks(content).filter(
    (chunk) => chunk.type === "iTXt" && isOwnedTextData(chunk.data),
  );
  if (owned.length === 0) return undefined;
  if (owned.length !== 1) throw pngError("PNG contains duplicate dsh-science metadata chunks");
  const data = owned[0]?.data;
  if (!data) return undefined;
  return validatePngMetadataData(data);
}

export function injectPngMetadata(
  content: Uint8Array,
  metadata: FigureReproducibilityMetadata,
): Buffer {
  extractPngMetadata(content);
  const metadataChunk = createPngMetadataChunk(metadata);
  const chunks = parseChunks(content);
  const output: Uint8Array[] = [PNG_SIGNATURE];
  for (const chunk of chunks) {
    if (chunk.type === "iTXt" && isOwnedTextData(chunk.data)) continue;
    if (chunk.type === "IEND") output.push(metadataChunk);
    output.push(chunk.raw);
  }
  return Buffer.concat(output);
}

function metadataError(message: string, cause?: unknown): ScienceError {
  return new ScienceError(
    message,
    "ARTIFACT_IO_FAILED",
    cause === undefined ? undefined : { cause },
  );
}

function canonicalMetadataBytes(metadata: FigureReproducibilityMetadata): Buffer {
  const parsed = figureReproducibilityMetadataSchema.parse(metadata);
  const bytes = Buffer.from(JSON.stringify(parsed), "utf8");
  if (bytes.length > MAX_ARTIFACT_METADATA_BYTES) {
    throw metadataError(`Artifact metadata exceeds the ${MAX_ARTIFACT_METADATA_BYTES} byte limit`);
  }
  return bytes;
}

function encodeXmlMetadata(metadata: FigureReproducibilityMetadata): string {
  return canonicalMetadataBytes(metadata).toString("base64");
}

function decodeXmlMetadata(value: string): FigureReproducibilityMetadata {
  const encoded = value.trim();
  if (!CANONICAL_BASE64.test(encoded)) {
    throw metadataError("Artifact metadata XML payload is not canonical base64");
  }
  const bytes = Buffer.from(encoded, "base64");
  if (bytes.toString("base64") !== encoded) {
    throw metadataError("Artifact metadata XML payload is not canonical base64");
  }
  if (bytes.length > MAX_ARTIFACT_METADATA_BYTES) {
    throw metadataError(`Artifact metadata exceeds the ${MAX_ARTIFACT_METADATA_BYTES} byte limit`);
  }
  try {
    const json = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return figureReproducibilityMetadataSchema.parse(JSON.parse(json));
  } catch (error) {
    if (error instanceof ScienceError) throw error;
    throw metadataError("Artifact metadata XML payload is invalid", error);
  }
}

function decodeXml(content: Uint8Array, format: "PDF XMP" | "SVG"): string {
  try {
    const xml = new TextDecoder("utf-8", { fatal: true }).decode(content);
    if (FORBIDDEN_XML_DECLARATION.test(xml)) {
      throw metadataError(`${format} may not contain a document type or entity declaration`);
    }
    return xml;
  } catch (error) {
    if (error instanceof ScienceError) throw error;
    throw metadataError(`${format} must be valid UTF-8`, error);
  }
}

function parseXml(
  xml: string,
  mime: "application/xml" | "image/svg+xml",
  format: string,
): Document {
  try {
    return new DOMParser({
      locator: false,
      onError(_level, message) {
        throw new Error(message);
      },
    }).parseFromString(xml, mime);
  } catch (error) {
    throw metadataError(`${format} XML is invalid`, error);
  }
}

function elementsByNamespace(
  parent: Document | Element,
  namespace: string,
  name: string,
): Element[] {
  const nodes = parent.getElementsByTagNameNS(namespace, name);
  return Array.from({ length: nodes.length }, (_, index) => nodes.item(index)).filter(
    (node): node is Element => node !== null,
  );
}

function parseSvg(content: Uint8Array): Document {
  const document = parseXml(decodeXml(content, "SVG"), "image/svg+xml", "SVG");
  const root = document.documentElement;
  if (!root || root.namespaceURI !== SVG_NAMESPACE || root.localName !== "svg") {
    throw metadataError("Artifact metadata injection requires an SVG root element");
  }
  return document;
}

function ownedSvgMetadata(document: Document): Element[] {
  return elementsByNamespace(document, SVG_NAMESPACE, "metadata").filter(
    (element) => element.getAttribute("id") === ARTIFACT_METADATA_KEYWORD,
  );
}

export function countSvgMetadataRecords(content: Uint8Array): number {
  return ownedSvgMetadata(parseSvg(content)).length;
}

export function extractSvgMetadata(content: Uint8Array): FigureReproducibilityMetadata | undefined {
  const document = parseSvg(content);
  const owned = ownedSvgMetadata(document);
  if (owned.length === 0) return undefined;
  if (owned.length !== 1)
    throw metadataError("SVG contains duplicate dsh-science metadata records");
  const payloads = elementsByNamespace(owned[0] as Element, DSH_NAMESPACE, "provenance");
  if (payloads.length !== 1) throw metadataError("SVG dsh-science metadata payload is invalid");
  return decodeXmlMetadata(payloads[0]?.textContent ?? "");
}

export function injectSvgMetadata(
  content: Uint8Array,
  metadata: FigureReproducibilityMetadata,
): Buffer {
  const document = parseSvg(content);
  const owned = ownedSvgMetadata(document);
  if (owned.length > 1) throw metadataError("SVG contains duplicate dsh-science metadata records");
  owned[0]?.parentNode?.removeChild(owned[0]);

  const container = document.createElementNS(SVG_NAMESPACE, "metadata");
  container.setAttribute("id", ARTIFACT_METADATA_KEYWORD);
  const payload = document.createElementNS(DSH_NAMESPACE, "dsh:provenance");
  payload.setAttributeNS(XMLNS_NAMESPACE, "xmlns:dsh", DSH_NAMESPACE);
  payload.setAttribute("encoding", "base64-json");
  payload.appendChild(document.createTextNode(encodeXmlMetadata(metadata)));
  container.appendChild(payload);
  const root = document.documentElement;
  if (!root) throw metadataError("Artifact metadata injection requires an SVG root element");
  root.insertBefore(container, root.firstChild);
  return Buffer.from(new XMLSerializer().serializeToString(document), "utf8");
}

function pdfMetadataStream(document: PDFDocument): PDFStream | undefined {
  return document.catalog.lookupMaybe(PDFName.of("Metadata"), PDFStream);
}

function decodePdfMetadataStream(stream: PDFStream): Buffer {
  try {
    return Buffer.from(
      stream instanceof PDFRawStream ? decodePDFRawStream(stream).decode() : stream.getContents(),
    );
  } catch (error) {
    throw metadataError("PDF XMP metadata stream is invalid", error);
  }
}

async function loadPdf(content: Uint8Array): Promise<PDFDocument> {
  try {
    return await PDFDocument.load(content, {
      ignoreEncryption: false,
      throwOnInvalidObject: true,
      updateMetadata: false,
    });
  } catch (error) {
    throw metadataError("Artifact metadata injection requires a valid unencrypted PDF", error);
  }
}

function pdfObjectContainsSignature(
  document: PDFDocument,
  value: unknown,
  visited: Set<string>,
  depth = 0,
): boolean {
  if (depth > 64) throw metadataError("PDF object graph exceeds the metadata safety limit");
  if (value instanceof PDFRef) {
    const key = value.toString();
    if (visited.has(key)) return false;
    visited.add(key);
    return pdfObjectContainsSignature(document, document.context.lookup(value), visited, depth + 1);
  }
  if (value instanceof PDFDict) {
    const type = value.lookupMaybe(PDFName.of("Type"), PDFName);
    if (type?.asString() === "/Sig" || value.has(PDFName.of("ByteRange"))) return true;
    return value
      .entries()
      .some(([, entry]) => pdfObjectContainsSignature(document, entry, visited, depth + 1));
  }
  if (value instanceof PDFArray) {
    return value
      .asArray()
      .some((entry) => pdfObjectContainsSignature(document, entry, visited, depth + 1));
  }
  return false;
}

function assertPdfIsMutable(document: PDFDocument): void {
  if (document.isEncrypted)
    throw metadataError("Encrypted PDF metadata injection is not supported");
  if (pdfObjectContainsSignature(document, document.catalog, new Set())) {
    throw metadataError("Signed PDF metadata injection would invalidate its signature");
  }
}

function newXmpPacket(): string {
  return [
    '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>',
    `<x:xmpmeta xmlns:x="${XMP_META_NAMESPACE}">`,
    `<rdf:RDF xmlns:rdf="${RDF_NAMESPACE}"/>`,
    "</x:xmpmeta>",
    '<?xpacket end="w"?>',
  ].join("");
}

function parseXmp(content: Uint8Array): Document {
  const xml = decodeXml(content, "PDF XMP");
  const document = parseXml(xml, "application/xml", "PDF XMP");
  if (elementsByNamespace(document, RDF_NAMESPACE, "RDF").length !== 1) {
    throw metadataError("PDF XMP requires exactly one rdf:RDF element");
  }
  return document;
}

function ownedXmpMetadata(document: Document): Element[] {
  return elementsByNamespace(document, DSH_NAMESPACE, "provenance");
}

function assertPdfaExtension(document: Document): void {
  if (elementsByNamespace(document, PDFA_ID_NAMESPACE, "part").length === 0) return;
  const declared = elementsByNamespace(document, PDFA_SCHEMA_NAMESPACE, "namespaceURI").some(
    (element) => element.textContent?.trim() === DSH_NAMESPACE,
  );
  if (!declared) {
    throw metadataError("PDF/A requires a declared dsh-science XMP extension schema");
  }
}

function writeXmpMetadata(
  content: Uint8Array | undefined,
  metadata: FigureReproducibilityMetadata,
): Buffer {
  const document = parseXmp(content ?? Buffer.from(newXmpPacket()));
  assertPdfaExtension(document);
  const owned = ownedXmpMetadata(document);
  if (owned.length > 1) throw metadataError("PDF contains duplicate dsh-science metadata records");
  owned[0]?.parentNode?.removeChild(owned[0]);
  const rdf = elementsByNamespace(document, RDF_NAMESPACE, "RDF")[0];
  if (!rdf) throw metadataError("PDF XMP requires one rdf:RDF element");
  const description = document.createElementNS(RDF_NAMESPACE, "rdf:Description");
  description.setAttributeNS(XMLNS_NAMESPACE, "xmlns:dsh", DSH_NAMESPACE);
  description.setAttributeNS(RDF_NAMESPACE, "rdf:about", "");
  const payload = document.createElementNS(DSH_NAMESPACE, "dsh:provenance");
  payload.setAttribute("encoding", "base64-json");
  payload.appendChild(document.createTextNode(encodeXmlMetadata(metadata)));
  description.appendChild(payload);
  rdf.appendChild(description);
  return Buffer.from(new XMLSerializer().serializeToString(document), "utf8");
}

export async function countPdfMetadataRecords(content: Uint8Array): Promise<number> {
  const document = await loadPdf(content);
  const stream = pdfMetadataStream(document);
  return stream ? ownedXmpMetadata(parseXmp(decodePdfMetadataStream(stream))).length : 0;
}

export async function extractPdfMetadata(
  content: Uint8Array,
): Promise<FigureReproducibilityMetadata | undefined> {
  const document = await loadPdf(content);
  const stream = pdfMetadataStream(document);
  if (!stream) return undefined;
  const owned = ownedXmpMetadata(parseXmp(decodePdfMetadataStream(stream)));
  if (owned.length === 0) return undefined;
  if (owned.length !== 1)
    throw metadataError("PDF contains duplicate dsh-science metadata records");
  return decodeXmlMetadata(owned[0]?.textContent ?? "");
}

export async function injectPdfMetadata(
  content: Uint8Array,
  metadata: FigureReproducibilityMetadata,
): Promise<Buffer> {
  const document = await loadPdf(content);
  assertPdfIsMutable(document);
  const existing = pdfMetadataStream(document);
  const xmp = writeXmpMetadata(existing ? decodePdfMetadataStream(existing) : undefined, metadata);
  const stream = document.context.stream(xmp, {
    Type: PDFName.of("Metadata"),
    Subtype: PDFName.of("XML"),
  });
  document.catalog.set(PDFName.of("Metadata"), document.context.register(stream));
  return Buffer.from(
    await document.save({
      addDefaultPage: false,
      objectsPerTick: Number.POSITIVE_INFINITY,
      updateFieldAppearances: false,
      useObjectStreams: false,
    }),
  );
}

export function isArtifactMetadataMime(mime: string): mime is ArtifactMetadataMime {
  return (ARTIFACT_METADATA_MIMES as readonly string[]).includes(mime);
}

export async function injectArtifactMetadata(
  content: Uint8Array,
  mime: ArtifactMetadataMime,
  metadata: FigureReproducibilityMetadata,
): Promise<Buffer> {
  if (mime === "image/png") return injectPngMetadata(content, metadata);
  if (mime === "image/svg+xml") return injectSvgMetadata(content, metadata);
  return injectPdfMetadata(content, metadata);
}

export async function extractArtifactMetadata(
  content: Uint8Array,
  mime: string,
): Promise<FigureReproducibilityMetadata | undefined> {
  if (!isArtifactMetadataMime(mime)) {
    throw metadataError(`Artifact metadata MIME ${mime} is unsupported`);
  }
  if (mime === "image/png") return extractPngMetadata(content);
  if (mime === "image/svg+xml") return extractSvgMetadata(content);
  return extractPdfMetadata(content);
}
