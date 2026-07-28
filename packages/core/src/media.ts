import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { ContentBlock, PromptCapabilities } from "@agentclientprotocol/sdk";
import { MediaAttachmentSchema } from "./types.js";
import type { MediaAttachment, MediaAttachmentKind } from "./types.js";

export const MAX_MEDIA_ATTACHMENT_BYTES = 100 * 1024 * 1024;
export const MAX_MEDIA_ATTACHMENTS = 20;
export const MAX_MEDIA_TURN_BYTES = 500 * 1024 * 1024;
export const MAX_INLINE_MEDIA_BYTES = 50 * 1024 * 1024;
export const MAX_INLINE_TEXT_DOCUMENT_BYTES = 5 * 1024 * 1024;
export const MEDIA_SNIFF_BYTES = 64;

export interface LoadedMediaAttachment {
  attachment: MediaAttachment;
  bytes: Buffer;
  base64: string;
}

export type InlineMediaLoader = (
  attachment: MediaAttachment,
  capBytes?: number,
) => Promise<LoadedMediaAttachment>;

/** Budgeted loader: tracks cumulative inline bytes across attachments for one turn. */
export function createInlineMediaLoader(budgetBytes = MAX_INLINE_MEDIA_BYTES): InlineMediaLoader {
  let loadedBytes = 0;
  return async (attachment, capBytes) => {
    const remaining = Math.max(0, budgetBytes - loadedBytes);
    const loaded = await loadMediaAttachment(
      attachment,
      capBytes === undefined ? remaining : Math.min(capBytes, remaining),
    );
    loadedBytes += loaded.bytes.byteLength;
    return loaded;
  };
}

export interface AcpPromptContentInput {
  text: string;
  attachments?: readonly MediaAttachment[];
  promptCapabilities?: PromptCapabilities | null;
  meta?: Record<string, unknown>;
}

export async function loadMediaAttachment(
  input: MediaAttachment,
  maxBytes = MAX_MEDIA_ATTACHMENT_BYTES,
): Promise<LoadedMediaAttachment> {
  const attachment = MediaAttachmentSchema.parse(input);
  const filePath = mediaAttachmentFilePath(attachment.uri);
  const info = await stat(filePath).catch(() => null);
  if (!info?.isFile()) {
    throw new Error(`Attachment "${attachment.name}" is no longer available.`);
  }
  if (info.size !== attachment.sizeBytes) {
    throw new Error(`Attachment "${attachment.name}" changed after it was added.`);
  }
  if (info.size > maxBytes) {
    throw new Error(`Attachment "${attachment.name}" exceeds the ${formatBytes(maxBytes)} limit.`);
  }
  const bytes = await readFile(filePath);
  if (bytes.byteLength !== attachment.sizeBytes) {
    throw new Error(`Attachment "${attachment.name}" changed while it was being read.`);
  }
  const detectedMime = detectMediaMimeType(bytes, attachment.name);
  if (detectedMime && detectedMime !== attachment.mimeType) {
    throw new Error(`Attachment "${attachment.name}" no longer matches its detected media type.`);
  }
  return { attachment, bytes, base64: bytes.toString("base64") };
}

export async function buildAcpPromptContent(input: AcpPromptContentInput): Promise<ContentBlock[]> {
  const attachments = validateMediaAttachments(input.attachments);
  const prompt: ContentBlock[] = [
    {
      type: "text",
      text: input.text,
      ...(input.meta && Object.keys(input.meta).length > 0 ? { _meta: input.meta } : {}),
    },
  ];
  const loadInline = createInlineMediaLoader();

  for (const attachment of attachments) {
    if (attachment.kind === "image" && input.promptCapabilities?.image) {
      const loaded = await loadInline(attachment);
      prompt.push({
        type: "image",
        data: loaded.base64,
        mimeType: attachment.mimeType,
        uri: attachment.uri,
      });
      continue;
    }
    if (attachment.kind === "audio" && input.promptCapabilities?.audio) {
      const loaded = await loadInline(attachment);
      prompt.push({
        type: "audio",
        data: loaded.base64,
        mimeType: attachment.mimeType,
      });
      continue;
    }
    if (input.promptCapabilities?.embeddedContext) {
      const loaded = await loadInline(attachment);
      prompt.push({
        type: "resource",
        resource: {
          uri: attachment.uri,
          blob: loaded.base64,
          mimeType: attachment.mimeType,
        },
      });
      continue;
    }
    prompt.push({
      type: "resource_link",
      uri: attachment.uri,
      name: attachment.name,
      title: attachment.name,
      mimeType: attachment.mimeType,
      size: attachment.sizeBytes,
      description:
        attachment.kind === "video"
          ? "Video attachment. ACP has no dedicated video content block."
          : `${attachment.kind} attachment`,
    });
  }

  return prompt;
}

export function mediaAttachmentFilePath(uri: string): string {
  if (uri.startsWith("file:")) {
    try {
      return fileURLToPath(uri);
    } catch {
      throw new Error("Attachment has an invalid local file URI.");
    }
  }
  if (path.isAbsolute(uri)) return uri;
  throw new Error("Attachment URI must be an absolute local path or file URI.");
}

export function detectMediaMimeType(bytes: Uint8Array, filename = ""): string | null {
  if (startsWith(bytes, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return "image/png";
  if (startsWith(bytes, [0xff, 0xd8, 0xff])) return "image/jpeg";
  if (ascii(bytes, 0, 6) === "GIF87a" || ascii(bytes, 0, 6) === "GIF89a") return "image/gif";
  if (ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WEBP") return "image/webp";
  if (ascii(bytes, 0, 5) === "%PDF-") return "application/pdf";
  if (ascii(bytes, 0, 4) === "OggS") return "audio/ogg";
  if (ascii(bytes, 0, 4) === "fLaC") return "audio/flac";
  if (ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 12) === "WAVE") return "audio/wav";
  if (ascii(bytes, 0, 3) === "ID3" || looksLikeMp3Frame(bytes)) return "audio/mpeg";
  if (ascii(bytes, 4, 8) === "ftyp") return isoBaseMediaMime(bytes, filename);
  if (startsWith(bytes, [0x1a, 0x45, 0xdf, 0xa3])) {
    return fileExtension(filename) === ".webm" ? "video/webm" : "video/x-matroska";
  }
  return mimeFromExtension(filename);
}

export function mediaKindFromMimeType(mimeType: string): MediaAttachmentKind {
  if (mimeType.startsWith("image/")) return "image";
  if (mimeType === "application/pdf") return "pdf";
  if (mimeType.startsWith("audio/")) return "audio";
  if (mimeType.startsWith("video/")) return "video";
  if (
    mimeType.startsWith("text/") ||
    mimeType === "application/json" ||
    mimeType === "application/xml" ||
    mimeType === "application/javascript" ||
    mimeType === "application/typescript"
  ) {
    return "text";
  }
  return "file";
}

export function attachmentFallbackText(attachment: MediaAttachment): string {
  const note =
    attachment.kind === "video"
      ? "Video content is available as a local resource; this provider path has no native video input."
      : "This provider path has no native input block for this attachment.";
  return `[Attachment: ${attachment.name}; type=${attachment.mimeType}; size=${attachment.sizeBytes} bytes; uri=${attachment.uri}. ${note}]`;
}

export function validateMediaAttachments(
  attachments: readonly unknown[] | undefined,
): MediaAttachment[] {
  if (!attachments) return [];
  if (attachments.length > MAX_MEDIA_ATTACHMENTS) {
    throw new Error(`A prompt can include at most ${MAX_MEDIA_ATTACHMENTS} attachments.`);
  }
  const validated = attachments.map((attachment) => MediaAttachmentSchema.parse(attachment));
  for (const attachment of validated) {
    if (attachment.sizeBytes > MAX_MEDIA_ATTACHMENT_BYTES) {
      throw new Error(`Attachment "${attachment.name}" exceeds the 100 MiB limit.`);
    }
  }
  if (
    validated.reduce((total, attachment) => total + attachment.sizeBytes, 0) > MAX_MEDIA_TURN_BYTES
  ) {
    throw new Error("Attachments must total 500 MiB or less.");
  }
  return validated;
}

function startsWith(bytes: Uint8Array, signature: readonly number[]): boolean {
  return signature.every((value, index) => bytes[index] === value);
}

function ascii(bytes: Uint8Array, start: number, end: number): string {
  return Buffer.from(bytes.subarray(start, end)).toString("latin1");
}

function looksLikeMp3Frame(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0xff && (bytes[1] & 0xe0) === 0xe0;
}

function isoBaseMediaMime(bytes: Uint8Array, filename: string): string {
  const brand = ascii(bytes, 8, 12).toLowerCase();
  if (brand === "m4a " || brand === "m4b ") return "audio/mp4";
  if (brand === "qt  ") return "video/quicktime";
  return fileExtension(filename) === ".m4a" ? "audio/mp4" : "video/mp4";
}

function mimeFromExtension(filename: string): string | null {
  return EXTENSION_MIME_TYPES[fileExtension(filename)] ?? null;
}

function fileExtension(filename: string): string {
  return path.extname(filename).toLowerCase();
}

function formatBytes(bytes: number): string {
  return `${Math.floor(bytes / (1024 * 1024))} MiB`;
}

const EXTENSION_MIME_TYPES: Readonly<Record<string, string>> = {
  ".aac": "audio/aac",
  ".avi": "video/x-msvideo",
  ".c": "text/x-c",
  ".cc": "text/x-c++",
  ".cpp": "text/x-c++",
  ".css": "text/css",
  ".csv": "text/csv",
  ".doc": "application/msword",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".flac": "audio/flac",
  ".gif": "image/gif",
  ".html": "text/html",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".js": "application/javascript",
  ".json": "application/json",
  ".m4a": "audio/mp4",
  ".md": "text/markdown",
  ".mkv": "video/x-matroska",
  ".mov": "video/quicktime",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
  ".ogg": "audio/ogg",
  ".pdf": "application/pdf",
  ".png": "image/png",
  ".ppt": "application/vnd.ms-powerpoint",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ".py": "text/x-python",
  ".rtf": "application/rtf",
  ".svg": "image/svg+xml",
  ".toml": "application/toml",
  ".ts": "application/typescript",
  ".tsx": "text/tsx",
  ".txt": "text/plain",
  ".wav": "audio/wav",
  ".webm": "video/webm",
  ".webp": "image/webp",
  ".xls": "application/vnd.ms-excel",
  ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  ".xml": "application/xml",
  ".yaml": "application/yaml",
  ".yml": "application/yaml",
};
