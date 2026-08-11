import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type Anthropic from "@anthropic-ai/sdk";
import type OpenAI from "openai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Agent } from "../src/agent.js";
import {
  buildAcpPromptContent,
  createInlineMediaLoader,
  detectMediaMimeType,
  loadMediaAttachment,
  MAX_INLINE_MEDIA_BYTES,
  mediaAttachmentFilePath,
  mediaKindFromMimeType,
  validateMediaAttachments,
} from "../src/media.js";
import {
  callAnthropicMessages,
  callOpenAIResponses,
  type NativeProtocolContext,
} from "../src/native-model.js";
import type { MediaAttachment } from "../src/types.js";

const temporaryDirectories = new Set<string>();

afterEach(async () => {
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("media attachments", () => {
  it("detects common media signatures and classifies canonical kinds", () => {
    expect(
      detectMediaMimeType(
        Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        "image.bin",
      ),
    ).toBe("image/png");
    expect(detectMediaMimeType(Buffer.from("%PDF-1.7"), "report.bin")).toBe("application/pdf");
    expect(mediaKindFromMimeType("audio/wav")).toBe("audio");
    expect(mediaKindFromMimeType("application/json")).toBe("text");
    expect(mediaKindFromMimeType("application/zip")).toBe("file");
  });

  it("covers media URI validation, signatures, extension fallbacks, and byte formatting", async () => {
    expect(mediaAttachmentFilePath("/tmp/notes.txt")).toBe("/tmp/notes.txt");
    expect(mediaAttachmentFilePath("file:///tmp/notes.txt")).toBe("/tmp/notes.txt");
    expect(() => mediaAttachmentFilePath("file:///%zz")).toThrow(/invalid local file URI/i);
    expect(() => mediaAttachmentFilePath("relative.txt")).toThrow(/absolute local path/i);

    expect(detectMediaMimeType(Buffer.from([0xff, 0xd8, 0xff]), "image.bin")).toBe("image/jpeg");
    expect(detectMediaMimeType(Buffer.from("GIF89a"), "image.bin")).toBe("image/gif");
    expect(
      detectMediaMimeType(
        Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WEBP")]),
        "image.bin",
      ),
    ).toBe("image/webp");
    expect(detectMediaMimeType(Buffer.from("OggS"), "audio.bin")).toBe("audio/ogg");
    expect(detectMediaMimeType(Buffer.from("fLaC"), "audio.bin")).toBe("audio/flac");
    expect(detectMediaMimeType(Buffer.from("ID3"), "audio.bin")).toBe("audio/mpeg");
    expect(detectMediaMimeType(Buffer.from([0xff, 0xe0]), "audio.bin")).toBe("audio/mpeg");
    expect(
      detectMediaMimeType(Buffer.concat([Buffer.alloc(4), Buffer.from("ftypm4a ")]), "audio.bin"),
    ).toBe("audio/mp4");
    expect(
      detectMediaMimeType(Buffer.concat([Buffer.alloc(4), Buffer.from("ftypqt  ")]), "video.bin"),
    ).toBe("video/quicktime");
    expect(
      detectMediaMimeType(
        Buffer.concat([Buffer.alloc(4), Buffer.from("ftyp"), Buffer.alloc(4)]),
        "clip.mp4",
      ),
    ).toBe("video/mp4");
    expect(detectMediaMimeType(Buffer.from([0x1a, 0x45, 0xdf, 0xa3]), "clip.webm")).toBe(
      "video/webm",
    );
    expect(detectMediaMimeType(Buffer.from([0x1a, 0x45, 0xdf, 0xa3]), "clip.mkv")).toBe(
      "video/x-matroska",
    );
    expect(detectMediaMimeType(Buffer.from("unknown"), "notes.md")).toBe("text/markdown");
    expect(detectMediaMimeType(Buffer.from("unknown"), "unknown.bin")).toBeNull();

    const root = await temporaryDirectory();
    const attachment = await fixtureAttachment(
      root,
      "tiny.txt",
      Buffer.from("tiny"),
      "text/plain",
      "text",
    );
    await expect(loadMediaAttachment(attachment, 1)).rejects.toThrow(/1 B limit/i);
    await expect(
      loadMediaAttachment({ ...attachment, sizeBytes: attachment.sizeBytes + 1 }),
    ).rejects.toThrow(/changed after it was added/i);
    await expect(
      loadMediaAttachment({ ...attachment, uri: "file:///missing.txt" }),
    ).rejects.toThrow(/no longer available/i);

    expect(() => validateMediaAttachments(undefined)).not.toThrow();
    await expect(createInlineMediaLoader(1024)({ ...attachment, sizeBytes: 1025 })).rejects.toThrow(
      /1 KiB total limit/i,
    );
    await expect(
      createInlineMediaLoader(MAX_INLINE_MEDIA_BYTES + 1)({
        ...attachment,
        sizeBytes: MAX_INLINE_MEDIA_BYTES + 1,
      }),
    ).rejects.toThrow(/50 MiB total limit/i);
  });

  it("bounds MIME sniffing and cumulative inline attachment bytes", async () => {
    expect(
      detectMediaMimeType(Buffer.concat([Buffer.alloc(64), Buffer.from("%PDF-1.7")]), "report.bin"),
    ).toBeNull();

    const root = await temporaryDirectory();
    const first = await fixtureAttachment(
      root,
      "first.txt",
      Buffer.from("abc"),
      "text/plain",
      "text",
    );
    const second = await fixtureAttachment(
      root,
      "second.txt",
      Buffer.from("def"),
      "text/plain",
      "text",
    );
    const loadInline = createInlineMediaLoader(5);
    await expect(loadInline(first)).resolves.toMatchObject({ attachment: first });
    await expect(loadInline(second)).rejects.toThrow(/5 B total limit/i);
  });

  it("maps optional ACP image/audio capabilities and preserves safe resource fallbacks", async () => {
    const root = await temporaryDirectory();
    const image = await fixtureAttachment(
      root,
      "diagram.png",
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]),
      "image/png",
      "image",
    );
    const audio = await fixtureAttachment(
      root,
      "sample.wav",
      Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVE"), Buffer.alloc(4)]),
      "audio/wav",
      "audio",
    );
    const pdf = await fixtureAttachment(
      root,
      "brief.pdf",
      Buffer.from("%PDF-1.7\nfixture"),
      "application/pdf",
      "pdf",
    );

    const capable = await buildAcpPromptContent({
      text: "Inspect these files",
      attachments: [image, audio, pdf],
      promptCapabilities: { image: true, audio: true },
    });
    expect(capable.map((block) => block.type)).toEqual(["text", "image", "audio", "resource_link"]);
    expect(capable[1]).toMatchObject({ mimeType: "image/png", uri: image.uri });
    expect(capable[2]).toMatchObject({ mimeType: "audio/wav" });
    expect(capable[3]).toMatchObject({
      type: "resource_link",
      uri: pdf.uri,
      name: "brief.pdf",
    });

    const embedded = await buildAcpPromptContent({
      text: "",
      attachments: [pdf],
      promptCapabilities: { embeddedContext: true },
    });
    expect(embedded[1]).toMatchObject({
      type: "resource",
      resource: {
        uri: pdf.uri,
        mimeType: "application/pdf",
      },
    });
  });

  it("rejects files that changed after attachment metadata was captured", async () => {
    const root = await temporaryDirectory();
    const attachment = await fixtureAttachment(
      root,
      "note.txt",
      Buffer.from("first"),
      "text/plain",
      "text",
    );
    await writeFile(path.join(root, "note.txt"), "longer second value");

    await expect(loadMediaAttachment(attachment)).rejects.toThrow(/changed after it was added/i);
  });

  it("validates unknown native-message attachments without a trusted type assertion", () => {
    const untrusted: readonly unknown[] = [
      {
        id: "forged",
        name: "forged.png",
        kind: "image",
        mimeType: "image/png",
        sizeBytes: "not-a-number",
        uri: "file:///tmp/forged.png",
        source: "user",
      },
    ];

    expect(() => validateMediaAttachments(untrusted)).toThrow();
  });

  it("maps image and file inputs for OpenAI Responses with explicit unsupported fallbacks", async () => {
    const root = await temporaryDirectory();
    const image = await fixtureAttachment(
      root,
      "diagram.png",
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 4]),
      "image/png",
      "image",
    );
    const pdf = await fixtureAttachment(
      root,
      "brief.pdf",
      Buffer.from("%PDF-1.7\nfixture"),
      "application/pdf",
      "pdf",
    );
    const video = await fixtureAttachment(
      root,
      "clip.mp4",
      Buffer.concat([Buffer.alloc(4), Buffer.from("ftyp"), Buffer.from("isom")]),
      "video/mp4",
      "video",
    );
    const requests: Array<Record<string, unknown>> = [];
    const context = nativeContext({
      openai: {
        responses: {
          create: async (request: Record<string, unknown>) => {
            requests.push(request);
            return openAIResponse();
          },
        },
      } as unknown as OpenAI,
    });

    await callOpenAIResponses(context, {
      messages: [{ role: "user", content: "", attachments: [image, pdf, video] }],
    });

    const input = requests[0]?.input as Array<{ role?: string; content?: unknown[] }>;
    expect(input[0]?.content).not.toContainEqual({ type: "input_text", text: "" });
    expect(input[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "input_image",
          image_url: expect.stringMatching(/^data:image\/png;base64,/),
        }),
        expect.objectContaining({
          type: "input_file",
          filename: "brief.pdf",
          file_data: expect.stringMatching(/^data:application\/pdf;base64,/),
        }),
        expect.objectContaining({
          type: "input_text",
          text: expect.stringContaining("no native video input"),
        }),
      ]),
    );
  });

  it("maps supported Anthropic image, PDF, and text document blocks", async () => {
    const root = await temporaryDirectory();
    const image = await fixtureAttachment(
      root,
      "photo.webp",
      Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WEBP"), Buffer.alloc(4)]),
      "image/webp",
      "image",
    );
    const pdf = await fixtureAttachment(
      root,
      "brief.pdf",
      Buffer.from("%PDF-1.7\nfixture"),
      "application/pdf",
      "pdf",
    );
    const text = await fixtureAttachment(
      root,
      "notes.txt",
      Buffer.from("source notes"),
      "text/plain",
      "text",
    );
    const requests: Array<Record<string, unknown>> = [];
    const context = nativeContext({
      anthropic: {
        messages: {
          create: async (request: Record<string, unknown>) => {
            requests.push(request);
            return anthropicResponse();
          },
        },
      } as unknown as Anthropic,
    });

    await callAnthropicMessages(context, {
      messages: [{ role: "user", content: "", attachments: [image, pdf, text] }],
    });

    const messages = requests[0]?.messages as Array<{ content: unknown[] }>;
    expect(messages[0]?.content).not.toContainEqual({ type: "text", text: "" });
    expect(messages[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "image",
          source: expect.objectContaining({ type: "base64", media_type: "image/webp" }),
        }),
        expect.objectContaining({
          type: "document",
          source: expect.objectContaining({ type: "base64", media_type: "application/pdf" }),
        }),
        expect.objectContaining({
          type: "document",
          source: { type: "text", media_type: "text/plain", data: "source notes" },
        }),
      ]),
    );
  });

  it("maps image, audio, and file blocks for OpenAI Chat while degrading video", async () => {
    const root = await temporaryDirectory();
    const image = await fixtureAttachment(
      root,
      "diagram.png",
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 4]),
      "image/png",
      "image",
    );
    const audio = await fixtureAttachment(
      root,
      "sample.wav",
      Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WAVE"), Buffer.alloc(4)]),
      "audio/wav",
      "audio",
    );
    const pdf = await fixtureAttachment(
      root,
      "brief.pdf",
      Buffer.from("%PDF-1.7\nfixture"),
      "application/pdf",
      "pdf",
    );
    const video = await fixtureAttachment(
      root,
      "clip.mp4",
      Buffer.concat([Buffer.alloc(4), Buffer.from("ftyp"), Buffer.from("isom")]),
      "video/mp4",
      "video",
    );
    const agent = new Agent({ name: "chat_media", model: "gpt-media" });
    const create = vi.fn(async (_request: { messages: Array<{ content: unknown }> }) => ({
      async *[Symbol.asyncIterator]() {
        yield { choices: [{ delta: { content: "Reviewed." } }] };
      },
    }));
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    await agent.callStream(
      {
        messages: [{ role: "user", content: "", attachments: [image, audio, pdf, video] }],
      },
      () => undefined,
    );

    const messages = create.mock.calls[0]?.[0]?.messages as Array<{ content: unknown[] }>;
    expect(messages[0]?.content).not.toContainEqual({ type: "text", text: "" });
    expect(messages[0]?.content).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "image_url",
          image_url: expect.objectContaining({
            url: expect.stringMatching(/^data:image\/png;base64,/),
          }),
        }),
        expect.objectContaining({
          type: "input_audio",
          input_audio: expect.objectContaining({ format: "wav" }),
        }),
        expect.objectContaining({
          type: "file",
          file: expect.objectContaining({ filename: "brief.pdf" }),
        }),
        expect.objectContaining({
          type: "text",
          text: expect.stringContaining("no native video input"),
        }),
      ]),
    );
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "swarmx-media-test-"));
  temporaryDirectories.add(directory);
  return directory;
}

async function fixtureAttachment(
  root: string,
  name: string,
  bytes: Buffer,
  mimeType: string,
  kind: MediaAttachment["kind"],
): Promise<MediaAttachment> {
  const filePath = path.join(root, name);
  await writeFile(filePath, bytes);
  return {
    id: name,
    name,
    kind,
    mimeType,
    sizeBytes: bytes.byteLength,
    uri: pathToFileURL(filePath).href,
    source: "user",
  };
}

function nativeContext(
  clients: Partial<Pick<NativeProtocolContext, "openai" | "anthropic">>,
): NativeProtocolContext {
  return {
    agentName: "media-agent",
    model: "media-model",
    instructions: "",
    parameters: {},
    maxOutputTokens: 256,
    apiMode: "standard",
    openai: (clients.openai ?? {}) as OpenAI,
    anthropic: (clients.anthropic ?? {}) as Anthropic,
    providerHostedWebSearch: false,
    tools: [],
    async callTool() {
      throw new Error("No tools expected");
    },
  };
}

function openAIResponse() {
  return {
    id: "resp_media",
    status: "completed",
    error: null,
    usage: null,
    output: [
      {
        id: "msg_media",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [{ type: "output_text", text: "Reviewed.", annotations: [] }],
      },
    ],
  } as never;
}

function anthropicResponse() {
  return {
    id: "msg_media",
    type: "message",
    role: "assistant",
    model: "media-model",
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: {},
    content: [{ type: "text", text: "Reviewed." }],
  } as never;
}
