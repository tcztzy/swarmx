# Multimedia attachments and previews

SwarmX Desktop supports typed local attachments end to end: select, drag, or
paste a file into the composer; send it with or without text; reopen it from the
conversation; and inspect supported formats in the persistent right workspace.
The interaction follows the Codex desktop pattern rather than a modal-first
viewer.

## Product research and design choice

Research was refreshed on 2026-07-26 against first-party documentation and the
current desktop products:

- Codex accepts images through upload, drag-and-drop, and paste. OpenAI also
  describes rich document, PDF, spreadsheet, and slide previews in the app's
  sidebar/summary workspace. See the [Codex image-input
  documentation](https://developers.openai.com/codex/app/features#image-inputs)
  and [Codex app announcement](https://openai.com/index/codex-for-almost-everything/).
- Claude Desktop exposes an **Add files or photos** action, attachment cards
  with removal, and an image inspection overlay. Its official help describes
  file upload and generated-file workflows. See [Upload files to
  Claude](https://support.claude.com/en/articles/8241126-upload-files-to-claude)
  and [Claude Desktop](https://code.claude.com/docs/en/desktop).

The selected SwarmX behavior is Codex-first:

1. The composer uses one restrained attachment entry point plus drag and paste.
2. Sent attachments remain assets on the conversation turn.
3. Selecting an asset opens the existing right workspace, preserving the
   conversation beside it.
4. The right workspace is resizable, persistent until closed, keyboard
   reachable, and never replaces the whole conversation with a modal.

Claude's removable attachment card remains a useful composer convention; its
modal-first preview is not the primary SwarmX interaction.

## User-facing support

| Attachment kind | Send from Desktop | Right-workspace preview | External action |
|---|---:|---|---|
| PNG, JPEG, GIF, WebP, SVG, and other detected images | Yes | Contained image | Open / reveal |
| PDF | Yes | Embedded PDF viewer | Open / reveal |
| MP3, WAV, FLAC, OGG, M4A, and other detected audio | Yes | Native audio controls when Electron has a codec | Open / reveal |
| MP4, WebM, MOV, MKV, AVI, and other detected video | Yes | Native video controls when Electron has a codec | Open / reveal |
| Text, Markdown, source, JSON, XML, CSV, YAML, and TOML | Yes | Bounded text preview | Open / reveal |
| Office, archive, and other files | Yes | Metadata card | Open / reveal |

The desktop accepts at most 20 attachments per turn, 100 MiB per attachment,
and 500 MiB total per turn. Text previews read at most 512 KiB. Provider and
protocol inline payloads are additionally limited to 50 MiB per request.

## Transport capability matrix

“Fallback” means SwarmX sends an explicit text/resource reference rather than
silently dropping the attachment.

| Execution path | Images | PDF / documents | Audio | Video |
|---|---|---|---|---|
| ACP 0.22 | Native `image` only when advertised | Embedded `resource` when advertised; otherwise `resource_link` | Native `audio` only when advertised | `resource_link`; ACP has no video content block |
| OpenAI Responses | `input_image` data URL | `input_file` data URL | Fallback in the current SwarmX route | Fallback |
| OpenAI Chat Completions | `image_url` data URL | `file` data URL | `input_audio` for MP3/WAV; other audio falls back | Fallback |
| Anthropic Messages | Base64 image for JPEG/PNG/GIF/WebP | Base64 PDF and inline text-document blocks | Fallback | Fallback |
| Echo backend | Metadata remains on the Session message | Same | Same | Same |

ACP defines `text`, `image`, `audio`, `resource_link`, and embedded `resource`
content blocks. Images, audio, and embedded context are negotiated through
`promptCapabilities`; there is no dedicated video block. See the official
[ACP content types](https://agentclientprotocol.com/protocol/v1/content),
[initialization](https://agentclientprotocol.com/protocol/v1/initialization),
and [prompt turn](https://agentclientprotocol.com/protocol/v1/prompt-turn)
specifications.

For native OpenAI execution, file inputs use base64 data URLs. OpenAI documents
that PDF input combines extracted text with page images, while non-PDF
documents and spreadsheets use format-specific processing. See [OpenAI file
inputs](https://developers.openai.com/api/docs/guides/file-inputs) and [image
and vision inputs](https://developers.openai.com/api/docs/guides/images-vision).
Anthropic image restrictions follow its [vision
documentation](https://docs.anthropic.com/en/docs/build-with-claude/vision).

This matrix describes the implemented SwarmX route, not every modality that a
provider may expose through another endpoint or model. ACP support is
negotiated explicitly. Native Provider mappings describe the selected API
shape; the selected model remains authoritative and may reject a modality it
does not implement. SwarmX surfaces that Provider error instead of claiming the
model understood the previewed file.

## Storage and security boundary

The renderer never receives arbitrary filesystem access. The Electron main
process imports selections and browser `File` bytes into a content-addressed
store:

```text
~/.swarmx/media/<sha256>/<sanitized-name>
```

Session messages persist only canonical metadata and the managed `file:` URI.
Before send, save, preview, open, or reveal, main validates the schema, size,
store containment, real path, and current file size. A dedicated
`swarmx-media://asset/...` protocol exposes only validated managed files to the
renderer; traversal and paths outside the store are rejected.

The original source is copied. Editing or deleting it after attachment does
not mutate an already-imported conversation asset. This also makes retry,
message history, and Session replay deterministic.

Base64 is request-scoped transport only. SwarmX never stores it in Session
JSONL, the Session index, activity records, or external ACP Session binding
metadata.

Writable main tasks reuse one external ACP Session only for text-only turns,
and only while Harness adapter, Model, ModelSupply, Agent profile, and canonical
working directory still match. Editing history, switching that identity, or
sending an attachment invalidates the binding. A new text Session rebuilds
context from canonical text history and attachment metadata, never historical
attachment bytes. Side chats, child Agents, background activations, workflows,
forks, and promoted tasks do not inherit the binding.

An attachment-bearing native Codex ACP turn runs with a private temporary
`CODEX_HOME` below `~/.swarmx/acp-ephemeral/`. SwarmX copies bounded auth/config
inputs and bounded read-only Agent/Skill/rule inputs, points adapter logs into
that same root, and removes the exact root after the request. A later attachment
turn also removes crash residue older than 24 hours. Protected Codex runs use
the existing `container run --rm` boundary instead. This prevents adapter
rollouts containing inline image data from entering the user's long-lived
`~/.codex/sessions`; it intentionally does not delete pre-existing Codex
history.

## Degradation and compatibility

- Unsupported provider modalities are represented by explicit fallback text or
  ACP resource links. They are not presented as if the model consumed their
  bytes.
- A missing, changed, oversized, or forged managed file fails closed with an
  attachment-specific error.
- Audio/video preview depends on codecs bundled with the installed Electron
  build. The **Open** action remains available when inline playback is not.
- General binary files remain sendable for routes that support file blocks,
  but Desktop intentionally previews only metadata and delegates full viewing
  to the operating system.
