import { File, FileAudio, FileImage, FileText, FileVideo, type LucideIcon } from "lucide-react";
import type { DesktopMediaAttachment } from "../../shared/desktop-api.js";

export function MessageAttachments({
  attachments,
  onPreview,
}: {
  attachments: DesktopMediaAttachment[];
  onPreview?: (attachment: DesktopMediaAttachment) => void;
}) {
  return (
    <div className="message-attachments" aria-label="Message attachments">
      {attachments.map((attachment) => (
        <button
          key={attachment.id}
          type="button"
          className="message-attachment"
          onClick={() => onPreview?.(attachment)}
          disabled={!onPreview}
          aria-label={`Preview ${attachment.name}`}
        >
          <AttachmentIcon attachment={attachment} />
          <span>
            <strong>{attachment.name}</strong>
            <small>
              {attachment.kind} · {formatMediaBytes(attachment.sizeBytes)}
            </small>
          </span>
        </button>
      ))}
    </div>
  );
}

export function AttachmentIcon({ attachment }: { attachment: DesktopMediaAttachment }) {
  const Icon = attachmentIcon(attachment);
  return <Icon aria-hidden="true" />;
}

function attachmentIcon(attachment: DesktopMediaAttachment): LucideIcon {
  if (attachment.kind === "image") return FileImage;
  if (attachment.kind === "audio") return FileAudio;
  if (attachment.kind === "video") return FileVideo;
  if (attachment.kind === "pdf" || attachment.kind === "text") return FileText;
  return File;
}

export function formatMediaBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0)} MB`;
}
